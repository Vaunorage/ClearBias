import time
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

import multiprocessing as mp
from scipy.optimize import basinhopping
import errno

from tqdm import tqdm

from paths import HERE


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def my_queue_get(queue, block=True, timeout=None):
    while True:
        try:
            return queue.get(block, timeout)
        except IOError as e:
            if e.errno != errno.EINTR:
                raise


def worker(fully_direct, local_inputs, minimizer, local_iteration_limit):
    results = []
    for inp in local_inputs:
        basinhopping(fully_direct.evaluate_local, inp, stepsize=1.0, take_step=fully_direct.local_perturbation,
                     minimizer_kwargs=minimizer, niter=local_iteration_limit)
    results.append([fully_direct.local_disc_inputs, fully_direct.local_disc_inputs_list, fully_direct.tot_inputs])
    return results


def mp_basinhopping(fully_direct, minimizer, local_iteration_limit):
    divided_lists = chunks(fully_direct.global_disc_inputs_list["discr_input"], 4)
    results = []

    # Adding tqdm to the loop for progress tracking
    for inputs in tqdm(list(divided_lists), desc="Processing inputs"):
        result = worker(fully_direct, inputs, minimizer, local_iteration_limit)
        results.extend(result)

    local_inputs = set()
    local_inputs_list = {"discr_input": [], "counter_discr_input": [], 'magnitude': []}
    tot_inputs_out = set()

    # Looping through results to collect data
    for res in results:
        set_inputs, list_inputs, tot_inputs = res
        for item in set_inputs:
            local_inputs.add(item)
        for k, item in enumerate(list_inputs['discr_input']):
            if item not in local_inputs_list['discr_input']:
                local_inputs_list['discr_input'].append(item)
                local_inputs_list['counter_discr_input'].append(list_inputs['counter_discr_input'][k])
                local_inputs_list['magnitude'].append(list_inputs['magnitude'][k])
        for item in tot_inputs:
            tot_inputs_out.add(item)

    fully_direct.local_disc_inputs = local_inputs
    fully_direct.local_disc_inputs_list = local_inputs_list
    fully_direct.tot_inputs = tot_inputs_out

    return fully_direct


class Dataset:
    def __init__(self, df: pd.DataFrame, col_to_be_predicted, sensitive_param_name_list=None):
        self.sensitive_param_name_list = sensitive_param_name_list

        self.num_params = df.shape[1] - 1

        self.col_to_be_predicted = col_to_be_predicted

        self.df = df
        self.column_names = self.get_column_names()
        self.input_bounds = self.get_input_bounds()
        self.col_to_be_predicted_idx = self.get_idx_of_col_to_be_predicted(col_to_be_predicted)

        self.sensitive_param_idx_list = [df.columns.get_loc(name) for name in self.sensitive_param_name_list if
                                         name in df.columns]

    def get_idx_of_col_to_be_predicted(self, col_to_be_predicted):
        return list(self.df.columns).index(col_to_be_predicted)

    def get_column_names(self):
        return list(self.df.columns)

    def get_input_bounds(self):
        input_bounds = []
        for col in self.df:
            numUniqueVals = self.df[col].nunique()
            input_bounds.append([0, numUniqueVals - 1])  # bound is inclusive
        return input_bounds


class Fully_Direct:
    def __init__(self, dataset: Dataset, perturbation_unit, threshold, global_iteration_limit, local_iteration_limit,
                 sensitive_param_idx, model):
        random.seed(time.time())
        self.start_time = time.time()

        self.column_names = dataset.column_names
        self.num_params = dataset.num_params
        self.input_bounds = dataset.input_bounds
        self.sensitive_param_idx = sensitive_param_idx
        self.col_to_be_predicted_idx = dataset.col_to_be_predicted_idx

        self.perturbation_unit = perturbation_unit
        self.threshold = threshold
        self.global_iteration_limit = global_iteration_limit
        self.local_iteration_limit = local_iteration_limit

        self.init_prob = 0.5
        self.cov = 0
        self.direction_probability_change_size = 0.001
        self.param_probability_change_size = 0.001

        self.direction_probability = [self.init_prob] * len(self.input_bounds)
        self.direction_probability[self.col_to_be_predicted_idx] = 0  # nullify the y col
        self.param_probability = [1.0 / self.num_params] * len(self.input_bounds)
        self.param_probability[self.col_to_be_predicted_idx] = 0

        self.global_disc_inputs = set()
        self.global_disc_inputs_list = {"discr_input": [], "counter_discr_input": [], 'magnitude': []}

        self.local_disc_inputs = set()
        self.local_disc_inputs_list = {"discr_input": [], "counter_discr_input": [], 'magnitude': []}

        self.tot_inputs = set()

        self.model = model

    def normalise_probability(self):
        probability_sum = 0.0
        for prob in self.param_probability:
            probability_sum = probability_sum + prob

        for i in range(self.num_params):
            self.param_probability[i] = float(self.param_probability[i]) / float(probability_sum)

    def evaluate_input(self, inp):
        inp = np.array([int(k) for k in inp])
        sensValue = inp[self.sensitive_param_idx]
        inp0 = np.delete(inp, self.col_to_be_predicted_idx).reshape(1, -1)
        out0 = self.model.predict(inp0)

        for i in range(self.input_bounds[self.sensitive_param_idx][1] + 1):
            if sensValue != i:
                modified_inp = inp.copy()
                modified_inp[self.sensitive_param_idx] = i
                modified_inp = np.delete(modified_inp, self.col_to_be_predicted_idx).reshape(1, -1)
                out1 = self.model.predict(modified_inp)

                if abs(out1 - out0) > self.threshold:
                    return abs(out1 - out0)

        return False

    def evaluate_global(self, inp):
        inp0 = [int(i) for i in inp]
        sensValue = inp0[self.sensitive_param_idx]

        inp0np = np.reshape(np.asarray(inp0), (1, -1))
        self.tot_inputs.add(tuple(map(tuple, inp0np)))

        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))

        # Returns early if input is already in the global discriminatory inputs set
        if (tuple(map(tuple, inp0)) in self.global_disc_inputs):
            return 0

        inp0delY = np.delete(inp0, [self.col_to_be_predicted_idx])
        inp0delY = np.reshape(inp0delY, (1, -1))
        out0 = self.model.predict(inp0delY)

        # Loops through all values of the sensitive parameter
        for i in range(self.input_bounds[self.sensitive_param_idx][1] + 1):
            if i != sensValue:
                inp1 = [int(k) for k in inp]
                inp1[self.sensitive_param_idx] = i
                inp1 = np.reshape(np.asarray(inp1), (1, -1))

                # drop y column here
                inp1delY = np.delete(inp1, [self.col_to_be_predicted_idx])
                inp1delY = np.reshape(inp1delY, (1, -1))
                out1 = self.model.predict(inp1delY)
                diff = abs(out0 - out1)

                if (diff > self.threshold):
                    self.global_disc_inputs.add(tuple(map(tuple, inp0)))  # add the entire input, including original y
                    self.global_disc_inputs_list['discr_input'].append(inp0.tolist()[0])
                    self.global_disc_inputs_list['counter_discr_input'].append(inp1.tolist()[0])
                    self.global_disc_inputs_list['magnitude'].append(diff)
                    return diff
        return 0

    def evaluate_local(self, inp):
        inp0 = [int(i) for i in inp]
        sensValue = inp0[self.sensitive_param_idx]

        inp0np = np.reshape(np.asarray(inp0), (1, -1))

        self.tot_inputs.add(tuple(map(tuple, inp0np)))

        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))

        # Returns early if input is already in the global or local discriminatory inputs set
        if ((tuple(map(tuple, inp0)) in self.global_disc_inputs) or (
                tuple(map(tuple, inp0)) in self.local_disc_inputs)):
            return 0

        inp0delY = np.delete(inp0, [self.col_to_be_predicted_idx])
        inp0delY = np.reshape(inp0delY, (1, -1))
        out0 = self.model.predict(inp0delY)

        # Loops through all values of the sensitive parameter
        for i in range(self.input_bounds[self.sensitive_param_idx][1] + 1):
            if sensValue != i:
                inp1 = [int(k) for k in inp]
                inp1[self.sensitive_param_idx] = i
                inp1 = np.asarray(inp1)
                inp1 = np.reshape(inp1, (1, -1))

                # drop y column here
                inp1delY = np.delete(inp1, [self.col_to_be_predicted_idx])
                inp1delY = np.reshape(inp1delY, (1, -1))

                out1 = self.model.predict(inp1delY)

                diff = abs(out0 - out1)

                if diff > self.threshold:
                    self.local_disc_inputs.add(tuple(map(tuple, inp0)))
                    self.local_disc_inputs_list['discr_input'].append(inp0.tolist()[0])
                    self.local_disc_inputs_list['counter_discr_input'].append(inp1.tolist()[0])
                    self.local_disc_inputs_list['magnitude'].append(diff)
                    return diff
        return 0

    def global_discovery(self, x):
        try:
            random.seed(time.time())
            x = [random.randint(low, high) for [low, high] in self.input_bounds]
            x[self.sensitive_param_idx] = 0
            return x
        except:  # unknown error
            return x

    def local_perturbation(self, x):
        columns = [i for i in range(len(self.input_bounds))]  # we're only perturbing non-y columns right?
        param_choice = np.random.choice(columns, p=self.param_probability)
        act = [-1, 1]
        direction_choice = np.random.choice(act, p=[self.direction_probability[param_choice],
                                                    (1 - self.direction_probability[param_choice])])

        if (x[param_choice] == self.input_bounds[param_choice][0]) \
                or (x[param_choice] == self.input_bounds[param_choice][1]):
            direction_choice = np.random.choice(act)

        x[param_choice] = x[param_choice] + (direction_choice * self.perturbation_unit)

        x[param_choice] = max(self.input_bounds[param_choice][0], x[param_choice])
        x[param_choice] = min(self.input_bounds[param_choice][1], x[param_choice])

        ei = self.evaluate_input(x)

        if (ei and direction_choice == -1) or (not ei and direction_choice == 1):
            self.direction_probability[param_choice] = min(
                self.direction_probability[param_choice]
                + (self.direction_probability_change_size * self.perturbation_unit), 1)

        elif (not ei and direction_choice == -1) or (ei and direction_choice == 1):
            self.direction_probability[param_choice] = max(
                self.direction_probability[param_choice]
                - (self.direction_probability_change_size * self.perturbation_unit), 0)

        if ei:
            self.param_probability[param_choice] = self.param_probability[param_choice] + \
                                                   self.param_probability_change_size
            self.normalise_probability()
        else:
            self.param_probability[param_choice] = max(
                self.param_probability[param_choice] - self.param_probability_change_size, 0)
            self.normalise_probability()

        return x


def generate_sklearn_classifier(dataset: Dataset, model_type: str):
    le = LabelEncoder()

    col_to_be_predicted = dataset.col_to_be_predicted

    df = dataset.df
    cat_feature = list(df.columns)

    for col in cat_feature:
        df.loc[:, col] = le.fit_transform(df[col])

    X = df.drop([col_to_be_predicted], axis=1)
    y = df[col_to_be_predicted]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    if model_type == "DecisionTree":
        model = DecisionTreeClassifier(random_state=42, criterion='entropy', splitter='random')
        model_name = 'DecisionTreeClassifier'
    elif model_type == "MLPC":
        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7, 5), random_state=1)
        model_name = 'MLPClassifier'
    elif model_type == "SVM":
        model = SVC(gamma=0.0025)
        model_name = 'SVC'
    elif model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=4)
        model_name = 'RandomForestClassifier'
    else:
        error_message = 'The chosen types of model is not supported yet. Please choose from one of the following: \
                            DecisionTree, MLPC, SVM and RandomForest'
        raise ValueError(error_message)

    model.fit(X_train.values, y_train.values)
    pred = model.predict(X_test.values)

    scores = []
    ss = {
        'model': model_name,
        'score': model.score(X_test, y_test),
        'f1_score': f1_score(y_test, pred, average='weighted')
    }
    print(ss)
    scores.append(ss)

    return model, scores


def collect_results(fully_direct, sensitive_attribute):
    # Calculate percentage of discriminatory inputs
    total_disc_inputs = len(fully_direct.global_disc_inputs_list["discr_input"]) + len(
        fully_direct.local_disc_inputs_list["discr_input"])
    percentage_disc_inputs = total_disc_inputs / len(fully_direct.tot_inputs) * 100 if fully_direct.tot_inputs else 0

    # Collect results in a dictionary, now including the sensitive attribute and detailed lists of discriminatory inputs
    results = {
        "Sensitive Attribute": sensitive_attribute,
        "Total Inputs": len(fully_direct.tot_inputs),
        "Discriminatory Inputs": total_disc_inputs,
        "Percentage Discriminatory Inputs": percentage_disc_inputs,
        "Global Discriminatory Inputs": fully_direct.global_disc_inputs_list["discr_input"],
        "Global Counter Discriminatory Inputs": fully_direct.global_disc_inputs_list["counter_discr_input"],
        "Global Magnitude": fully_direct.global_disc_inputs_list["magnitude"],
        "Local Discriminatory Inputs": fully_direct.local_disc_inputs_list["discr_input"],
        "Local Counter Discriminatory Inputs": fully_direct.local_disc_inputs_list["counter_discr_input"],
        "Local Magnitude": fully_direct.local_disc_inputs_list["magnitude"],
    }
    return results


def aequitas_fully_directed_sklearn(dataset, perturbation_unit, threshold, global_iteration_limit,
                                    local_iteration_limit, sensitive_param_id, model, sensitive_attribute):
    print(f"Aequitas Fully Directed Started for {sensitive_attribute}...\n")
    initial_input = [random.randint(low, high) for [low, high] in dataset.input_bounds]
    minimizer = {"method": "L-BFGS-B"}

    fully_direct = Fully_Direct(dataset, perturbation_unit, threshold, global_iteration_limit, local_iteration_limit,
                                sensitive_param_id, model)

    basinhopping(fully_direct.evaluate_global, initial_input, stepsize=1.0, take_step=fully_direct.global_discovery,
                 minimizer_kwargs=minimizer, niter=global_iteration_limit)
    print("Finished Global Search")
    global_results = collect_results(fully_direct, sensitive_attribute)

    fully_direct = mp_basinhopping(fully_direct, minimizer, local_iteration_limit)
    print("Local Search Finished")
    local_results = collect_results(fully_direct, sensitive_attribute)

    return {**global_results, **local_results}


def run_aequitas(df, col_to_be_predicted, sensitive_param_name_list, perturbation_unit, model_type, threshold=0,
                 global_iteration_limit=1000, local_iteration_limit=100):
    results = []
    dataset = Dataset(df, col_to_be_predicted=col_to_be_predicted, sensitive_param_name_list=sensitive_param_name_list)
    model, model_scores = generate_sklearn_classifier(dataset, model_type)
    for sensitive_param_id, sensitive_attribute in zip(dataset.sensitive_param_idx_list, sensitive_param_name_list):
        result = aequitas_fully_directed_sklearn(dataset, perturbation_unit, threshold, global_iteration_limit,
                                                 local_iteration_limit, sensitive_param_id, model, sensitive_attribute)
        results.append(result)

    results_df = pd.DataFrame(results)

    ress_df = []
    for row in results_df.iterrows():
        sub1_discr = pd.DataFrame(row[1]['Local Discriminatory Inputs'], columns=df.columns)

        subgroup_num = [f"{row[0]}{e}" for e in range(sub1_discr.shape[0])]

        sub1_discr['subgroup_num'] = subgroup_num
        sub1_discr['diff_outcome'] = np.array(row[1]['Local Magnitude'])

        sub2_discr = pd.DataFrame(row[1]['Local Counter Discriminatory Inputs'], columns=df.columns)
        sub2_discr['subgroup_num'] = subgroup_num
        sub2_discr['diff_outcome'] = np.array(row[1]['Local Magnitude'])

        res_df = pd.concat([sub1_discr.reset_index(drop=True), sub2_discr.reset_index(drop=True)])

        res_df['Sensitive Attribute'] = row[1]['Sensitive Attribute']
        res_df['Total Inputs'] = row[1]['Total Inputs']
        res_df['Discriminatory Inputs'] = row[1]['Discriminatory Inputs']
        res_df['Percentage Discriminatory Inputs'] = row[1]['Percentage Discriminatory Inputs']

        ress_df.append(res_df.reset_index(drop=True))

    res = pd.concat(ress_df)

    res = res.sort_values(['Sensitive Attribute', 'subgroup_num'])
    res['subgroup_id'] = res.groupby(['subgroup_num']).cumcount() + 1

    attr = list(df.columns)
    attr.remove(col_to_be_predicted)
    res['indv_key'] = res.apply(lambda x: '|'.join(list(map(str, x[attr].values.tolist()))), axis=1)

    def transform_subgroup(x, protected_attr):
        res = x['indv_key'].tolist()
        res = ["*".join(res), "*".join(res[::-1])]
        return pd.Series(res, index=x.index)

    ress = res.groupby(['subgroup_num']).apply(lambda x: transform_subgroup(x, attr))
    if ress.shape[0] > 0:
        res['couple_key'] = ress.reset_index(level=0, drop=True)
    else:
        res['couple_key'] = None

    return res, model_scores
