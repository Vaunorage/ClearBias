import time
import random
import numpy as np
import pandas as pd
from typing import TypedDict
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from scipy.optimize import basinhopping
import errno

from tqdm import tqdm
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from typing import List, Dict, Union, Tuple


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
        self.df = df
        self.col_to_be_predicted = col_to_be_predicted
        self.sensitive_param_name_list = sensitive_param_name_list
        self.feature_names = [col for col in df.columns if col != col_to_be_predicted]
        self.protected_attributes = sensitive_param_name_list
        self.sensitive_param_idx_list = [self.feature_names.index(name) for name in sensitive_param_name_list]
        self.column_names = list(df.columns)

        self.num_params = df.shape[1] - 1

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

        self.normalise_probability()

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
        self.normalise_probability()
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


class AequitasResultRow(TypedDict, total=False):
    outcome: Union[int, float]
    diff_outcome: float
    type: str
    Sensitive_Attribute: str
    Total_Inputs: int
    Discriminatory_Inputs: int
    Percentage_Discriminatory_Inputs: float
    case_id: int
    indv_key: str
    couple_key: str


AequitasResultDF = DataFrame


def collect_results(fully_direct, sensitive_attribute, start_time):
    # Calculate percentage of discriminatory inputs
    total_disc_inputs = len(fully_direct.global_disc_inputs_list["discr_input"]) + len(
        fully_direct.local_disc_inputs_list["discr_input"])
    percentage_disc_inputs = total_disc_inputs / len(fully_direct.tot_inputs) * 100 if fully_direct.tot_inputs else 0

    # Calculate additional metrics
    performance_metrics = calculate_metrics(fully_direct, start_time)

    # Collect results in a dictionary
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
        **performance_metrics  # Add the new metrics
    }
    return results


def aequitas_fully_directed_sklearn(dataset, perturbation_unit, threshold, global_iteration_limit,
                                    local_iteration_limit, sensitive_param_id, model, sensitive_attribute):
    print(f"Aequitas Fully Directed Started for {sensitive_attribute}...\n")
    start_time = time.time()  # Track start time
    initial_input = [random.randint(low, high) for [low, high] in dataset.input_bounds]
    minimizer = {"method": "L-BFGS-B"}

    fully_direct = Fully_Direct(dataset, perturbation_unit, threshold, global_iteration_limit, local_iteration_limit,
                                sensitive_param_id, model)

    basinhopping(fully_direct.evaluate_global, initial_input, stepsize=1.0, take_step=fully_direct.global_discovery,
                 minimizer_kwargs=minimizer, niter=global_iteration_limit)
    print("Finished Global Search")

    fully_direct = mp_basinhopping(fully_direct, minimizer, local_iteration_limit)
    print("Local Search Finished")

    results = collect_results(fully_direct, sensitive_attribute, start_time)

    # Create DataFrames with the results
    res_global = pd.DataFrame({k: results[k] for k in
                               ['Global Discriminatory Inputs', 'Global Counter Discriminatory Inputs',
                                'Global Magnitude']})
    res_global['type'] = 'global'

    res_local = pd.DataFrame({k: results[k] for k in
                              ['Local Discriminatory Inputs', 'Local Counter Discriminatory Inputs',
                               'Local Magnitude']})
    res_local['type'] = 'local'

    res = pd.DataFrame(np.concatenate([res_global.values, res_local.values]),
                       columns=['discrimination', 'counter_discrimination', 'magnitude', 'type'])

    # Add all metrics to the results DataFrame
    metric_columns = ['Sensitive Attribute', 'Total Inputs', 'Discriminatory Inputs',
                      'Percentage Discriminatory Inputs', 'TSN', 'DSN', 'DSS', 'SUR']
    for column in metric_columns:
        res[column] = results[column]

    return res


def calculate_metrics(fully_direct, start_time) -> Dict[str, float]:
    """
    Calculate key performance metrics for Aequitas

    Parameters:
    fully_direct: Fully_Direct object containing results
    start_time: float, the start time of the search process

    Returns:
    Dict containing TSN, DSN, DSS, and SUR metrics
    """
    # Calculate Total Sample Number (TSN)
    tsn = len(fully_direct.tot_inputs)

    # Calculate Discriminatory Sample Number (DSN)
    global_dsn = len(fully_direct.global_disc_inputs_list["discr_input"])
    local_dsn = len(fully_direct.local_disc_inputs_list["discr_input"])
    total_dsn = global_dsn + local_dsn

    # Calculate execution time
    execution_time = time.time() - start_time

    # Calculate Discriminatory Sample Search (DSS) - average time per discriminatory sample
    dss = execution_time / total_dsn if total_dsn > 0 else 0

    # Calculate Success Rate (SUR) - percentage of discriminatory samples
    sur = (total_dsn / tsn * 100) if tsn > 0 else 0

    return {
        "TSN": tsn,
        "DSN": total_dsn,
        "DSS": round(dss, 2),  # seconds
        "SUR": round(sur, 2)  # percentage
    }


def run_aequitas(df: DataFrame,
                 col_to_be_predicted: str,
                 sensitive_param_name_list: List[str],
                 perturbation_unit: float,
                 model_type: str,
                 threshold: float = 0,
                 global_iteration_limit: int = 1000,
                 local_iteration_limit: int = 100) -> Tuple[AequitasResultDF, Dict[str, Union[str, float]]]:
    """
    Run Aequitas analysis on the provided dataset.

    Args:
        df: Input DataFrame
        col_to_be_predicted: Target column name
        sensitive_param_name_list: List of sensitive parameter names
        perturbation_unit: Size of perturbation steps
        model_type: Type of model to use
        threshold: Discrimination threshold
        global_iteration_limit: Maximum global iterations
        local_iteration_limit: Maximum local iterations

    Returns:
        Tuple containing results DataFrame and model scores
    """
    # Initialize dataset and model
    dataset = Dataset(df, col_to_be_predicted=col_to_be_predicted,
                      sensitive_param_name_list=sensitive_param_name_list)
    model, model_scores = generate_sklearn_classifier(dataset, model_type)

    # Run analysis for each sensitive parameter
    results = []
    for sensitive_param_id, sensitive_attribute in zip(dataset.sensitive_param_idx_list,
                                                       sensitive_param_name_list):
        result = aequitas_fully_directed_sklearn(
            dataset,
            perturbation_unit,
            threshold,
            global_iteration_limit,
            local_iteration_limit,
            sensitive_param_id,
            model,
            sensitive_attribute
        )
        results.append(result)

    # If no results, return empty
    if not results:
        return pd.DataFrame(), {}

    # Combine all results
    combined_results: AequitasResultDF = pd.concat(results)
    combined_results['case_id'] = np.arange(combined_results.shape[0])

    # Rename magnitude to diff_outcome for clarity
    combined_results = combined_results.rename(columns={'magnitude': 'diff_outcome'})

    # Process discriminatory and counter-discriminatory cases
    discriminatory_cases = combined_results.copy()
    discriminatory_cases['discrimination'] = discriminatory_cases['counter_discrimination']
    discriminatory_cases = discriminatory_cases.drop(columns=['counter_discrimination'])

    counter_cases = combined_results.drop(columns=['counter_discrimination'])

    # Combine all cases
    all_cases: AequitasResultDF = pd.concat([discriminatory_cases, counter_cases],
                                            ignore_index=True)
    all_cases.sort_values(by=['case_id'], inplace=True)

    try:
        # Convert discrimination data to proper format
        discrimination_data = []
        for disc in all_cases['discrimination']:
            if isinstance(disc, (list, np.ndarray)):
                discrimination_data.append(disc)
            else:
                discrimination_data.append([disc] * len(dataset.column_names))

        # Create final results DataFrame
        disc_df = pd.DataFrame(discrimination_data, columns=dataset.column_names)
        final_results: AequitasResultDF = pd.concat(
            [disc_df, all_cases.drop(columns=['discrimination'])],
            axis=1
        )

        # Predict outcomes
        prediction_columns = [col for col in dataset.column_names
                              if col != dataset.col_to_be_predicted]
        final_results['outcome'] = model.predict(final_results[prediction_columns])

        # Process diff_outcome to ensure consistent format
        final_results['diff_outcome'] = final_results['diff_outcome'].apply(
            lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x
        )

        # Generate individual and couple keys
        attributes = [col for col in df.columns if col != col_to_be_predicted]
        final_results['indv_key'] = final_results.apply(
            lambda x: '|'.join(map(str, x[attributes].values.tolist())),
            axis=1
        )

        # Generate couple keys
        couple_keys = final_results.groupby(['case_id']).apply(
            lambda x: pd.Series([
                '-'.join(x['indv_key'].tolist()),
                '-'.join(x['indv_key'].tolist()[::-1])
            ], index=x.index)
        )

        if not couple_keys.empty:
            final_results['couple_key'] = couple_keys.reset_index(level=0, drop=True)
        else:
            final_results['couple_key'] = None

        return final_results, model_scores[0]

    except Exception as e:
        print(f"Error processing results: {str(e)}")
        print(f"Shape of discrimination data: {len(discrimination_data)} rows")
        print(f"Number of columns expected: {len(dataset.column_names)}")
        raise e