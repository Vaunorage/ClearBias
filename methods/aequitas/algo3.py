import logging
import numpy as np
import random
from itertools import product
import pandas as pd
from scipy.optimize import basinhopping
import time

from data_generator.main import get_real_data, DiscriminationData, generate_from_real_data
from methods.utils import train_sklearn_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def aequitas(dataset: DiscriminationData, sensitive_param, max_global, max_local, step_size,
             init_prob, direction_probability_change_size, param_probability_change_size,
             all_inputs, all_exec_times, results_df):
    """
    The implementation of AEQUITAS_Fully_Connected
    :param dataset: the name of testing dataset
    :param sensitive_param: the name of testing dataset
    :param model_path: the path of testing model
    :param max_global: the maximum number of samples for global search
    :param max_local: the maximum number of samples for local search
    :param step_size: the step size of perturbation
    :return:
    """

    all_discrimination = set()

    class Local_Perturbation(object):
        """
        The  implementation of local perturbation
        """

        def __init__(self, preds, conf: DiscriminationData, sensitive_param, param_probability,
                     param_probability_change_size,
                     direction_probability, direction_probability_change_size, step_size):
            """
            Initial function of local perturbation
            :param sess: TF session
            :param preds: the model's symbolic output
            :param x: input placeholder
            :param conf: the configuration of dataset
            :param sensitive_param: the index of sensitive feature
            :param param_probability: the probabilities of features
            :param param_probability_change_size: the step size for changing probability
            :param direction_probability: the probabilities of perturbation direction
            :param direction_probability_change_size:
            :param step_size: the step size of perturbation
            """
            self.preds = preds
            self.conf = conf
            self.sensitive_param = sensitive_param
            self.param_probability = param_probability
            self.param_probability_change_size = param_probability_change_size
            self.direction_probability = direction_probability
            self.direction_probability_change_size = direction_probability_change_size
            self.step_size = step_size
            self.perturbation_unit = 1

        def __call__(self, x):
            """
            Local perturbation
            :param x: input instance for local perturbation
            :return: new potential individual discriminatory instance
            """
            # randomly choose the feature for perturbation
            param_choice = np.random.choice(range(len(self.conf.attr_columns)), p=self.param_probability)

            # randomly choose the direction for perturbation
            perturbation_options = [-1, 1]
            direction_choice = np.random.choice(perturbation_options, p=[self.direction_probability[param_choice],
                                                                         (1 - self.direction_probability[
                                                                             param_choice])])
            if (x[param_choice] == self.conf.input_bounds[param_choice][0]) or (
                    x[param_choice] == self.conf.input_bounds[param_choice][1]):
                direction_choice = np.random.choice(perturbation_options)

            # perturbation
            x[param_choice] = x[param_choice] + (direction_choice * self.step_size)

            # clip the generating instance with each feature to make sure it is valid
            x[param_choice] = max(self.conf.input_bounds[param_choice][0], x[param_choice])
            x[param_choice] = min(self.conf.input_bounds[param_choice][1], x[param_choice])

            # check whether the test case is an individual discriminatory instance
            ei, max_row = check_for_error_condition(self.conf, self.preds, x)

            # update the probabilities of directions
            if (ei != 0 and direction_choice == -1) or (not ei != 0 and direction_choice == 1):
                self.direction_probability[param_choice] = min(self.direction_probability[param_choice] + (
                        self.direction_probability_change_size * self.perturbation_unit), 1)

            elif (not ei != 0 and direction_choice == -1) or (ei != 0 and direction_choice == 1):
                self.direction_probability[param_choice] = max(self.direction_probability[param_choice] - (
                        self.direction_probability_change_size * self.perturbation_unit), 0)

            # update the probabilities of features
            if ei != 0:
                self.param_probability[param_choice] = self.param_probability[
                                                           param_choice] + self.param_probability_change_size
                self.normalise_probability()
            else:
                self.param_probability[param_choice] = max(
                    self.param_probability[param_choice] - self.param_probability_change_size, 0)
                self.normalise_probability()

            return x

        def normalise_probability(self):
            """
            Normalize the probability
            :return: probability
            """
            probability_sum = 0.0
            for prob in self.param_probability:
                probability_sum = probability_sum + prob

            for i in range(len(self.conf.attr_columns)):
                self.param_probability[i] = float(self.param_probability[i]) / float(probability_sum)

    class Global_Discovery(object):
        """
        The  implementation of global perturbation
        """

        def __init__(self, conf: DiscriminationData):
            """
            Initial function of global perturbation
            :param conf: the configuration of dataset
            """
            self.conf = conf

        def __call__(self, x):
            """
            Global perturbation
            :param x: input instance for local perturbation
            :return: new potential individual discriminatory instance
            """
            # clip the generating instance with each feature to make sure it is valid
            for i in range(len(self.conf.attr_columns)):
                x[i] = random.randint(self.conf.input_bounds[i][0], self.conf.input_bounds[i][1])
            return x

    def check_for_error_condition(conf: DiscriminationData, preds, t):
        inp_df = pd.DataFrame([t], columns=conf.attr_columns)
        original_pred = preds(inp_df)[0]

        # Get all unique values for each protected attribute
        protected_values = {}
        for attr in conf.protected_attributes:
            protected_values[attr] = sorted(conf.dataframe[attr].unique())

        # Generate all possible combinations of protected attributes
        attr_names = list(protected_values.keys())
        attr_values = list(protected_values.values())
        combinations = list(product(*attr_values))

        test_cases = []
        for combination in combinations:
            # Skip if it's identical to the input case
            if all(inp_df[attr].iloc[0] == value
                   for attr, value in zip(attr_names, combination)):
                continue

            new_case = inp_df.copy()
            for attr, value in zip(attr_names, combination):
                new_case[attr] = value
            test_cases.append(new_case)

        if not test_cases:
            return 0, None

        test_cases_df = pd.concat(test_cases)
        test_predictions = preds(test_cases_df)
        test_cases_df['outcome'] = test_predictions

        discriminations = test_cases_df['outcome'] - original_pred
        max_discrimination = discriminations.max()
        max_discrimination_idx = discriminations.idxmax()
        max_discrimination_row = test_cases_df.loc[max_discrimination_idx]

        discriminations_df = test_cases_df[discriminations > 0]

        inp_df['outcome'] = original_pred

        for el in test_cases_df.to_numpy():
            tot_inputs.add((tuple(map(float, inp_df.to_numpy()[0])), tuple(map(float, el))))

        if discriminations_df.shape[0] != 0:
            for el in discriminations_df.to_numpy():
                all_discrimination.add((tuple(map(float, inp_df.to_numpy()[0])), tuple(map(float, el))))

        current_tsn = len(set(list(all_inputs) + list(tot_inputs)))
        current_dsn = len(all_discrimination) + sum(map(lambda x: x.shape[0], results_df))
        logger.info(f"TSN: {current_tsn} - DSN: {current_dsn} - {current_dsn / current_tsn:.2f}")

        return max_discrimination, max_discrimination_row

    start = time.time()

    # Train the model using the new function
    model, X_train, X_test, y_train, y_test, feature_names = train_sklearn_model(
        data=dataset.training_dataframe,
        model_type='rf',  # You can change this to 'svm', 'lr', or 'dt'
        sensitive_attrs=dataset.protected_attributes,
        target_col=dataset.outcome_column
    )

    params = len(dataset.attr_columns)

    # hyper-parameters for initial probabilities of directions

    direction_probability = [init_prob] * params

    # hyper-parameters for features
    param_probability = [1.0 / params] * params

    # prepare the testing data and model
    preds = model.predict

    # store the result of fairness testing
    global_disc_inputs = set()
    global_disc_inputs_list = []
    local_disc_inputs = set()
    local_disc_inputs_list = []
    tot_inputs = set()

    minimizer = {"method": "L-BFGS-B"}

    def evaluate_local(inp):
        result, max_row = check_for_error_condition(dataset, preds, inp)
        inp_key = pd.DataFrame([inp.astype('int').tolist()], columns=dataset.attr_columns)
        inp_key = list(inp_key.drop(columns=dataset.protected_attributes).to_numpy())[0]
        tot_inputs.add(tuple(inp_key))

        # count = 300
        end = time.time()
        use_time = end - start
        if use_time >= 3900:
            return float('inf')  # Return a large number instead of None

        if result > 0 and (tuple(inp_key) not in global_disc_inputs) and (
                tuple(inp_key) not in local_disc_inputs):
            local_disc_inputs.add(tuple(inp_key))
            local_disc_inputs_list.append(inp_key)

        return float(result)  # Return result as a float instead of boolean

    global_discovery = Global_Discovery(dataset)
    local_perturbation = Local_Perturbation(preds, dataset, sensitive_param, param_probability,
                                            param_probability_change_size, direction_probability,
                                            direction_probability_change_size, step_size)

    for i in range(max_global):
        # global generation
        inp = global_discovery(list(dataset.xdf.sample(1).to_numpy()[0]))
        inp_key = pd.DataFrame([inp], columns=dataset.attr_columns)
        inp_key = list(inp_key.drop(columns=dataset.protected_attributes).to_numpy())[0]
        tot_inputs.add(tuple(inp_key))

        result, max_row = check_for_error_condition(dataset, preds, inp)

        # if get an individual discriminatory instance
        if result > 0 and (tuple(inp_key) not in global_disc_inputs) and (tuple(inp_key) not in local_disc_inputs):
            global_disc_inputs_list.append(inp_key)
            global_disc_inputs.add(tuple(inp_key))

            basinhopping(evaluate_local, inp, stepsize=1.0, take_step=local_perturbation,
                         minimizer_kwargs=minimizer, niter=max_local)

    res_df = []
    case_id = 0
    for org, counter_org in all_discrimination:
        # Ensure org and counter_org are properly shaped
        org = np.array(org).reshape(1, -1)  # Reshape to 1 row with all columns
        counter_org = np.array(counter_org).reshape(1, -1)
        
        indv1 = pd.DataFrame(org, columns=dataset.attr_columns)
        indv2 = pd.DataFrame(counter_org, columns=dataset.attr_columns)

        indv_key1 = "|".join(str(x) for x in indv1[dataset.attr_columns].iloc[0])
        indv_key2 = "|".join(str(x) for x in indv2[dataset.attr_columns].iloc[0])

        # Add the additional columns
        indv1['indv_key'] = indv_key1
        indv2['indv_key'] = indv_key2

        # Create couple_key as before

        couple_key = f"{indv_key1}-{indv_key2}"
        diff_outcome = abs(indv1['outcome'] - indv2['outcome'])

        df_res = pd.concat([indv1, indv2])
        df_res['couple_key'] = couple_key
        df_res['diff_outcome'] = diff_outcome
        df_res['case_id'] = case_id
        res_df.append(df_res)
        case_id += 1

    if len(res_df) != 0:
        res_df = pd.concat(res_df)
    else:
        res_df = pd.DataFrame([])

    end_time = time.time()
    execution_time = end_time - start
    return res_df, execution_time, tot_inputs


def run_aequitas(data: DiscriminationData, max_global=100, max_local=1000, step_size=1,
                 init_prob=0.5, direction_probability_change_size=0.001, param_probability_change_size=0.001):
    start = time.time()
    all_inputs = []
    all_exec_times = []
    results_df = []
    ITER = 1

    logger.info("Starting AEQUITAS algorithm execution")
    for i in range(ITER):
        logger.info(f"Starting iteration {i + 1}/{ITER}")
        res_df, execution_time, total_inputs = aequitas(dataset=data,
                                                        sensitive_param=data.sensitive_indices,
                                                        max_global=max_global,
                                                        max_local=max_local,
                                                        step_size=step_size,
                                                        init_prob=init_prob,
                                                        direction_probability_change_size=direction_probability_change_size,
                                                        param_probability_change_size=param_probability_change_size,
                                                        all_inputs=all_inputs,
                                                        all_exec_times=all_exec_times,
                                                        results_df=results_df)
        end = time.time()
        all_inputs.extend(total_inputs)
        all_exec_times.append(execution_time)
        results_df.append(res_df)
        logger.info(f"Total time for iteration {i + 1}: {end - start:.2f}s")

    results_df = pd.concat(results_df)
    results_df.drop_duplicates(subset='couple_key', keep='first', inplace=True)

    tsn = len(all_inputs)
    dsn = results_df.shape[0] / len(all_inputs)

    metrics = {
        "TSN": tsn,
        "DSN": dsn,
        "DSS": round(sum(all_exec_times) / len(all_exec_times), 2),
        "SUR": round(dsn / tsn, 2)
    }

    logger.info("Algorithm execution completed")
    logger.info(f"Final metrics: {metrics}")

    return results_df, metrics


if __name__ == '__main__':
    ge, ge_schema = get_real_data('adult')
    # ge, schema = generate_from_real_data('adult')
    results_df, metrics = run_aequitas(ge, max_global=1000, max_local=10000, step_size=1,
                                       init_prob=0.5, direction_probability_change_size=0.001,
                                       param_probability_change_size=0.001)
    print(results_df)
