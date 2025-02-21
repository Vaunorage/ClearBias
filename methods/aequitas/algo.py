import numpy as np
import random
import time
from itertools import product
import pandas as pd
from scipy.optimize import basinhopping
import copy
import logging
import sys
from data_generator.main import get_real_data, generate_from_real_data, DiscriminationData
from methods.utils import train_sklearn_model

logger = logging.getLogger('aequitas')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_input_bounds(discrimination_data):
    """Get input bounds for each feature based on training data"""
    bounds = []
    for column in discrimination_data.training_dataframe.columns:
        if column != discrimination_data.outcome_column:
            min_val = discrimination_data.training_dataframe[column].min()
            max_val = discrimination_data.training_dataframe[column].max()
            bounds.append((min_val, max_val))
    return bounds


def run_aequitas(discrimination_data: DiscriminationData, model_type='rf', max_global=1000, max_local=1000,
                 step_size=1.0, init_prob=0.5, random_seed=None):
    """
    Main AEQUITAS implementation using custom data generation and model training

    Args:
        discrimination_data: DiscriminationData object containing training data and metadata
        model_type: Type of model to train ('rf' for Random Forest)
        max_global: Maximum number of global search iterations
        max_local: Maximum number of local search iterations
        step_size: Step size for local perturbation
        init_prob: Initial probability for direction choice
        random_seed: Random seed for reproducibility (default: None)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    start_time = time.time()
    # Train model using provided training function
    model, X_train, X_test, y_train, y_test, feature_names = train_sklearn_model(
        data=discrimination_data.training_dataframe,
        model_type=model_type,
        sensitive_attrs=discrimination_data.protected_attributes,
        target_col=discrimination_data.outcome_column
    )

    # Get input bounds and number of parameters
    input_bounds = get_input_bounds(discrimination_data)
    params = len(input_bounds)

    # Get sensitive parameter indices for all protected attributes
    sensitive_params = [feature_names.index(attr) for attr in discrimination_data.protected_attributes]

    # Initialize probabilities
    param_probability = [1.0 / params] * params
    direction_probability = [init_prob] * params
    param_probability_change_size = 0.001
    direction_probability_change_size = 0.001

    # Initialize result tracking
    global_disc_inputs = set()
    global_disc_inputs_list = []
    local_disc_inputs = set()
    local_disc_inputs_list = []
    tot_inputs = set()
    count = [1]  # For tracking periodic output
    all_discriminations = set()

    def check_for_error_condition(model, instance, protected_indices, input_bounds):
        """
        Check for discrimination across all combinations of protected attributes
        """
        instance = pd.DataFrame([instance], columns=discrimination_data.attr_columns)
        label = model.predict(instance)[0]

        # Generate all possible combinations of protected attribute values
        protected_values = []
        for idx in protected_indices:
            values = range(int(input_bounds[idx][0]), int(input_bounds[idx][1]) + 1)
            protected_values.append(list(values))

        # Try all combinations
        new_df = []
        for values in product(*protected_values):
            if tuple(instance[discrimination_data.protected_attributes]) != values:
                new_instance = instance.copy()
                new_instance[discrimination_data.protected_attributes] = values
                new_df.append(new_instance)
        new_df = pd.concat(new_df)
        new_df['outcome'] = model.predict(new_df)

        for row in new_df.to_numpy():
            tot_inputs.add(tuple(row.astype(int)))

        discrimination_df = new_df[new_df['outcome'] != label]

        for _, row in discrimination_df.iterrows():
            all_discriminations.add(tuple((tuple(instance), int(label),
                                           tuple(row[discrimination_data.attr_columns]),
                                           int(row['outcome']))))

        return discrimination_df.shape[0] != 0

    class Local_Perturbation:
        """
        Local perturbation class modified to handle multiple protected attributes
        """

        def __init__(self, model, input_bounds, sensitive_params, param_probability,
                     param_probability_change_size, direction_probability,
                     direction_probability_change_size, step_size):
            self.model = model
            self.input_bounds = input_bounds
            self.sensitive_params = sensitive_params
            self.param_probability = param_probability
            self.param_probability_change_size = param_probability_change_size
            self.direction_probability = direction_probability
            self.direction_probability_change_size = direction_probability_change_size
            self.step_size = step_size
            self.perturbation_unit = 1
            self.params = len(input_bounds)

        def __call__(self, x):
            # Randomly choose the feature for perturbation
            param_choice = np.random.choice(range(self.params), p=self.param_probability)

            # Randomly choose direction for perturbation
            perturbation_options = [-1, 1]
            direction_choice = np.random.choice(
                perturbation_options,
                p=[self.direction_probability[param_choice],
                   1 - self.direction_probability[param_choice]]
            )

            # If at bounds, choose random direction
            if (x[param_choice] == self.input_bounds[param_choice][0]) or \
                    (x[param_choice] == self.input_bounds[param_choice][1]):
                direction_choice = np.random.choice(perturbation_options)

            # Perform perturbation
            x[param_choice] = x[param_choice] + (direction_choice * self.step_size)

            # Clip to bounds
            x[param_choice] = max(self.input_bounds[param_choice][0],
                                  min(self.input_bounds[param_choice][1], x[param_choice]))

            # Check for discrimination
            error_condition = check_for_error_condition(self.model, x, self.sensitive_params, self.input_bounds)

            # Check if any protected attribute values changed
            # error_condition = any(error_values[i] != int(x[self.sensitive_params[i]])
            #                       for i in range(len(self.sensitive_params)))

            # Update direction probabilities
            if (error_condition and direction_choice == -1) or \
                    (not error_condition and direction_choice == 1):
                self.direction_probability[param_choice] = min(
                    self.direction_probability[param_choice] +
                    (self.direction_probability_change_size * self.perturbation_unit),
                    1
                )
            elif (not error_condition and direction_choice == -1) or \
                    (error_condition and direction_choice == 1):
                self.direction_probability[param_choice] = max(
                    self.direction_probability[param_choice] -
                    (self.direction_probability_change_size * self.perturbation_unit),
                    0
                )

            # Update parameter probabilities
            if error_condition:
                self.param_probability[param_choice] += self.param_probability_change_size
                self._normalize_probability()
            else:
                self.param_probability[param_choice] = max(
                    self.param_probability[param_choice] - self.param_probability_change_size,
                    0
                )
                self._normalize_probability()

            return x

        def _normalize_probability(self):
            probability_sum = sum(self.param_probability)
            for i in range(self.params):
                self.param_probability[i] = float(self.param_probability[i]) / float(probability_sum)

    class Global_Discovery:
        """
        Direct port of the original Global_Discovery class
        """

        def __init__(self, input_bounds):
            self.input_bounds = input_bounds

        def __call__(self, x):
            for i in range(len(x)):
                x[i] = random.randint(
                    int(self.input_bounds[i][0]),
                    int(self.input_bounds[i][1])
                )
            return x

    def evaluate_global(inp):
        """Global search evaluation function"""
        error_condition = check_for_error_condition(model, inp, sensitive_params, input_bounds)

        # Check if any protected attribute values changed
        # error_condition = any(result[i] != int(inp[sensitive_params[i]])
        #                       for i in range(len(sensitive_params)))

        temp = tuple(inp.tolist())
        tot_inputs.add(temp)

        # More frequent logging - every 2 seconds
        end = time.time()
        use_time = end - start_time

        # Store last_log_time in function's closure
        if not hasattr(evaluate_global, 'last_log_time'):
            evaluate_global.last_log_time = start_time

        # Log every 2 seconds
        if use_time - (evaluate_global.last_log_time - start_time) >= 2:
            # Clear previous line
            sys.stdout.write('\r' + ' ' * 100 + '\r')
            # Print metrics
            sys.stdout.write(
                f"Time: {use_time:.1f}s | "
                f"Tested: {len(tot_inputs)} | "
                f"Found: {len(global_disc_inputs_list) + len(local_disc_inputs_list)} | "
                f"Rate: {float(len(global_disc_inputs_list) + len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100:.2f}%"
            )
            sys.stdout.flush()
            evaluate_global.last_log_time = end

        if error_condition:
            global_disc_inputs.add(temp)
            global_disc_inputs_list.append(list(inp))

            # Run local search
            try:
                local_minimizer = {"method": "L-BFGS-B"}
                basinhopping(
                    evaluate_local,
                    inp,
                    stepsize=step_size,
                    take_step=Local_Perturbation(
                        model, input_bounds, sensitive_params,
                        param_probability.copy(), param_probability_change_size,
                        direction_probability.copy(), direction_probability_change_size,
                        step_size
                    ),
                    minimizer_kwargs=local_minimizer,
                    niter=max_local
                )
            except Exception as e:
                logger.error(f"Local search error: {e}")

        return float(not error_condition)

    def evaluate_local(inp):
        """Local search evaluation function"""
        error_condition = check_for_error_condition(model, inp, sensitive_params, input_bounds)

        # Check if any protected attribute values changed
        # error_condition = any(result[i] != int(inp[sensitive_params[i]])
        #                       for i in range(len(sensitive_params)))

        temp = tuple(inp.tolist())
        tot_inputs.add(temp)

        if error_condition and temp not in global_disc_inputs and temp not in local_disc_inputs:
            local_disc_inputs.add(temp)
            local_disc_inputs_list.append(list(inp))

        # Share the same logging mechanism with global search
        end = time.time()
        use_time = end - start_time

        if not hasattr(evaluate_local, 'last_log_time'):
            evaluate_local.last_log_time = start_time

        if use_time - (evaluate_local.last_log_time - start_time) >= 2:
            sys.stdout.write('\r' + ' ' * 100 + '\r')
            sys.stdout.write(
                f"Time: {use_time:.1f}s | "
                f"Tested: {len(tot_inputs)} | "
                f"Found: {len(global_disc_inputs_list) + len(local_disc_inputs_list)} | "
                f"Rate: {float(len(global_disc_inputs_list) + len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100:.2f}%"
            )
            sys.stdout.flush()
            evaluate_local.last_log_time = end

        return float(not error_condition)

    # Initial input - use first training example
    initial_input = X_train.iloc[0].values.astype('int')

    # Run global search
    logger.info("Starting global search...")
    minimizer = {"method": "L-BFGS-B"}
    basinhopping(evaluate_global, initial_input, stepsize=step_size, take_step=Global_Discovery(input_bounds),
                 minimizer_kwargs=minimizer, niter=max_global)

    # Calculate final results
    end_time = time.time()
    total_time = end_time - start_time

    disc_inputs = len(global_disc_inputs_list) + len(local_disc_inputs_list)
    success_rate = (disc_inputs / len(tot_inputs)) * 100 if len(tot_inputs) > 0 else 0

    logger.info("\nFinal Results:")
    logger.info(f"Total inputs tested: {len(tot_inputs)}")
    logger.info(f"Global discriminatory inputs: {len(global_disc_inputs_list)}")
    logger.info(f"Local discriminatory inputs: {len(local_disc_inputs_list)}")
    logger.info(f"Success rate: {success_rate:.4f}%")
    logger.info(f"Total time: {total_time:.2f} seconds")

    tsn = len(tot_inputs)  # Total Sample Number
    dsn = len(global_disc_inputs) + len(local_disc_inputs)  # Discriminatory Sample Number
    sur = dsn / tsn if tsn > 0 else 0  # Success Rate
    dss = total_time / dsn if dsn > 0 else float('inf')  # Discriminatory Sample Search time

    metrics = {
        'TSN': tsn,
        'DSN': dsn,
        'SUR': sur,
        'DSS': dss
    }

    logger.info(f"Total Inputs: {len(tot_inputs)}")
    logger.info(f"Global Search Discriminatory Inputs: {len(global_disc_inputs)}")
    logger.info(f"Local Search Discriminatory Inputs: {len(local_disc_inputs)}")
    logger.info(f"Success Rate (SUR): {metrics['SUR']:.4f}")
    logger.info(f"Average Search Time per Discriminatory Sample (DSS): {metrics['DSS']:.4f} seconds")
    logger.info(f"Total Discriminatory Pairs Found: {len(all_discriminations)}")

    res_df = []
    case_id = 0
    for org, org_res, counter_org, counter_org_res in all_discriminations:
        indv1 = pd.DataFrame([list(org)], columns=discrimination_data.attr_columns)
        indv2 = pd.DataFrame([list(counter_org)], columns=discrimination_data.attr_columns)

        indv_key1 = "|".join(str(x) for x in indv1[discrimination_data.attr_columns].iloc[0])
        indv_key2 = "|".join(str(x) for x in indv2[discrimination_data.attr_columns].iloc[0])

        # Add the additional columns
        indv1['indv_key'] = indv_key1
        indv1['outcome'] = org_res
        indv2['indv_key'] = indv_key2
        indv2['outcome'] = counter_org_res

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

    res_df['TSN'] = tsn
    res_df['DSN'] = dsn
    res_df['SUR'] = sur
    res_df['DSS'] = dss

    return res_df, metrics


if __name__ == '__main__':
    # Get data using provided generator
    discrimination_data, schema = get_real_data('adult')
    # discrimination_data, schema = generate_from_real_data(data_generator)

    # For the adult dataset
    results, global_cases = run_aequitas(
        discrimination_data=discrimination_data,
        model_type='rf', max_global=200, max_local=1000, step_size=1.0
    )
