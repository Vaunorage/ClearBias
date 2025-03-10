import numpy as np
import random
import time
from itertools import product
import pandas as pd
from scipy.optimize import basinhopping
import logging
import sys
from data_generator.main import get_real_data, generate_from_real_data, DiscriminationData
from methods.utils import train_sklearn_model
from sklearnex import patch_sklearn

patch_sklearn()

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
                 step_size=1.0, init_prob=0.5, random_seed=None, max_total_iterations=None, time_limit_seconds=None,
                 max_tsn=None):
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
        max_total_iterations: Maximum total number of iterations across both global and local search (default: None)
        time_limit_seconds: Maximum execution time in seconds (default: None)
        max_tsn: Maximum number of total samples to test before stopping (default: None)
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
        target_col=discrimination_data.outcome_column,
        random_state=random_seed
    )

    # Get input bounds and number of parameters
    input_bounds = get_input_bounds(discrimination_data)
    params = len(input_bounds)

    # Get sensitive parameter indices for all protected attributes
    sensitive_params = [discrimination_data.attr_columns.index(attr) for attr in
                        discrimination_data.protected_attributes]

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
    total_iterations = [0]  # Track total iterations across both phases

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
            if tuple(instance[discrimination_data.protected_attributes].values[0]) != values:
                new_instance = instance.copy()
                new_instance[discrimination_data.protected_attributes] = values
                new_df.append(new_instance)

        if not new_df:  # If no combinations were found
            return False

        new_df = pd.concat(new_df)
        new_df['outcome'] = model.predict(new_df)

        for row in new_df.to_numpy():
            tot_inputs.add(tuple(row.astype(int)))

        discrimination_df = new_df[new_df['outcome'] != label]

        for _, row in discrimination_df.iterrows():
            all_discriminations.add(tuple((tuple(instance.values[0]), int(label),
                                           tuple(row[discrimination_data.attr_columns]),
                                           int(row['outcome']))))

        return discrimination_df.shape[0] != 0

    class Local_Perturbation:
        """
        Local perturbation class modified to handle multiple protected attributes with controlled randomness
        """

        def __init__(self, model, input_bounds, sensitive_params, param_probability,
                     param_probability_change_size, direction_probability,
                     direction_probability_change_size, step_size, random_seed=None):
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
            self.random_seed = random_seed
            self.call_count = 0  # Add a counter to create deterministic "randomness"

            # Initialize with a fixed random state
            if random_seed is not None:
                self.random_state = np.random.RandomState(random_seed)
            else:
                self.random_state = np.random.RandomState()

        def __call__(self, x):
            # Check if time limit is reached
            if time_limit_seconds and (time.time() - start_time) > time_limit_seconds:
                return x  # Return unmodified if time limit reached

            # Check if max_tsn is reached
            if max_tsn and len(tot_inputs) >= max_tsn:
                return x  # Return unmodified if max_tsn is reached

            # Increment call count for deterministic seeding
            self.call_count += 1

            # If using a seed, we create a deterministic seed for each call
            # by combining the original seed with the call count
            if self.random_seed is not None:
                derived_seed = self.random_seed + self.call_count
                # Reset the random state for deterministic behavior
                self.random_state = np.random.RandomState(derived_seed)
                # Also set the global random generators
                np.random.seed(derived_seed)
                random.seed(derived_seed)

            # Use self.random_state instead of np.random for all random operations
            param_choice = self.random_state.choice(range(self.params), p=self.param_probability)

            # Randomly choose direction for perturbation
            perturbation_options = [-1, 1]
            direction_choice = self.random_state.choice(
                perturbation_options,
                p=[self.direction_probability[param_choice],
                   1 - self.direction_probability[param_choice]]
            )

            # If at bounds, choose random direction
            if (x[param_choice] == self.input_bounds[param_choice][0]) or \
                    (x[param_choice] == self.input_bounds[param_choice][1]):
                direction_choice = self.random_state.choice(perturbation_options)

            # Perform perturbation
            x[param_choice] = x[param_choice] + (direction_choice * self.step_size)

            # Clip to bounds
            x[param_choice] = max(self.input_bounds[param_choice][0],
                                  min(self.input_bounds[param_choice][1], x[param_choice]))

            # Check for discrimination
            error_condition = check_for_error_condition(self.model, x, self.sensitive_params, self.input_bounds)

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
        Global discovery with controlled randomness
        """

        def __init__(self, input_bounds, random_seed=None):
            self.input_bounds = input_bounds
            self.random_seed = random_seed
            self.call_count = 0

            # Initialize random state
            if random_seed is not None:
                self.random_state = random.Random(random_seed)
            else:
                self.random_state = random.Random()

        def __call__(self, x):
            # Check if time limit is reached
            if time_limit_seconds and (time.time() - start_time) > time_limit_seconds:
                return x  # Return unmodified if time limit reached

            # Check if max_tsn is reached
            if max_tsn and len(tot_inputs) >= max_tsn:
                return x  # Return unmodified if max_tsn is reached

            # Increment call count for deterministic seeding
            self.call_count += 1

            # If using a seed, create a new seed for each call
            if self.random_seed is not None:
                derived_seed = self.random_seed + self.call_count
                # Reset the random state
                self.random_state = random.Random(derived_seed)
                # Also set global random state
                random.seed(derived_seed)
                np.random.seed(derived_seed)

            for i in range(len(x)):
                x[i] = self.random_state.randint(
                    int(self.input_bounds[i][0]),
                    int(self.input_bounds[i][1])
                )
            return x

    def evaluate_global(inp):
        """Global search evaluation function"""
        # Check if time limit is reached
        if time_limit_seconds and (time.time() - start_time) > time_limit_seconds:
            return 0.0  # Stop the search when time limit is reached

        # Check if max_tsn is reached
        if max_tsn and len(tot_inputs) >= max_tsn:
            return 0.0  # Stop the search when max_tsn is reached

        if max_total_iterations and total_iterations[0] >= max_total_iterations:
            return 0.0  # Stop the search when max total iterations is reached

        total_iterations[0] += 1
        error_condition = check_for_error_condition(model, inp, sensitive_params, input_bounds)

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

                if random_seed is not None:
                    np.random.seed(random_seed)
                    random.seed(random_seed)

                basinhopping(
                    evaluate_local,
                    inp,
                    stepsize=step_size,
                    take_step=Local_Perturbation(
                        model, input_bounds, sensitive_params,
                        param_probability.copy(), param_probability_change_size,
                        direction_probability.copy(), direction_probability_change_size,
                        step_size,
                        random_seed=random_seed  # Pass the random seed
                    ),
                    minimizer_kwargs=local_minimizer,
                    niter=max_local,
                    seed=random_seed if random_seed is not None else None,
                    callback=lambda x, f, accept: (time_limit_seconds and (
                            time.time() - start_time) > time_limit_seconds) or
                                                  (max_tsn and len(tot_inputs) >= max_tsn)
                )
            except Exception as e:
                logger.error(f"Local search error: {e}")

        return float(not error_condition)

    def evaluate_local(inp):
        """Local search evaluation function"""
        # Check if time limit is reached
        if time_limit_seconds and (time.time() - start_time) > time_limit_seconds:
            return 0.0  # Stop the search when time limit is reached

        # Check if max_tsn is reached
        if max_tsn and len(tot_inputs) >= max_tsn:
            return 0.0  # Stop the search when max_tsn is reached

        if max_total_iterations and total_iterations[0] >= max_total_iterations:
            return 0.0  # Stop the search when max total iterations is reached

        total_iterations[0] += 1
        error_condition = check_for_error_condition(model, inp, sensitive_params, input_bounds)

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

    # Create a callback function to check termination conditions
    def check_termination_conditions(x, f, accept):
        if time_limit_seconds and (time.time() - start_time) > time_limit_seconds:
            return True  # True will stop the optimization
        if max_tsn and len(tot_inputs) >= max_tsn:
            return True  # True will stop the optimization
        return False

    # Initial input - use first training example
    initial_input = X_train.iloc[0].values.astype('int')

    # Keep running global searches until we hit max_tsn or time limit
    logger.info("Starting search process...")

    # Continue searching until we meet our stopping conditions
    search_iteration = 0
    while True:
        # Check termination conditions
        if time_limit_seconds and (time.time() - start_time) > time_limit_seconds:
            logger.info(f"\nTime limit of {time_limit_seconds} seconds reached. Stopping search.")
            break

        if max_tsn and len(tot_inputs) >= max_tsn:
            logger.info(f"\nMax TSN of {max_tsn} reached. Stopping search.")
            break

        search_iteration += 1
        logger.info(f"\nStarting global search iteration {search_iteration}...")

        # For subsequent iterations, use random training examples to diversify starting points
        if search_iteration > 1:
            initial_input = X_train.sample(1).iloc[0].values.astype('int')

        # Create a seeded random number generator for basinhopping
        if random_seed is not None:
            # Add iteration to seed for variation in subsequent runs
            iter_seed = random_seed + (search_iteration * 1000)
            np.random.seed(iter_seed)
            random.seed(iter_seed)

        # Run global search with basinhopping
        minimizer = {
            "method": "L-BFGS-B",
            "args": (),
            "options": {"maxiter": 100}
        }

        try:
            basinhopping(
                evaluate_global,
                initial_input,
                niter=max_global,
                T=1.0,
                stepsize=step_size,
                minimizer_kwargs=minimizer,
                take_step=Global_Discovery(input_bounds, random_seed=iter_seed if random_seed is not None else None),
                seed=iter_seed if random_seed is not None else None,
                callback=check_termination_conditions
            )
        except Exception as e:
            logger.error(f"Global search error: {e}")
            # Continue with next iteration rather than stopping completely
            continue

    # Calculate final results
    end_time = time.time()
    total_time = end_time - start_time

    disc_inputs = len(global_disc_inputs_list) + len(local_disc_inputs_list)
    success_rate = (disc_inputs / len(tot_inputs)) * 100 if len(tot_inputs) > 0 else 0

    # Log termination reason
    if time_limit_seconds and total_time >= time_limit_seconds:
        logger.info(f"\nExecution terminated after reaching time limit of {time_limit_seconds} seconds")
    elif max_tsn and len(tot_inputs) >= max_tsn:
        logger.info(f"\nExecution terminated after reaching max TSN of {max_tsn}")

    logger.info("\nFinal Results:")
    logger.info(f"Total inputs tested: {len(tot_inputs)}")
    logger.info(f"Global discriminatory inputs: {len(global_disc_inputs_list)}")
    logger.info(f"Local discriminatory inputs: {len(local_disc_inputs_list)}")
    logger.info(f"Success rate: {success_rate:.4f}%")
    logger.info(f"Total time: {total_time:.2f} seconds")

    tsn = len(tot_inputs)  # Total Sample Number
    dsn = len(all_discriminations)  # Discriminatory Sample Number
    sur = dsn / tsn if tsn > 0 else 0  # Success Rate
    dss = total_time / dsn if dsn > 0 else float('inf')  # Discriminatory Sample Search time

    metrics = {
        'TSN': tsn,
        'DSN': dsn,
        'SUR': sur,
        'DSS': dss,
        'total_time': total_time,
        'time_limit_reached': time_limit_seconds is not None and total_time >= time_limit_seconds,
        'max_tsn_reached': max_tsn is not None and tsn >= max_tsn
    }

    logger.info(f"Total Inputs: {len(tot_inputs)}")
    logger.info(f"Discriminatory Inputs: {len(all_discriminations)}")
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

    # For the adult dataset with max_tsn parameter
    results, global_cases = run_aequitas(
        discrimination_data=discrimination_data,
        model_type='rf', max_global=200,
        max_local=1000, step_size=1.0,
        time_limit_seconds=3600, max_tsn=10000
    )
