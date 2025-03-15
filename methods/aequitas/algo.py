import numpy as np
import random
import time
from itertools import product
import pandas as pd
from scipy.optimize import basinhopping
import logging
import sys
from data_generator.main import get_real_data, DiscriminationData
from methods.utils import train_sklearn_model
from sklearnex import patch_sklearn

patch_sklearn()

# Configure logging
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


def run_aequitas(discrimination_data: DiscriminationData, model_type='rf', max_global=500, max_local=2000,
                 step_size=1.0, init_prob=0.5, random_seed=42, max_total_iterations=100000, time_limit_seconds=3600,
                 max_tsn=10000, param_probability_change_size=0.005, direction_probability_change_size=0.005,
                 one_attr_at_a_time=False):
    """
    Improved AEQUITAS implementation with better search strategies and numerical stability

    Args:
        discrimination_data: DiscriminationData object containing training data and metadata
        model_type: Type of model to train ('rf' for Random Forest)
        max_global: Maximum number of global search iterations
        max_local: Maximum number of local search iterations
        step_size: Base step size for local perturbation
        init_prob: Initial probability for direction choice
        random_seed: Random seed for reproducibility (default: 42)
        time_limit_seconds: Maximum execution time in seconds (default: 3600)
        max_tsn: Maximum number of total samples to test before stopping (default: 10000)
    """
    # Set random seeds for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    start_time = time.time()

    # Train model using provided training function
    logger.info("Training the model...")
    model, X_train, X_test, y_train, y_test, feature_names = train_sklearn_model(
        data=discrimination_data.training_dataframe,
        model_type=model_type,
        sensitive_attrs=discrimination_data.protected_attributes,
        target_col=discrimination_data.outcome_column,
        random_state=random_seed
    )
    logger.info(f"Model trained. Features: {feature_names}")

    dsn_per_protected_attr = {e: 0 for e in discrimination_data.protected_attributes}
    dsn_per_protected_attr['total'] = 0

    # Get input bounds and number of parameters
    input_bounds = get_input_bounds(discrimination_data)
    params = len(input_bounds)
    logger.info(f"Total parameters: {params}")
    logger.info(f"Input bounds: {input_bounds}")

    # Get sensitive parameter indices for all protected attributes
    sensitive_params = [discrimination_data.attr_columns.index(attr) for attr in
                        discrimination_data.protected_attributes]
    logger.info(f"Protected attribute indices: {sensitive_params}")

    # Initialize probabilities with higher change rates for faster adaptation
    param_probability = [1.0 / params] * params
    direction_probability = [init_prob] * params

    # Initialize result tracking
    global_disc_inputs = set()
    global_disc_inputs_list = []
    local_disc_inputs = set()
    local_disc_inputs_list = []
    tot_inputs = set()
    all_discriminations = set()
    total_iterations = [0]  # Track total iterations
    last_log_time = start_time  # For periodic logging

    # Initialize dictionary to track dsn by attribute value
    dsn_per_protected_attr = {e: 0 for e in discrimination_data.protected_attributes}
    dsn_per_protected_attr['total'] = 0

    def check_for_error_condition(model, instance, protected_indices, input_bounds, one_attr_at_a_time=False):
        """
        Check for discrimination across protected attributes

        Args:
            model: Trained model to evaluate
            instance: Input instance to check
            protected_indices: Indices of protected attributes
            input_bounds: Bounds for input values
            one_attr_at_a_time: If True, vary only one protected attribute at a time

        Returns:
            bool: True if discrimination is found, False otherwise
        """
        # Ensure instance is integer and within bounds
        instance = np.round(instance).astype(int)
        for i, (low, high) in enumerate(input_bounds):
            instance[i] = max(int(low), min(int(high), instance[i]))

        # Convert to DataFrame for prediction
        instance = pd.DataFrame([instance], columns=discrimination_data.attr_columns)

        # Get original prediction
        label = model.predict(instance)[0]

        new_df = []

        if one_attr_at_a_time:
            # Vary one attribute at a time
            for i, attr_idx in enumerate(protected_indices):
                attr_name = discrimination_data.protected_attributes[i]
                current_value = instance[attr_name].values[0]

                # Get all possible values for this attribute
                values = range(int(input_bounds[attr_idx][0]), int(input_bounds[attr_idx][1]) + 1)

                # Create variants with different values for this attribute only
                for value in values:
                    if int(current_value) == value:
                        continue

                    new_instance = instance.copy()
                    new_instance[attr_name] = value
                    new_df.append(new_instance)
        else:
            # Generate all possible combinations of protected attribute values
            protected_values = []
            for idx in protected_indices:
                values = range(int(input_bounds[idx][0]), int(input_bounds[idx][1]) + 1)
                protected_values.append(list(values))

            # Create variants with all combinations of protected attributes
            for values in product(*protected_values):
                if tuple(instance[discrimination_data.protected_attributes].values[0]) != values:
                    new_instance = instance.copy()
                    for i, attr in enumerate(discrimination_data.protected_attributes):
                        new_instance[attr] = values[i]
                    new_df.append(new_instance)

        if not new_df:  # If no combinations were found
            return False

        new_df = pd.concat(new_df)
        new_predictions = model.predict(new_df)
        new_df['outcome'] = new_predictions

        # Add to total inputs
        for row in new_df.to_numpy():
            tot_inputs.add(tuple(row.astype(int)))

        # Find discriminatory instances (different outcome)
        discrimination_df = new_df[new_df['outcome'] != label]

        # Record discriminatory pairs and update attribute value counts
        for _, row in discrimination_df.iterrows():
            # Create the discrimination pair tuple
            disc_pair = (tuple(instance.values[0].astype(int)), int(label),
                         tuple(row[discrimination_data.attr_columns].astype(int)),
                         int(row['outcome']))

            # Only count if this is a new discrimination
            if disc_pair not in all_discriminations:
                all_discriminations.add(disc_pair)

                n_inp = pd.DataFrame(np.expand_dims(disc_pair[0], 0), columns=discrimination_data.attr_columns)
                n_counter = pd.DataFrame(np.expand_dims(disc_pair[2], 0), columns=discrimination_data.attr_columns)

                # Update counts for each protected attribute value in both original and variant
                for i, attr in enumerate(discrimination_data.protected_attributes):
                    if n_inp[attr].iloc[0] != n_counter[attr].iloc[0]:
                        dsn_per_protected_attr[attr] += 1
                        dsn_per_protected_attr['total'] += 1

        return discrimination_df.shape[0] > 0

    class ImprovedLocalPerturbation:
        """
        Local perturbation with improved search strategies
        """

        def __init__(self, model, input_bounds, sensitive_params, param_probability,
                     param_probability_change_size, direction_probability,
                     direction_probability_change_size, step_size, random_seed=None):
            self.model = model
            self.input_bounds = input_bounds
            self.sensitive_params = sensitive_params
            self.param_probability = param_probability.copy()
            self.param_probability_change_size = param_probability_change_size
            self.direction_probability = direction_probability.copy()
            self.direction_probability_change_size = direction_probability_change_size
            self.step_size = step_size
            self.step_multipliers = [0.5, 1.0, 2.0]  # Vary step sizes
            self.params = len(input_bounds)
            self.random_seed = random_seed
            self.call_count = 0
            self.visited_inputs = set()  # Track visited inputs

            # Initialize random state
            if random_seed is not None:
                self.random_state = np.random.RandomState(random_seed)
            else:
                self.random_state = np.random.RandomState()

        def __call__(self, x):
            # Check termination conditions
            if time_limit_seconds and (time.time() - start_time) > time_limit_seconds:
                return x
            if max_tsn and len(tot_inputs) >= max_tsn:
                return x

            # Increment call count and track iterations
            self.call_count += 1
            total_iterations[0] += 1

            # Ensure deterministic behavior if using seed
            if self.random_seed is not None:
                derived_seed = self.random_seed + self.call_count
                self.random_state = np.random.RandomState(derived_seed)
                np.random.seed(derived_seed)
                random.seed(derived_seed)

            # Convert to integers to avoid floating point issues
            x = np.round(x).astype(float)  # Keep as float for perturbation but ensure integer values

            # Try multiple perturbations to find a new input
            for attempt in range(3):  # Try a few times to find new input
                # Select parameter to perturb
                param_choice = self.random_state.choice(range(self.params), p=self.param_probability)

                # Select step size multiplier
                step_multiplier = self.random_state.choice(self.step_multipliers)
                current_step = self.step_size * step_multiplier

                # Select direction
                direction_choice = self.random_state.choice(
                    [-1, 1],
                    p=[self.direction_probability[param_choice],
                       1 - self.direction_probability[param_choice]]
                )

                # If at bounds, reverse direction
                if ((x[param_choice] <= self.input_bounds[param_choice][0] and direction_choice == -1) or
                        (x[param_choice] >= self.input_bounds[param_choice][1] and direction_choice == 1)):
                    direction_choice *= -1

                # Create new input
                x_new = x.copy()
                x_new[param_choice] = x_new[param_choice] + (direction_choice * current_step)

                # Clip to bounds
                x_new[param_choice] = max(self.input_bounds[param_choice][0],
                                          min(self.input_bounds[param_choice][1], x_new[param_choice]))

                # Check if new input has been visited
                new_input_tuple = tuple(np.round(x_new).astype(int))
                if new_input_tuple not in self.visited_inputs:
                    self.visited_inputs.add(new_input_tuple)
                    x = x_new
                    break

            # Check for discrimination
            rounded_x = np.round(x).astype(int)
            error_condition = check_for_error_condition(self.model, rounded_x, self.sensitive_params, self.input_bounds,
                                                        one_attr_at_a_time=one_attr_at_a_time)

            # Update direction probabilities
            if (error_condition and direction_choice == -1) or (not error_condition and direction_choice == 1):
                self.direction_probability[param_choice] = min(
                    self.direction_probability[param_choice] + self.direction_probability_change_size,
                    0.9  # Cap at 0.9 to ensure some exploration
                )
            elif (not error_condition and direction_choice == -1) or (error_condition and direction_choice == 1):
                self.direction_probability[param_choice] = max(
                    self.direction_probability[param_choice] - self.direction_probability_change_size,
                    0.1  # Floor at 0.1 to ensure some exploration
                )

            # Update parameter probabilities
            if error_condition:
                self.param_probability[param_choice] += self.param_probability_change_size
                self._normalize_probability()
            else:
                self.param_probability[param_choice] = max(
                    self.param_probability[param_choice] - self.param_probability_change_size,
                    0.01  # Minimum probability to ensure all parameters have a chance
                )
                self._normalize_probability()

            return x

        def _normalize_probability(self):
            """Normalize probability distribution to sum to 1"""
            probability_sum = sum(self.param_probability)
            if probability_sum > 0:  # Avoid division by zero
                for i in range(self.params):
                    self.param_probability[i] = float(self.param_probability[i]) / float(probability_sum)

    class ImprovedGlobalDiscovery:
        """
        Global discovery with improved diversity and numerical stability
        """

        def __init__(self, input_bounds, sensitive_params, random_seed=None):
            self.input_bounds = input_bounds
            self.sensitive_params = sensitive_params
            self.random_seed = random_seed
            self.call_count = 0
            self.previous_inputs = set()  # Track previous inputs
            self.diversification_attempts = 5  # Number of attempts to generate diverse input

            # Initialize random state
            if random_seed is not None:
                self.random_state = random.Random(random_seed)
            else:
                self.random_state = random.Random()

        def __call__(self, x):
            # Check termination conditions
            if time_limit_seconds and (time.time() - start_time) > time_limit_seconds:
                return x
            if max_tsn and len(tot_inputs) >= max_tsn:
                return x

            # Increment call count
            self.call_count += 1
            total_iterations[0] += 1

            # Update random state for deterministic behavior
            if self.random_seed is not None:
                derived_seed = self.random_seed + self.call_count
                self.random_state = random.Random(derived_seed)
                random.seed(derived_seed)
                np.random.seed(derived_seed)

            # Try multiple times to generate a diverse input
            for _ in range(self.diversification_attempts):
                # Generate completely random input
                for i in range(len(x)):
                    x[i] = self.random_state.randint(
                        int(self.input_bounds[i][0]),
                        int(self.input_bounds[i][1])
                    )

                # Use different strategy for every other attempt
                if _ % 2 == 1:
                    # Focus on exploring sensitive parameters more thoroughly
                    for param_idx in self.sensitive_params:
                        # Try a different value for sensitive parameters
                        low, high = self.input_bounds[param_idx]
                        x[param_idx] = self.random_state.randint(int(low), int(high))

                # Convert to integer
                x_int = np.round(x).astype(int)
                temp = tuple(x_int.tolist())

                # If this input hasn't been seen before, use it
                if temp not in self.previous_inputs:
                    self.previous_inputs.add(temp)
                    # Copy the integer values back to x
                    for i in range(len(x)):
                        x[i] = float(x_int[i])
                    break

            return x

    def evaluate_global(inp):
        """Global search evaluation function with improved logging and termination"""
        # Check termination conditions
        if time_limit_seconds and (time.time() - start_time) > time_limit_seconds:
            return 0.0
        if max_tsn and len(tot_inputs) >= max_tsn:
            return 0.0

        # Ensure inp is integer
        inp = np.round(inp).astype(int)

        # Check for discrimination
        error_condition = check_for_error_condition(model, inp, sensitive_params, input_bounds,
                                                    one_attr_at_a_time=one_attr_at_a_time)

        # Track inputs
        temp = tuple(inp.tolist())
        tot_inputs.add(temp)

        # Periodic logging (every 2 seconds)
        nonlocal last_log_time
        current_time = time.time()
        elapsed_time = current_time - start_time

        if current_time - last_log_time >= 2:
            # Clear previous line and print metrics
            sys.stdout.write('\r' + ' ' * 100 + '\r')
            sys.stdout.write(
                f"Time: {elapsed_time:.1f}s | "
                f"Iterations: {total_iterations[0]} | "
                f"Tested: {len(tot_inputs)} | "
                f"Found: {len(global_disc_inputs_list) + len(local_disc_inputs_list)} | "
                f"Rate: {float(len(global_disc_inputs_list) + len(local_disc_inputs_list)) / max(1, len(tot_inputs)) * 100:.2f}%"
            )
            sys.stdout.flush()
            last_log_time = current_time

        # If discriminatory input found
        if error_condition:
            global_disc_inputs.add(temp)
            global_disc_inputs_list.append(list(inp))

            # Run local search with current max_local parameter
            try:
                local_minimizer = {"method": "L-BFGS-B"}

                # Reset random seed for local search
                if random_seed is not None:
                    local_seed = random_seed + len(global_disc_inputs_list)
                    np.random.seed(local_seed)
                    random.seed(local_seed)

                # Run local search
                basinhopping(
                    evaluate_local,
                    inp.astype(float),  # Convert to float for optimization
                    stepsize=step_size,
                    take_step=ImprovedLocalPerturbation(
                        model, input_bounds, sensitive_params,
                        param_probability.copy(), param_probability_change_size,
                        direction_probability.copy(), direction_probability_change_size,
                        step_size,
                        random_seed=local_seed  # Pass unique seed for this local search
                    ),
                    minimizer_kwargs=local_minimizer,
                    niter=max_local,
                    seed=local_seed,
                    callback=lambda x, f, accept: (
                            (time_limit_seconds and (time.time() - start_time) > time_limit_seconds) or
                            (max_tsn and len(tot_inputs) >= max_tsn)
                    )
                )
            except Exception as e:
                logger.error(f"Local search error: {e}")

        return float(not error_condition)

    def evaluate_local(inp):
        """Local search evaluation function"""
        # Check termination conditions
        if time_limit_seconds and (time.time() - start_time) > time_limit_seconds:
            return 0.0
        if max_tsn and len(tot_inputs) >= max_tsn:
            return 0.0

        # Convert to integer for evaluation
        inp = np.round(inp).astype(int)

        # Check for discrimination
        error_condition = check_for_error_condition(model, inp, sensitive_params, input_bounds,
                                                    one_attr_at_a_time=one_attr_at_a_time)

        # Track inputs
        temp = tuple(inp.tolist())
        tot_inputs.add(temp)

        # Track local discriminatory inputs (avoiding duplicates)
        if error_condition and temp not in global_disc_inputs and temp not in local_disc_inputs:
            local_disc_inputs.add(temp)
            local_disc_inputs_list.append(list(inp))

        # Periodic logging (shares same mechanism with global search)
        nonlocal last_log_time
        current_time = time.time()
        elapsed_time = current_time - start_time

        if current_time - last_log_time >= 2:
            sys.stdout.write('\r' + ' ' * 100 + '\r')
            sys.stdout.write(
                f"Time: {elapsed_time:.1f}s | "
                f"Iterations: {total_iterations[0]} | "
                f"Tested: {len(tot_inputs)} | "
                f"Found: {len(global_disc_inputs_list) + len(local_disc_inputs_list)} | "
                f"Rate: {float(len(global_disc_inputs_list) + len(local_disc_inputs_list)) / max(1, len(tot_inputs)) * 100:.2f}%"
            )
            sys.stdout.flush()
            last_log_time = current_time

        return float(not error_condition)

    # Define callback function to check termination conditions
    def check_termination_conditions(x, f, accept):
        if time_limit_seconds and (time.time() - start_time) > time_limit_seconds:
            return True
        if max_tsn and len(tot_inputs) >= max_tsn:
            return True
        return False

    # Initial input - use first training example
    initial_input = X_train.iloc[0].values.astype('int')

    # Main search loop
    logger.info("Starting search process...")
    search_iteration = 0
    consecutive_no_improvement = 0
    current_max_local = max_local  # Dynamic adjustment of local search depth

    while True:
        # Check termination conditions
        current_time = time.time()
        elapsed_time = current_time - start_time

        if time_limit_seconds and elapsed_time > time_limit_seconds:
            logger.info(f"\nTime limit of {time_limit_seconds} seconds reached. Stopping search.")
            break

        if max_tsn and len(tot_inputs) >= max_tsn:
            logger.info(f"\nMax TSN of {max_tsn} reached. Stopping search.")
            break

        # Track disc inputs before iteration for measuring improvement
        previous_disc_count = len(global_disc_inputs_list) + len(local_disc_inputs_list)

        search_iteration += 1
        logger.info(f"\nStarting global search iteration {search_iteration}...")

        # Select diverse starting points
        if search_iteration > 1:
            # Increase randomness over time
            if search_iteration % 5 == 0:
                # Every 5th iteration, use completely random input
                initial_input = np.array([random.randint(int(low), int(high))
                                          for low, high in input_bounds])
            elif search_iteration % 3 == 0:
                # Every 3rd iteration, use maximally different input from previous
                if len(global_disc_inputs_list) > 0:
                    # Use most recently found discriminatory input
                    base_input = global_disc_inputs_list[-1]
                    # Flip values for non-protected attributes
                    initial_input = np.array(base_input).astype(int)
                    for i in range(len(initial_input)):
                        if i not in sensitive_params:
                            low, high = input_bounds[i]
                            # Generate value far from current
                            current_val = initial_input[i]
                            if current_val < (low + high) / 2:
                                initial_input[i] = int(high)
                            else:
                                initial_input[i] = int(low)
                else:
                    # Use random sample from training if no disc inputs yet
                    initial_input = X_train.sample(1).iloc[0].values.astype('int')
            else:
                # Otherwise use random training example
                initial_input = X_train.sample(1).iloc[0].values.astype('int')

        # Periodic reset of probabilities
        if search_iteration % 10 == 0:
            logger.info("Periodic reset of search probabilities to avoid local optima")
            param_probability = [1.0 / params] * params
            direction_probability = [random.random() for _ in range(params)]
            # Normalize direction probabilities
            sum_dir_prob = sum(direction_probability)
            direction_probability = [p / sum_dir_prob for p in direction_probability]

        # Create seed for this iteration
        iter_seed = None
        if random_seed is not None:
            iter_seed = random_seed + (search_iteration * 1000)
            np.random.seed(iter_seed)
            random.seed(iter_seed)

        # Run global search
        try:
            minimizer = {
                "method": "L-BFGS-B",
                "options": {"maxiter": 100}
            }

            basinhopping(
                evaluate_global,
                initial_input.astype(float),  # Convert to float for optimization
                niter=max_global,
                T=1.0,
                stepsize=step_size,
                minimizer_kwargs=minimizer,
                take_step=ImprovedGlobalDiscovery(
                    input_bounds,
                    sensitive_params,
                    random_seed=iter_seed
                ),
                seed=iter_seed,
                callback=check_termination_conditions
            )
        except Exception as e:
            logger.error(f"Global search error: {e}")
            # Continue to next iteration

        # Check if we found new discriminatory inputs
        current_disc_count = len(global_disc_inputs_list) + len(local_disc_inputs_list)
        if current_disc_count > previous_disc_count:
            # Found new inputs - focus on exploitation
            consecutive_no_improvement = 0
            current_max_local = min(current_max_local * 1.5, max_local * 3)
            logger.info(f"Found {current_disc_count - previous_disc_count} new discriminatory inputs. "
                        f"Increasing local search depth to {current_max_local}")
        else:
            # No new inputs - focus on exploration
            consecutive_no_improvement += 1
            current_max_local = max(current_max_local * 0.7, max_local * 0.3)
            logger.info(f"No new discriminatory inputs found. "
                        f"Decreasing local search depth to {current_max_local}")

            # After several unsuccessful iterations, try more drastic changes
            if consecutive_no_improvement >= 5:
                logger.info("Multiple iterations without improvement. Trying more diverse search strategies.")
                # Double the probability change size temporarily
                param_probability_change_size *= 2
                direction_probability_change_size *= 2
                # But cap at reasonable values
                param_probability_change_size = min(param_probability_change_size, 0.02)
                direction_probability_change_size = min(direction_probability_change_size, 0.02)
                # Reset counter
                consecutive_no_improvement = 0

        # Update max_local for next iteration
        max_local = int(current_max_local)

    # Calculate final results
    end_time = time.time()
    total_time = end_time - start_time

    # Log final results
    tsn = len(tot_inputs)  # Total Sample Number
    dsn = len(all_discriminations)  # Discriminatory Sample Number
    sur = dsn / tsn if tsn > 0 else 0  # Success Rate
    dss = total_time / dsn if dsn > 0 else float('inf')  # Discriminatory Sample Search time

    for k, v in dsn_per_protected_attr.items():
        dsn_per_protected_attr[k] = v / tsn

    # Log dsn_by_attr_value counts
    logger.info("\nDiscrimination counts by protected attribute values:")
    metrics = {
        'TSN': tsn,
        'DSN': dsn,
        'SUR': sur,
        'DSS': dss,
        'total_time': total_time,
        'time_limit_reached': time_limit_seconds is not None and total_time >= time_limit_seconds,
        'max_tsn_reached': max_tsn is not None and tsn >= max_tsn,
        'dsn_by_attr_value': dsn_per_protected_attr
    }

    logger.info("\nFinal Results:")
    logger.info(f"Total inputs tested: {tsn}")
    logger.info(f"Global discriminatory inputs: {len(global_disc_inputs_list)}")
    logger.info(f"Local discriminatory inputs: {len(local_disc_inputs_list)}")
    logger.info(f"Total discriminatory pairs: {dsn}")
    logger.info(f"Success rate (SUR): {sur:.4f}")
    logger.info(f"Avg. search time per discriminatory sample (DSS): {dss:.4f} seconds")
    logger.info(f"Discrimination by attribute value: {dsn_per_protected_attr}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Total iterations: {total_iterations[0]}")

    # Generate result dataframe
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

        # Create couple_key
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

    # Add metrics to result dataframe
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
    results, metrics = run_aequitas(
        discrimination_data=discrimination_data,
        model_type='rf', max_global=200,
        max_local=1000, step_size=1.0,
        time_limit_seconds=3600, max_tsn=10000, one_attr_at_a_time=True
    )

    print(metrics)
