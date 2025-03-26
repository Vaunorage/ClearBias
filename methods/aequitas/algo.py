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
            min_val = 0 if min_val < 0 else min_val
            max_val = discrimination_data.training_dataframe[column].max()
            bounds.append((min_val, max_val))
    return bounds


def run_aequitas(discrimination_data: DiscriminationData, model_type='rf', max_global=500, max_local=2000,
                 step_size=1.0, init_prob=0.5, random_seed=42, time_limit_seconds=3600,
                 max_tsn=10000, param_probability_change_size=0.005, direction_probability_change_size=0.005,
                 one_attr_at_a_time=False):
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

    dsn_by_attr_value = {e: {'TSN': 0, 'DSN': 0} for e in discrimination_data.protected_attributes}
    dsn_by_attr_value['total'] = 0

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
    local_disc_inputs = set()
    tot_inputs = set()
    all_discriminations = set()
    total_iterations = 0  # Track total iterations
    last_log_time = start_time  # For periodic logging

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

            # Get or initialize consecutive_same_count to track when we're stuck
            consecutive_same_count = getattr(self, 'consecutive_same_count', 0)

            # Ensure deterministic behavior if using seed
            if self.random_seed is not None:
                derived_seed = self.random_seed + self.call_count
                self.random_state = np.random.RandomState(derived_seed)
                np.random.seed(derived_seed)
                random.seed(derived_seed)

            # Convert to integers to avoid floating point issues
            x = np.round(x).astype(float)  # Keep as float for perturbation but ensure integer values

            # Store original x to check if we end up returning the same value
            original_x = x.copy()

            # ENHANCEMENT 1: Use larger step multipliers
            # Dynamically increase step size if we're getting stuck
            if consecutive_same_count > 3:
                # Increase step multipliers when stuck
                self.step_multipliers = [5.0, 10.0, 20.0]
            elif consecutive_same_count > 0:
                # Moderately increased step multipliers
                self.step_multipliers = [2.0, 5.0, 10.0]
            else:
                # Normal step multipliers - already much larger than original
                self.step_multipliers = [1.0, 3.0, 5.0]

            # ENHANCEMENT 2: More attempts to find new input
            max_attempts = 5 if consecutive_same_count > 2 else 3
            found_new_input = False

            for attempt in range(max_attempts):
                # ENHANCEMENT 3: Parameter choice strategy
                # If we're stuck, try parameters we haven't changed recently
                if consecutive_same_count > 2 and hasattr(self, 'recently_changed_params'):
                    # Avoid parameters we've tried recently
                    available_params = [i for i in range(self.params)
                                        if i not in self.recently_changed_params]

                    # If all parameters have been tried recently, reset
                    if not available_params:
                        available_params = list(range(self.params))

                    param_choice = self.random_state.choice(available_params)
                else:
                    # Normal parameter selection based on probabilities
                    param_choice = self.random_state.choice(range(self.params), p=self.param_probability)

                # Track recently changed parameters (last 3)
                if not hasattr(self, 'recently_changed_params'):
                    self.recently_changed_params = []
                self.recently_changed_params.append(param_choice)
                if len(self.recently_changed_params) > 3:
                    self.recently_changed_params.pop(0)

                # ENHANCEMENT 4: Step size selection
                # Larger step size when stuck
                if consecutive_same_count > 5:
                    # Force largest step size when very stuck
                    step_multiplier = self.step_multipliers[-1]
                else:
                    step_multiplier = self.random_state.choice(self.step_multipliers)

                current_step = self.step_size * step_multiplier

                # ENHANCEMENT 5: Direction selection strategy
                # If we're stuck, alternate directions
                if consecutive_same_count > 3 and hasattr(self, 'last_direction'):
                    # Use opposite direction from last time
                    direction_choice = -self.last_direction
                else:
                    # Normal direction selection
                    direction_choice = self.random_state.choice(
                        [-1, 1],
                        p=[self.direction_probability[param_choice],
                           1 - self.direction_probability[param_choice]]
                    )

                # Store last direction
                self.last_direction = direction_choice

                # ENHANCEMENT 6: Better handling of bounds
                # If at bounds, reverse direction with a larger step
                if ((x[param_choice] <= self.input_bounds[param_choice][0] and direction_choice == -1) or
                        (x[param_choice] >= self.input_bounds[param_choice][1] and direction_choice == 1)):
                    direction_choice *= -1
                    # When hitting bounds, increase step size
                    current_step *= 1.5

                # Create new input
                x_new = x.copy()
                x_new[param_choice] = x_new[param_choice] + (direction_choice * current_step)

                # Clip to bounds
                x_new[param_choice] = max(self.input_bounds[param_choice][0],
                                          min(self.input_bounds[param_choice][1], x_new[param_choice]))

                # ENHANCEMENT 7: Random jitter for getting unstuck
                # Add small random jitter to other parameters when stuck
                if consecutive_same_count > 4:
                    # Add jitter to a few random parameters
                    jitter_count = min(3, self.params - 1)  # Jitter up to 3 other params
                    jitter_params = self.random_state.choice(
                        [i for i in range(self.params) if i != param_choice],
                        size=jitter_count, replace=False
                    )

                    for jitter_param in jitter_params:
                        # Small jitter (10-20% of parameter range)
                        param_range = self.input_bounds[jitter_param][1] - self.input_bounds[jitter_param][0]
                        jitter_size = (param_range * 0.1) * self.random_state.uniform(1.0, 2.0)
                        jitter_dir = self.random_state.choice([-1, 1])

                        x_new[jitter_param] += jitter_dir * jitter_size
                        # Clip to bounds
                        x_new[jitter_param] = max(self.input_bounds[jitter_param][0],
                                                  min(self.input_bounds[jitter_param][1], x_new[jitter_param]))

                # Check if new input has been visited
                new_input_tuple = tuple(np.round(x_new).astype(int))
                if new_input_tuple not in self.visited_inputs:
                    self.visited_inputs.add(new_input_tuple)
                    x = x_new
                    found_new_input = True
                    break

            # ENHANCEMENT 8: If all attempts failed, force a larger jump
            if not found_new_input:
                if consecutive_same_count > 10:
                    # Desperate measure: Make a big jump to a completely different region
                    # Choose half the parameters randomly and change them significantly
                    change_count = max(1, self.params // 2)
                    params_to_change = self.random_state.choice(range(self.params), size=change_count, replace=False)

                    x_new = x.copy()
                    for param_idx in params_to_change:
                        param_range = self.input_bounds[param_idx][1] - self.input_bounds[param_idx][0]
                        # Jump to a random point in the parameter space
                        x_new[param_idx] = self.random_state.uniform(
                            self.input_bounds[param_idx][0],
                            self.input_bounds[param_idx][1]
                        )

                    # Force acceptance of this jump even if visited before
                    x = x_new
                    new_input_tuple = tuple(np.round(x).astype(int))
                    self.visited_inputs.add(new_input_tuple)
                    found_new_input = True

            # ENHANCEMENT 9: Track same-result counter
            if np.array_equal(np.round(x).astype(int), np.round(original_x).astype(int)):
                self.consecutive_same_count = consecutive_same_count + 1
            else:
                self.consecutive_same_count = 0

            # Debug print for tracking when stuck (uncomment if needed)
            # if self.consecutive_same_count > 5 and self.call_count % 10 == 0:
            #     print(f"WARNING: Stuck for {self.consecutive_same_count} iterations")

            # Check for discrimination
            rounded_x = np.round(x).astype(int)
            error_condition, discrimination_df = check_for_error_condition(self.model, rounded_x, self.sensitive_params,
                                                                           self.input_bounds, tot_inputs,
                                                                           all_discriminations,
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
        error_condition, discrimination_df = check_for_error_condition(model, inp, sensitive_params, input_bounds,
                                                                       tot_inputs, all_discriminations,
                                                                       one_attr_at_a_time=one_attr_at_a_time)

        # Track inputs
        temp = tuple(inp.tolist())
        tot_inputs.add(temp)

        if error_condition and temp not in global_disc_inputs and temp not in local_disc_inputs:
            local_disc_inputs.add(temp)

        # Periodic logging (shares same mechanism with global search)
        nonlocal last_log_time
        current_time = time.time()
        elapsed_time = current_time - start_time

        if current_time - last_log_time >= 2:
            sys.stdout.write('\r' + ' ' * 100 + '\r')
            sys.stdout.write(
                f"Time: {elapsed_time:.1f}s | "
                f"Iterations: {total_iterations} | "
                f"Tested: {len(tot_inputs)} | "
                f"Found: {len(global_disc_inputs) + len(local_disc_inputs)} | "
                f"Rate: {float(len(global_disc_inputs) + len(local_disc_inputs)) / max(1, len(tot_inputs)) * 100:.2f}%"
            )
            sys.stdout.flush()
            last_log_time = current_time

        return float(not error_condition)

    def check_for_error_condition(model, instance, protected_indices, input_bounds, tot_inputs, all_discriminations,
                                  one_attr_at_a_time=False):
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
                    dsn_by_attr_value[attr_name]['TSN'] += 1
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
                        dsn_by_attr_value[attr]['TSN'] += 1
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
                         tuple(row[discrimination_data.attr_columns].astype(int)), int(row['outcome']))

            # Only count if this is a new discrimination
            if disc_pair not in all_discriminations:
                all_discriminations.add(disc_pair)

                n_inp = pd.DataFrame(np.expand_dims(disc_pair[0], 0), columns=discrimination_data.attr_columns)
                n_counter = pd.DataFrame(np.expand_dims(disc_pair[2], 0), columns=discrimination_data.attr_columns)

                # Update counts for each protected attribute value in both original and variant
                for i, attr in enumerate(discrimination_data.protected_attributes):
                    if n_inp[attr].iloc[0] != n_counter[attr].iloc[0]:
                        dsn_by_attr_value[attr]['DSN'] += 1
                        dsn_by_attr_value['total'] += 1

        return discrimination_df.shape[0] > 0, discrimination_df

    # GLOBAL DISCRIMINATION DISCOVERY :

    global_inputs = discrimination_data.get_random_rows(max_global)

    for i, global_inp in global_inputs.iterrows():
        result, discrimination_df = check_for_error_condition(model, global_inp.to_numpy(),
                                                              list(discrimination_data.sensitive_indices_dict.values()),
                                                              input_bounds, tot_inputs, all_discriminations,
                                                              one_attr_at_a_time=one_attr_at_a_time)
        if result:
            global_disc_inputs.add(tuple(global_inp.to_numpy()))

    # LOCAL DISCOVERY
    for global_inp_i, global_inp in enumerate(global_disc_inputs):

        try:
            local_minimizer = {"method": "L-BFGS-B"}

            # Reset random seed for local search
            if random_seed is not None:
                local_seed = random_seed + global_inp_i
                np.random.seed(local_seed)
                random.seed(local_seed)

            # Run local search
            basinhopping(
                evaluate_local,
                np.array(global_inp).astype(float),  # Convert to float for optimization
                stepsize=step_size,
                take_step=ImprovedLocalPerturbation(
                    model, input_bounds, sensitive_params,
                    param_probability.copy(), param_probability_change_size,
                    direction_probability.copy(), direction_probability_change_size,
                    step_size
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

    # Calculate final results
    end_time = time.time()
    total_time = end_time - start_time

    # Log final results
    tsn = len(tot_inputs)  # Total Sample Number
    dsn = len(all_discriminations)  # Discriminatory Sample Number
    sur = dsn / tsn if tsn > 0 else 0  # Success Rate
    dss = total_time / dsn if dsn > 0 else float('inf')  # Discriminatory Sample Search time

    for k, v in dsn_by_attr_value.items():
        if k != 'total':
            dsn_by_attr_value[k]['SUR'] = dsn_by_attr_value[k]['DSN'] / dsn_by_attr_value[k]['TSN']
            dsn_by_attr_value[k]['DSS'] = dss

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
        'dsn_by_attr_value': dsn_by_attr_value
    }

    logger.info("\nFinal Results:")
    logger.info(f"Total inputs tested: {tsn}")
    logger.info(f"Global discriminatory inputs: {len(global_disc_inputs)}")
    logger.info(f"Local discriminatory inputs: {len(local_disc_inputs)}")
    logger.info(f"Total discriminatory pairs: {dsn}")
    logger.info(f"Success rate (SUR): {sur:.4f}")
    logger.info(f"Avg. search time per discriminatory sample (DSS): {dss:.4f} seconds")
    logger.info(f"Discrimination by attribute value: {dsn_by_attr_value}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Total iterations: {total_iterations}")

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
    discrimination_data, schema = get_real_data('adult', use_cache=True)
    # discrimination_data, schema = generate_from_real_data(data_generator)

    # For the adult dataset with max_tsn parameter
    results, metrics = run_aequitas(
        discrimination_data=discrimination_data,
        model_type='rf', max_global=200,
        max_local=10000, step_size=1.0,
        time_limit_seconds=3600, max_tsn=50000, one_attr_at_a_time=True
    )

    print(metrics)
