import numpy as np
import random
import time
from scipy.optimize import basinhopping
import logging
import sys
from data_generator.main import get_real_data, DiscriminationData
from methods.utils import train_sklearn_model, check_for_error_condition, make_final_metrics_and_dataframe
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


def run_aequitas(data: DiscriminationData, model_type='rf', max_global=500, max_local=2000, step_size=1.0,
                 init_prob=0.5, random_seed=42, max_runtime_seconds=3600, max_tsn=10000,
                 param_probability_change_size=0.005, direction_probability_change_size=0.005, one_attr_at_a_time=False,
                 db_path=None, analysis_id=None, use_cache=True):
    early_termination = False

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    start_time = time.time()

    # Train model using provided training function
    logger.info("Training the model...")
    model, X_train, X_test, y_train, y_test, feature_names = train_sklearn_model(
        data=data.training_dataframe,
        model_type=model_type,
        sensitive_attrs=data.protected_attributes,
        target_col=data.outcome_column,
        random_state=random_seed,
        use_cache=use_cache
    )
    logger.info(f"Model trained. Features: {feature_names}")

    dsn_by_attr_value = {e: {'TSN': 0, 'DSN': 0} for e in data.protected_attributes}
    dsn_by_attr_value['total'] = 0

    # Get input bounds and number of parameters
    input_bounds = get_input_bounds(data)
    params = len(input_bounds)
    logger.info(f"Total parameters: {params}")
    logger.info(f"Input bounds: {input_bounds}")

    # Get sensitive parameter indices for all protected attributes
    sensitive_params = [data.attr_columns.index(attr) for attr in
                        data.protected_attributes]
    logger.info(f"Protected attribute indices: {sensitive_params}")

    # Initialize probabilities with higher change rates for faster adaptation
    param_probability = [1.0 / params] * params
    direction_probability = [init_prob] * params

    # Initialize result tracking
    global_disc_inputs = set()
    local_disc_inputs = set()
    tot_inputs = set()
    all_discriminations = set()
    all_tot_inputs = []

    def should_terminate() -> bool:
        nonlocal early_termination
        current_runtime = time.time() - start_time
        time_limit_exceeded = current_runtime > max_runtime_seconds
        tsn_threshold_reached = max_tsn is not None and len(tot_inputs) >= max_tsn

        if time_limit_exceeded or tsn_threshold_reached:
            early_termination = True
            return True
        return False

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
            # Check termination conditions using should_terminate function
            if should_terminate():
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
                # Add termination check inside attempt loop
                if should_terminate():
                    return x

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

            # Add termination check after attempt loop
            if should_terminate():
                return x

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

            # Check for discrimination
            rounded_x = np.round(x).astype(int)
            error_condition, discrimination_df, max_discr, org_df, tested_inp = check_for_error_condition(logger=logger,
                                                                                                          discrimination_data=data,
                                                                                                          model=self.model,
                                                                                                          dsn_by_attr_value=dsn_by_attr_value,
                                                                                                          instance=rounded_x,
                                                                                                          tot_inputs=tot_inputs,
                                                                                                          all_discriminations=all_discriminations,
                                                                                                          one_attr_at_a_time=one_attr_at_a_time,
                                                                                                          db_path=db_path,
                                                                                                          analysis_id=analysis_id)

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
        # Check termination conditions using should_terminate function
        if should_terminate():
            return 0.0

        # Convert to integer for evaluation
        inp = np.round(inp).astype(int)

        # Check for discrimination
        error_condition, discrimination_df, max_discr, org_df, tested_inp = check_for_error_condition(logger=logger,
                                                                                                      discrimination_data=data,
                                                                                                      model=model,
                                                                                                      dsn_by_attr_value=dsn_by_attr_value,
                                                                                                      instance=inp,
                                                                                                      tot_inputs=tot_inputs,
                                                                                                      all_discriminations=all_discriminations,
                                                                                                      one_attr_at_a_time=one_attr_at_a_time,
                                                                                                      db_path=db_path,
                                                                                                      analysis_id=analysis_id)

        # Track inputs
        temp = tuple(inp.tolist())
        tot_inputs.add(temp)

        if error_condition and temp not in global_disc_inputs and temp not in local_disc_inputs:
            local_disc_inputs.add(temp)

        return float(not error_condition)

    # GLOBAL DISCRIMINATION DISCOVERY:
    global_inputs = data.get_random_rows(max_global)

    for i, global_inp in global_inputs.iterrows():
        # Add termination check inside global discovery loop
        if should_terminate():
            break

        result, discrimination_df, max_discr, org_df, tested_inp = check_for_error_condition(logger=logger,
                                                                                             discrimination_data=data,
                                                                                             model=model,
                                                                                             dsn_by_attr_value=dsn_by_attr_value,
                                                                                             instance=global_inp.to_numpy(),
                                                                                             tot_inputs=tot_inputs,
                                                                                             all_discriminations=all_discriminations,
                                                                                             one_attr_at_a_time=one_attr_at_a_time,
                                                                                             db_path=db_path,
                                                                                             analysis_id=analysis_id)
        if result:
            global_disc_inputs.add(tuple(global_inp.to_numpy()))

    # LOCAL DISCOVERY
    for global_inp_i, global_inp in enumerate(global_disc_inputs):
        # Add termination check before starting each local discovery
        if should_terminate():
            break

        try:
            local_minimizer = {"method": "L-BFGS-B"}

            # Reset random seed for local search
            if random_seed is not None:
                local_seed = random_seed + global_inp_i
                np.random.seed(local_seed)
                random.seed(local_seed)

            # Create a basin-hopping callback that checks for termination
            def basin_callback(x, f, accept):
                return should_terminate()

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
                callback=basin_callback,
                accept_test=lambda f_new, x_new, f_old, x_old: not should_terminate(),
            )
        except Exception as e:
            logger.error(f"Local search error: {e}")

        # Add termination check after each local search completes
        if should_terminate():
            break

    res_df, metrics = make_final_metrics_and_dataframe(data, tot_inputs, all_discriminations,
                                                       dsn_by_attr_value, start_time, logger=logger)

    return res_df, metrics


if __name__ == '__main__':
    # Get data using provided generator
    discrimination_data, schema = get_real_data('adult', use_cache=True)
    # discrimination_data, schema = generate_from_real_data(data_generator)

    # For the adult dataset with max_tsn parameter
    results, metrics = run_aequitas(data=discrimination_data, model_type='rf', max_global=200, max_local=10000,
                                    step_size=1.0, max_runtime_seconds=200, max_tsn=3300, one_attr_at_a_time=True)

    print(metrics)
