from __future__ import division

import itertools
import pandas as pd
import numpy as np
import random
import time
from scipy.optimize import basinhopping
import logging
import sys

from data_generator.main import get_real_data
from methods.utils import train_sklearn_model

logger = logging.getLogger('aequitas')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def run_aequitas(discrimination_data, model_type='rf', init_prob=0.5,
                 threshold=0.2, global_iteration_limit=100, local_iteration_limit=100):
    """
    Run Aequitas Fully Directed algorithm on the given discrimination data.
    
    Args:
        discrimination_data: DiscriminationData object containing the dataset and metadata
        model_type: Type of sklearn model to use ('rf', 'svm', 'lr', 'dt')
        init_prob: Initial probability for direction probability
        perturbation_unit: Size of perturbation for feature values
        threshold: Threshold for discrimination
        global_iteration_limit: Maximum number of global iterations
        local_iteration_limit: Maximum number of local iterations
        direction_probability_change_size: Change size for direction probability
        param_probability_change_size: Change size for parameter probability
    
    Returns:
        tuple: (results_dataframe, metrics_dict)
    """
    # Configure logger

    random.seed(time.time())

    # Train the model using the new function
    model, X_train, X_test, y_train, y_test, feature_names = train_sklearn_model(
        data=discrimination_data.dataframe,
        model_type=model_type,
        sensitive_attrs=discrimination_data.protected_attributes,
        target_col=discrimination_data.outcome_column
    )

    params = len(feature_names)
    direction_probability = [init_prob] * params
    param_probability = [1.0 / params] * params

    # Initialize tracking variables
    start_time = time.time()
    global_disc_inputs = set()
    global_disc_inputs_list = []
    local_disc_inputs = set()
    local_disc_inputs_list = []
    tot_inputs = set()
    discrimination_cases = []

    def normalise_probability():
        """Normalize the probability distribution to ensure it sums to 1"""
        probability_sum = 0.0
        for prob in param_probability:
            probability_sum = probability_sum + prob

        for i in range(params):
            param_probability[i] = float(param_probability[i]) / float(probability_sum)

    def calculate_metrics():
        """Calculate and return the testing metrics"""
        nonlocal start_time, total_samples, discriminatory_samples

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate Success Rate (SUR)
        sur = discriminatory_samples / len(tot_inputs) if len(tot_inputs) > 0 else 0

        # Calculate Discriminatory Sample Search time (DSS)
        dss = total_time / discriminatory_samples if discriminatory_samples > 0 else float('inf')

        return {
            'TSN': len(tot_inputs),
            'DSN': discriminatory_samples,
            'SUR': sur,
            'DSS': dss,
            'Total_Time': total_time
        }

    def print_progress(search_type=""):
        """Print current progress and metrics in a single line"""
        nonlocal current_global_iter, current_local_iter
        metrics = calculate_metrics()

        # Clear previous line
        sys.stdout.write('\r')
        sys.stdout.flush()

        # Create progress message
        if search_type == "global":
            progress = f"Global: {current_global_iter}/{global_iteration_limit} ({(current_global_iter / global_iteration_limit) * 100:.1f}%)"
        elif search_type == "local":
            progress = f"Local: {current_local_iter}/{local_iteration_limit} ({(current_local_iter / local_iteration_limit) * 100:.1f}%)"
        else:
            progress = "Initializing..."

        # Format the status line
        status = (
            f"\r{progress} | "
            f"Tested: {metrics['TSN']} | "
            f"Found: {metrics['DSN']} | "
            f"Success: {metrics['SUR']:.4f} | "
            f"Time/Sample: {metrics['DSS']:.2f}s | "
            f"Total: {metrics['Total_Time']:.1f}s"
        )

        logger.info(status)

    def is_case_tested(features):
        """Check if a case has already been tested"""
        return (tuple(features),) in tot_inputs

    def evaluate_input(inp):
        """Evaluate discrimination for a given input"""
        protected_attrs = discrimination_data.protected_attributes
        results = []

        # Create base input as DataFrame
        base_input = pd.DataFrame([inp], columns=feature_names)
        base_pred = model.predict(base_input)[0]

        # Test each protected attribute
        for attr in protected_attrs:
            attr_idx = feature_names.index(attr)
            unique_vals = sorted(discrimination_data.dataframe[attr].unique())

            # Test all values of protected attribute
            predictions = []
            for val in unique_vals:
                test_input = base_input.copy()
                test_input[attr].iloc[0] = val
                pred = model.predict(test_input)[0]
                predictions.append(pred)

            # Calculate maximum discrimination
            max_disc = max(abs(p1 - p2) for p1, p2 in itertools.combinations(predictions, 2))
            results.append(max_disc)

        return max(results)

    def evaluate_global(inp):
        """
        Evaluate discrimination across all possible combinations of protected attributes.
        Returns the maximum discrimination found between any pair of protected attribute combinations.
        """
        nonlocal total_samples, discriminatory_samples, current_global_iter
        current_global_iter += 1
        # Check if this input has already been tested
        if is_case_tested(inp):
            logger.info(f"\rSkipping already tested case in global search")
            return 0

        # Convert input to DataFrame
        inp_df = pd.DataFrame([inp], columns=feature_names)
        original_pred = model.predict(inp_df)[0]

        # Add original case to total inputs
        tot_inputs.add((tuple(inp),))

        # Get values for protected attributes
        protected_values = {}
        for attr in discrimination_data.protected_attributes:
            protected_values[attr] = sorted(discrimination_data.dataframe[attr].unique())

        # Generate all possible combinations of protected attribute values
        value_combinations = list(
            itertools.product(*[protected_values[attr] for attr in discrimination_data.protected_attributes]))

        # Create test cases for all combinations
        test_cases = []
        for values in value_combinations:
            # Skip if combination is identical to original
            if all(inp_df[attr].iloc[0] == value for attr, value in
                   zip(discrimination_data.protected_attributes, values)):
                continue

            new_case = inp_df.copy()
            for attr, value in zip(discrimination_data.protected_attributes, values):
                new_case[attr] = value

            # Skip if this case has already been tested
            if is_case_tested(new_case.iloc[0].values):
                continue

            test_cases.append(new_case)

        if not test_cases:  # If no new combinations were generated
            return 0

        # Update total samples count
        total_samples += len(test_cases) + 1  # +1 for original case

        # Combine all test cases and get predictions
        test_df = pd.concat(test_cases)
        predictions = model.predict(test_df)

        # Find discriminatory cases
        discriminatory_cases = test_df[predictions != original_pred]
        max_discrimination = 0

        if len(discriminatory_cases) > 0:
            # Update discriminatory samples count
            discriminatory_samples += len(discriminatory_cases)

            # Print progress after finding discrimination
            print_progress("global")

            for _, disc_case in discriminatory_cases.iterrows():
                disc_tuple = tuple(disc_case.values)

                # Add to total inputs before processing
                tot_inputs.add((disc_tuple,))

                counter_pred = model.predict(disc_case.to_frame().T)[0]

                # Store discrimination case
                discrimination_cases.append((
                    list(inp_df.iloc[0].values),  # original_features
                    original_pred,  # original_outcome
                    list(disc_case.values),  # counter_features
                    counter_pred  # counter_outcome
                ))

                if (disc_tuple,) not in global_disc_inputs:
                    global_disc_inputs.add((disc_tuple,))
                    global_disc_inputs_list.append(list(disc_case.values))

                # Calculate discrimination magnitude
                discrimination = abs(original_pred - counter_pred)
                max_discrimination = max(max_discrimination, discrimination)

                if discrimination > threshold:
                    logger.info(
                        f"\nDiscrimination found - Original: {dict(zip(discrimination_data.protected_attributes, inp_df[discrimination_data.protected_attributes].iloc[0]))} "
                        f"({original_pred}) → New: {dict(zip(discrimination_data.protected_attributes, disc_case[discrimination_data.protected_attributes]))} "
                        f"({counter_pred}) | Magnitude: {discrimination:.4f}"
                    )
                    print_progress("global")

        # Print progress periodically
        if current_global_iter % 10 == 0:
            print_progress("global")

        return max_discrimination

    def evaluate_local(inp):
        """
        Evaluate local discrimination by checking individual changes in protected attributes.
        Returns the maximum discrimination found from changing any single protected attribute.
        """
        nonlocal total_samples, discriminatory_samples, current_local_iter
        current_local_iter += 1

        # Check if this input has already been tested
        if is_case_tested(inp):
            logger.info(f"\rSkipping already tested case in local search")
            return 0

        # Convert input to DataFrame
        inp_df = pd.DataFrame([inp], columns=feature_names)
        original_pred = model.predict(inp_df)[0]

        # Add original case to total inputs
        tot_inputs.add((tuple(inp),))

        # Get values for protected attributes
        protected_values = {}
        for attr in discrimination_data.protected_attributes:
            protected_values[attr] = sorted(discrimination_data.dataframe[attr].unique())

        # Create test cases by changing one protected attribute at a time
        test_cases = []
        for attr in discrimination_data.protected_attributes:
            current_value = inp_df[attr].iloc[0]
            # Try each possible value for this attribute while keeping others constant
            for value in protected_values[attr]:
                if value != current_value:  # Skip if it's the current value
                    new_case = inp_df.copy()
                    new_case[attr] = value

                    # Skip if this case has already been tested
                    if is_case_tested(new_case.iloc[0].values):
                        continue

                    test_cases.append({
                        'data': new_case,
                        'changed_attr': attr,
                        'new_value': value
                    })

        if not test_cases:  # If no new combinations were generated
            return 0

        # Update total samples count
        total_samples += len(test_cases) + 1  # +1 for original case

        max_discrimination = 0

        # Evaluate each test case
        for test_case in test_cases:
            test_df = test_case['data']

            # Add to total inputs before processing
            tot_inputs.add((tuple(test_df.iloc[0].values),))

            counter_pred = model.predict(test_df)[0]

            # Calculate discrimination for this case
            discrimination = abs(original_pred - counter_pred)

            if discrimination > max_discrimination:
                max_discrimination = discrimination

            # If this case shows discrimination above threshold
            if discrimination > threshold:
                # Update discriminatory samples count
                discriminatory_samples += 1

                disc_tuple = tuple(test_df.iloc[0].values)

                # Store discrimination case
                discrimination_cases.append((
                    list(inp_df.iloc[0].values),  # original_features
                    original_pred,  # original_outcome
                    list(test_df.iloc[0].values),  # counter_features
                    counter_pred  # counter_outcome
                ))

                if (disc_tuple,) not in local_disc_inputs:
                    local_disc_inputs.add((disc_tuple,))
                    local_disc_inputs_list.append(list(test_df.iloc[0].values))

                    logger.info(
                        f"\nLocal discrimination found - Attribute: {test_case['changed_attr']} | "
                        f"Change: {inp_df[test_case['changed_attr']].iloc[0]} → {test_case['new_value']} | "
                        f"Prediction: {original_pred} → {counter_pred} | "
                        f"Magnitude: {discrimination:.4f}"
                    )
                    print_progress("local")

        # Print progress periodically
        if current_local_iter % 10 == 0:
            print_progress("local")

        return max_discrimination

    class Local_Perturbation(object):

        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, x):
            """
            Local perturbation of inputs by making small changes to feature values
            """
            s = self.stepsize
            for i in range(len(x)):
                # Skip protected attributes
                if feature_names[i] in discrimination_data.protected_attributes:
                    continue

                # Randomly perturb other features
                random.seed(time.time())
                if random.random() < direction_probability[i]:
                    if x[i] + s <= 1:
                        x[i] = x[i] + s
                else:
                    if x[i] - s >= 0:
                        x[i] = x[i] - s

            # Randomly set one protected attribute to a random valid value
            if discrimination_data.protected_attributes:  # If there are any protected attributes
                attr = random.choice(discrimination_data.protected_attributes)
                idx = feature_names.index(attr)
                possible_values = sorted(discrimination_data.dataframe[attr].unique())
                x[idx] = random.choice(possible_values)

            return x

    class Global_Discovery(object):
        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, x):
            """
            Global discovery of discriminatory inputs by randomly perturbing feature values
            """
            for i in range(len(x)):
                random.seed(time.time())
                x[i] = random.random()

            # Set all protected attributes to their minimum values
            for attr in discrimination_data.protected_attributes:
                idx = feature_names.index(attr)
                possible_values = sorted(discrimination_data.dataframe[attr].unique())
                x[idx] = possible_values[0]  # Use the first (minimum) value

            return x

    # Initialize start time and progress tracking
    logger.info("Starting Global Search...")
    print_progress()

    initial_input = list(discrimination_data.xdf.sample(n=1).to_numpy()[0])

    minimizer = {"method": "L-BFGS-B"}

    global_discovery = Global_Discovery()
    local_perturbation = Local_Perturbation()

    basinhopping(evaluate_global, initial_input, stepsize=1.0, take_step=global_discovery, minimizer_kwargs=minimizer,
                 niter=global_iteration_limit)

    logger.info("Finished Global Search")
    metrics = calculate_metrics()
    logger.info(f"TSN: {metrics['TSN']}")
    logger.info(f"DSN: {metrics['DSN']}")
    logger.info(f"Success Rate (SUR = DSN/TSN): {metrics['SUR']:.4f}")
    logger.info(f"Discriminatory Sample Search time (DSS): {metrics['DSS']:.4f} seconds")
    logger.info(f"Total Time: {metrics['Total_Time']:.2f} seconds")
    logger.info("")
    logger.info("Starting Local Search")

    for inp in global_disc_inputs_list[:1]:
        basinhopping(evaluate_local, inp, stepsize=1.0, take_step=local_perturbation, minimizer_kwargs=minimizer,
                     niter=local_iteration_limit)
        metrics = calculate_metrics()
        logger.info(f"TSN: {metrics['TSN']}")
        logger.info(f"DSN: {metrics['DSN']}")
        logger.info(f"Success Rate (SUR = DSN/TSN): {metrics['SUR']:.4f}")
        logger.info(f"Discriminatory Sample Search time (DSS): {metrics['DSS']:.4f} seconds")
        logger.info(f"Total Time: {metrics['Total_Time']:.2f} seconds")

    logger.info("")
    logger.info("Local Search Finished")
    metrics = calculate_metrics()
    logger.info(f"TSN: {metrics['TSN']}")
    logger.info(f"DSN: {metrics['DSN']}")
    logger.info(f"Success Rate (SUR = DSN/TSN): {metrics['SUR']:.4f}")
    logger.info(f"Discriminatory Sample Search time (DSS): {metrics['DSS']:.4f} seconds")
    logger.info(f"Total Time: {metrics['Total_Time']:.2f} seconds")
    logger.info("")
    logger.info(f"Total Inputs are {len(tot_inputs)}")
    logger.info(f"Number of discriminatory inputs are {len(global_disc_inputs_list) + len(local_disc_inputs_list)}")
    logger.info(f"Time running : {(time.time() - start_time):.2f} seconds")

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate metrics
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
    logger.info(f"Total Discriminatory Pairs Found: {len(discrimination_cases)}")

    res_df = []
    case_id = 0
    for org, org_outcome, counter_org, counter_org_outcome in discrimination_cases:
        indv1 = pd.DataFrame([list(org)], columns=discrimination_data.attr_columns)
        indv2 = pd.DataFrame([list(counter_org)], columns=discrimination_data.attr_columns)

        indv_key1 = "|".join(str(x) for x in indv1[discrimination_data.attr_columns].iloc[0])
        indv_key2 = "|".join(str(x) for x in indv2[discrimination_data.attr_columns].iloc[0])

        # Add the additional columns
        indv1['indv_key'] = indv_key1
        indv1['outcome'] = int(org_outcome)
        indv2['indv_key'] = indv_key2
        indv2['outcome'] = int(counter_org_outcome)

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
    ge, schema = get_real_data('adult')
    ll = run_aequitas(discrimination_data=ge)
    print('ddd')
