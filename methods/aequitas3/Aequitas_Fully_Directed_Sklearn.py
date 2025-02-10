from __future__ import division

import itertools
import pandas as pd
from data_generator.main import get_real_data

import numpy as np
import random
import time
from scipy.optimize import basinhopping
import config
from methods.utils import train_sklearn_model

import logging
import sys

# Configure logger
logger = logging.getLogger('aequitas')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

random.seed(time.time())

# Add argument parsing before the main logic

ge, ge_schema = get_real_data('adult')

# Train the model using the new function
model, X_train, X_test, y_train, y_test, feature_names = train_sklearn_model(
    data=ge.dataframe,
    model_type='rf',  # You can change this to 'svm', 'lr', or 'dt'
    sensitive_attrs=ge.protected_attributes,
    target_col=ge.outcome_column
)

# Save the trained model and feature names
model_data = {
    'model': model,
    'feature_names': feature_names
}

init_prob = 0.5
params = len(feature_names)
direction_probability = [init_prob] * params
direction_probability_change_size = 0.001

param_probability = [1.0 / params] * params
param_probability_change_size = 0.001

def normalise_probability():
    """Normalize the probability distribution to ensure it sums to 1"""
    probability_sum = 0.0
    for prob in param_probability:
        probability_sum = probability_sum + prob

    for i in range(params):
        param_probability[i] = float(param_probability[i]) / float(probability_sum)

sensitive_param = config.sensitive_param
name = 'sex'
cov = 0

perturbation_unit = config.perturbation_unit

threshold = config.threshold

global_disc_inputs = set()
global_disc_inputs_list = []

local_disc_inputs = set()
local_disc_inputs_list = []

tot_inputs = set()

# Global variables for storing discrimination cases
discrimination_cases = []  # List of (original_features, original_outcome, counter_features, counter_outcome)

global_iteration_limit = 100
local_iteration_limit = 100

input_bounds = config.input_bounds

# Global variables for metrics and progress tracking
start_time = None
total_samples = 0  # TSN
discriminatory_samples = 0  # DSN
current_global_iter = 0
current_local_iter = 0

def calculate_metrics():
    """Calculate and return the testing metrics"""
    global start_time, total_samples, discriminatory_samples
    
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
    metrics = calculate_metrics()
    global current_global_iter, current_local_iter
    
    # Clear previous line
    sys.stdout.write('\r')
    sys.stdout.flush()
    
    # Create progress message
    if search_type == "global":
        progress = f"Global: {current_global_iter}/{global_iteration_limit} ({(current_global_iter/global_iteration_limit)*100:.1f}%)"
    elif search_type == "local":
        progress = f"Local: {current_local_iter}/{local_iteration_limit} ({(current_local_iter/local_iteration_limit)*100:.1f}%)"
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
    inp0 = [int(i) for i in inp]
    inp1 = [int(i) for i in inp]

    inp0[feature_names.index(sensitive_param)] = 0
    inp1[feature_names.index(sensitive_param)] = 1

    inp0 = np.asarray(inp0)
    inp0 = np.reshape(inp0, (1, -1))

    inp1 = np.asarray(inp1)
    inp1 = np.reshape(inp1, (1, -1))

    out0 = model.predict(inp0)
    out1 = model.predict(inp1)

    return abs(out1 + out0)


def evaluate_global(inp):
    """
    Evaluate discrimination across all possible combinations of protected attributes.
    Returns the maximum discrimination found between any pair of protected attribute combinations.
    """
    global total_samples, discriminatory_samples, current_global_iter
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
    for attr in ge.protected_attributes:
        protected_values[attr] = sorted(ge.dataframe[attr].unique())

    # Generate all possible combinations of protected attribute values
    value_combinations = list(itertools.product(*[protected_values[attr] for attr in ge.protected_attributes]))

    # Create test cases for all combinations
    test_cases = []
    for values in value_combinations:
        # Skip if combination is identical to original
        if all(inp_df[attr].iloc[0] == value for attr, value in zip(ge.protected_attributes, values)):
            continue

        new_case = inp_df.copy()
        for attr, value in zip(ge.protected_attributes, values):
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
                original_pred,                 # original_outcome
                list(disc_case.values),        # counter_features
                counter_pred                   # counter_outcome
            ))
            
            if (disc_tuple,) not in global_disc_inputs:
                global_disc_inputs.add((disc_tuple,))
                global_disc_inputs_list.append(list(disc_case.values))
            
            # Calculate discrimination magnitude
            discrimination = abs(original_pred - counter_pred)
            max_discrimination = max(max_discrimination, discrimination)
            
            if discrimination > threshold:
                logger.info(
                    f"\nDiscrimination found - Original: {dict(zip(ge.protected_attributes, inp_df[ge.protected_attributes].iloc[0]))} "
                    f"({original_pred}) → New: {dict(zip(ge.protected_attributes, disc_case[ge.protected_attributes]))} "
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
    global total_samples, discriminatory_samples, current_local_iter
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
    for attr in ge.protected_attributes:
        protected_values[attr] = sorted(ge.dataframe[attr].unique())

    # Create test cases by changing one protected attribute at a time
    test_cases = []
    for attr in ge.protected_attributes:
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
                original_pred,                 # original_outcome
                list(test_df.iloc[0].values),  # counter_features
                counter_pred                   # counter_outcome
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
            if feature_names[i] in ge.protected_attributes:
                continue
                
            # Randomly perturb other features
            random.seed(time.time())
            if random.random() < direction_probability[i]:
                if x[i] + s <= input_bounds[i][1]:
                    x[i] = x[i] + s
            else:
                if x[i] - s >= input_bounds[i][0]:
                    x[i] = x[i] - s

        # Randomly set one protected attribute to a random valid value
        if ge.protected_attributes:  # If there are any protected attributes
            attr = random.choice(ge.protected_attributes)
            idx = feature_names.index(attr)
            possible_values = sorted(ge.dataframe[attr].unique())
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
            x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])

        # Set all protected attributes to their minimum values
        for attr in ge.protected_attributes:
            idx = feature_names.index(attr)
            possible_values = sorted(ge.dataframe[attr].unique())
            x[idx] = possible_values[0]  # Use the first (minimum) value

        return x


# Initialize start time and progress tracking
start_time = time.time()
current_global_iter = 0
current_local_iter = 0

logger.info("Starting Global Search...")
print_progress()

initial_input = list(ge.xdf.sample(n=1).to_numpy()[0])

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

for inp in global_disc_inputs_list:
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


