from methods.aequitas2.utils import get_adult_dataset, prepare_data_for_fairness_testing
from methods.aequitas2 import utils as ut
import numpy as np
import random
import time
from scipy.optimize import basinhopping
from methods.aequitas2 import config
from methods.aequitas2 import loss_funcs as lf

random.seed(time.time())
start_time = time.time()

perturbation_unit = config.perturbation_unit
threshold = config.threshold

global_disc_inputs = set()
global_disc_inputs_list = []
local_disc_inputs = set()
local_disc_inputs_list = []
tot_inputs = set()

global_iteration_limit = 6000
local_iteration_limit = 1000

# Get and prepare the dataset
X, Y, sensitive = get_adult_dataset()
X, Y, sensitive, configs = prepare_data_for_fairness_testing(X, Y, sensitive, 'sex')

# Access the configurations
params = configs['params']
sensitive_param = configs['sensitive_param']
input_bounds = configs['input_bounds']
name = configs['name']
cov = configs['cov']
sensitive_attrs = configs['sensitive_attrs']
sensitive_attrs_to_cov_thresh = configs['sensitive_attrs_to_cov_thresh']

# Use the original loss function and constraints
loss_function = lf._logistic_loss
sep_constraint = 0
gamma = None

# Train model using original utility function
model = ut.train_model(X, Y, sensitive, loss_function, 1, 0, sep_constraint, sensitive_attrs,
                       sensitive_attrs_to_cov_thresh, gamma)

initial_input = configs['initial_input']
init_prob = 0.5
direction_probability = [init_prob] * params
direction_probability_change_size = 0.001

param_probability = [1.0 / params] * params
param_probability_change_size = 0.001


def normalise_probability():
    probability_sum = sum(param_probability)
    for i in range(params):
        param_probability[i] = float(param_probability[i]) / float(probability_sum)


def evaluate_input(inp):
    """Original evaluation function"""
    inp0 = [int(i) for i in inp]
    inp1 = [int(i) for i in inp]

    inp0[sensitive_param - 1] = 0
    inp1[sensitive_param - 1] = 1

    out0 = np.sign(np.dot(model, inp0))
    out1 = np.sign(np.dot(model, inp1))

    return float(abs(out0 - out1) > threshold)


class FullyDirectedLocalPerturbation:
    def __init__(self, stepsize=1):
        self.stepsize = stepsize

    def __call__(self, x):
        param_choice = np.random.choice(range(params), p=param_probability)
        perturbation_options = [-1, 1]

        direction_choice = np.random.choice(perturbation_options,
                                            p=[direction_probability[param_choice],
                                               1 - direction_probability[param_choice]])

        if (x[param_choice] == input_bounds[param_choice][0] or
                x[param_choice] == input_bounds[param_choice][1]):
            direction_choice = np.random.choice(perturbation_options)

        x[param_choice] = x[param_choice] + (direction_choice * perturbation_unit)
        x[param_choice] = max(input_bounds[param_choice][0], x[param_choice])
        x[param_choice] = min(input_bounds[param_choice][1], x[param_choice])

        ei = evaluate_input(x)

        if (ei and direction_choice == -1) or (not ei and direction_choice == 1):
            direction_probability[param_choice] = min(
                direction_probability[param_choice] + (direction_probability_change_size * perturbation_unit), 1)
        elif (not ei and direction_choice == -1) or (ei and direction_choice == 1):
            direction_probability[param_choice] = max(
                direction_probability[param_choice] - (direction_probability_change_size * perturbation_unit), 0)

        if ei:
            param_probability[param_choice] += param_probability_change_size
            normalise_probability()
        else:
            param_probability[param_choice] = max(param_probability[param_choice] - param_probability_change_size, 0)
            normalise_probability()

        return x


class GlobalDiscovery:
    def __init__(self, stepsize=1):
        self.stepsize = stepsize

    def __call__(self, x):
        for i in range(params):
            random.seed(time.time())
            x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])
        x[sensitive_param - 1] = 0
        return x


def evaluate_global(inp):
    inp0 = [int(i) for i in inp]
    inp1 = [int(i) for i in inp]

    inp0[sensitive_param - 1] = 0
    inp1[sensitive_param - 1] = 1

    out0 = np.sign(np.dot(model, inp0))
    out1 = np.sign(np.dot(model, inp1))

    tot_inputs.add(tuple(inp0))

    if abs(out0 - out1) > threshold and tuple(inp0) not in global_disc_inputs:
        global_disc_inputs.add(tuple(inp0))
        global_disc_inputs_list.append(inp0)

    return float(not abs(out0 - out1) > threshold)


def evaluate_local(inp):
    inp0 = [int(i) for i in inp]
    inp1 = [int(i) for i in inp]

    inp0[sensitive_param - 1] = 0
    inp1[sensitive_param - 1] = 1

    out0 = np.sign(np.dot(model, inp0))
    out1 = np.sign(np.dot(model, inp1))

    tot_inputs.add(tuple(inp0))

    if (abs(out0 - out1) > threshold and
            tuple(inp0) not in global_disc_inputs and
            tuple(inp0) not in local_disc_inputs):
        local_disc_inputs.add(tuple(inp0))
        local_disc_inputs_list.append(inp0)

    return float(not abs(out0 - out1) > threshold)


# Print initial model info
print(model, -1.0, {'sex': sensitive['sex'], 'race': sensitive['race']},
      loss_function, 1, 0, 0, sensitive_attrs, sensitive_attrs_to_cov_thresh)

minimizer = {"method": "L-BFGS-B"}
global_discovery = GlobalDiscovery()
local_perturbation = FullyDirectedLocalPerturbation()

basinhopping(evaluate_global, initial_input, stepsize=1.0, take_step=global_discovery,
             minimizer_kwargs=minimizer, niter=global_iteration_limit)

print("Finished Global Search")
print(
    f"Percentage discriminatory inputs - {(len(global_disc_inputs_list) + len(local_disc_inputs_list)) / len(tot_inputs) * 100}")
print("\nStarting Local Search")

for inp in global_disc_inputs_list:
    basinhopping(evaluate_local, inp, stepsize=1.0, take_step=local_perturbation,
                 minimizer_kwargs=minimizer, niter=local_iteration_limit)
    print(
        f"Percentage discriminatory inputs - {(len(global_disc_inputs_list) + len(local_disc_inputs_list)) / len(tot_inputs) * 100}")

print("\nLocal Search Finished")
print(
    f"Percentage discriminatory inputs - {(len(global_disc_inputs_list) + len(local_disc_inputs_list)) / len(tot_inputs) * 100}")
print(f"\nTotal Inputs are {len(tot_inputs)}")
print(f"Number of discriminatory inputs are {len(global_disc_inputs_list) + len(local_disc_inputs_list)}")