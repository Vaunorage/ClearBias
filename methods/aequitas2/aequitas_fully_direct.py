from methods.aequitas2.utils import get_adult_dataset, prepare_data_for_fairness_testing
from methods.aequitas2 import utils as ut
import numpy as np
from methods.aequitas2 import loss_funcs as lf
import random
import time
from scipy.optimize import basinhopping
from methods.aequitas2 import config
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

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

# Get the generated initial input and optimizer settings
initial_input = configs['initial_input']
init_prob = 0.5
direction_probability = [init_prob] * params
direction_probability_change_size = 0.001

param_probability = [1.0 / params] * params
param_probability_change_size = 0.001

loss_function = lf._logistic_loss
sep_constraint = 0


class ModelFactory:
    def __init__(self):
        self.models = {
            'logistic': self._create_logistic_regression,
            'svm': self._create_svm,
            'random_forest': self._create_random_forest,
            'neural_network': self._create_neural_network
        }

    def _create_logistic_regression(self):
        return LogisticRegression(random_state=42, max_iter=1000)

    def _create_svm(self):
        return SVC(kernel='rbf', probability=True, random_state=42)

    def _create_random_forest(self):
        return RandomForestClassifier(n_estimators=100, random_state=42)

    def _create_neural_network(self):
        return MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)

    def get_model(self, model_type='logistic'):
        model_creator = self.models.get(model_type.lower())
        if model_creator is None:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model_creator()


def train_model_sklearn(X, y, model_type='logistic'):
    """
    Train a sklearn model of the specified type
    """
    factory = ModelFactory()
    model = factory.get_model(model_type)
    model.fit(X, y)
    return model


def evaluate_input_sklearn(inp, model, sensitive_param, threshold):
    """
    Evaluate input using sklearn models
    """
    inp0 = [int(i) for i in inp]
    inp1 = [int(i) for i in inp]

    inp0[sensitive_param - 1] = 0
    inp1[sensitive_param - 1] = 1

    # Reshape for sklearn prediction
    inp0 = np.array(inp0).reshape(1, -1)
    inp1 = np.array(inp1).reshape(1, -1)

    # Get probabilities instead of just predictions
    out0 = model.predict_proba(inp0)[:, 1]
    out1 = model.predict_proba(inp1)[:, 1]

    return float(abs(out0 - out1) > threshold)


# gamma = None
# model = ut.train_model(X, Y, sensitive, loss_function, 1, 0, sep_constraint, sensitive_attrs,
#                        sensitive_attrs_to_cov_thresh, gamma)

model = train_model_sklearn(X, Y, 'logistic')


def normalise_probability():
    probability_sum = 0.0
    for prob in param_probability:
        probability_sum = probability_sum + prob

    for i in range(params):
        param_probability[i] = float(param_probability[i]) / float(probability_sum)


# First, define the base perturbation class
class BasePerturbation(object):
    def __init__(self, stepsize=1):
        self.stepsize = stepsize


# Fully Directed implementation
class FullyDirectedLocalPerturbation(BasePerturbation):
    def __call__(self, x):
        s = self.stepsize
        param_choice = np.random.choice(range(params), p=param_probability)
        perturbation_options = [-1, 1]

        direction_choice = np.random.choice(perturbation_options,
                                            p=[direction_probability[param_choice],
                                               (1 - direction_probability[param_choice])])

        if (x[param_choice] == input_bounds[param_choice][0]) or (x[param_choice] == input_bounds[param_choice][1]):
            direction_choice = np.random.choice(perturbation_options)

        x[param_choice] = x[param_choice] + (direction_choice * perturbation_unit)
        x[param_choice] = max(input_bounds[param_choice][0], x[param_choice])
        x[param_choice] = min(input_bounds[param_choice][1], x[param_choice])

        ei = evaluate_input(x)

        if (ei and direction_choice == -1) or (not ei and direction_choice == 1):
            direction_probability[param_choice] = min(direction_probability[param_choice] +
                                                      (direction_probability_change_size * perturbation_unit), 1)
        elif (not ei and direction_choice == -1) or (ei and direction_choice == 1):
            direction_probability[param_choice] = max(direction_probability[param_choice] -
                                                      (direction_probability_change_size * perturbation_unit), 0)

        if ei:
            param_probability[param_choice] = param_probability[param_choice] + param_probability_change_size
            normalise_probability()
        else:
            param_probability[param_choice] = max(param_probability[param_choice] - param_probability_change_size, 0)
            normalise_probability()

        return x


# Semi-directed implementation
class SemiDirectedLocalPerturbation(BasePerturbation):
    def __call__(self, x):
        s = self.stepsize
        param_choice = random.randint(0, params - 1)
        perturbation_options = [-1, 1]

        direction_choice = np.random.choice(perturbation_options,
                                            p=[direction_probability[param_choice],
                                               (1 - direction_probability[param_choice])])

        if (x[param_choice] == input_bounds[param_choice][0]) or (x[param_choice] == input_bounds[param_choice][1]):
            direction_choice = np.random.choice(perturbation_options)

        x[param_choice] = x[param_choice] + (direction_choice * perturbation_unit)
        x[param_choice] = max(input_bounds[param_choice][0], x[param_choice])
        x[param_choice] = min(input_bounds[param_choice][1], x[param_choice])

        ei = evaluate_input(x)

        if (ei and direction_choice == -1) or (not ei and direction_choice == 1):
            direction_probability[param_choice] = min(direction_probability[param_choice] +
                                                      (direction_probability_change_size * perturbation_unit), 1)
        elif (not ei and direction_choice == -1) or (ei and direction_choice == 1):
            direction_probability[param_choice] = max(direction_probability[param_choice] -
                                                      (direction_probability_change_size * perturbation_unit), 0)

        return x


# Random implementation
class RandomLocalPerturbation(BasePerturbation):
    def __call__(self, x):
        s = self.stepsize
        param_choice = random.randint(0, params - 1)
        direction_choice = random.choice([-1, 1])

        x[param_choice] = x[param_choice] + (direction_choice * perturbation_unit)
        x[param_choice] = max(input_bounds[param_choice][0], x[param_choice])
        x[param_choice] = min(input_bounds[param_choice][1], x[param_choice])

        return x


def get_perturbation_strategy(strategy='fully-directed'):
    strategies = {
        'fully-directed': FullyDirectedLocalPerturbation,
        'semi-directed': SemiDirectedLocalPerturbation,
        'random': RandomLocalPerturbation
    }
    return strategies.get(strategy, FullyDirectedLocalPerturbation)()


class Global_Discovery(object):
    def __init__(self, stepsize=1):
        self.stepsize = stepsize

    def __call__(self, x):
        s = self.stepsize
        # Updated xrange to range
        for i in range(params):
            random.seed(time.time())
            x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])

        x[sensitive_param - 1] = 0
        return x


def evaluate_input(inp):
    return evaluate_input_sklearn(inp, model, sensitive_param, threshold)


def evaluate_global(inp):
    inp0 = [int(i) for i in inp]
    inp1 = [int(i) for i in inp]

    inp0[sensitive_param - 1] = 0
    inp1[sensitive_param - 1] = 1

    # Reshape for sklearn prediction
    inp0_reshaped = np.array(inp0).reshape(1, -1)
    inp1_reshaped = np.array(inp1).reshape(1, -1)

    # Get probabilities instead of just predictions
    out0 = model.predict_proba(inp0_reshaped)[:, 1]
    out1 = model.predict_proba(inp1_reshaped)[:, 1]

    tot_inputs.add(tuple(inp0))

    if (abs(out0 - out1) > threshold and tuple(inp0) not in global_disc_inputs):
        global_disc_inputs.add(tuple(inp0))
        global_disc_inputs_list.append(inp0)

    return float(not abs(out0 - out1) > threshold)


def evaluate_local(inp):
    inp0 = [int(i) for i in inp]
    inp1 = [int(i) for i in inp]

    inp0[sensitive_param - 1] = 0
    inp1[sensitive_param - 1] = 1

    # Reshape for sklearn prediction
    inp0_reshaped = np.array(inp0).reshape(1, -1)
    inp1_reshaped = np.array(inp1).reshape(1, -1)

    # Get probabilities instead of just predictions
    out0 = model.predict_proba(inp0_reshaped)[:, 1]
    out1 = model.predict_proba(inp1_reshaped)[:, 1]

    tot_inputs.add(tuple(inp0))

    if (abs(out0 - out1) > threshold and (tuple(inp0) not in global_disc_inputs)
            and (tuple(inp0) not in local_disc_inputs)):
        local_disc_inputs.add(tuple(inp0))
        local_disc_inputs_list.append(inp0)

    return float(not abs(out0 - out1) > threshold)

minimizer = {"method": "L-BFGS-B"}
global_discovery = Global_Discovery()
local_perturbation = get_perturbation_strategy(strategy='fully-directed')

basinhopping(evaluate_global, initial_input, stepsize=1.0, take_step=global_discovery,
             minimizer_kwargs=minimizer, niter=global_iteration_limit)

print("Finished Global Search")
print("Percentage discriminatory inputs - " +
      str(float(len(global_disc_inputs_list) + len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100))
print("")
print("Starting Local Search")

for inp in global_disc_inputs_list:
    basinhopping(evaluate_local, inp, stepsize=1.0, take_step=local_perturbation,
                 minimizer_kwargs=minimizer, niter=local_iteration_limit)
    print("Percentage discriminatory inputs - " +
          str(float(len(global_disc_inputs_list) + len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100))

print("")
print("Local Search Finished")
print("Percentage discriminatory inputs - " +
      str(float(len(global_disc_inputs_list) + len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100))

print("")
print("Total Inputs are " + str(len(tot_inputs)))
print("Number of discriminatory inputs are " + str(len(global_disc_inputs_list) + len(local_disc_inputs_list)))
