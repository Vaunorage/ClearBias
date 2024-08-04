import math
import warnings

import numpy as np
import random
import logging

import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
from methods.exp_ga.genetic_algorithm import GA
from methods.exp_ga.config import census, credit, bank

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def construct_explainer(train_vectors, feature_names, class_names):
    return LimeTabularExplainer(
        train_vectors, feature_names=feature_names, class_names=class_names, discretize_continuous=False
    )


def search_seed(model, feature_names, sens_name, explainer, train_vectors, num, threshold_l):
    seed = []
    for x in train_vectors:
        exp = explainer.explain_instance(x, model.predict_proba, num_features=num)
        exp_result = exp.as_list(label=exp.available_labels()[0])
        rank = [item[0] for item in exp_result]
        loc = rank.index(sens_name)
        if loc < math.ceil(len(exp_result) * threshold_l):
            seed.append(x)
        if len(seed) >= 100:
            break
    return seed


class GlobalDiscovery:
    def __init__(self, step_size=1):
        self.step_size = step_size

    def __call__(self, iteration, params, input_bounds, sensitive_param):
        samples = []
        for _ in range(iteration):
            sample = [random.randint(bounds[0], bounds[1]) for bounds in input_bounds]
            sample[sensitive_param - 1] = 0
            samples.append(np.array(sample))
        return samples


def xai_fair_testing(dataset, threshold, threshold_rank, sensitive_param, max_global, max_local):
    data_config = {"census": census, "credit": credit, "bank": bank}
    config = data_config[dataset]()

    # Load data and prepare model
    X, Y = config.get_dataframe(preprocess=True)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, Y)

    global_disc_inputs = set()
    local_disc_inputs = set()
    total_inputs = set()

    results = []

    def evaluate_local(input_sample):
        input_sample = input_sample.squeeze()
        input_array = input_sample.squeeze()
        total_inputs.add(tuple(input_array))

        for val in range(*config.input_bounds[config.sensitive_indices[sensitive_param]]):
            if val != input_sample[config.sensitive_indices[sensitive_param]]:
                altered_input = input_array.copy()
                altered_input[config.sensitive_indices[sensitive_param]] = val

                output_original = model.predict(pd.DataFrame(input_array.reshape(1, -1), columns=config.feature_name))
                output_altered = model.predict(pd.DataFrame(altered_input.reshape(1, -1), columns=config.feature_name))

                if (abs(output_original - output_altered) > threshold and tuple(input_array)
                        not in global_disc_inputs.union(local_disc_inputs)):
                    local_disc_inputs.add(tuple(input_array))
                    results.append((input_array, altered_input, output_original[0], output_altered[0]))
                    return 2 * abs(output_altered - output_original) + 1
        return 2 * abs(output_altered - output_original) + 1

    global_discovery = GlobalDiscovery()
    train_samples = global_discovery(max_global, config.params, config.input_bounds,
                                     config.sensitive_indices[sensitive_param])

    explainer = construct_explainer(X, config.feature_name, config.class_name)
    seed = search_seed(model, config.feature_name, config.sens_name[sensitive_param], explainer, train_samples,
                       config.params, threshold_rank)

    if not seed:
        logger.info("No seeds found. Exiting...")
        return

    for input_sample in seed:
        input_array = np.array([int(i) for i in input_sample]).reshape(1, -1)
        global_disc_inputs.add(tuple(map(tuple, input_array)))

    # Local search
    ga = GA(
        nums=list(global_disc_inputs), bound=config.input_bounds, func=evaluate_local,
        DNA_SIZE=len(config.input_bounds), cross_rate=0.9, mutation=0.05
    )

    for _ in tqdm(range(max_local), desc="Local search progress"):
        ga.evolve()

    logger.info(f"Total Inputs: {len(total_inputs)}")
    logger.info(f"Discriminatory inputs: {len(local_disc_inputs)}")
    logger.info(
        f"Percentage discriminatory inputs: {float(len(local_disc_inputs)) / float(len(total_inputs)) * 100:.2f}%"
    )

    # Create DataFrame from results
    df = pd.DataFrame(results, columns=["Original Input", "Altered Input", "Original Outcome", "Altered Outcome"])
    df["Original Input"] = df["Original Input"].apply(lambda x: list(x))
    df["Altered Input"] = df["Altered Input"].apply(lambda x: list(x))

    return df

    # Display DataFrame

# Usage example:
df = xai_fair_testing("census", 0.5, 0.5, "age", 50, 50)
print('ddd')
