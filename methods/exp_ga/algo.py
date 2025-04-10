import time
from itertools import product
from typing import TypedDict, List, Tuple, Dict, Any, Union, Set
import math
import uuid
import warnings
import numpy as np
import random
import logging
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from lime.lime_tabular import LimeTabularExplainer

from data_generator.main import DiscriminationData
from methods.exp_ga.genetic_algorithm import GA
from methods.utils import check_for_error_condition, make_final_metrics_and_dataframe

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class Metrics(TypedDict):
    TSN: int  # Total Sample Number
    DSN: int  # Discriminatory Sample Number
    DSS: float  # Discriminatory Sample Search (avg time)
    SUR: float  # Success Rate


ExpGAResultDF = DataFrame


def get_model(model_type: str, **kwargs) -> BaseEstimator:
    """
    Factory function to create different types of models with specified parameters.

    Args:
        model_type: One of 'rf' (Random Forest), 'dt' (Decision Tree),
                   'mlp' (Multi-layer Perceptron), or 'svm' (Support Vector Machine)
        **kwargs: Model-specific parameters
    """
    models = {
        'rf': RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 10),
            random_state=kwargs.get('random_state', 42)
        ),
        'dt': DecisionTreeClassifier(
            random_state=kwargs.get('random_state', 42)
        ),
        'mlp': MLPClassifier(
            hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (100,)),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42)
        ),
        'svm': SVC(
            kernel=kwargs.get('kernel', 'rbf'),
            probability=True,  # Required for LIME
            random_state=kwargs.get('random_state', 42)
        )
    }

    if model_type not in models:
        raise ValueError(f"Model type '{model_type}' not supported. Choose from: {list(models.keys())}")

    return models[model_type]


def construct_explainer(train_vectors: np.ndarray, feature_names: List[str],
                        class_names: List[str]) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        train_vectors, feature_names=feature_names, class_names=class_names, discretize_continuous=False
    )


def search_seed(model: BaseEstimator, feature_names: List[str], sens_name: str,
                explainer: LimeTabularExplainer, train_vectors: np.ndarray, num: int,
                threshold_l: float, nb_seed=100) -> List[np.ndarray]:
    seed: List[np.ndarray] = []
    for i, x in enumerate(train_vectors):
        exp = explainer.explain_instance(x, model.predict_proba, num_features=num)
        exp_result = exp.as_list(label=exp.available_labels()[0])
        rank = [item[0] for item in exp_result]
        loc = rank.index(sens_name)
        if loc < math.ceil(len(exp_result) * threshold_l):
            seed.append(x)
        if len(seed) >= nb_seed:
            break
    return seed


class GlobalDiscovery:
    def __init__(self, step_size: int = 1):
        self.step_size = step_size

    def __call__(self, iteration: int, params: int, input_bounds: List[Tuple[int, int]],
                 sensitive_param: int) -> List[np.ndarray]:
        samples = []
        for _ in range(iteration):
            sample = [random.randint(bounds[0], bounds[1]) for bounds in input_bounds]
            sample[sensitive_param - 1] = 0
            samples.append(np.array(sample))
        return samples


def run_expga(dataset: DiscriminationData, threshold_rank: float,
              max_global: int, max_local: int, model_type: str = 'rf', cross_rate=0.9, mutation=0.1,
              max_runtime_seconds: float = None, max_tsn: int = None, nb_seed=100, one_attr_at_a_time=False,
              **model_kwargs) -> Tuple[pd.DataFrame, Metrics]:
    """
    XAI fairness testing that works on all protected attributes simultaneously.
    Uses a centralized termination check to improve code structure.
    """

    logger.info("Starting ExpGA with all protected attributes")

    start_time = time.time()
    disc_times: List[float] = []

    X, Y = dataset.xdf, dataset.ydf
    model = get_model(model_type, **model_kwargs)
    model.fit(X, Y)

    global_disc_inputs: Set[Tuple[float, ...]] = set()
    local_disc_inputs: Set[Tuple[float, ...]] = set()
    total_inputs: Set[Tuple[float, ...]] = set()
    tot_inputs = set()
    all_discriminations = set()

    early_termination = False

    def should_terminate() -> bool:
        nonlocal early_termination
        current_runtime = time.time() - start_time
        max_runtime_seconds_exceeded = max_runtime_seconds is not None and current_runtime > max_runtime_seconds
        tsn_threshold_reached = max_tsn is not None and len(tot_inputs) >= max_tsn

        if max_runtime_seconds_exceeded or tsn_threshold_reached:
            early_termination = True
            return True
        return False

    dsn_by_attr_value = {e: {'TSN': 0, 'DSN': 0} for e in dataset.protected_attributes}
    dsn_by_attr_value['total'] = 0

    def evaluate_local(input_sample) -> float:
        """
        Evaluates a sample for discrimination across all protected attributes at once.
        """
        if should_terminate():
            return 0.0

        input_array = np.array(input_sample, dtype=float).flatten()

        result, results_df, max_diff, org_df, tested_inp = check_for_error_condition(logger=logger,
                                                                                     discrimination_data=dataset,
                                                                                     dsn_by_attr_value=dsn_by_attr_value,
                                                                                     model=model, instance=input_sample,
                                                                                     tot_inputs=tot_inputs,
                                                                                     all_discriminations=all_discriminations,
                                                                                     one_attr_at_a_time=one_attr_at_a_time)

        if result and tuple(input_array) not in global_disc_inputs.union(local_disc_inputs):
            local_disc_inputs.add(tuple(input_array))
            disc_times.append(time.time() - start_time)
            return 2 * max_diff + 1
        return 0.0

    # Generate seeds for each protected attribute
    seeds = []

    # Use the original global discovery approach
    global_discovery = GlobalDiscovery()

    # Create seeds for each protected attribute
    for p_attr in dataset.protected_attributes:
        if should_terminate():
            break

        sensitive_idx = dataset.sensitive_indices_dict[p_attr]

        # Generate samples focused on this attribute
        attr_samples = global_discovery(
            max_global // len(dataset.protected_attributes),
            len(dataset.feature_names),
            dataset.input_bounds,
            sensitive_idx
        )

        # Initialize explainer if needed
        explainer = construct_explainer(X.values, dataset.feature_names, dataset.outcome_column)

        # Find promising seeds
        attr_seeds = search_seed(
            model,
            dataset.feature_names,
            p_attr,
            explainer,
            np.array(attr_samples),
            len(dataset.feature_names),
            threshold_rank,
            nb_seed
        )

        seeds.extend(attr_seeds)

        # Limit total seeds
        if len(seeds) >= 100:
            break

    # Ensure we have at least some seeds if we haven't terminated
    if not seeds and not should_terminate():
        for _ in range(10):
            if should_terminate():
                break

            random_seed = np.array([float(random.randint(bounds[0], bounds[1]))
                                    for bounds in dataset.input_bounds])
            seeds.append(random_seed)

    # Format seeds for GA
    formatted_seeds = []
    for seed in seeds:
        if should_terminate():
            break

        try:
            seed_array = np.array(seed, dtype=float).flatten()
            if len(seed_array) == len(dataset.feature_names):
                formatted_seeds.append(seed_array)
                global_disc_inputs.add(tuple(seed_array))
                total_inputs.add(tuple(seed_array))
        except:
            continue

    # Run genetic algorithm only if we haven't terminated
    if not should_terminate():
        ga = GA(
            nums=formatted_seeds,
            bound=dataset.input_bounds,
            func=evaluate_local,
            DNA_SIZE=len(dataset.input_bounds),
            cross_rate=cross_rate,
            mutation=mutation
        )

        # Evolve population
        for iteration in range(max_local):
            if should_terminate():
                break

            ga.evolve()

            # Log progress
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: TSN={len(tot_inputs)}, DSN={len(all_discriminations)}")

    # Calculate final results
    res_df, metrics = make_final_metrics_and_dataframe(dataset, tot_inputs, all_discriminations, dsn_by_attr_value,
                                                       start_time, logger=logger)

    # Log whether we terminated early
    if early_termination:
        logger.info("Early termination triggered: either max runtime or max TSN reached")

    return res_df, metrics


if __name__ == "__main__":
    from data_generator.main import get_real_data, DiscriminationData

    data_obj, schema = get_real_data('adult', use_cache=True)

    results_df, metrics = run_expga(
        data_obj,
        threshold_rank=0.5,
        max_global=20000,
        max_local=100,
        cluster_num=50,
        max_runtime_seconds=3600,
        max_tsn=20000,
        step_size=0.05,
        one_attr_at_a_time=True
    )

    print(f"\nTesting Metrics: {metrics}")
