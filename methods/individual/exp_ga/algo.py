import time
from typing import TypedDict, List, Tuple, Set
import math
import warnings
import numpy as np
import random
import logging
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator
from lime.lime_tabular import LimeTabularExplainer

from data_generator.main import DiscriminationData
from methods.utils import (check_for_error_condition, make_final_metrics_and_dataframe,
                           train_sklearn_model)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class Metrics(TypedDict):
    TSN: int
    DSN: int
    DSS: float
    SUR: float


ExpGAResultDF = DataFrame


class GA:
    def __init__(self, nums, bound, func, DNA_SIZE=None, cross_rate=0.8, mutation=0.003):
        # Convert input nums to proper numpy array shape
        self.nums = np.array(nums, dtype=np.int32)
        if len(self.nums.shape) == 2:
            self.nums = self.nums.reshape(len(nums), -1)

        self.bound = np.array(bound)
        self.func = func
        self.cross_rate = cross_rate
        self.mutation = mutation

        self.min_nums, self.max_nums = self.bound[:, 0], self.bound[:, 1]
        self.var_len = self.max_nums - self.min_nums

        if DNA_SIZE is None:
            self.DNA_SIZE = int(np.ceil(np.max(np.log2(self.var_len + 1))))
        else:
            self.DNA_SIZE = DNA_SIZE

        self.POP_SIZE = len(nums)
        self.POP = self.nums.copy()
        self.copy_POP = self.nums.copy()

        # Ensure POP has correct shape
        if len(self.POP.shape) == 1:
            self.POP = self.POP.reshape(self.POP_SIZE, -1)
            self.copy_POP = self.copy_POP.reshape(self.POP_SIZE, -1)

    def get_fitness(self, non_negative=False):
        result = np.array([self.func(individual) for individual in self.POP])
        if non_negative:
            result = result - np.min(result)
        return result

    def select(self):
        fitness = self.get_fitness()

        if len(fitness) == 0:
            raise ValueError("Fitness array is empty.")

        total_fitness = np.sum(fitness)
        if total_fitness == 0:
            probabilities = np.ones(len(fitness)) / len(fitness)
        else:
            probabilities = fitness / total_fitness

        probabilities = probabilities.squeeze()
        if probabilities.ndim == 0:
            probabilities = np.array([probabilities])

        probabilities = probabilities / np.sum(probabilities)

        selected_indices = np.random.choice(np.arange(self.POP_SIZE), size=self.POP_SIZE, replace=True, p=probabilities)
        self.POP = self.POP[selected_indices]

    def crossover(self):
        for i in range(self.POP_SIZE):
            if np.random.rand() < self.cross_rate:
                partner_idx = np.random.randint(0, self.POP_SIZE)

                # Ensure proper array shapes for crossover
                if len(self.POP[i].shape) == 1 and len(self.POP[partner_idx].shape) == 1:
                    cross_points = np.random.randint(0, self.POP[i].size)
                    end_points = np.random.randint(cross_points + 1, self.POP[i].size + 1)

                    # Create copies to avoid modifying original arrays
                    temp = self.POP[i].copy()
                    temp[cross_points:end_points] = self.POP[partner_idx][cross_points:end_points]
                    self.POP[i] = temp

    def mutate(self):
        for i in range(len(self.POP)):
            individual = self.POP[i].flatten()  # Ensure 1D array for mutation

            for gene_idx in range(individual.size):
                if np.random.rand() < self.mutation:
                    if gene_idx < len(self.bound):  # Check if we have bounds for this gene
                        low = int(self.bound[gene_idx][0])
                        high = int(self.bound[gene_idx][1])

                        if high > low:
                            new_value = np.random.randint(low, high + 1)
                            individual[gene_idx] = new_value

            # Reshape back if necessary and update population
            self.POP[i] = individual.reshape(self.POP[i].shape)

    def evolve(self):
        self.select()
        self.crossover()
        self.mutate()

    def reset(self):
        self.POP = self.copy_POP.copy()

    def log(self):
        fitness = self.get_fitness()
        population_log = [ind.flatten().tolist() for ind in self.POP]
        return population_log, fitness.tolist()


def construct_explainer(train_vectors: np.ndarray, feature_names: List[str],
                        class_names: List[str]) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        train_vectors, feature_names=feature_names, class_names=class_names, discretize_continuous=False
    )


def search_seed(model: BaseEstimator, feature_names: List[str], sens_name: str,
                explainer: LimeTabularExplainer, train_vectors: np.ndarray, num: int,
                threshold_l: float, nb_seed=100, start_time=None, max_runtime_seconds=None) -> List[np.ndarray]:
    """
    Searches for seed instances where the sensitive attribute is influential in model prediction.

    Args:
        model: The trained model
        feature_names: List of feature names
        sens_name: Name of the sensitive attribute
        explainer: LIME explainer instance
        train_vectors: Training data vectors
        num: Number of features for LIME explanation
        threshold_l: Threshold for considering a feature influential
        nb_seed: Maximum number of seeds to find
        start_time: Start time of the entire process (for time limit checking)
        max_runtime_seconds: Maximum runtime allowed in seconds

    Returns:
        List of seed instances
    """
    logger.info(f"Starting search for seeds with sensitive attribute: {sens_name}")

    seed: List[np.ndarray] = []

    for i, x in enumerate(train_vectors):
        # Check time limit if applicable
        if start_time is not None and max_runtime_seconds is not None:
            current_runtime = time.time() - start_time
            if current_runtime > max_runtime_seconds:
                logger.info(f"Time limit of {max_runtime_seconds} seconds reached during seed search. "
                            f"Found {len(seed)} seeds so far.")
                break

        # Log progress periodically (every 100 instances)
        if i > 0 and i % 100 == 0:
            logger.info(f"Processed {i}/{len(train_vectors)} instances, found {len(seed)}/{nb_seed} seeds")

        exp = explainer.explain_instance(x, model.predict_proba, num_features=num, num_samples=1000)
        exp_result = exp.as_list(label=exp.available_labels()[0])
        rank = [item[0] for item in exp_result]

        if sens_name in rank:
            loc = rank.index(sens_name)
            if loc < math.ceil(len(exp_result) * threshold_l):
                seed.append(x)

        if len(seed) >= nb_seed:
            logger.info(f"Found all {nb_seed} seeds required. Search complete.")
            break

    logger.info(f"Search complete: Found {len(seed)}/{nb_seed} seeds after examining {i + 1} instances")

    return seed


class GlobalDiscovery:
    def __init__(self, step_size: int = 1):
        self.step_size = step_size

    def __call__(self, iteration: int, params: int, input_bounds: List[Tuple[int, int]],
                 sensitive_param: int) -> List[np.ndarray]:
        samples = []
        unique_tuples = set()

        attempts = 0
        max_attempts = iteration * 10  # Set a reasonable limit to prevent infinite loops

        while len(samples) < iteration and attempts < max_attempts:
            attempts += 1
            sample = [random.randint(bounds[0], bounds[1]) for bounds in input_bounds]
            sample[sensitive_param - 1] = 0

            sample_tuple = tuple(sample)

            if sample_tuple not in unique_tuples:
                unique_tuples.add(sample_tuple)
                samples.append(np.array(sample))

        return samples


def run_expga(data: DiscriminationData, threshold_rank: float, max_global: int, max_local: int, model_type: str = 'rf',
              cross_rate=0.9, mutation=0.1, max_runtime_seconds: float = None, max_tsn: int = None, random_seed=100,
              one_attr_at_a_time=False, db_path=None, analysis_id=None, use_gpu=False,
              **model_kwargs) -> Tuple[pd.DataFrame, Metrics]:
    """
    XAI fairness testing that works on all protected attributes simultaneously.
    Uses a centralized termination check to improve code structure.
    """

    logger.info("Starting ExpGA with all protected attributes")

    start_time = time.time()
    disc_times: List[float] = []

    X, Y = data.xdf, data.ydf

    model, X_train, X_test, y_train, y_test, feature_names = train_sklearn_model(
        data=data.training_dataframe.copy(),
        model_type=model_type,
        target_col=data.outcome_column,
        sensitive_attrs=list(data.protected_attributes),
        random_state=random_seed,
        use_cache=True,
        use_gpu=use_gpu
    )

    global_disc_inputs: Set[Tuple[float, ...]] = set()
    local_disc_inputs: Set[Tuple[float, ...]] = set()
    total_inputs: Set[Tuple[float, ...]] = set()
    tot_inputs = set()
    all_discriminations = set()
    all_tot_inputs = []

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

    dsn_by_attr_value = {e: {'TSN': 0, 'DSN': 0} for e in data.protected_attributes}
    dsn_by_attr_value['total'] = 0

    def evaluate_local(input_sample) -> float:
        """
        Evaluates a sample for discrimination across all protected attributes at once.
        """
        if should_terminate():
            return 0.0

        input_array = np.array(input_sample, dtype=float).flatten()

        result, results_df, max_diff, org_df, tested_inp = check_for_error_condition(logger=logger,
                                                                                     discrimination_data=data,
                                                                                     dsn_by_attr_value=dsn_by_attr_value,
                                                                                     model=model, instance=input_sample,
                                                                                     tot_inputs=tot_inputs,
                                                                                     all_discriminations=all_discriminations,
                                                                                     one_attr_at_a_time=one_attr_at_a_time,
                                                                                     db_path=db_path,
                                                                                     analysis_id=analysis_id)

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
    for p_attr in data.protected_attributes:
        if should_terminate():
            break

        sensitive_idx = data.sensitive_indices_dict[p_attr]

        # Generate samples focused on this attribute
        attr_samples = global_discovery(
            max_global // len(data.protected_attributes),
            len(data.feature_names),
            data.input_bounds,
            sensitive_idx
        )

        # Initialize explainer if needed
        explainer = construct_explainer(X.values, data.feature_names, [data.outcome_column])

        # Find promising seeds
        attr_seeds = search_seed(
            model,
            data.feature_names,
            p_attr,
            explainer,
            np.array(attr_samples),
            len(data.feature_names),
            threshold_rank,
            random_seed,
            start_time=start_time,
            max_runtime_seconds=max_runtime_seconds
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
                                    for bounds in data.input_bounds])
            seeds.append(random_seed)

    # Format seeds for GA
    formatted_seeds = []
    for seed in seeds:
        if should_terminate():
            break

        try:
            seed_array = np.array(seed, dtype=float).flatten()
            if len(seed_array) == len(data.feature_names):
                formatted_seeds.append(seed_array)
                global_disc_inputs.add(tuple(seed_array))
                total_inputs.add(tuple(seed_array))
        except:
            continue

    # Run genetic algorithm only if we haven't terminated
    if not should_terminate():
        ga = GA(
            nums=formatted_seeds,
            bound=data.input_bounds,
            func=evaluate_local,
            DNA_SIZE=len(data.input_bounds),
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
    res_df, metrics = make_final_metrics_and_dataframe(data, tot_inputs, all_discriminations, dsn_by_attr_value,
                                                       start_time, logger=logger)

    # Log whether we terminated early
    if early_termination:
        logger.info("Early termination triggered: either max runtime or max TSN reached")

    return res_df, metrics


if __name__ == "__main__":
    from data_generator.main import get_real_data, DiscriminationData

    data_obj, schema = get_real_data('adult', use_cache=True)

    results_df, metrics = run_expga(data_obj, threshold_rank=0.5, max_global=20000, max_local=100,
                                    max_runtime_seconds=1500, max_tsn=20000, one_attr_at_a_time=True, cluster_num=50,
                                    step_size=0.05)

    print(f"\nTesting Metrics: {metrics}")
