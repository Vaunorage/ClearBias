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
    """
    GPU-accelerated Genetic Algorithm implementation using CuPy when available
    """

    def __init__(self, nums, bound, func, DNA_SIZE=None, cross_rate=0.8, mutation=0.003, use_gpu=True):
        self.use_gpu = use_gpu

        try:
            if self.use_gpu:
                import cupy as cp
                self.xp = cp
                # Convert input nums to proper CuPy array shape
                self.nums = cp.array(nums, dtype=cp.int32)
                if len(self.nums.shape) == 2:
                    self.nums = self.nums.reshape(len(nums), -1)

                self.bound = cp.array(bound)
            else:
                self.xp = np
                # Use NumPy arrays
                self.nums = np.array(nums, dtype=np.int32)
                if len(self.nums.shape) == 2:
                    self.nums = self.nums.reshape(len(nums), -1)

                self.bound = np.array(bound)
        except ImportError:
            logger.warning("CuPy not available. Using NumPy instead.")
            self.use_gpu = False
            self.xp = np
            # Use NumPy arrays
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
        # Move data to CPU for evaluation if using GPU
        if self.use_gpu:
            import cupy as cp
            try:
                pop_cpu = cp.asnumpy(self.POP)
                result = np.array([self.func(individual) for individual in pop_cpu])
                result = cp.array(result)

                if non_negative:
                    result = result - cp.min(result)
                return result
            except Exception as e:
                logger.warning(f"Error in GPU fitness calculation: {e}. Using CPU.")
                result = np.array([self.func(individual) for individual in cp.asnumpy(self.POP)])
                if non_negative:
                    result = result - np.min(result)
                return result
        else:
            result = np.array([self.func(individual) for individual in self.POP])
            if non_negative:
                result = result - np.min(result)
            return result

    def select(self):
        fitness = self.get_fitness()

        if len(fitness) == 0:
            raise ValueError("Fitness array is empty.")

        total_fitness = self.xp.sum(fitness)
        if total_fitness == 0:
            probabilities = self.xp.ones(len(fitness)) / len(fitness)
        else:
            probabilities = fitness / total_fitness

        probabilities = probabilities.squeeze()
        if probabilities.ndim == 0:
            probabilities = self.xp.array([probabilities])

        probabilities = probabilities / self.xp.sum(probabilities)

        # Handle GPU/CPU selection
        if self.use_gpu:
            import cupy as cp
            try:
                selected_indices = cp.random.choice(
                    cp.arange(self.POP_SIZE),
                    size=self.POP_SIZE,
                    replace=True,
                    p=probabilities
                )
                self.POP = self.POP[selected_indices]
            except Exception as e:
                logger.warning(f"Error in GPU selection: {e}. Using CPU.")
                # Fall back to CPU
                selected_indices = np.random.choice(
                    np.arange(self.POP_SIZE),
                    size=self.POP_SIZE,
                    replace=True,
                    p=cp.asnumpy(probabilities)
                )
                self.POP = self.POP[cp.array(selected_indices)]
        else:
            selected_indices = np.random.choice(
                np.arange(self.POP_SIZE),
                size=self.POP_SIZE,
                replace=True,
                p=probabilities
            )
            self.POP = self.POP[selected_indices]

    def crossover(self):
        for i in range(self.POP_SIZE):
            if self.xp.random.rand() < self.cross_rate:
                partner_idx = self.xp.random.randint(0, self.POP_SIZE)

                # Ensure proper array shapes for crossover
                if len(self.POP[i].shape) == 1 and len(self.POP[partner_idx].shape) == 1:
                    cross_points = self.xp.random.randint(0, self.POP[i].size)
                    end_points = self.xp.random.randint(cross_points + 1, self.POP[i].size + 1)

                    # Create copies to avoid modifying original arrays
                    temp = self.POP[i].copy()
                    temp[cross_points:end_points] = self.POP[partner_idx][cross_points:end_points]
                    self.POP[i] = temp

    def mutate(self):
        for i in range(len(self.POP)):
            individual = self.POP[i].flatten()  # Ensure 1D array for mutation

            for gene_idx in range(individual.size):
                if self.xp.random.rand() < self.mutation:
                    if gene_idx < len(self.bound):  # Check if we have bounds for this gene
                        low = int(self.bound[gene_idx][0])
                        high = int(self.bound[gene_idx][1])

                        if high > low:
                            if self.use_gpu:
                                import cupy as cp
                                new_value = cp.random.randint(low, high + 1)
                            else:
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

        # Convert to CPU for logging if using GPU
        if self.use_gpu:
            import cupy as cp
            try:
                population_log = [ind.get().flatten().tolist() for ind in self.POP]
                fitness_log = fitness.get().tolist()
            except:
                population_log = [cp.asnumpy(ind).flatten().tolist() for ind in self.POP]
                fitness_log = cp.asnumpy(fitness).tolist()
        else:
            population_log = [ind.flatten().tolist() for ind in self.POP]
            fitness_log = fitness.tolist()

        return population_log, fitness_log

def construct_explainer(train_vectors: np.ndarray, feature_names: List[str],
                        class_names: List[str]) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        train_vectors, feature_names=feature_names, class_names=class_names, discretize_continuous=False
    )


def search_seed(model: BaseEstimator, feature_names: List[str], sens_name: str,
                explainer: LimeTabularExplainer, train_vectors: np.ndarray, num: int,
                threshold_l: float, nb_seed=100, use_gpu=False) -> List[np.ndarray]:
    """
    Search for seed data points with significant influence from sensitive attributes
    """
    seed: List[np.ndarray] = []

    # Use GPU acceleration for model predictions if available
    if use_gpu:
        try:
            import cupy as cp
            import cudf

            # Process batches for efficiency
            batch_size = min(100, len(train_vectors))
            for i in range(0, len(train_vectors), batch_size):
                batch_end = min(i + batch_size, len(train_vectors))
                batch = train_vectors[i:batch_end]

                for j, x in enumerate(batch):
                    exp = explainer.explain_instance(x, model.predict_proba, num_features=num)
                    exp_result = exp.as_list(label=exp.available_labels()[0])
                    rank = [item[0] for item in exp_result]

                    try:
                        loc = rank.index(sens_name)
                        if loc < math.ceil(len(exp_result) * threshold_l):
                            seed.append(x)
                    except ValueError:
                        # sens_name not in rank
                        continue

                    if len(seed) >= nb_seed:
                        return seed
        except (ImportError, Exception) as e:
            logger.warning(f"Error using GPU for seed search: {e}. Using CPU.")
            # Fall back to CPU processing

    # CPU processing
    for i, x in enumerate(train_vectors):
        exp = explainer.explain_instance(x, model.predict_proba, num_features=num)
        exp_result = exp.as_list(label=exp.available_labels()[0])
        rank = [item[0] for item in exp_result]

        try:
            loc = rank.index(sens_name)
            if loc < math.ceil(len(exp_result) * threshold_l):
                seed.append(x)
        except ValueError:
            # sens_name not in rank
            continue

        if len(seed) >= nb_seed:
            break

    return seed


class GPUGlobalDiscovery:
    """
    GPU-accelerated global discovery for input samples
    """

    def __init__(self, step_size: int = 1, use_gpu: bool = True):
        self.step_size = step_size
        self.use_gpu = use_gpu

        # Initialize GPU if requested
        if self.use_gpu:
            try:
                import cupy as cp
                self.xp = cp
            except ImportError:
                logger.warning("CuPy not available. Using NumPy instead.")
                self.use_gpu = False
                self.xp = np
        else:
            self.xp = np

    def __call__(self, iteration: int, params: int, input_bounds: List[Tuple[int, int]],
                 sensitive_param: int) -> List[np.ndarray]:
        samples = []

        # Generate samples using GPU if available
        if self.use_gpu:
            try:
                import cupy as cp

                # Generate all samples at once using GPU
                shape = (iteration, params)

                # Create random samples within bounds
                random_samples = cp.zeros(shape, dtype=cp.int32)

                for param_idx in range(params):
                    low, high = input_bounds[param_idx]
                    random_samples[:, param_idx] = cp.random.randint(low, high + 1, size=iteration)

                # Set sensitive parameter to 0
                random_samples[:, sensitive_param - 1] = 0

                # Convert to numpy arrays for return
                samples = [np.array(sample) for sample in cp.asnumpy(random_samples)]

            except Exception as e:
                logger.warning(f"Error using GPU for global discovery: {e}. Using CPU.")
                # Fall back to CPU implementation
                for _ in range(iteration):
                    sample = [random.randint(bounds[0], bounds[1]) for bounds in input_bounds]
                    sample[sensitive_param - 1] = 0
                    samples.append(np.array(sample))
        else:
            # CPU implementation
            for _ in range(iteration):
                sample = [random.randint(bounds[0], bounds[1]) for bounds in input_bounds]
                sample[sensitive_param - 1] = 0
                samples.append(np.array(sample))

        return samples


def run_expga(data: DiscriminationData, threshold_rank: float, max_global: int, max_local: int, model_type: str = 'rf',
              cross_rate=0.9, mutation=0.1, max_runtime_seconds: float = None, max_tsn: int = None, nb_seed=100,
              one_attr_at_a_time=False, db_path=None, analysis_id=None, use_gpu=True, **model_kwargs) -> Tuple[
    pd.DataFrame, Metrics]:
    """
    GPU-accelerated XAI fairness testing that works on all protected attributes simultaneously.
    """
    logger.info(f"Starting ExpGA with all protected attributes (GPU: {'enabled' if use_gpu else 'disabled'})")

    start_time = time.time()
    disc_times: List[float] = []

    X, Y = data.xdf, data.ydf

    # Initialize GPU acceleration if requested
    if use_gpu:
        try:
            import cupy as cp
            import cudf
            gpu_available = True
        except ImportError:
            logger.warning("GPU libraries not available. Falling back to CPU.")
            use_gpu = False
            gpu_available = False

    model, X_train, X_test, y_train, y_test, feature_names = train_sklearn_model(
        data=data.dataframe,
        model_type=model_type,
        target_col=data.outcome_column,
        sensitive_attrs=list(data.protected_attributes),
        random_state=nb_seed,
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

        result, results_df, max_diff, org_df, tested_inp = check_for_error_condition(
            logger=logger,
            discrimination_data=data,
            dsn_by_attr_value=dsn_by_attr_value,
            model=model, instance=input_sample,
            tot_inputs=tot_inputs,
            all_discriminations=all_discriminations,
            one_attr_at_a_time=one_attr_at_a_time,
            db_path=db_path,
            analysis_id=analysis_id
        )

        if result and tuple(input_array) not in global_disc_inputs.union(local_disc_inputs):
            local_disc_inputs.add(tuple(input_array))
            disc_times.append(time.time() - start_time)
            return 2 * max_diff + 1
        return 0.0

    # Generate seeds for each protected attribute
    seeds = []

    # Use the GPU-optimized global discovery approach
    global_discovery = GPUGlobalDiscovery(use_gpu=use_gpu)

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
            nb_seed,
            use_gpu=use_gpu
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
        # Use GPU-accelerated GA if available
        ga = CudaGA(
            nums=formatted_seeds,
            bound=data.input_bounds,
            func=evaluate_local,
            DNA_SIZE=len(data.input_bounds),
            cross_rate=cross_rate,
            mutation=mutation,
            use_gpu=use_gpu
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
    res_df, metrics = make_final_metrics_and_dataframe(
        data, tot_inputs, all_discriminations, dsn_by_attr_value,
        start_time, logger=logger
    )

    # Log whether we terminated early
    if early_termination:
        logger.info("Early termination triggered: either max runtime or max TSN reached")

    return res_df, metrics


if __name__ == "__main__":
    from data_generator.main import get_real_data, DiscriminationData

    data_obj, schema = get_real_data('adult', use_cache=True)

    results_df, metrics = run_expga(data_obj, threshold_rank=0.5, max_global=20000, max_local=100,
                                    max_runtime_seconds=3600, max_tsn=20000, one_attr_at_a_time=True, cluster_num=50,
                                    step_size=0.05)

    print(f"\nTesting Metrics: {metrics}")
