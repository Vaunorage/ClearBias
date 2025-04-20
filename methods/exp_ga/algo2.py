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
import torch
from torch import cuda
from cuml.ensemble import RandomForestClassifier
from cuml.svm import SVC
from cuml import DecisionTreeClassifier
import cudf
from cuml.metrics import accuracy_score
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lime.lime_tabular import LimeTabularExplainer

from data_generator.main import DiscriminationData
from methods.exp_ga.genetic_algorithm import GA
from methods.utils import check_for_error_condition, make_final_metrics_and_dataframe

# Check GPU availability
device = torch.device('cuda' if cuda.is_available() else 'cpu')
use_gpu = cuda.is_available()

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


# GPU-accelerated MLP model using PyTorch
class TorchMLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=(100,), output_size=2, random_state=42):
        super(TorchMLPClassifier, self).__init__()
        torch.manual_seed(random_state)

        layers = []
        prev_size = input_size

        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size

        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.model(x)

    def fit(self, X, y, batch_size=64, epochs=100, lr=0.001):
        X_tensor = torch.FloatTensor(X).to(self.device)
        # Convert y to one-hot if needed
        if len(y.shape) == 1:
            num_classes = len(np.unique(y))
            y_one_hot = np.zeros((len(y), num_classes))
            y_one_hot[np.arange(len(y)), y] = 1
            y_tensor = torch.FloatTensor(y_one_hot).to(self.device)
        else:
            y_tensor = torch.FloatTensor(y).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, torch.argmax(batch_y, dim=1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}')

        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            probs = self(X_tensor)
            return probs.cpu().numpy()


def get_model(model_type: str, **kwargs) -> Any:
    """
    Factory function to create different types of GPU-accelerated models with specified parameters.

    Args:
        model_type: One of 'rf' (Random Forest), 'dt' (Decision Tree),
                   'mlp' (Multi-layer Perceptron), or 'svm' (Support Vector Machine)
        **kwargs: Model-specific parameters
    """
    if not use_gpu:
        # Fallback to CPU models if GPU is not available
        from sklearn.ensemble import RandomForestClassifier as CPURandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier as CPUDecisionTreeClassifier
        from sklearn.neural_network import MLPClassifier as CPUMLPClassifier
        from sklearn.svm import SVC as CPUSVCClassifier

        models = {
            'rf': CPURandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 10),
                random_state=kwargs.get('random_state', 42)
            ),
            'dt': CPUDecisionTreeClassifier(
                random_state=kwargs.get('random_state', 42)
            ),
            'mlp': CPUMLPClassifier(
                hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (100,)),
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42)
            ),
            'svm': CPUSVCClassifier(
                kernel=kwargs.get('kernel', 'rbf'),
                probability=True,  # Required for LIME
                random_state=kwargs.get('random_state', 42)
            )
        }
    else:
        # GPU-accelerated models
        models = {
            'rf': RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 10),
                random_state=kwargs.get('random_state', 42)
            ),
            'dt': DecisionTreeClassifier(
                random_state=kwargs.get('random_state', 42)
            ),
            'mlp': TorchMLPClassifier(
                input_size=kwargs.get('input_size', 10),
                hidden_sizes=kwargs.get('hidden_layer_sizes', (100,)),
                output_size=kwargs.get('output_size', 2),
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


def search_seed(model: Any, feature_names: List[str], sens_name: str,
                explainer: LimeTabularExplainer, train_vectors: np.ndarray, num: int,
                threshold_l: float, nb_seed=100) -> List[np.ndarray]:
    seed: List[np.ndarray] = []

    # Process in batches for GPU efficiency
    batch_size = 32
    total_samples = len(train_vectors)

    for batch_start in range(0, total_samples, batch_size):
        if len(seed) >= nb_seed:
            break

        batch_end = min(batch_start + batch_size, total_samples)
        batch = train_vectors[batch_start:batch_end]

        for x in batch:
            exp = explainer.explain_instance(x, model.predict_proba, num_features=num)
            exp_result = exp.as_list(label=exp.available_labels()[0])
            rank = [item[0] for item in exp_result]

            try:
                loc = rank.index(sens_name)
                if loc < math.ceil(len(exp_result) * threshold_l):
                    seed.append(x)
                if len(seed) >= nb_seed:
                    break
            except ValueError:
                # sens_name not in the rank list, skip this sample
                continue

    return seed


# GPU-accelerated GlobalDiscovery
class GlobalDiscovery:
    def __init__(self, step_size: int = 1):
        self.step_size = step_size
        self.device = device

    def __call__(self, iteration: int, params: int, input_bounds: List[Tuple[int, int]],
                 sensitive_param: int) -> List[np.ndarray]:
        if use_gpu:
            # GPU-accelerated sample generation
            samples = []
            for _ in range(iteration):
                # Generate random samples on GPU
                sample = [random.randint(bounds[0], bounds[1]) for bounds in input_bounds]
                sample[sensitive_param - 1] = 0
                samples.append(np.array(sample))
            return samples
        else:
            # CPU fallback
            samples = []
            for _ in range(iteration):
                sample = [random.randint(bounds[0], bounds[1]) for bounds in input_bounds]
                sample[sensitive_param - 1] = 0
                samples.append(np.array(sample))
            return samples


# GPU-accelerated GA implementation
class GPUGA(GA):
    def __init__(self, nums, bound, func, DNA_SIZE, cross_rate=0.9, mutation=0.1):
        super().__init__(nums, bound, func, DNA_SIZE, cross_rate, mutation)

        # Move population to GPU if available
        if use_gpu:
            # Convert numpy arrays to torch tensors
            self.pop = torch.FloatTensor(self.pop).to(device)
            self.bound = [(float(min_val), float(max_val)) for min_val, max_val in bound]

    def select(self):
        if use_gpu and isinstance(self.pop, torch.Tensor):
            # GPU accelerated selection
            fitness = self.get_fitness()
            fitness = torch.FloatTensor(fitness).to(device)
            idx = torch.multinomial(fitness, self.pop.shape[0], replacement=True)
            return self.pop[idx]
        else:
            # Fall back to CPU implementation
            return super().select()

    def crossover(self, parent, pop):
        if use_gpu and isinstance(self.pop, torch.Tensor):
            # GPU accelerated crossover
            if torch.rand(1).item() < self.cross_rate:
                # Select another individual from pop
                i = torch.randint(0, self.pop.shape[0], (1,)).item()
                # Choose crossover points
                cross_points = torch.rand(self.DNA_SIZE) < 0.5
                # Apply crossover
                parent_copy = parent.clone()
                parent_copy[cross_points] = pop[i, cross_points]
                return parent_copy
            return parent
        else:
            # Fall back to CPU implementation
            return super().crossover(parent, pop)

    def mutate(self, child):
        if use_gpu and isinstance(self.pop, torch.Tensor):
            # GPU accelerated mutation
            for point in range(self.DNA_SIZE):
                if torch.rand(1).item() < self.mutation:
                    min_val, max_val = self.bound[point]
                    child[point] = torch.rand(1).item() * (max_val - min_val) + min_val
            return child
        else:
            # Fall back to CPU implementation
            return super().mutate(child)

    def evolve(self):
        if use_gpu and isinstance(self.pop, torch.Tensor):
            # Create new population
            pop = self.select()
            pop_copy = pop.clone()

            # Apply crossover and mutation
            for parent in pop:
                child = self.crossover(parent, pop_copy)
                child = self.mutate(child)
                parent[:] = child

            # Move back to numpy for fitness evaluation
            self.pop = pop.cpu().numpy() if hasattr(pop, 'cpu') else np.array(pop)

            # Calculate fitness - convert back to GPU if necessary
            self.fitness = self.get_fitness()

            # Move back to GPU
            self.pop = torch.FloatTensor(self.pop).to(device)
        else:
            # Fall back to CPU implementation
            super().evolve()


def run_expga(data: DiscriminationData, threshold_rank: float, max_global: int, max_local: int, model_type: str = 'rf',
              cross_rate=0.9, mutation=0.1, max_runtime_seconds: float = None, max_tsn: int = None, nb_seed=100,
              one_attr_at_a_time=False, db_path=None, analysis_id=None, **model_kwargs) -> Tuple[pd.DataFrame, Metrics]:
    """
    GPU-accelerated XAI fairness testing that works on all protected attributes simultaneously.
    Uses a centralized termination check to improve code structure.
    """

    logger.info(f"Starting GPU-accelerated ExpGA with all protected attributes (GPU available: {use_gpu})")

    start_time = time.time()
    disc_times: List[float] = []

    X, Y = data.xdf, data.ydf

    # Add feature dimensions to model kwargs if using MLP
    if model_type == 'mlp':
        model_kwargs['input_size'] = X.shape[1]
        model_kwargs['output_size'] = len(np.unique(Y))

    model = get_model(model_type, **model_kwargs)
    model.fit(X, Y)

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

        # Convert to appropriate type for GPU/CPU
        if use_gpu and isinstance(input_sample, torch.Tensor):
            input_array = input_sample.cpu().numpy().flatten()
        else:
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

    # Use the GPU-accelerated global discovery approach
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
        explainer = construct_explainer(X.values, data.feature_names, data.outcome_column)

        # Find promising seeds
        attr_seeds = search_seed(
            model,
            data.feature_names,
            p_attr,
            explainer,
            np.array(attr_samples),
            len(data.feature_names),
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

    # Run GPU-accelerated genetic algorithm only if we haven't terminated
    if not should_terminate():
        ga = GPUGA(
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

    # Clean up GPU memory if needed
    if use_gpu:
        torch.cuda.empty_cache()

    return res_df, metrics


if __name__ == "__main__":
    from data_generator.main import get_real_data, DiscriminationData

    data_obj, schema = get_real_data('adult', use_cache=True)

    results_df, metrics = run_expga(data_obj, threshold_rank=0.5, max_global=20000, max_local=100,
                                    max_runtime_seconds=3600, max_tsn=20000, one_attr_at_a_time=True, cluster_num=50,
                                    step_size=0.05)

    print(f"\nTesting Metrics: {metrics}")
    print(f"Used GPU: {use_gpu}")