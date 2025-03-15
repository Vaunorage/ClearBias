import time
from typing import TypedDict, List, Tuple, Dict, Any, Union, Set
import math
import uuid
import warnings
import numpy as np
import random
import logging
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from lime.lime_tabular import LimeTabularExplainer

from data_generator.main import DiscriminationData
from methods.exp_ga.genetic_algorithm import GA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class ExpGAResultRow(TypedDict, total=False):
    case_id: int
    outcome: float
    diff_outcome: float
    indv_key: str
    couple_key: str


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
                threshold_l: float) -> List[np.ndarray]:
    seed: List[np.ndarray] = []
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


def xai_fair_testing(dataset: DiscriminationData, threshold: float, threshold_rank: float,
                     max_global: int, max_local: int, model_type: str = 'rf',
                     time_limit: float = None, max_tsn: int = None, **model_kwargs) -> Tuple[
    ExpGAResultDF, Metrics]:
    """
    XAI fairness testing that works on all protected attributes simultaneously.
    Fixed to handle array shape issues properly.
    """
    start_time = time.time()
    disc_times: List[float] = []

    X, Y = dataset.xdf, dataset.ydf
    model = get_model(model_type, **model_kwargs)
    model.fit(X, Y)

    global_disc_inputs: Set[Tuple[float, ...]] = set()
    local_disc_inputs: Set[Tuple[float, ...]] = set()
    total_inputs: Set[Tuple[float, ...]] = set()

    results: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
    dsn_per_protected_attr = {e: 0 for e in dataset.protected_attributes}
    dsn_per_protected_attr['total'] = 0

    def evaluate_local(input_sample) -> float:
        """
        Evaluates a sample for discrimination across all protected attributes at once.
        """
        if time_limit and time.time() - start_time > time_limit:
            return 0.0

        try:
            # Standardize input format
            input_array = np.array(input_sample, dtype=float).flatten()

            # Verify dimensions
            if len(input_array) != len(dataset.feature_names):
                return 0.0

            # Add to total inputs
            total_inputs.add(tuple(input_array))

            # Get original prediction
            input_df = pd.DataFrame([input_array], columns=dataset.feature_names)
            output_original = model.predict(input_df)[0]

            # Track best discrimination found
            max_diff = 0.0
            best_altered_input = None
            best_output = None

            # Check each protected attribute
            for attr_name in dataset.protected_attributes:
                attr_idx = dataset.sensitive_indices[attr_name]
                current_value = input_array[attr_idx]

                # Try each possible value
                for val in range(int(dataset.input_bounds[attr_idx][0]),
                                 int(dataset.input_bounds[attr_idx][1]) + 1):
                    if val == current_value:
                        continue

                    # Create altered input
                    altered_input = input_array.copy()
                    altered_input[attr_idx] = val

                    # Predict outcome
                    altered_df = pd.DataFrame([altered_input], columns=dataset.feature_names)
                    output_altered = model.predict(altered_df)[0]

                    # Check for discrimination
                    diff = abs(output_original - output_altered)

                    # If this is the best discrimination found so far
                    if diff > threshold and diff > max_diff:
                        max_diff = diff
                        best_altered_input = altered_input
                        best_output = output_altered

                        dsn_per_protected_attr[attr_name] += 1
                        dsn_per_protected_attr['total'] += 1

                        # Early return for efficiency - follows original algorithm approach
                        if tuple(input_array) not in global_disc_inputs.union(local_disc_inputs):
                            local_disc_inputs.add(tuple(input_array))
                            results.append((input_array, altered_input, output_original, output_altered))
                            disc_times.append(time.time() - start_time)
                            return 2 * diff + 1

            # If we found discrimination but didn't do early return
            if max_diff > 0 and best_altered_input is not None:
                if tuple(input_array) not in global_disc_inputs.union(local_disc_inputs):
                    local_disc_inputs.add(tuple(input_array))
                    results.append((input_array, best_altered_input, output_original, best_output))
                    disc_times.append(time.time() - start_time)
                return 2 * max_diff + 1

            # No discrimination found
            return 0.0

        except Exception as e:
            logger.error(f"Error in evaluate_local: {str(e)}")
            return 0.0

    # Generate seeds for each protected attribute
    seeds = []

    # Use the original global discovery approach
    global_discovery = GlobalDiscovery()

    # Create seeds for each protected attribute
    for p_attr in dataset.protected_attributes:
        sensitive_idx = dataset.sensitive_indices[p_attr]

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
            threshold_rank
        )

        seeds.extend(attr_seeds)

        # Limit total seeds
        if len(seeds) >= 100:
            break

    # Ensure we have at least some seeds
    if not seeds:
        for _ in range(10):
            random_seed = np.array([float(random.randint(bounds[0], bounds[1]))
                                    for bounds in dataset.input_bounds])
            seeds.append(random_seed)

    # Format seeds for GA
    formatted_seeds = []
    for seed in seeds:
        try:
            seed_array = np.array(seed, dtype=float).flatten()
            if len(seed_array) == len(dataset.feature_names):
                formatted_seeds.append(seed_array)
                global_disc_inputs.add(tuple(seed_array))
                total_inputs.add(tuple(seed_array))
        except:
            continue

    # Run genetic algorithm
    try:
        ga = GA(
            nums=formatted_seeds,
            bound=dataset.input_bounds,
            func=evaluate_local,
            DNA_SIZE=len(dataset.input_bounds),
            cross_rate=0.9,
            mutation=0.1
        )

        # Evolve population
        for iteration in tqdm(range(max_local), desc="Local search progress"):
            if (max_tsn and len(total_inputs) >= max_tsn) or (time_limit and time.time() - start_time > time_limit):
                break

            ga.evolve()

            # Log progress
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: TSN={len(total_inputs)}, DSN={len(local_disc_inputs)}")

    except Exception as e:
        logger.error(f"Error in GA: {str(e)}")

    # Calculate metrics
    tsn = len(total_inputs)
    dsn = len(local_disc_inputs)
    sur = (dsn / tsn * 100) if tsn > 0 else 0.0
    dss = np.mean(np.diff(disc_times)) if len(disc_times) > 1 else 0.0

    metrics: Metrics = {
        "TSN": tsn,
        "DSN": dsn,
        "SUR": sur,
        "DSS": dss
    }

    # Log results
    logger.info(f"Total Inputs (TSN): {tsn}")
    logger.info(f"Discriminatory inputs (DSN): {dsn}")
    logger.info(f"Average time to find discriminatory sample (DSS): {dss:.2f} seconds")
    logger.info(f"Success Rate (SUR): {sur:.2f}%")

    # Process results into DataFrame
    if not results:
        empty_df = pd.DataFrame(columns=[
                                            'case_id', 'outcome', 'diff_outcome', 'indv_key', 'couple_key'
                                        ] + list(dataset.feature_names))
        return empty_df, metrics

    # Create DataFrame from results
    df = pd.DataFrame(results, columns=["Original Input", "Altered Input", "Original Outcome", "Altered Outcome"])

    # Process outcome difference
    df['Outcome Difference'] = df['Altered Outcome'] - df['Original Outcome']
    df['case_id'] = [str(uuid.uuid4())[:8] for _ in range(df.shape[0])]

    # Split into original and altered
    df1 = df[['case_id', "Original Input", 'Original Outcome', 'Outcome Difference']].copy()
    df1.rename(columns={'Original Input': 'input', 'Original Outcome': 'outcome'}, inplace=True)

    df2 = df[['case_id', "Altered Input", 'Altered Outcome', 'Outcome Difference']].copy()
    df2.rename(columns={'Altered Input': 'input', 'Altered Outcome': 'outcome'}, inplace=True)

    # Combine
    df = pd.concat([df1, df2])
    df.rename(columns={'Outcome Difference': 'diff_outcome'}, inplace=True)
    df['diff_outcome'] = df['diff_outcome'].apply(abs)

    # Extract features
    try:
        # Convert input arrays to lists for DataFrame construction
        input_lists = []
        for inp in df['input']:
            if hasattr(inp, 'tolist'):
                input_lists.append(inp.tolist())
            else:
                input_lists.append(list(inp))

        df_attr = pd.DataFrame(input_lists, columns=dataset.feature_names)
        df = pd.concat([df.reset_index(drop=True), df_attr.reset_index(drop=True)], axis=1)
        df.drop(columns=['input'], inplace=True)
        df.sort_values(by=['case_id'], inplace=True)

        # Generate keys
        valid_attrs = [col for col in dataset.attributes if col in df.columns]
        df['indv_key'] = df[valid_attrs].apply(
            lambda row: '|'.join(str(int(x)) for x in row),
            axis=1
        )

        df['couple_key'] = df.groupby(df.index // 2)['indv_key'].transform(lambda x: '-'.join(x))
    except Exception as e:
        logger.error(f"Error in DataFrame processing: {str(e)}")

    # Add metrics
    for k, v in metrics.items():
        df[k] = v

    metrics['dsn_per_protected_attr'] = dsn_per_protected_attr

    return df, metrics


def run_expga(dataset: DiscriminationData, model_type: str = 'rf', threshold: float = 0.5,
              threshold_rank: float = 0.5, max_global: int = 50, max_local: int = 50,
              time_limit: float = None, max_tsn: int = None, **model_kwargs) -> Tuple[
    ExpGAResultDF, Dict[str, Any]]:
    """
    Simplified run_expga that tests all protected attributes at once.
    """
    logger.info("Starting ExpGA with all protected attributes")

    # Run the improved testing function
    df, metrics = xai_fair_testing(
        dataset,
        threshold,
        threshold_rank,
        max_global,
        max_local,
        model_type,
        time_limit=time_limit,
        max_tsn=max_tsn,
        **model_kwargs
    )

    # Format metrics as expected by caller
    if 'TSN' in metrics:
        metrics_dict = {attr: metrics for attr in dataset.protected_attributes}
        all_metrics_r = pd.DataFrame(metrics_dict.values()).agg({
            'DSN': 'sum',
            'TSN': 'sum',
            'SUR': 'mean',
            'DSS': 'mean'
        }).to_dict()
    else:
        all_metrics_r = metrics

    # Empty DataFrame case
    if df.empty:
        empty_df = pd.DataFrame(columns=['case_id', 'outcome', 'diff_outcome', 'indv_key', 'couple_key'] +
                                        list(dataset.feature_names))
        return empty_df, all_metrics_r

    # Process case IDs to be sequential
    df['case_id'] = df['case_id'].replace({v: k for k, v in enumerate(df['case_id'].unique())})

    return df, all_metrics_r
