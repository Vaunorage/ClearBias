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
                     sensitive_param: str, max_global: int, max_local: int,
                     model_type: str = 'rf', time_limit: float = None, max_tsn: int = None, **model_kwargs) -> Tuple[
    ExpGAResultDF, Metrics]:
    """
    Improved implementation of the XAI fairness testing algorithm with better
    sample discovery and more efficient evaluation.
    """
    # Track execution time
    start_time = time.time()
    disc_times: List[float] = []

    # Prepare data and model
    X, Y = dataset.xdf, dataset.ydf
    model = get_model(model_type, **model_kwargs)
    model.fit(X, Y)

    # Initialize sets to track inputs
    global_disc_inputs: Set[Tuple[float, ...]] = set()
    local_disc_inputs: Set[Tuple[float, ...]] = set()
    total_inputs: Set[Tuple[float, ...]] = set()

    # Store results
    results: List[Tuple[np.ndarray, np.ndarray, float, float]] = []

    def evaluate_local(input_sample: np.ndarray) -> float:
        """
        Check for discrimination across protected attributes.
        Returns a fitness score for GA optimization.
        """
        if time_limit and time.time() - start_time > time_limit:
            return 0.0

        # Handle various input shapes that the GA might provide
        if len(input_sample.shape) > 1:
            # If we have a multi-dimensional array, flatten it to 1D
            if input_sample.shape[0] == 1:
                input_sample = input_sample.flatten()
            else:
                # If it's a complex shape with multiple samples, take the first one
                input_sample = input_sample[0].flatten() if len(input_sample.shape) > 2 else input_sample[0]

        # Ensure input sample has the right length
        if len(input_sample) != len(dataset.feature_names):
            logger.warning(
                f"Input sample shape mismatch: got {len(input_sample)}, expected {len(dataset.feature_names)}")
            # Return low fitness for invalid inputs
            return 0.0

        # Convert input to the right format for the model
        input_df = pd.DataFrame([input_sample], columns=dataset.feature_names)

        # Add to total inputs set (for metrics)
        original_input_tuple = tuple(input_sample)
        total_inputs.add(original_input_tuple)

        # Get original prediction
        original_outcome = model.predict(input_df)[0]
        found_discrimination = False
        max_diff = 0.0

        # Check each protected attribute individually
        for attr_name, idx in dataset.sensitive_indices.items():
            current_value = input_sample[idx]

            # Try all possible values for this protected attribute
            min_val = int(dataset.input_bounds[idx][0])
            max_val = int(dataset.input_bounds[idx][1])

            for val in range(min_val, max_val + 1):
                if val == current_value:
                    continue  # Skip the original value

                # Create altered input with changed protected attribute
                altered_input = input_sample.copy()
                altered_input[idx] = val

                # Add to total inputs set
                altered_tuple = tuple(altered_input)
                total_inputs.add(altered_tuple)

                # Predict with altered input
                altered_df = pd.DataFrame([altered_input], columns=dataset.feature_names)
                altered_outcome = model.predict(altered_df)[0]

                # Calculate difference in outcomes
                diff = abs(original_outcome - altered_outcome)

                # Check if this is discriminatory
                if diff > threshold:
                    found_discrimination = True

                    # Record the discriminatory sample
                    if original_input_tuple not in global_disc_inputs.union(local_disc_inputs):
                        local_disc_inputs.add(original_input_tuple)
                        disc_times.append(time.time() - start_time)

                    # Add to results
                    results.append((input_sample, altered_input, original_outcome, altered_outcome))

                    # Update maximum difference
                    if diff > max_diff:
                        max_diff = diff

        # Return fitness score for GA (higher is better)
        return 2 * max_diff + (1 if found_discrimination else 0)

    # Global discovery strategy
    global_discovery = GlobalDiscovery()

    # Initialize variables for seed collection
    explainer = None
    seed = []

    # Main search loop
    while (not max_tsn or len(total_inputs) < max_tsn) and (not time_limit or time.time() - start_time < time_limit):
        # Generate global samples for exploration
        train_samples = global_discovery(max_global, len(dataset.feature_names), dataset.input_bounds,
                                         dataset.sensitive_indices[sensitive_param])

        # Initialize LIME explainer if needed
        if explainer is None:
            explainer = construct_explainer(X.values, dataset.feature_names, dataset.outcome_column)

        # Get promising seeds using LIME
        new_seeds = search_seed(model, dataset.feature_names, sensitive_param, explainer,
                                np.array(train_samples), len(dataset.feature_names), threshold_rank)

        seed.extend(new_seeds)

        # Check if we should continue
        if not seed and (time_limit and time.time() - start_time > time_limit):
            logger.info("No seeds found or time limit exceeded. Exiting...")
            metrics: Metrics = {
                "TSN": 0,
                "DSN": 0,
                "SUR": 0.0,
                "DSS": 0.0,
            }
            empty_df = pd.DataFrame(
                columns=['case_id', 'outcome', 'diff_outcome', 'indv_key', 'couple_key'] + list(dataset.feature_names))
            return empty_df, metrics

        # Process seeds
        for input_sample in seed:
            if (max_tsn and len(total_inputs) >= max_tsn) or (time_limit and time.time() - start_time > time_limit):
                break

            # Ensure proper shape
            if len(input_sample.shape) == 1:
                input_array = input_sample.reshape(1, -1)
            else:
                input_array = input_sample

            # Add to global discriminatory inputs
            global_disc_inputs.add(tuple(map(float, input_array.flatten())))

        # Skip GA if we already have enough samples
        if max_tsn and len(total_inputs) >= max_tsn:
            break

        # Run genetic algorithm search
        try:
            ga = GA(
                nums=[np.array(list(map(float, num))) for num in global_disc_inputs],
                bound=dataset.input_bounds,
                func=evaluate_local,
                DNA_SIZE=len(dataset.input_bounds),
                cross_rate=0.9,
                mutation=0.05
            )

            # Run GA evolution
            for _ in tqdm(range(max_local), desc="Local search progress"):
                if (max_tsn and len(total_inputs) >= max_tsn) or (time_limit and time.time() - start_time > time_limit):
                    break
                ga.evolve()

                # Report progress every 10 iterations
                if _ % 10 == 0:
                    logger.info(f"GA iteration {_}: TSN={len(total_inputs)}, DSN={len(local_disc_inputs)}")

        except Exception as e:
            logger.error(f"Error in GA: {str(e)}")
            # Continue with the next seed batch

        # If we have enough samples or we didn't find any new ones, exit the loop
        if (max_tsn and len(total_inputs) >= max_tsn) or len(seed) == 0:
            break

        # Clear seed for next iteration
        seed = []

        logger.info(
            f"Current TSN: {len(total_inputs)}, DSN: {len(local_disc_inputs)}, Target: {max_tsn if max_tsn else 'None'}")

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

    logger.info(f"Total Inputs (TSN): {tsn}")
    logger.info(f"Discriminatory inputs (DSN): {dsn}")
    logger.info(f"Average time to find discriminatory sample (DSS): {dss:.2f} seconds")
    logger.info(f"Success Rate (SUR): {sur:.2f}%")

    # Prepare results dataframe
    if not results:
        empty_df = pd.DataFrame(columns=[
                                            'case_id', 'outcome', 'diff_outcome', 'indv_key', 'couple_key'
                                        ] + list(dataset.feature_names))
        return empty_df, metrics

    df: ExpGAResultDF = pd.DataFrame(results,
                                     columns=["Original Input", "Altered Input", "Original Outcome", "Altered Outcome"])

    if df.empty:
        empty_df = pd.DataFrame(columns=[
                                            'case_id', 'outcome', 'diff_outcome', 'indv_key', 'couple_key'
                                        ] + list(dataset.feature_names))
        return empty_df, metrics

    # Process the dataframe
    df['Outcome Difference'] = df['Altered Outcome'] - df['Original Outcome']
    df['case_id'] = [str(uuid.uuid4())[:8] for _ in range(df.shape[0])]

    # Split into original and altered inputs
    df1 = df[['case_id', "Original Input", 'Original Outcome', 'Outcome Difference']].copy()
    df1.rename(columns={'Original Input': 'input', 'Original Outcome': 'outcome'}, inplace=True)

    df2 = df[['case_id', "Altered Input", 'Altered Outcome', 'Outcome Difference']].copy()
    df2.rename(columns={'Altered Input': 'input', 'Altered Outcome': 'outcome'}, inplace=True)

    # Combine and format
    df = pd.concat([df1, df2])
    df.rename(columns={'Outcome Difference': 'diff_outcome'}, inplace=True)
    df['diff_outcome'] = df['diff_outcome'].apply(abs)

    # Extract feature values
    try:
        df_attr = pd.DataFrame(df['input'].apply(lambda x: list(x)).tolist(), columns=dataset.feature_names)
        df = pd.concat([df.reset_index(drop=True), df_attr.reset_index(drop=True)], axis=1)
        df.drop(columns=['input'], inplace=True)
        df.sort_values(by=['case_id'], inplace=True)

        # Create individual and couple keys
        valid_attrs = [col for col in dataset.attributes if col in df.columns]
        df['indv_key'] = df[valid_attrs].apply(
            lambda row: '|'.join(str(int(x)) for x in row),
            axis=1
        )

        df['couple_key'] = df.groupby(df.index // 2)['indv_key'].transform(lambda x: '-'.join(x))

        # Add metrics to dataframe
        for k, v in metrics.items():
            df[k] = v
    except Exception as e:
        logger.error(f"Error in processing dataframe: {str(e)}")
        # Return a basic dataframe with the metrics
        df = pd.DataFrame(columns=['case_id', 'outcome', 'diff_outcome', 'indv_key', 'couple_key'] +
                                  list(dataset.feature_names))
        for k, v in metrics.items():
            df[k] = v

    return df, metrics


def run_expga(dataset: DiscriminationData, model_type: str = 'rf', threshold: float = 0.5,
              threshold_rank: float = 0.5, max_global: int = 50, max_local: int = 50,
              time_limit: float = None, max_tsn: int = None, **model_kwargs) -> Tuple[
    ExpGAResultDF, Dict[str, Metrics]]:
    start_time = time.time()
    dfs: List[ExpGAResultDF] = []
    all_metrics: Dict[str, Metrics] = {}

    for p_attr in dataset.protected_attributes:
        remaining_time = None if time_limit is None else max(0, time_limit - (time.time() - start_time))
        if remaining_time == 0:
            logger.warning(f"Time limit of {time_limit} seconds exceeded. Stopping early.")
            break

        df, metrics = xai_fair_testing(dataset, threshold, threshold_rank, p_attr, max_global, max_local,
                                       model_type, time_limit=remaining_time, max_tsn=max_tsn, **model_kwargs)
        dfs.append(df)
        all_metrics[p_attr] = metrics

    if not dfs:
        empty_df = pd.DataFrame(columns=['case_id', 'outcome', 'diff_outcome', 'indv_key', 'couple_key'] +
                                        list(dataset.feature_names))
        empty_metrics = {'DSN': 0, 'TSN': 0, 'SUR': 0.0, 'DSS': 0.0}
        return empty_df, empty_metrics

    all_res = pd.concat(dfs)
    all_res['case_id'] = all_res['case_id'].replace({v: k for k, v in enumerate(all_res['case_id'].unique())})
    all_metrics_r = pd.DataFrame(all_metrics.values()).agg(
        {'DSN': 'sum', 'TSN': 'sum', 'SUR': 'mean', 'DSS': 'mean'}).to_dict()

    return all_res, all_metrics_r
