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
                     sensitive_param: str, max_global: int, max_local: int,
                     model_type: str = 'rf', time_limit: float = None, max_tsn: int = None, **model_kwargs) -> Tuple[
    ExpGAResultDF, Metrics]:
    start_time = time.time()
    disc_times: List[float] = []

    X, Y = dataset.xdf, dataset.ydf
    model = get_model(model_type, **model_kwargs)
    model.fit(X, Y)

    global_disc_inputs: Set[Tuple[float, ...]] = set()
    local_disc_inputs: Set[Tuple[float, ...]] = set()
    total_inputs: Set[Tuple[float, ...]] = set()

    results: List[Tuple[np.ndarray, np.ndarray, float, float]] = []

    def evaluate_local(input_sample: np.ndarray) -> float:
        if time_limit and time.time() - start_time > time_limit:
            return 0.0

        input_sample = input_sample.squeeze()
        input_array = input_sample.squeeze()
        total_inputs.add(tuple(input_array))

        output_original = model.predict(pd.DataFrame(input_array.reshape(1, -1), columns=dataset.feature_names))
        output_altered = None

        for val in range(*dataset.input_bounds[dataset.sensitive_indices[sensitive_param]]):
            if val != input_sample[dataset.sensitive_indices[sensitive_param]]:
                altered_input = input_array.copy()
                altered_input[dataset.sensitive_indices[sensitive_param]] = val

                output_altered = model.predict(
                    pd.DataFrame(altered_input.reshape(1, -1), columns=dataset.feature_names))

                if (abs(output_original - output_altered) > threshold and tuple(input_array)
                        not in global_disc_inputs.union(local_disc_inputs)):
                    local_disc_inputs.add(tuple(input_array))
                    results.append((input_array, altered_input, output_original[0], output_altered[0]))
                    disc_times.append(time.time() - start_time)
                    return 2 * abs(output_altered - output_original) + 1

        if output_altered is None:
            output_altered = output_original

        return 2 * abs(output_altered - output_original) + 1

    global_discovery = GlobalDiscovery()

    explainer = None
    seed = []

    # Continue searching as long as we need more samples and have time
    while (not max_tsn or len(total_inputs) < max_tsn) and (not time_limit or time.time() - start_time < time_limit):
        # Generate a new batch of global samples
        train_samples = global_discovery(max_global, len(dataset.feature_names), dataset.input_bounds,
                                         dataset.sensitive_indices[sensitive_param])

        # Initialize explainer if this is the first iteration
        if explainer is None:
            explainer = construct_explainer(X, dataset.feature_names, dataset.outcome_column)

        # Search for new seeds using the current batch
        new_seeds = search_seed(model, dataset.feature_names, sensitive_param, explainer, train_samples,
                                len(dataset.feature_names), threshold_rank)

        seed.extend(new_seeds)

        if not seed and (time_limit and time.time() - start_time > time_limit):
            logger.info("No seeds found or time limit exceeded. Exiting...")
            metrics: Metrics = {
                "TSN": 0,
                "DSN": 0,
                "DSS": 0.0,
                "SUR": 0.0
            }
            empty_df = pd.DataFrame(columns=[
                                                'case_id', 'outcome', 'diff_outcome', 'indv_key', 'couple_key'
                                            ] + list(dataset.feature_names))
            return empty_df, metrics

        # Process the new seeds
        for input_sample in seed:
            if (max_tsn and len(total_inputs) >= max_tsn) or (time_limit and time.time() - start_time > time_limit):
                break
            input_array = np.array([int(i) for i in input_sample]).reshape(1, -1)
            global_disc_inputs.add(tuple(map(tuple, input_array)))

        # Skip GA if we already have enough samples
        if max_tsn and len(total_inputs) >= max_tsn:
            break

        # Initialize and run GA with current seeds
        ga = GA(
            nums=list(global_disc_inputs), bound=dataset.input_bounds, func=evaluate_local,
            DNA_SIZE=len(dataset.input_bounds), cross_rate=0.9, mutation=0.05
        )

        for _ in tqdm(range(max_local), desc="Local search progress"):
            if (max_tsn and len(total_inputs) >= max_tsn) or (time_limit and time.time() - start_time > time_limit):
                break
            ga.evolve()

        # If we have enough samples or we didn't find any new ones, exit the loop
        if (max_tsn and len(total_inputs) >= max_tsn) or len(seed) == 0:
            break

        # Clear seed for next iteration to avoid duplicate work
        seed = []

        logger.info(f"Current TSN: {len(total_inputs)}, Target: {max_tsn if max_tsn else 'None'}")

    tsn = len(total_inputs)
    dsn = len(local_disc_inputs)
    dss = np.mean(np.diff(disc_times)) if len(disc_times) > 1 else 0.0
    sur = (dsn / tsn * 100) if tsn > 0 else 0.0

    metrics: Metrics = {
        "TSN": tsn,
        "DSN": dsn,
        "DSS": dss,
        "SUR": sur
    }

    logger.info(f"Total Inputs (TSN): {tsn}")
    logger.info(f"Discriminatory inputs (DSN): {dsn}")
    logger.info(f"Average time to find discriminatory sample (DSS): {dss:.2f} seconds")
    logger.info(f"Success Rate (SUR): {sur:.2f}%")

    df: ExpGAResultDF = pd.DataFrame(results,
                                     columns=["Original Input", "Altered Input", "Original Outcome", "Altered Outcome"])

    if df.empty:
        empty_df = pd.DataFrame(columns=[
                                            'case_id', 'outcome', 'diff_outcome', 'indv_key', 'couple_key'
                                        ] + list(dataset.feature_names))
        return empty_df, metrics

    df['Outcome Difference'] = df['Altered Outcome'] - df['Original Outcome']
    df['case_id'] = [str(uuid.uuid4())[:8] for _ in range(df.shape[0])]

    df1 = df[['case_id', "Original Input", 'Original Outcome', 'Outcome Difference']].copy()
    df1.rename(columns={'Original Input': 'input', 'Original Outcome': 'outcome'}, inplace=True)

    df2 = df[['case_id', "Altered Input", 'Altered Outcome', 'Outcome Difference']].copy()
    df2.rename(columns={'Altered Input': 'input', 'Altered Outcome': 'outcome'}, inplace=True)

    df = pd.concat([df1, df2])
    df.rename(columns={'Outcome Difference': 'diff_outcome'}, inplace=True)
    df['diff_outcome'] = df['diff_outcome'].apply(abs)

    df_attr = pd.DataFrame(df['input'].apply(lambda x: list(x)).tolist(), columns=dataset.feature_names)
    df = pd.concat([df.reset_index(drop=True), df_attr.reset_index(drop=True)], axis=1)
    df.drop(columns=['input'], inplace=True)
    df.sort_values(by=['case_id'], inplace=True)

    valid_attrs = [col for col in dataset.attributes if col in df.columns]
    df['indv_key'] = df[valid_attrs].apply(
        lambda row: '|'.join(str(int(x)) for x in row),
        axis=1
    )

    df['couple_key'] = df.groupby(df.index // 2)['indv_key'].transform(lambda x: '-'.join(x))

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
