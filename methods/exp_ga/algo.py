from typing import TypedDict, List, Tuple, Dict, Any, Union
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
from lime.lime_tabular import LimeTabularExplainer

from data_generator.main import DiscriminationData
from methods.exp_ga.genetic_algorithm import GA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class ExpGAResultRow(TypedDict, total=False):
    group_id: str
    outcome: float
    diff_outcome: float
    indv_key: str
    couple_key: str

ExpGAResultDF = DataFrame

def construct_explainer(train_vectors: np.ndarray, feature_names: List[str], class_names: List[str]) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        train_vectors, feature_names=feature_names, class_names=class_names, discretize_continuous=False
    )

def search_seed(model: RandomForestClassifier, feature_names: List[str], sens_name: str, 
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
                     random_forest_n_estimators: int = 10, 
                     random_forest_random_state: int = 42) -> ExpGAResultDF:
    # Load data and prepare model
    X, Y = dataset.xdf, dataset.ydf
    model = RandomForestClassifier(n_estimators=random_forest_n_estimators, random_state=random_forest_random_state)
    model.fit(X, Y)

    global_disc_inputs: Set[Tuple[float, ...]] = set()
    local_disc_inputs: Set[Tuple[float, ...]] = set()
    total_inputs: Set[Tuple[float, ...]] = set()

    results: List[Tuple[np.ndarray, np.ndarray, float, float]] = []

    def evaluate_local(input_sample: np.ndarray) -> float:
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
                    return 2 * abs(output_altered - output_original) + 1

        if output_altered is None:
            output_altered = output_original

        return 2 * abs(output_altered - output_original) + 1

    global_discovery = GlobalDiscovery()
    train_samples = global_discovery(max_global, len(dataset.feature_names), dataset.input_bounds,
                                     dataset.sensitive_indices[sensitive_param])

    explainer = construct_explainer(X, dataset.feature_names, dataset.outcome_column)
    seed = search_seed(model, dataset.feature_names, sensitive_param, explainer, train_samples,
                       len(dataset.feature_names), threshold_rank)

    if not seed:
        logger.info("No seeds found. Exiting...")
        return pd.DataFrame()

    for input_sample in seed:
        input_array = np.array([int(i) for i in input_sample]).reshape(1, -1)
        global_disc_inputs.add(tuple(map(tuple, input_array)))

    # Local search
    ga = GA(
        nums=list(global_disc_inputs), bound=dataset.input_bounds, func=evaluate_local,
        DNA_SIZE=len(dataset.input_bounds), cross_rate=0.9, mutation=0.05
    )

    for _ in tqdm(range(max_local), desc="Local search progress"):
        ga.evolve()

    logger.info(f"Total Inputs: {len(total_inputs)}")
    logger.info(f"Discriminatory inputs: {len(local_disc_inputs)}")
    logger.info(
        f"Percentage discriminatory inputs: {float(len(local_disc_inputs)) / float(len(total_inputs)) * 100:.2f}%"
    )

    # Create DataFrame from results
    df: ExpGAResultDF = pd.DataFrame(results, columns=["Original Input", "Altered Input", "Original Outcome", "Altered Outcome"])
    df['Outcome Difference'] = df['Altered Outcome'] - df['Original Outcome']
    df['group_id'] = [str(uuid.uuid4())[:8] for _ in range(df.shape[0])]

    df1 = df[['group_id', "Original Input", 'Original Outcome', 'Outcome Difference']].copy()
    df1.rename(columns={'Original Input': 'input', 'Original Outcome': 'outcome'}, inplace=True)

    df2 = df[['group_id', "Altered Input", 'Altered Outcome', 'Outcome Difference']].copy()
    df2.rename(columns={'Altered Input': 'input', 'Altered Outcome': 'outcome'}, inplace=True)

    df = pd.concat([df1, df2])

    df.rename(columns={'Outcome Difference': 'diff_outcome'}, inplace=True)

    df['diff_outcome'] = df['diff_outcome'].apply(abs)

    df_attr = pd.DataFrame(df['input'].apply(lambda x: list(x)).tolist(), columns=dataset.feature_names)

    df = pd.concat([df.reset_index(drop=True), df_attr.reset_index(drop=True)], axis=1)

    df.drop(columns=['input'], inplace=True)

    df.sort_values(by=['group_id'], inplace=True)

    df['indv_key'] = df.apply(lambda row: '|'.join(str(int(row[col])) for col in list(dataset.attributes)), axis=1)
    df['couple_key'] = df.groupby(df.index // 2)['indv_key'].transform('*'.join)

    return df

def run_expga(dataset: DiscriminationData, threshold: float = 0.5, threshold_rank: float = 0.5, 
              max_global: int = 50, max_local: int = 50,
              random_forest_n_estimators: int = 10, 
              random_forest_random_state: int = 42) -> ExpGAResultDF:
    dfs: List[ExpGAResultDF] = []

    for p_attr in dataset.protected_attributes:
        df = xai_fair_testing(dataset, threshold, threshold_rank, p_attr, max_global, max_local,
                              random_forest_n_estimators, random_forest_random_state)
        dfs.append(df)

    return pd.concat(dfs)