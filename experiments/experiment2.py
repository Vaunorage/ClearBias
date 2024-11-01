import threading
import pandas as pd
from typing import Set, Dict, Tuple, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sqlite3
import hashlib
import json
import logging
import time
from experiments.experiement1 import TimeoutException
from methods.biasscan.algo import run_bias_scan
from methods.exp_ga.algo import run_expga
from methods.ml_check.algo import run_mlcheck
from methods.aequitas.algo import run_aequitas
from path import HERE
from data_generator.main2 import generate_data, DiscriminationData

# Constants
DB_NAME = HERE.joinpath('experiments/experiment_results2.db')
NUM_DATASETS = 10
NUM_RUNS = 5

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def timeout_handler(signum, frame):
    raise TimeoutException("Function call timed out")


def run_with_timeout(func, *args, timeout=300, **kwargs):
    result = [None]

    def target():
        result[0] = func(*args, **kwargs)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        logger.warning(f"Function {func.__name__} timed out after {timeout} seconds")
        return pd.DataFrame(), {}

    return result[0] if result[0] is not None else (pd.DataFrame(), {})


def param_hash(params: Dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()


def save_experiment(params: Dict[str, Any], results_dfs: Dict, results_metas: Dict, results: Dict):
    conn = sqlite3.connect(DB_NAME)

    summary_df = pd.DataFrame({
        'param_hash': [param_hash(params)],
        'params': [json.dumps(params)],
        'results': [json.dumps(results)],
        'meta': [json.dumps(results_metas)]
    })
    summary_df.to_sql('experiments', conn, if_exists='append', index=False)

    # Save each method's detailed results if they exist as DataFrames
    for method, result in results_dfs.items():
        if isinstance(result, pd.DataFrame):
            timestamp = int(time.time())
            table_name = f"{method.lower()}_results_{param_hash(params)}_{timestamp}"

            result['param_hash'] = param_hash(params)  # Add a reference to the parameters
            result.to_sql(table_name, conn, if_exists='fail', index=False)  # Create a new table

    conn.close()
    logger.debug(f"Experiment results saved for parameters: {params}")


def save_dataset(dataset: DiscriminationData, params: Dict[str, Any]):
    conn = sqlite3.connect(DB_NAME)

    # Save the dataset DataFrame
    dataset_table_name = f"dataset_{param_hash(params)}"
    dataset.dataframe.to_sql(dataset_table_name, conn, if_exists='replace', index=False)

    # Ensure attributes match the actual columns in the dataframe
    actual_columns = set(dataset.dataframe.columns)
    protected_attributes = {k: v for k, v in dataset.attributes.items() if k in actual_columns}
    categorical_columns = [col for col in dataset.categorical_columns if col in actual_columns]

    # Save the dataset metadata
    metadata = {
        'param_hash': param_hash(params),
        'params': json.dumps(params),
        'outcome_column': dataset.outcome_column,
        'protected_attributes': json.dumps(protected_attributes),
        'categorical_columns': json.dumps(categorical_columns),
        'collisions': dataset.collisions,
        'nb_groups': dataset.nb_groups,
        'max_group_size': dataset.max_group_size,
        'hiddenlayers_depth': dataset.hiddenlayers_depth,
        'dataset_table_name': dataset_table_name
    }
    pd.DataFrame([metadata]).to_sql('datasets_metadata', conn, if_exists='append', index=False)

    # Save relevance metrics if available
    if hasattr(dataset, 'relevance_metrics') and isinstance(dataset.relevance_metrics, pd.DataFrame):
        relevance_table_name = f"relevance_{param_hash(params)}"
        dataset.relevance_metrics.to_sql(relevance_table_name, conn, if_exists='replace', index=True)
        metadata['relevance_table_name'] = relevance_table_name

    conn.close()
    logger.debug(f"Dataset saved with parameters: {params}")


def load_all_datasets() -> List[Tuple[DiscriminationData, Dict[str, Any]]]:
    conn = sqlite3.connect(DB_NAME)

    # Load all metadata
    metadata_df = pd.read_sql("SELECT * FROM datasets_metadata", conn)

    datasets = []
    for _, metadata in metadata_df.iterrows():
        # Load the dataset
        dataset_df = pd.read_sql(f"SELECT * FROM {metadata['dataset_table_name']}", conn)

        # Parse the metadata
        params = json.loads(metadata['params'])
        protected_attributes = json.loads(metadata['protected_attributes'])
        categorical_columns = json.loads(metadata['categorical_columns'])

        # Load relevance metrics if available
        relevance_metrics = None
        if 'relevance_table_name' in metadata:
            relevance_metrics = pd.read_sql(f"SELECT * FROM {metadata['relevance_table_name']}", conn,
                                            index_col='group_key')

        # Create a DiscriminationData object
        dataset = DiscriminationData(
            dataframe=dataset_df,
            categorical_columns=categorical_columns,
            attributes=protected_attributes,
            collisions=metadata['collisions'],
            nb_groups=metadata['nb_groups'],
            max_group_size=metadata['max_group_size'],
            hiddenlayers_depth=metadata['hiddenlayers_depth'],
            outcome_column=metadata['outcome_column']
        )

        if relevance_metrics is not None:
            dataset.relevance_metrics = relevance_metrics

        datasets.append((dataset, params))

    conn.close()

    logger.info(f"Loaded {len(datasets)} datasets from the database")
    return datasets


def generate_and_save_datasets(num_datasets: int, param_ranges: Dict[str, Tuple[float, float]]) -> List[Dict[str, Any]]:
    dataset_params = []
    for _ in tqdm(range(num_datasets), desc="Generating and saving datasets"):
        if param_ranges is not None:
            params = {k: np.random.uniform(v[0], v[1]) if isinstance(v[0], (int, float)) else np.random.choice(v)
                      for k, v in param_ranges.items()}
            dataset = generate_data(**params)
        else:
            dataset = generate_data()
            params = {}  # Empty dict if no param_ranges provided

        save_dataset(dataset, params)
        dataset_params.append(params)

    return dataset_params


def run_algorithm(algorithm, dataset, **kwargs):
    if algorithm == 'MLCheck':
        return run_mlcheck(dataset, iteration_no=1, **kwargs)
    elif algorithm == 'Aequitas':
        return run_aequitas(dataset.training_dataframe, col_to_be_predicted=dataset.outcome_column,
                            sensitive_param_name_list=dataset.protected_attributes,
                            perturbation_unit=1, model_type="DecisionTree", threshold=0,
                            global_iteration_limit=100, local_iteration_limit=10, **kwargs)
    elif algorithm == 'BiasScan':
        return run_bias_scan(dataset, test_size=0.3, random_state=42, n_estimators=200,
                             bias_scan_num_iters=100, bias_scan_scoring='Poisson',
                             bias_scan_favorable_value='high', bias_scan_mode='ordinal', **kwargs)
    elif algorithm == 'ExpGA':
        return run_expga(dataset, threshold=0.5, threshold_rank=0.5,
                         max_global=50, max_local=50, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def analyze_method(method_couple_keys: Set[str], original_couple_keys: Set[str]) -> Dict[str, int]:
    matching = len(method_couple_keys.intersection(original_couple_keys))
    new = len(method_couple_keys - original_couple_keys)
    return {'matching': matching, 'new': new}


def run_experiments(datasets: List[Tuple[Any, Dict]], num_runs: int):
    methods = ['MLCheck', 'Aequitas', 'BiasScan', 'ExpGA']
    results = []

    for dataset, params in tqdm(datasets, desc="Processing datasets"):
        original_couple_keys = set(dataset.dataframe['couple_key'])

        for method in methods:
            for run in range(num_runs):
                try:
                    result_df, result_meta = run_with_timeout(run_algorithm, method, dataset)

                    if result_df is not None and isinstance(result_df,
                                                            pd.DataFrame) and not result_df.empty and 'couple_key' in result_df:
                        method_result = analyze_method(set(result_df['couple_key']), original_couple_keys)
                    else:
                        method_result = {'matching': 0, 'new': 0}

                    results.append({
                        'dataset_params': params,
                        'method': method,
                        'run': run,
                        'matching': method_result['matching'],
                        'new': method_result['new'],
                        'meta': result_meta
                    })

                    # Save partial results to database
                    save_experiment(params, {method: result_df}, {method: result_meta}, {method: method_result})

                except Exception as e:
                    logger.error(
                        f"An error occurred while running {method} on dataset {datasets.index((dataset, params))}, run {run}: {str(e)}")
                    results.append({
                        'dataset_params': params,
                        'method': method,
                        'run': run,
                        'matching': 0,
                        'new': 0,
                        'meta': {'error': str(e)}
                    })

    return pd.DataFrame(results)


def analyze_results(results_df: pd.DataFrame):
    # Compute average performance for each method across all datasets and runs
    avg_performance = results_df.groupby('method')[['matching', 'new']].mean()

    # Find the best performing method
    best_method = avg_performance.sum(axis=1).idxmax()

    # Analyze impact of dataset parameters on method performance
    param_columns = [col for col in results_df.columns if col.startswith('dataset_params.')]
    param_impact = {}
    for param in param_columns:
        correlation = results_df[[param, 'matching', 'new']].groupby('method').corr()[param].drop(param)
        param_impact[param] = correlation

    return avg_performance, best_method, param_impact


def plot_results(avg_performance, best_method, param_impact):
    # Plot average performance
    plt.figure(figsize=(12, 6))
    avg_performance.plot(kind='bar')
    plt.title('Average Performance of Each Method')
    plt.xlabel('Method')
    plt.ylabel('Average Count')
    plt.legend(['Matching', 'New'])
    plt.tight_layout()
    plt.savefig('average_performance.png')
    plt.close()

    # Plot parameter impact
    plt.figure(figsize=(15, 10))
    sns.heatmap(pd.DataFrame(param_impact), cmap='coolwarm', center=0, annot=True)
    plt.title('Impact of Dataset Parameters on Method Performance')
    plt.tight_layout()
    plt.savefig('parameter_impact.png')
    plt.close()

    logger.info(f"Best performing method overall: {best_method}")


if __name__ == "__main__":
    logger.info("Starting efficient experiment with dataset storage using to_sql")

    # Uncomment the following line if you want to generate new datasets
    dataset_params = generate_and_save_datasets(NUM_DATASETS, None)

    # Load all datasets
    datasets = load_all_datasets()

    # Run experiments
    results_df = run_experiments(datasets, NUM_RUNS)

    # Analyze and plot results
    avg_performance, best_method, param_impact = analyze_results(results_df)
    plot_results(avg_performance, best_method, param_impact)

    # Save final results
    results_df.to_csv('experiment_results.csv', index=False)

    logger.info(f"Results stored in CSV: experiment_results.csv")
    logger.info(f"Results also stored in SQLite database: {DB_NAME}")
    logger.info("Experiment completed successfully")
