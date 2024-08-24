import signal

import pandas as pd
from typing import Set, Dict, List, Tuple, Any, Union
from scipy.stats import qmc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sqlite3
import hashlib
import json
import logging
import time
from datetime import timedelta

from methods.biasscan.algo import run_bias_scan
from methods.exp_ga.algo import run_expga
from methods.ml_check.algo import run_mlcheck
from data_generator.main import generate_data
from methods.aequitas.algo import run_aequitas
from paths import HERE

# Constants
DB_NAME = HERE.joinpath('experiments/experiment_results.db')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Database functions
def create_database():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS experiments
                 (param_hash TEXT PRIMARY KEY, params TEXT, results TEXT, meta TEXT)''')
    conn.commit()
    conn.close()
    logger.info(f"Database '{DB_NAME}' initialized")


def param_hash(params: Dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()


def experiment_exists(params: Dict[str, Any]) -> bool:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hash_value = param_hash(params)
    c.execute("SELECT 1 FROM experiments WHERE param_hash = ?", (hash_value,))
    result = c.fetchone() is not None
    conn.close()
    return result


def save_experiment(params: Dict[str, Any], results_dfs: Dict, results_metas: Dict, results: Dict):
    conn = sqlite3.connect(DB_NAME)

    # Save the experiment summary
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


def get_experiment_results(params: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hash_value = param_hash(params)
    c.execute("SELECT results FROM experiments WHERE param_hash = ?", (hash_value,))
    result = c.fetchone()
    conn.close()
    return json.loads(result[0]) if result else None


def analyze_comprehensive_results(results_df):
    logger.info("Analyzing comprehensive results")
    methods = ['MLCheck', 'Aequitas', 'BiasScan', 'ExpGA']
    avg_performance = results_df[[f"{method}_matching" for method in methods] +
                                 [f"{method}_new" for method in methods]].mean()

    best_method = avg_performance.idxmax().split('_')[0]

    param_impact = {}
    for param in results_df.columns:
        if param not in [f"{method}_{metric}" for method in methods for metric in ['matching', 'new']]:
            correlation = results_df[[param] + [f"{method}_matching" for method in methods] +
                                     [f"{method}_new" for method in methods]].corr()[param].drop(param)
            param_impact[param] = correlation

    return avg_performance, best_method, param_impact


def plot_comprehensive_results(avg_performance, best_method, param_impact):
    logger.info("Plotting comprehensive results")
    plt.figure(figsize=(12, 6))
    avg_performance.plot(kind='bar')
    plt.title('Average Performance of Each Method')
    plt.xlabel('Method and Metric')
    plt.ylabel('Average Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 10))
    sns.heatmap(pd.DataFrame(param_impact).T, cmap='coolwarm', center=0, annot=True)
    plt.title('Impact of Parameters on Method Performance')
    plt.tight_layout()
    plt.show()

    print(f"Best performing method overall: {best_method}")


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Function call timed out")


def run_with_timeout(func, *args, timeout=300, **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        result = func(*args, **kwargs)
    except TimeoutException:
        logger.warning(f"Function {func.__name__} timed out after {timeout} seconds")
        result = None
    finally:
        signal.alarm(0)  # Disable the alarm

    return result


def run_experiment_with_params(params: Dict[str, Any]) -> Dict[str, Union[Dict[str, int], pd.DataFrame]]:
    logger.info(f"Running experiment with parameters: {params}")
    ge = generate_data(**params)
    original_couple_keys: Set[str] = set(ge.dataframe['couple_key'])
    methods = ['MLCheck', 'Aequitas', 'BiasScan', 'ExpGA']
    method_results = {}

    for method in methods:
        logger.info(f"Running {method}")
        try:
            if method == 'MLCheck':
                result = run_with_timeout(run_mlcheck, ge, iteration_no=1)
                method_results[method] = result if result is not None else None
            elif method == 'Aequitas':
                result = run_with_timeout(run_aequitas,
                                          ge.training_dataframe, col_to_be_predicted=ge.outcome_column,
                                          sensitive_param_name_list=ge.protected_attributes,
                                          perturbation_unit=1, model_type="DecisionTree", threshold=0,
                                          global_iteration_limit=100, local_iteration_limit=10
                                          )
                method_results[method] = result if result is not None else None
            elif method == 'BiasScan':
                result = run_with_timeout(run_bias_scan,
                                          ge, test_size=0.3, random_state=42, n_estimators=200,
                                          bias_scan_num_iters=100, bias_scan_scoring='Poisson',
                                          bias_scan_favorable_value='high', bias_scan_mode='ordinal'
                                          )
                method_results[method] = result if result is not None else None
            elif method == 'ExpGA':
                result = run_with_timeout(run_expga, ge, threshold=0.5, threshold_rank=0.5,
                                          max_global=50, max_local=50)
                method_results[method] = result if result is not None else None
        except Exception as e:
            logger.error(f"An error occurred while running {method}: {str(e)}")
            continue

    results = {
        method: analyze_method(set(res[0]['couple_key']),
                               original_couple_keys) if res[0] is not None and 'couple_key' in res[0] else {
            'matching': 0,
            'new': 0}
        for method, res in method_results.items() if isinstance(res[0], pd.DataFrame)
    }

    results_dfs = {method: res[0] for method, res in method_results.items() if isinstance(res[0], pd.DataFrame)}
    results_metas = {method: res[1] for method, res in method_results.items() if isinstance(res[0], pd.DataFrame)}

    save_experiment(params, results_dfs, results_metas, results)
    return results


def analyze_method(method_couple_keys: Set[str], original_couple_keys: Set[str]) -> Dict[str, int]:
    matching = len(method_couple_keys.intersection(original_couple_keys))
    new = len(method_couple_keys - original_couple_keys)
    return {'matching': matching, 'new': new}


def generate_efficient_parameter_combinations(param_ranges: Dict[str, Tuple[float, float]], num_samples: int):
    continuous_params = [param for param, (low, high) in param_ranges.items() if isinstance(low, (int, float))]
    categorical_params = [param for param in param_ranges if param not in continuous_params]

    # Generate Latin Hypercube samples for continuous parameters
    sampler = qmc.LatinHypercube(d=len(continuous_params))
    samples = sampler.random(n=num_samples)

    param_combinations = []
    for sample in samples:
        params = {}
        for i, param in enumerate(continuous_params):
            low, high = param_ranges[param]
            value = low + (high - low) * sample[i]
            if isinstance(param_ranges[param][0], int):
                value = int(round(value))
            params[param] = value

        # Add categorical parameters
        for param in categorical_params:
            params[param] = np.random.choice(param_ranges[param])

        # Ensure min and max are the same for paired parameters
        for param in ['similarity', 'alea_uncertainty', 'epis_uncertainty', 'magnitude', 'frequency']:
            if f'min_{param}' in params and f'max_{param}' in params:
                params[f'max_{param}'] = params[f'min_{param}']

        param_combinations.append(params)

    return param_combinations


def run_efficient_experiment(param_ranges: Dict[str, Tuple[float, float]], num_samples: int = 100):
    # create_database()

    param_combinations = generate_efficient_parameter_combinations(param_ranges, num_samples)
    total_experiments = len(param_combinations)
    logger.info(f"Total number of experiments to run: {total_experiments}")

    results = []
    start_time = time.time()
    completed_experiments = 0
    skipped_experiments = 0

    for params in tqdm(param_combinations, desc="Running experiments", unit="experiment"):
        if experiment_exists(params):
            skipped_experiments += 1
            experiment_results = get_experiment_results(params)
        else:
            experiment_results = run_experiment_with_params(params)
            completed_experiments += 1

        results.append({**params, **{f"{method}_{key}": value
                                     for method, method_results in experiment_results.items()
                                     for key, value in method_results.items()}})

        elapsed_time = time.time() - start_time
        experiments_done = completed_experiments + skipped_experiments
        estimated_total_time = (elapsed_time / experiments_done) * total_experiments if experiments_done > 0 else 0
        estimated_remaining_time = estimated_total_time - elapsed_time

        logger.info(f"Progress: {experiments_done}/{total_experiments} experiments processed")
        logger.info(f"Completed: {completed_experiments}, Skipped: {skipped_experiments}")
        logger.info(f"Elapsed time: {timedelta(seconds=int(elapsed_time))}")
        logger.info(f"Estimated time remaining: {timedelta(seconds=int(estimated_remaining_time))}")

    results_df = pd.DataFrame(results)

    logger.info(f"Experiment completed. Total time: {timedelta(seconds=int(time.time() - start_time))}")
    logger.info(f"Experiments run: {completed_experiments}, Experiments skipped: {skipped_experiments}")

    return results_df


# The analyze_comprehensive_results and plot_comprehensive_results functions remain unchanged

if __name__ == "__main__":
    logger.info("Starting efficient experiment")

    # Define parameter ranges
    param_ranges = {
        'min_number_of_classes': (2, 4),
        'max_number_of_classes': (4, 6),
        'nb_attributes': (4, 8),
        'prop_protected_attr': (0.1, 0.5),
        'nb_groups': (100, 500),
        'max_group_size': (30, 70),
        'hiddenlayers_depth': (2, 4),
        'min_similarity': (0.0, 1.0),
        'max_similarity': (0.0, 1.0),
        'min_alea_uncertainty': (0.0, 1.0),
        'max_alea_uncertainty': (0.0, 1.0),
        'min_epis_uncertainty': (0.0, 1.0),
        'max_epis_uncertainty': (0.0, 1.0),
        'min_magnitude': (0.0, 1.0),
        'max_magnitude': (0.0, 1.0),
        'min_frequency': (0.0, 1.0),
        'max_frequency': (0.0, 1.0),
        'categorical_outcome': [True, True],
        'nb_categories_outcome': (2, 6)
    }

    num_samples = 100  # Adjust this number based on your computational resources and time constraints
    efficient_results = run_efficient_experiment(param_ranges, num_samples)

    avg_performance, best_method, param_impact = analyze_comprehensive_results(efficient_results)
    plot_comprehensive_results(avg_performance, best_method, param_impact)

    logger.info(f"Results also stored in SQLite database: {DB_NAME}")
    logger.info("Experiment completed successfully")
