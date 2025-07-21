import sqlite3
import json
import random
import optuna

from data_generator.main import generate_optimal_discrimination_data
from methods.group.fair_naive_bayes.main import run_naive_bayes
from methods.group.verifair.run_verifair import run_verifair

from methods.individual.adf.main import run_adf
from methods.individual.aequitas.algo import run_aequitas
from methods.individual.exp_ga.algo import run_expga
from methods.individual.fliptest.main import run_fliptest
from methods.individual.kosei.main import run_kosei
from methods.individual.limi.main import run_limi

from methods.subgroup.biasscan.algo import run_bias_scan
from methods.subgroup.divexplorer.main import run_divexploer
from methods.subgroup.gerryfair.main import run_gerryfair
from methods.subgroup.slicefinder.main import run_slicefinder
from methods.subgroup.sliceline.main import run_sliceline
from path import HERE

DATABASE_FILE = HERE.joinpath('experiments/synthetic_dataset_experiments2/synthetic_experiments_optuna.db')
N_TRIALS = 20  # Number of optimization trials for each method
MAX_RUNTIME_SECONDS = 300


def setup_database():
    """Create the database file if it doesn't exist."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.close()


def save_optuna_result(method, study, dataset_params):
    """Save the best result from an Optuna study to a method-specific table."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    table_name = f"{method}_results"

    # Create a table for the method if it doesn't exist
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            best_params TEXT,
            best_value REAL,
            dataset_params TEXT
        )
    ''')

    params_json = json.dumps(study.best_params)
    dataset_params_json = json.dumps(dataset_params)

    cursor.execute(f'''
        INSERT INTO {table_name} (best_params, best_value, dataset_params)
        VALUES (?, ?, ?)
    ''', (params_json, study.best_value, dataset_params_json))

    conn.commit()
    conn.close()


def create_objective(method_func, data, param_suggester):
    """Creates an Optuna objective function for a given method."""

    def objective(trial):
        params = param_suggester(trial)
        try:
            _, metrics = method_func(data=data, **params)
            # Objective: maximize DSN (return DSN directly since we're using 'maximize' direction)
            return metrics.get('DSN', float('-inf'))  # Return negative infinity for failures when maximizing
        except Exception as e:
            print(f"Error during trial for {method_func.__name__}: {e}")
            return float('-inf')  # Return negative infinity to penalize failures when maximizing

    return objective


def run_all_optimizations():
    """Run Optuna optimization for all specified methods."""
    print("--- Generating a random dataset for hyperparameter optimization ---")
    min_classes = random.randint(2, 4)
    min_group = random.randint(10, 50)
    min_diff_subgroup = random.uniform(0.05, 0.2)

    dataset_params = {
        'nb_groups': random.randint(50, 200),
        'nb_attributes': random.randint(10, 50),
        'prop_protected_attr': random.uniform(0.1, 0.5),
        'min_number_of_classes': min_classes,
        'max_number_of_classes': random.randint(min_classes, 7),
        'min_group_size': min_group,
        'max_group_size': random.randint(min_group, 150),
        'min_diff_subgroup_size': min_diff_subgroup,
        'max_diff_subgroup_size': random.uniform(min_diff_subgroup, 0.4),
        'nb_categories_outcome': 2
    }
    print("Generated dataset parameters:", json.dumps(dataset_params, indent=2))
    data = generate_optimal_discrimination_data(**dataset_params)

    def verifair_params(trial):
        return {
            'c': trial.suggest_float('c', 0.01, 0.5),
            'Delta': trial.suggest_float('Delta', 0.0, 0.1),
            'delta': trial.suggest_float('delta', 1e-12, 1e-8, log=True),
            'n_samples': trial.suggest_int('n_samples', 1, 10),
            'n_max': trial.suggest_int('n_max', 500, 2000),
            'is_causal': False,
            'log_iters': trial.suggest_int('log_iters', 500, 2000),
            'max_runtime_seconds': MAX_RUNTIME_SECONDS
        }

    def biasscan_params(trial):
        return {
            'test_size': trial.suggest_float('test_size', 0.1, 0.5),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'bias_scan_num_iters': trial.suggest_int('bias_scan_num_iters', 50, 200),
            'bias_scan_scoring': trial.suggest_categorical('bias_scan_scoring', ['Poisson', 'Entropy']),
            'bias_scan_favorable_value': 'high',
            'bias_scan_mode': 'ordinal',
            'max_runtime_seconds': MAX_RUNTIME_SECONDS
        }

    def slicefinder_params(trial):
        return {
            'approach': 'lattice',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'k': trial.suggest_int('k', 2, 10),
            'epsilon': trial.suggest_float('epsilon', 0.1, 0.5),
            'min_size': trial.suggest_int('min_size', 50, 200),
            'min_effect_size': trial.suggest_float('min_effect_size', 0.1, 0.5),
            'max_runtime_seconds': MAX_RUNTIME_SECONDS
        }

    def sliceline_params(trial):
        return {
            'K': trial.suggest_int('K', 2, 10),
            'max_runtime_seconds': MAX_RUNTIME_SECONDS
        }

    def adf_params(trial):
        return {
            'max_local': trial.suggest_int('max_local', 50, 200),
            'cluster_num': trial.suggest_int('cluster_num', 20, 100),
            'step_size': trial.suggest_float('step_size', 0.01, 0.2),
            'max_runtime_seconds': MAX_RUNTIME_SECONDS
        }

    def aequitas_params(trial):
        return {
            'max_local': trial.suggest_int('max_local', 5000, 15000),
            'step_size': trial.suggest_float('step_size', 0.5, 2.0),
            'max_runtime_seconds': MAX_RUNTIME_SECONDS
        }

    def expga_params(trial):
        return {
            'threshold_rank': trial.suggest_float('threshold_rank', 0.1, 0.9),
            'max_local': trial.suggest_int('max_local', 500, 2000),
            'threshold': trial.suggest_float('threshold', 0.1, 0.9),
            'max_runtime_seconds': MAX_RUNTIME_SECONDS
        }

    def kosei_params(trial):
        return {
            'num_samples': trial.suggest_int('num_samples', 100, 500),
            'local_search_limit': trial.suggest_int('local_search_limit', 50, 200),
            'max_runtime_seconds': MAX_RUNTIME_SECONDS
        }

    def limi_params(trial):
        return {
            'lambda_val': trial.suggest_float('lambda_val', 0.1, 0.9),
            'n_test_samples': trial.suggest_int('n_test_samples', 1000, 5000),
            'max_runtime_seconds': MAX_RUNTIME_SECONDS
        }

    def naive_bayes_params(trial):
        return {
            'delta': trial.suggest_float('delta', 1e-3, 1e-1, log=True),
            'k': trial.suggest_int('k', 5, 50),
            'max_runtime_seconds': MAX_RUNTIME_SECONDS
        }

    def fliptest_params(trial):
        return {
            'max_runs': trial.suggest_int('max_runs', 1, 10),
            'max_runtime_seconds': MAX_RUNTIME_SECONDS
        }

    def divexplorer_params(trial):
        return {
            'K': trial.suggest_int('K', 5, 50),
            'min_support': trial.suggest_float('min_support', 0.01, 0.2),
            'max_runtime_seconds': MAX_RUNTIME_SECONDS
        }

    def gerryfair_params(trial):
        return {
            'C': trial.suggest_float('C', 1, 100, log=True),
            'gamma': trial.suggest_float('gamma', 1e-3, 1e-1, log=True),
            'max_iters': trial.suggest_int('max_iters', 2, 10)
        }

    methods_to_optimize = {
        'verifair': (run_verifair, verifair_params),
        'bias_scan': (run_bias_scan, biasscan_params),
        'slicefinder': (run_slicefinder, slicefinder_params),
        'sliceline': (run_sliceline, sliceline_params),
        'adf': (run_adf, adf_params),
        'aequitas': (run_aequitas, aequitas_params),
        'expga': (run_expga, expga_params),
        'kosei': (run_kosei, kosei_params),
        'limi': (run_limi, limi_params),
        'naive_bayes': (run_naive_bayes, naive_bayes_params),
        'fliptest': (run_fliptest, fliptest_params),
        'divexplorer': (run_divexploer, divexplorer_params),
        'gerryfair': (run_gerryfair, gerryfair_params),
    }

    for name, (method_func, param_suggester) in methods_to_optimize.items():
        print(f"\n--- Optimizing {name} ---")
        objective_func = create_objective(method_func, data, param_suggester)
        # Changed direction to 'maximize' to maximize DSN
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_func, n_trials=N_TRIALS)

        print(f"Finished optimizing {name}.")
        print(f"  Best value: {study.best_value}")
        print(f"  Best params: {study.best_params}")
        save_optuna_result(name, study, dataset_params)


def get_best_params(method: str, dataset_params: dict) -> dict:
    """Get the best parameters for a method given dataset parameters.

    Args:
        method (str): The method name
        dataset_params (dict): The dataset parameters used during optimization

    Returns:
        dict: The best parameters found, or empty dict if not found
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    table_name = f"{method}_results"
    dataset_params_json = json.dumps(dataset_params)

    try:
        # Changed ORDER BY to DESC since we're now maximizing (higher values are better)
        cursor.execute(f'''
            SELECT best_params FROM {table_name}
            WHERE dataset_params = ?
            ORDER BY best_value DESC
            LIMIT 1
        ''', (dataset_params_json,))

        result = cursor.fetchone()

        if result:
            best_params = json.loads(result[0])
            return best_params
        else:
            print(f"No results found for method '{method}' with the specified dataset parameters.")
            return {}

    except sqlite3.OperationalError:
        print(f"Table '{table_name}' not found. Please run the optimization first.")
        return {}
    finally:
        conn.close()


if __name__ == "__main__":
    setup_database()
    run_all_optimizations()
    print("\n--- All optimizations completed. Results saved to 'synthetic_experiments_optuna.db' ---")
