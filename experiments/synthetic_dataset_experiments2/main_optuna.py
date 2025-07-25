import sqlite3
import json
import random
import uuid
import optuna
import numpy as np

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
from methods.individual.sg.main import run_sg
from path import HERE

DATABASE_FILE = HERE.joinpath('experiments/synthetic_dataset_experiments2/synthetic_experiments_optuna.db')
STUDIES_DATABASE_FILE = HERE.joinpath('experiments/synthetic_dataset_experiments2/optuna_studies.db')
N_TRIALS = 50  # Number of optimization trials for each method
MAX_RUNTIME_SECONDS = 300


def convert_numpy_types(obj):
    """Recursively convert NumPy types to native Python types in a dictionary or list."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def safe_json_serialize(obj):
    """Safely serialize objects to JSON, handling NumPy types and other non-serializable objects."""

    def convert_item(item):
        if isinstance(item, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(item)
        elif isinstance(item, (np.floating, np.float64, np.float32, np.float16)):
            return float(item)
        elif isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, (np.bool_, bool)):
            return bool(item)
        elif isinstance(item, dict):
            return {key: convert_item(value) for key, value in item.items()}
        elif isinstance(item, (list, tuple)):
            return [convert_item(x) for x in item]
        elif hasattr(item, '__dict__'):
            # For custom objects, try to convert their dict representation
            try:
                return convert_item(item.__dict__)
            except:
                return str(item)
        else:
            return item

    try:
        converted_obj = convert_item(obj)
        return json.dumps(converted_obj, sort_keys=True)
    except (TypeError, ValueError) as e:
        # Fallback: convert to string representation
        print(f"Warning: Could not serialize object, using string representation: {e}")
        return json.dumps(str(obj))


def setup_database():
    """Create the database file if it doesn't exist."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.close()


def get_study_storage_url():
    """Get the SQLite storage URL for Optuna studies."""
    return f"sqlite:///{STUDIES_DATABASE_FILE}"


def create_method_table(method_name):
    """Create a comprehensive table for storing all trial results for a method."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    table_name = f"{method_name}_results"

    # Create comprehensive table with all trial information
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_name TEXT,
            trial_number INTEGER,
            trial_params TEXT,
            trial_value REAL,
            trial_state TEXT,
            all_metrics TEXT,
            trial_datetime_start TEXT,
            trial_datetime_complete TEXT,
            trial_duration REAL,
            optuna_trial_id INTEGER,
            dataset_content TEXT,
            dataset_params TEXT,
            res_df_content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(study_name, trial_number)
        )
    ''')

    conn.commit()
    conn.close()


def save_trial_result_callback(study, trial):
    """Save all trial results from an Optuna study to the database."""
    method_name = study.study_name.split('_study_')[0]
    table_name = f"{method_name}_results"
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # Extract data from trial
    all_metrics = trial.user_attrs.get('all_metrics')
    dataset_params = trial.user_attrs.get('dataset_params')
    dataset_content = trial.user_attrs.get('dataset_content')
    res_df_content = trial.user_attrs.get('res_df_content')

    # Insert trial data into the specific method table
    cursor.execute(f'''
        INSERT OR REPLACE INTO "{table_name}" (
            study_name, trial_number, trial_params, trial_value, trial_state,
            all_metrics, trial_datetime_start, trial_datetime_complete, trial_duration,
            optuna_trial_id, dataset_content, dataset_params, res_df_content
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        study.study_name,
        trial.number,
        safe_json_serialize(trial.params),
        trial.value,
        str(trial.state).split('.')[-1],
        safe_json_serialize(all_metrics),
        trial.datetime_start.isoformat() if trial.datetime_start else None,
        trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        trial.duration.total_seconds() if trial.duration else None,
        trial._trial_id,
        dataset_content,
        safe_json_serialize(dataset_params),
        res_df_content
    ))

    conn.commit()
    conn.close()
    print(f"Saved results for trial {trial.number} of {study.study_name}")


def create_objective_with_metrics_tracking(method_name, method_func, param_suggester):
    """Creates an Optuna objective function that tracks all metrics and generates a new dataset for each trial."""

    def objective(trial):
        # Generate a new dataset for each trial

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
            'nb_categories_outcome': random.randint(2, 10)
        }

        if method_name == 'divexplorer':
            dataset_params['nb_categories_outcome'] = random.randint(2, 4)

        data = generate_optimal_discrimination_data(**dataset_params)

        # Store dataset and params in the tracker for later processing
        trial.set_user_attr('dataset_params', dataset_params)
        trial.set_user_attr('dataset_content', data.dataframe.to_json(orient='split'))

        # Suggest hyperparameters
        params = param_suggester(trial)

        res_df, all_metrics = method_func(data=data, **params)

        # Store all metrics in the tracker
        trial.set_user_attr('res_df_content', res_df.to_json(orient='split'))
        # Convert numpy types in metrics to native python types for JSON serialization
        all_metrics_serializable = convert_numpy_types(all_metrics)
        trial.set_user_attr('all_metrics', all_metrics_serializable)

        # Return the main objective value
        return all_metrics.get('DSN', float('-inf'))

    return objective


def generate_study_name(method_name):
    """Generate a unique study name based on method."""
    return f"{method_name}_study_{uuid.uuid4()}"


def get_completed_trials_count(method_name):
    """Get the number of completed trials for a method with given dataset params."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    table_name = f"{method_name}_results"

    try:
        cursor.execute(f'''
            SELECT COUNT(*) FROM {table_name} 
            AND trial_state = 'COMPLETE'
        ''', )
        result = cursor.fetchone()
        count = result[0] if result else 0
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        count = 0

    conn.close()
    return count


def run_all_optimizations():
    """Run Optuna optimization for all specified methods with persistence."""

    # Get storage URL for persistent studies
    storage_url = get_study_storage_url()
    print(f"Using storage: {storage_url}")

    # Parameter suggestion functions (same as before)
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
            'max_local': trial.suggest_int('max_local', 100, 4000),
            'max_global': trial.suggest_int('max_local', 100, 4000),
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
            'C': trial.suggest_float('C', 0.01, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0, log=True),
            'max_iters': trial.suggest_int('max_iterations', 5, 20)
        }

    def sg_params(trial):
        return {
            'cluster_num': trial.suggest_int('cluster_num', 10, 100),
            'max_tsn': trial.suggest_int('max_tsn', 50, 50000),
            'max_runtime_seconds': MAX_RUNTIME_SECONDS
        }

    methods_to_optimize = {
        # 'verifair': (run_verifair, verifair_params),
        # 'bias_scan': (run_bias_scan, biasscan_params),
        # 'slicefinder': (run_slicefinder, slicefinder_params),
        # 'sliceline': (run_sliceline, slicelinee_params),
        'adf': (run_adf, adf_params),
        # 'aequitas': (run_aequitas, aequitas_params),
        # 'expga': (run_expga, expga_params),
        # 'kosei': (run_kosei, kosei_params),
        # 'limi': (run_limi, limi_params),
        # 'naive_bayes': (run_naive_bayes, naive_bayes_params),
        # 'fliptest': (run_fliptest, fliptest_params),
        # 'divexplorer': (run_divexploer, divexplorer_params),
        # 'gerryfair': (run_gerryfair, gerryfair_params),
        # 'sg': (run_sg, sg_params),
    }

    for name, (method_func, param_suggester) in methods_to_optimize.items():
        print(f"\n--- Processing {name} ---")

        # Create table for this method
        create_method_table(name)

        # Generate unique study name
        study_name = generate_study_name(name)

        # Create the objective function
        objective_func = create_objective_with_metrics_tracking(name, method_func, param_suggester)

        # Create study with SQLite storage
        study = optuna.create_study(
            direction='maximize',
            storage=storage_url,
            study_name=study_name,
            load_if_exists=True
        )

        print(f"Study name: {study_name}")
        print(f"Current number of trials in study: {len(study.trials)}")

        # Run remaining trials with the callback
        study.optimize(objective_func, n_trials=N_TRIALS, callbacks=[save_trial_result_callback])

        print(f"Finished optimizing {name}.")
        if study.best_trial:
            print(f"  Best value: {study.best_value}")
            print(f"  Best params: {study.best_params}")
        print(f"  Total trials: {len(study.trials)}")




def get_experiment_summary():
    """Print a summary of all experiments."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # Get all method tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_results'")
    tables = cursor.fetchall()

    print("\n--- EXPERIMENT SUMMARY ---")
    for (table_name,) in tables:
        method_name = table_name.replace('_results', '')

        # Get trial counts by state
        cursor.execute(f'''
            SELECT trial_state, COUNT(*) FROM "{table_name}" 
            GROUP BY trial_state
        ''')
        state_counts = dict(cursor.fetchall())

        # Get best result
        cursor.execute(f'''
            SELECT MAX(trial_value), trial_params FROM "{table_name}" 
            WHERE trial_state = 'COMPLETE'
        ''')
        best_result = cursor.fetchone()

        print(f"\n{method_name.upper()}:")
        print(f"  Trial states: {state_counts}")
        if best_result and best_result[0] is not None:
            print(f"  Best DSN: {best_result[0]:.4f}")
            print(f"  Best params: {best_result[1]}")

    conn.close()


if __name__ == "__main__":
    setup_database()
    run_all_optimizations()
    print("\n--- All optimizations completed. Results saved to databases ---")
    print(f"Study data stored in: {STUDIES_DATABASE_FILE}")
    print(f"Results summary stored in: {DATABASE_FILE}")

    # Print experiment summary
    get_experiment_summary()
