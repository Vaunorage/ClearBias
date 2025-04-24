import time
import sqlite3
import optuna
from typing import Tuple, Optional, Dict, Any
from data_generator.main import generate_data
from methods.adf.main import run_adf
from methods.aequitas.algo import run_aequitas
from methods.exp_ga.algo import run_expga
from methods.sg.main import run_sg
from path import HERE

# Original method parameters (will be used as defaults and bounds)
method_params = {
    'expga': {'threshold_rank': 0.5, 'max_global': 3000, 'max_local': 1000, 'threshold': 0.5,
              'max_runtime_seconds': 800},
    'aequitas': {'max_global': 100, 'max_local': 1000, 'step_size': 1.0, 'max_runtime_seconds': 800},
    'adf': {'max_global': 20000, 'max_local': 100, 'cluster_num': 50, 'max_runtime_seconds': 800,
            'step_size': 0.05},
    'sg': {'cluster_num': 50, 'max_runtime_seconds': 800}
}

# Define search spaces for hyperparameters
param_search_space = {
    'expga': {
        'threshold_rank': ('float', 0.1, 0.9),
        'max_global': ('int', 1000, 5000),
        'max_local': ('int', 500, 2000),
        'threshold': ('float', 0.1, 0.9)
    },
    'aequitas': {
        'max_global': ('int', 50, 200),
        'max_local': ('int', 500, 2000),
        'step_size': ('float', 0.5, 2.0)
    },
    'adf': {
        'max_global': ('int', 10000, 30000),
        'max_local': ('int', 50, 200),
        'cluster_num': ('int', 20, 100),
        'step_size': ('float', 0.01, 0.1)
    },
    'sg': {
        'cluster_num': ('int', 20, 100)
    }
}

DB_PATH = HERE.joinpath("experiments/analyzing_methods/global/global_testing_res.db")
OPTUNA_DB_PATH = HERE.joinpath("experiments/analyzing_methods/global/optuna_studies.db")


class ExperimentTracker:
    """
    Class to track experiments and avoid running duplicates.
    """

    def __init__(self, db_path=DB_PATH):
        """Initialize the experiment tracker with a database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._ensure_experiments_table_exists()
        self._ensure_hyperparameter_table_exists()

    def _ensure_hyperparameter_table_exists(self):
        """Create the hyperparameter optimization table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS hyperparameter_optimization (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            method TEXT NOT NULL,
            model_type TEXT NOT NULL,
            nb_attributes INTEGER NOT NULL,
            prop_protected_attr REAL NOT NULL,
            nb_categories_outcome INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            best_params TEXT NOT NULL,
            metric_value REAL NOT NULL
        )
        ''')
        self.conn.commit()

    def _ensure_experiments_table_exists(self):
        """Create the experiments tracking table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiment_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            method TEXT NOT NULL,
            model_type TEXT NOT NULL,
            nb_attributes INTEGER NOT NULL,
            prop_protected_attr REAL NOT NULL,
            nb_categories_outcome INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            results_table TEXT NOT NULL,
            testdata_table TEXT NOT NULL,
            execution_time REAL,
            success BOOLEAN DEFAULT 1
        )
        ''')
        self.conn.commit()

    def experiment_exists(self, method: str, model_type: str, dataset_attr: dict) -> Optional[Tuple[str, str]]:
        """
        Check if an experiment with the given parameters has already been run.

        Args:
            method: The method name ('expga', 'sg', 'aequitas', 'adf')
            model_type: The model type ('rf', 'mlp', 'dt')
            dataset_attr: Dictionary with dataset attributes

        Returns:
            Tuple of (results_table, testdata_table) if experiment exists, None otherwise
        """
        # Skip ADF check for non-MLP models since it's not applicable
        if method == 'adf' and model_type != 'mlp':
            return None

        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT results_table, testdata_table 
        FROM experiment_tracking 
        WHERE method = ? AND model_type = ? AND nb_attributes = ? 
        AND prop_protected_attr = ? AND nb_categories_outcome = ?
        AND success = 1
        ORDER BY timestamp DESC
        LIMIT 1
        ''', (
            method,
            model_type,
            dataset_attr['nb_attributes'],
            dataset_attr['prop_protected_attr'],
            dataset_attr['nb_categories_outcome']
        ))

        result = cursor.fetchone()
        return result if result else None

    def register_experiment(self, method: str, model_type: str, dataset_attr: dict,
                            results_table: str, testdata_table: str,
                            execution_time: float = None, success: bool = True):
        """
        Register a completed experiment in the tracking database.

        Args:
            method: The method name
            model_type: The model type
            dataset_attr: Dictionary with dataset attributes
            results_table: Name of the results table in DB
            testdata_table: Name of the test data table in DB
            execution_time: Time taken to execute the experiment
            success: Whether the experiment completed successfully
        """
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO experiment_tracking 
        (method, model_type, nb_attributes, prop_protected_attr, nb_categories_outcome, 
         timestamp, results_table, testdata_table, execution_time, success)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            method,
            model_type,
            dataset_attr['nb_attributes'],
            dataset_attr['prop_protected_attr'],
            dataset_attr['nb_categories_outcome'],
            int(time.time()),
            results_table,
            testdata_table,
            execution_time,
            1 if success else 0
        ))

        self.conn.commit()

    def get_best_hyperparameters(self, method: str, model_type: str, dataset_attr: dict) -> Optional[Dict[str, Any]]:
        """
        Get the best hyperparameters for a given method, model, and dataset combination.

        Args:
            method: The method name
            model_type: The model type
            dataset_attr: Dictionary with dataset attributes

        Returns:
            Dictionary of best hyperparameters if found, None otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT best_params 
        FROM hyperparameter_optimization 
        WHERE method = ? AND model_type = ? AND nb_attributes = ? 
        AND prop_protected_attr = ? AND nb_categories_outcome = ?
        ORDER BY metric_value DESC
        LIMIT 1
        ''', (
            method,
            model_type,
            dataset_attr['nb_attributes'],
            dataset_attr['prop_protected_attr'],
            dataset_attr['nb_categories_outcome']
        ))

        result = cursor.fetchone()
        if result:
            import json
            return json.loads(result[0])

        return None

    def register_hyperparameters(self, method: str, model_type: str, dataset_attr: dict,
                                 best_params: Dict[str, Any], metric_value: float):
        """
        Register the best hyperparameters in the database.

        Args:
            method: The method name
            model_type: The model type
            dataset_attr: Dictionary with dataset attributes
            best_params: Dictionary of best hyperparameters
            metric_value: Value of the optimization metric
        """
        import json
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO hyperparameter_optimization 
        (method, model_type, nb_attributes, prop_protected_attr, nb_categories_outcome, 
         timestamp, best_params, metric_value)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            method,
            model_type,
            dataset_attr['nb_attributes'],
            dataset_attr['prop_protected_attr'],
            dataset_attr['nb_categories_outcome'],
            int(time.time()),
            json.dumps(best_params),
            metric_value
        ))

        self.conn.commit()

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()


def create_objective_function(method: str, model_type: str, dataset_attr: dict):
    """
    Create an objective function for Optuna to optimize.

    Args:
        method: The method name
        model_type: The model type
        dataset_attr: Dictionary with dataset attributes

    Returns:
        An objective function to be used by Optuna
    """
    # Generate data once for all evaluations of this objective function
    data_obj = generate_data(min_group_size=100, max_group_size=1000, nb_groups=20, **dataset_attr)

    def objective(trial):
        # Set up method-specific parameters
        args = method_params[method].copy()

        # Replace with suggested hyperparameters from trial
        for param_name, (param_type, lower, upper) in param_search_space[method].items():
            if param_type == 'int':
                args[param_name] = trial.suggest_int(param_name, lower, upper)
            elif param_type == 'float':
                args[param_name] = trial.suggest_float(param_name, lower, upper)
            elif param_type == 'categorical':
                args[param_name] = trial.suggest_categorical(param_name, lower)  # lower is choices here

        # Set reduced runtime for optimization
        args['max_runtime_seconds'] = 200  # Reduced runtime during optimization
        args['data'] = data_obj
        args['model_type'] = model_type
        args['db_path'] = None  # Don't save results during optimization
        args['analysis_id'] = f"optuna_trial_{trial.number}"

        try:
            # Run the appropriate method and extract metric
            if method == 'expga':
                _, metrics = run_expga(**args)
                # Higher discrimination found is better
                return metrics.get('discrimination_found', 0)

            elif method == 'sg':
                _, metrics = run_sg(**args)
                # Higher discrimination found is better
                return metrics.get('discrimination_found', 0)

            elif method == 'aequitas':
                _, metrics = run_aequitas(**args)
                # Higher discrimination found is better
                return metrics.get('discrimination_found', 0)

            elif method == 'adf' and model_type == 'mlp':
                _, metrics = run_adf(**args)
                # Higher discrimination found is better
                return metrics.get('discrimination_found', 0)

            # Default fallback value
            return 0

        except Exception as e:
            print(f"Error in trial {trial.number}: {str(e)}")
            # Return a poor value to discourage Optuna from exploring this region
            return float('-inf')

    return objective


def optimize_hyperparameters(method: str, model_type: str, tracker: ExperimentTracker, dataset_attr: dict,
                             n_trials: int = 30) -> Dict[str, Any]:
    """
    Run hyperparameter optimization for a method and model type.

    Args:
        method: The method name
        model_type: The model type
        tracker: ExperimentTracker instance
        dataset_attr: Dictionary with dataset attributes
        n_trials: Number of Optuna trials

    Returns:
        Dictionary of best hyperparameters
    """
    print(f"Optimizing hyperparameters for {method} on {model_type} with {dataset_attr}")

    # Skip ADF for non-MLP models
    if method == 'adf' and model_type != 'mlp':
        print(f"Skipping {method} for {model_type} model (not applicable)")
        return method_params[method]

    # Check if we already have optimized hyperparameters
    existing_params = tracker.get_best_hyperparameters(method, model_type, dataset_attr)
    if existing_params:
        print(f"Using previously optimized hyperparameters for {method} on {model_type} with {dataset_attr}")
        return existing_params

    # Create unique study name
    study_name = f"{method}_{model_type}_attr{dataset_attr['nb_attributes']}_prot{dataset_attr['prop_protected_attr']}_cat{dataset_attr['nb_categories_outcome']}"

    # Set up storage (SQLite)
    storage = f"sqlite:///{OPTUNA_DB_PATH}"

    # Create and run study
    objective = create_objective_function(method, model_type, dataset_attr)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",  # We want to maximize discrimination found
        load_if_exists=True  # Continue existing studies
    )

    study.optimize(objective, n_trials=n_trials)

    # Get best parameters and add the fixed parameters
    best_params = method_params[method].copy()
    best_params.update(study.best_params)

    # Register the best hyperparameters
    tracker.register_hyperparameters(
        method=method,
        model_type=model_type,
        dataset_attr=dataset_attr,
        best_params=best_params,
        metric_value=study.best_value
    )

    print(f"Best hyperparameters for {method} on {model_type}: {best_params}")
    print(f"Best metric value: {study.best_value}")

    return best_params


def run_experiment_for_model(method: str, model_type: str, tracker: ExperimentTracker, dataset_attr: dict,
                             optimize_params: bool = True, n_trials: int = 30) -> None:
    """Run experiment for a specific model and dataset combination if not already done."""

    # Skip ADF for non-MLP models
    if method == 'adf' and model_type != 'mlp':
        print(f"Skipping {method} for {model_type} model (not applicable)")
        return

    # Check if experiment already exists
    existing_experiment = tracker.experiment_exists(method, model_type, dataset_attr)

    if existing_experiment:
        results_table, testdata_table = existing_experiment
        print(f"Experiment for {method} on {model_type} with {dataset_attr} already exists.")
        print(f"Results available in tables: {results_table} and {testdata_table}")
        return

    # Generate new analysis ID
    timestamp = int(time.time())
    analysis_id_base = f"{method}_{model_type}_attr{dataset_attr['nb_attributes']}_prot{dataset_attr['prop_protected_attr']}_cat{dataset_attr['nb_categories_outcome']}_{timestamp}"
    results_table = f"{analysis_id_base}_results"
    testdata_table = f"{analysis_id_base}_testdata"

    # Optimize hyperparameters if requested
    if optimize_params:
        params = optimize_hyperparameters(method, model_type, tracker, dataset_attr, n_trials)
    else:
        params = method_params[method].copy()

    # Generate data
    print(f"Running new experiment: {method} on {model_type} with {dataset_attr}")
    data_obj = generate_data(min_group_size=1000, max_group_size=10000, nb_groups=50, **dataset_attr)
    df = data_obj.dataframe.copy()
    df['algorithm'] = method
    df['model'] = model_type

    # Save test data
    conn = sqlite3.connect(tracker.db_path)
    df.to_sql(name=testdata_table, con=conn, if_exists='replace')

    # Prepare method parameters
    args = params.copy()
    args['data'] = data_obj
    args['model_type'] = model_type
    args['db_path'] = tracker.db_path
    args['analysis_id'] = results_table

    success = True
    start_time = time.time()

    try:
        # Run the appropriate method
        if method == 'expga':
            print(f"Running ExpGA for {model_type} with params: {args}")
            expga_results, expga_metrics = run_expga(**args)
            print("EXPGA METRICS", expga_metrics)

        elif method == 'sg':
            print(f"Running SG for {model_type} with params: {args}")
            sg_results, sg_metrics = run_sg(**args)
            print("SG METRICS", sg_metrics)

        elif method == 'aequitas':
            print(f"Running Aequitas for {model_type} with params: {args}")
            aequitas_results, aequitas_metrics = run_aequitas(**args)
            print("AEQUITAS METRICS", aequitas_metrics)

        elif method == 'adf' and model_type == 'mlp':
            print(f"Running ADF for {model_type} with params: {args}")
            adf_results, adf_metrics = run_adf(**args)
            print("ADF METRICS", adf_metrics)

    except Exception as e:
        print(f"Error running {method} on {model_type}: {str(e)}")
        success = False

    execution_time = time.time() - start_time

    # Register the experiment in the tracker
    tracker.register_experiment(
        method=method,
        model_type=model_type,
        dataset_attr=dataset_attr,
        results_table=results_table,
        testdata_table=testdata_table,
        execution_time=execution_time,
        success=success
    )

    print(f"Experiment completed in {execution_time:.2f} seconds. Success: {success}")


def run_all_experiments(optimize_params=True, n_trials=30):
    """Run all experiments with tracking to avoid duplicates."""
    tracker = ExperimentTracker()

    try:
        nb_attributes = [5, 10, 15, 20]
        prop_protected_attr = [0.1, 0.2, 0.5]
        nb_categories_outcome = [2, 3, 4, 6]
        methods = ['expga', 'sg', 'aequitas', 'adf']
        models = ['rf', 'mlp', 'dt']

        total_experiments = len(methods) * len(models) * len(nb_attributes) * len(prop_protected_attr) * len(
            nb_categories_outcome)
        completed = 0

        print(f"Planning to run up to {total_experiments} experiments (if not already completed)")
        print(f"Hyperparameter optimization: {optimize_params}, Trials per optimization: {n_trials}")

        for method in methods:
            for model in models:
                for attr in nb_attributes:
                    for protected_attr in prop_protected_attr:
                        for cat in nb_categories_outcome:
                            dataset_attr = {
                                'nb_attributes': attr,
                                'prop_protected_attr': protected_attr,
                                'nb_categories_outcome': cat
                            }

                            run_experiment_for_model(
                                method, model, tracker, dataset_attr,
                                optimize_params=optimize_params, n_trials=n_trials
                            )
                            completed += 1
                            print(f"Progress: {completed}/{total_experiments} experiments processed")

    finally:
        tracker.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run experiments with hyperparameter optimization')
    parser.add_argument('--optimize', action='store_true', help='Enable hyperparameter optimization')
    parser.add_argument('--trials', type=int, default=30, help='Number of Optuna trials per optimization')

    args = parser.parse_args()

    run_all_experiments(optimize_params=args.optimize, n_trials=args.trials)