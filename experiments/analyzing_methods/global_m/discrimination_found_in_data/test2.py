import time
import sqlite3
import json
from functools import lru_cache

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from data_generator.main import generate_data
from methods.adf.main import run_adf
from methods.aequitas.algo import run_aequitas
from methods.exp_ga.algo import run_expga
from methods.sg.main import run_sg
from path import HERE
from experiments.analyzing_methods.global_m.discrimination_found_in_data.meta_learner import MetaLearner
# Import the MetaLearner

DB_PATH = HERE.joinpath("experiments/analyzing_methods/global_m/global_testing_res.db")


class EnhancedExperimentTracker:
    """
    Enhanced version of ExperimentTracker that integrates meta-learning capabilities.
    """

    def __init__(self, db_path=DB_PATH):
        """Initialize the experiment tracker with a database connection and meta-learner."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._ensure_experiments_table_exists()
        self._ensure_optuna_table_exists()

        # Initialize the meta-learner
        self.meta_learner = MetaLearner(db_path)

        # Track which methods have meta-models available
        self.available_meta_models = self._load_available_meta_models()

    def _load_available_meta_models(self):
        """Load all available meta-models."""
        available_models = {}

        for method in ['expga', 'sg', 'aequitas', 'adf']:
            # Try to load meta-model from database
            if self.meta_learner.load_meta_model(method):
                available_models[method] = True
            else:
                # If not available, try to train a new one
                if self.meta_learner.train_meta_model(method):
                    available_models[method] = True
                    # Save the newly trained model
                    self.meta_learner.save_meta_model(method)
                else:
                    available_models[method] = False

        return available_models

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
            success BOOLEAN DEFAULT 1,
            params TEXT, -- JSON string of parameters
            objective_value REAL, -- Objective value from optimization
            used_meta_learning BOOLEAN DEFAULT 0 -- Whether meta-learning was used
        )
        ''')
        self.conn.commit()

    def _ensure_optuna_table_exists(self):
        """Create a table to store Optuna trial results."""
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS optuna_trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            method TEXT NOT NULL,
            model_type TEXT NOT NULL,
            nb_attributes INTEGER NOT NULL,
            prop_protected_attr REAL NOT NULL,
            nb_categories_outcome INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            trial_number INTEGER NOT NULL,
            params TEXT NOT NULL, -- JSON string of parameters
            results_table TEXT NOT NULL,
            testdata_table TEXT NOT NULL,
            num_exact_couple_matches INTEGER NOT NULL,
            num_new_group_couples INTEGER NOT NULL,
            objective_value REAL NOT NULL,
            execution_time REAL,
            used_meta_learning BOOLEAN DEFAULT 0
        )
        ''')
        self.conn.commit()

    def experiment_exists(self, method: str, model_type: str, dataset_attr: dict, params: Optional[Dict] = None) -> \
    Optional[Tuple[str, str]]:
        """
        Check if an experiment with the given parameters has already been run.
        """
        # Skip ADF check for non-MLP models since it's not applicable
        if method == 'adf' and model_type != 'mlp':
            return None

        cursor = self.conn.cursor()

        # Handle the query differently if params is provided (for optimization runs)
        if params:
            params_json = json.dumps(params, sort_keys=True)
            cursor.execute('''
            SELECT results_table, testdata_table 
            FROM experiment_tracking 
            WHERE method = ? AND model_type = ? AND nb_attributes = ? 
            AND prop_protected_attr = ? AND nb_categories_outcome = ?
            AND params = ?
            AND success = 1
            ORDER BY timestamp DESC
            LIMIT 1
            ''', (
                method,
                model_type,
                dataset_attr['nb_attributes'],
                dataset_attr['prop_protected_attr'],
                dataset_attr['nb_categories_outcome'],
                params_json
            ))
        else:
            cursor.execute('''
            SELECT results_table, testdata_table 
            FROM experiment_tracking 
            WHERE method = ? AND model_type = ? AND nb_attributes = ? 
            AND prop_protected_attr = ? AND nb_categories_outcome = ?
            AND params IS NULL
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
                            execution_time: float = None, success: bool = True,
                            params: Optional[Dict] = None, objective_value: Optional[float] = None,
                            used_meta_learning: bool = False):
        """
        Register a completed experiment in the tracking database.
        """
        params_json = json.dumps(params, sort_keys=True) if params else None

        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO experiment_tracking 
        (method, model_type, nb_attributes, prop_protected_attr, nb_categories_outcome, 
         timestamp, results_table, testdata_table, execution_time, success, params, 
         objective_value, used_meta_learning)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            1 if success else 0,
            params_json,
            objective_value,
            1 if used_meta_learning else 0
        ))

        self.conn.commit()

    def register_optuna_trial(self, method: str, model_type: str, dataset_attr: dict,
                              trial_number: int, params: Dict, results_table: str, testdata_table: str,
                              num_exact_matches: int, num_new_couples: int, objective_value: float,
                              execution_time: float, used_meta_learning: bool = False):
        """
        Register an Optuna trial in the database.
        """
        params_json = json.dumps(params, sort_keys=True)

        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO optuna_trials 
        (method, model_type, nb_attributes, prop_protected_attr, nb_categories_outcome, 
         timestamp, trial_number, params, results_table, testdata_table, 
         num_exact_couple_matches, num_new_group_couples, objective_value, execution_time,
         used_meta_learning)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            method,
            model_type,
            dataset_attr['nb_attributes'],
            dataset_attr['prop_protected_attr'],
            dataset_attr['nb_categories_outcome'],
            int(time.time()),
            trial_number,
            params_json,
            results_table,
            testdata_table,
            num_exact_matches,
            num_new_couples,
            objective_value,
            execution_time,
            1 if used_meta_learning else 0
        ))

        self.conn.commit()

    def get_meta_optimized_params(self, method: str, model_type: str, dataset_attr: dict) -> Optional[Dict]:
        """
        Get optimized parameters using meta-learning for a specific experiment configuration.

        Args:
            method: The method name
            model_type: The model type
            dataset_attr: Dictionary with dataset attributes

        Returns:
            Dictionary of optimized parameters or None if not available
        """
        # Check if we have a meta-model for this method
        if method in self.available_meta_models and self.available_meta_models[method]:
            # Use meta-learning to predict good parameters
            predicted_params = self.meta_learner.predict_good_params(method, model_type, dataset_attr)

            if predicted_params:
                print(f"Using meta-learning predicted parameters for {method} on {model_type}")
                return predicted_params

        # Fall back to previously best parameters if available
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT params, objective_value
        FROM experiment_tracking
        WHERE method = ? AND model_type = ? AND nb_attributes = ? 
        AND prop_protected_attr = ? AND nb_categories_outcome = ?
        AND success = 1
        ORDER BY objective_value DESC
        LIMIT 1
        ''', (
            method,
            model_type,
            dataset_attr['nb_attributes'],
            dataset_attr['prop_protected_attr'],
            dataset_attr['nb_categories_outcome']
        ))

        result = cursor.fetchone()
        if result and result[0]:
            print(f"Using previously best parameters for {method} on {model_type}")
            return json.loads(result[0])

        return None

    def update_meta_models(self):
        """
        Update all meta-models with the latest experiment data.
        This should be called periodically to incorporate new knowledge.
        """
        print("Updating meta-models with latest experiment data...")

        for method in ['expga', 'sg', 'aequitas', 'adf']:
            print(f"Training meta-model for {method}...")
            success = self.meta_learner.train_meta_model(method)

            if success:
                # Save the updated model
                self.meta_learner.save_meta_model(method)
                self.available_meta_models[method] = True
                print(f"Meta-model for {method} updated successfully")
            else:
                print(f"Failed to update meta-model for {method}")

    def close(self):
        """Close the database connection and meta-learner."""
        if self.conn:
            self.conn.close()

        self.meta_learner.close()


@lru_cache(maxsize=4096)
def matches_pattern(pattern: str, value: str) -> bool:
    for sub_pat, sub_pat_val in zip(pattern.split("-"), value.split("-")):
        for el1, el2 in zip(sub_pat.split('|'), sub_pat_val.split('|')):
            if el1 == '*':
                continue
            elif el1 != el2:
                return False
    return True


def is_individual_part_of_the_original_indv(indv_key, indv_key_list):
    return indv_key in indv_key_list


def is_couple_part_of_a_group(couple_key, group_key_list):
    res = []

    couple_key_elems = couple_key.split('-')
    if len(couple_key_elems) != 2:
        print(f"Warning: Unexpected couple key format: {couple_key}")
        return res

    opt1 = f"{couple_key_elems[0]}-{couple_key_elems[1]}"
    opt2 = f"{couple_key_elems[1]}-{couple_key_elems[0]}"

    for grp_key in group_key_list:
        if matches_pattern(grp_key, opt1) or matches_pattern(grp_key, opt2):
            res.append(grp_key)
    return res


def analyze_results(results_df, synthetic_data) -> dict:
    """Analyze the results and calculate metrics for optimization objective."""
    # Convert to string type once
    synthetic_data = synthetic_data.astype({
        'indv_key': str,
        'group_key': str,
        'subgroup_key': str
    })

    results_df = results_df.astype({
        'indv_key': str,
        'couple_key': str
    })

    # Get unique groups
    unique_groups = synthetic_data['group_key'].unique()

    total_exact_couple_matches = 0
    total_new_group_couples = 0

    for group_key in unique_groups:
        # Get individuals in this group from synthetic data
        group_synthetic_indv = set(
            synthetic_data[synthetic_data['group_key'] == group_key]['indv_key']
        )

        # Find exact couple matches
        exact_couple_matches = []
        for couple_key in results_df['couple_key'].unique():
            try:
                indv1, indv2 = couple_key.split('-')
                if (is_individual_part_of_the_original_indv(indv1, group_synthetic_indv) and
                        is_individual_part_of_the_original_indv(indv2, group_synthetic_indv)):
                    exact_couple_matches.append(couple_key)
            except ValueError:
                continue  # Skip malformed couple keys

        # Find new couples matching group pattern but not in original data
        new_group_couples = []
        for key in results_df['couple_key'].unique():
            if is_couple_part_of_a_group(key, [group_key]) and key not in exact_couple_matches:
                new_group_couples.append(key)

        total_exact_couple_matches += len(exact_couple_matches)
        total_new_group_couples += len(new_group_couples)

    return {
        'num_exact_couple_matches': total_exact_couple_matches,
        'num_new_group_couples': total_new_group_couples
    }


def run_meta_optimization(method: str, model_type: str, dataset_attr: dict, tracker: EnhancedExperimentTracker,
                          n_trials: int = 30) -> Dict:
    """
    Run Optuna optimization with meta-learning to find the best hyperparameters.

    Args:
        method: The method name ('expga', 'sg', 'aequitas', 'adf')
        model_type: The model type ('rf', 'mlp', 'dt')
        dataset_attr: Dictionary with dataset attributes
        tracker: EnhancedExperimentTracker instance
        n_trials: Number of optimization trials to run

    Returns:
        Dictionary of best parameters found
    """
    # Skip ADF for non-MLP models
    if method == 'adf' and model_type != 'mlp':
        print(f"Skipping optimization for {method} on {model_type} (not applicable)")
        return {}

    # First, try to get meta-optimized parameters
    meta_params = tracker.get_meta_optimized_params(method, model_type, dataset_attr)

    if meta_params:
        print(f"Using meta-learning optimized parameters directly for {method} on {model_type}")
        return meta_params

    # Generate data once for the optimization
    data_obj = generate_data(min_group_size=1000, max_group_size=10000, nb_groups=50, **dataset_attr)

    # Get parameter search spaces
    param_search_spaces = {
        'expga': {
            'threshold_rank': (0.1, 0.9),
            'max_global': (1000, 5000),
            'max_local': (500, 2000),
            'threshold': (0.1, 0.9)
        },
        'aequitas': {
            'max_global': (50, 300),
            'max_local': (500, 2000),
            'step_size': (0.5, 2.0)
        },
        'adf': {
            'max_global': (10000, 30000),
            'max_local': (50, 200),
            'cluster_num': (20, 100),
            'step_size': (0.01, 0.1)
        },
        'sg': {
            'cluster_num': (20, 100)
        }
    }

    # Default parameters
    base_params = {
        'expga': {'threshold_rank': 0.5, 'max_global': 3000, 'max_local': 1000, 'threshold': 0.5,
                  'max_runtime_seconds': 800},
        'aequitas': {'max_global': 100, 'max_local': 1000, 'step_size': 1.0, 'max_runtime_seconds': 800},
        'adf': {'max_global': 20000, 'max_local': 100, 'cluster_num': 50, 'max_runtime_seconds': 800,
                'step_size': 0.05},
        'sg': {'cluster_num': 50, 'max_runtime_seconds': 800}
    }

    # Create study with meta-learning
    import optuna
    study = optuna.create_study(direction='maximize')

    # Define the objective function
    def objective(trial):
        # Get meta-predicted parameters if available
        best_guess_params = tracker.meta_learner.predict_good_params(method, model_type, dataset_attr)
        used_meta_learning = best_guess_params is not None

        # Generate parameters for this trial, starting from meta-learning predictions if available
        params = base_params[method].copy()
        search_space = param_search_spaces.get(method, {})

        for param_name, param_range in search_space.items():
            # If we have a meta-prediction, use it as a starting point
            if used_meta_learning and param_name in best_guess_params:
                meta_value = best_guess_params[param_name]

                # Define a narrower search range around the meta-predicted value
                if isinstance(param_range[0], int):
                    # For integers, use a percentage-based window
                    width = (param_range[1] - param_range[0]) // 4
                    lower = max(param_range[0], int(meta_value - width))
                    upper = min(param_range[1], int(meta_value + width))
                    params[param_name] = trial.suggest_int(param_name, lower, upper)
                else:
                    # For floats, use a standard deviation based approach
                    width = (param_range[1] - param_range[0]) / 4
                    lower = max(param_range[0], meta_value - width)
                    upper = min(param_range[1], meta_value + width)
                    params[param_name] = trial.suggest_float(param_name, lower, upper)
            else:
                # Regular parameter suggestion
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])

        # Check if experiment with these parameters already exists
        existing_experiment = tracker.experiment_exists(method, model_type, dataset_attr, params)

        if existing_experiment:
            results_table, testdata_table = existing_experiment
            print(f"Experiment for trial {trial.number} already exists: {results_table}")

            # Load results from existing experiment
            conn = sqlite3.connect(tracker.db_path)
            results_df = pd.read_sql(f"SELECT * FROM {results_table}", conn)
            synthetic_data = pd.read_sql(f"SELECT * FROM {testdata_table}", conn)

            # Calculate the metrics
            metrics = analyze_results(results_df, synthetic_data)

            # Calculate objective value (weighted sum of the metrics)
            objective_value = metrics['num_exact_couple_matches'] + metrics['num_new_group_couples'] * 1.5

            return objective_value

        # Generate new analysis ID for this trial
        timestamp = int(time.time())
        analysis_id_base = (
            f"{method}_{model_type}_attr{dataset_attr['nb_attributes']}"
            f"_prot{dataset_attr['prop_protected_attr']}_cat{dataset_attr['nb_categories_outcome']}"
            f"_trial{trial.number}_{timestamp}"
        )
        results_table = f"{analysis_id_base}_results"
        testdata_table = f"{analysis_id_base}_testdata"

        # Prepare DataFrame for saving test data
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

        # Initialize metrics with default values
        metrics = {'num_exact_couple_matches': 0, 'num_new_group_couples': 0}
        success = True
        start_time = time.time()

        try:
            # Run the appropriate method
            if method == 'expga':
                print(f"Running ExpGA trial {trial.number} for {model_type}...")
                expga_results, expga_metrics = run_expga(**args)
                print("EXPGA METRICS", expga_metrics)

            elif method == 'sg':
                print(f"Running SG trial {trial.number} for {model_type}...")
                sg_results, sg_metrics = run_sg(**args)
                print("SG METRICS", sg_metrics)

            elif method == 'aequitas':
                print(f"Running Aequitas trial {trial.number} for {model_type}...")
                aequitas_results, aequitas_metrics = run_aequitas(**args)
                print("AEQUITAS METRICS", aequitas_metrics)

            elif method == 'adf' and model_type == 'mlp':
                print(f"Running ADF trial {trial.number} for {model_type}...")
                adf_results, adf_metrics = run_adf(**args)
                print("ADF METRICS", adf_metrics)

            # Load the results
            results_df = pd.read_sql(f"SELECT * FROM {results_table}", conn)

            # Calculate the metrics
            metrics = analyze_results(results_df, df)

        except Exception as e:
            print(f"Error running {method} trial {trial.number} on {model_type}: {str(e)}")
            success = False

        execution_time = time.time() - start_time

        # Calculate objective value (weighted sum of the metrics)
        objective_value = metrics['num_exact_couple_matches'] + metrics['num_new_group_couples'] * 1.5

        # Register the trial results
        if success:
            tracker.register_optuna_trial(
                method=method,
                model_type=model_type,
                dataset_attr=dataset_attr,
                trial_number=trial.number,
                params=params,
                results_table=results_table,
                testdata_table=testdata_table,
                num_exact_matches=metrics['num_exact_couple_matches'],
                num_new_couples=metrics['num_new_group_couples'],
                objective_value=objective_value,
                execution_time=execution_time,
                used_meta_learning=used_meta_learning
            )

        # Register the experiment as well
        tracker.register_experiment(
            method=method,
            model_type=model_type,
            dataset_attr=dataset_attr,
            results_table=results_table,
            testdata_table=testdata_table,
            execution_time=execution_time,
            success=success,
            params=params,
            objective_value=objective_value,
            used_meta_learning=used_meta_learning
        )

        print(f"Trial {trial.number} completed. Metrics: {metrics}, Objective: {objective_value}")

        return objective_value

    # Run the optimization
    print(f"Starting meta-learning enhanced optimization for {method} on {model_type}")
    study.optimize(objective, n_trials=n_trials)

    # Get the best parameters
    best_params = {**base_params[method], **study.best_params}

    # After optimization, update the meta-learner with new knowledge
    tracker.update_meta_models()

    return best_params


def run_experiment_with_meta_learning(method: str, model_type: str, dataset_attr: dict,
                                      tracker: EnhancedExperimentTracker) -> None:
    """Run an experiment using meta-learning optimized parameters."""
    # Skip ADF for non-MLP models
    if method == 'adf' and model_type != 'mlp':
        print(f"Skipping {method} for {model_type} model (not applicable)")
        return

    # Get meta-optimized parameters
    best_params = tracker.get_meta_optimized_params(method, model_type, dataset_attr)
    used_meta_learning = best_params is not None

    # If no meta-optimized parameters are available, run optimization
    if not best_params:
        print(f"No meta-optimized parameters found for {method} on {model_type}. Running optimization...")
        best_params = run_meta_optimization(method, model_type, dataset_attr, tracker)
        used_meta_learning = True  # We're using the meta-optimization process

    # Check if experiment with these parameters already exists
    existing_experiment = tracker.experiment_exists(method, model_type, dataset_attr, best_params)

    if existing_experiment:
        results_table, testdata_table = existing_experiment
        print(f"Experiment for {method} on {model_type} with meta-optimized params already exists.")
        print(f"Results available in tables: {results_table} and {testdata_table}")
        return

    # Generate new analysis ID
    timestamp = int(time.time())
    analysis_id_base = (
        f"{method}_{model_type}_attr{dataset_attr['nb_attributes']}"
        f"_prot{dataset_attr['prop_protected_attr']}_cat{dataset_attr['nb_categories_outcome']}"
        f"_meta_{timestamp}"
    )
    results_table = f"{analysis_id_base}_results"
    testdata_table = f"{analysis_id_base}_testdata"

    # Generate data
    print(f"Running new experiment with meta-optimized params: {method} on {model_type}")
    print(f"Parameters: {best_params}")
    data_obj = generate_data(min_group_size=1000, max_group_size=10000, nb_groups=50, **dataset_attr)
    df = data_obj.dataframe.copy()
    df['algorithm'] = method
    df['model'] = model_type

    # Save test data
    conn = sqlite3.connect(tracker.db_path)
    df.to_sql(name=testdata_table, con=conn, if_exists='replace')

    # Prepare method parameters
    args = best_params.copy()
    args['data'] = data_obj
    args['model_type'] = model_type
    args['db_path'] = tracker.db_path
    args['analysis_id'] = results_table

    success = True
    start_time = time.time()
    objective_value = None

    try:
        # Run the appropriate method with meta-optimized params
        if method == 'expga':
            print(f"Running ExpGA with meta-optimized params for {model_type}...")
            expga_results, expga_metrics = run_expga(**args)
            print("EXPGA METRICS", expga_metrics)

        elif method == 'sg':
            print(f"Running SG with meta-optimized params for {model_type}...")
            sg_results, sg_metrics = run_sg(**args)
            print("SG METRICS", sg_metrics)

        elif method == 'aequitas':
            print(f"Running Aequitas with meta-optimized params for {model_type}...")
            aequitas_results, aequitas_metrics = run_aequitas(**args)
            print("AEQUITAS METRICS", aequitas_metrics)

        elif method == 'adf' and model_type == 'mlp':
            print(f"Running ADF with meta-optimized params for {model_type}...")
            adf_results, adf_metrics = run_adf(**args)
            print("ADF METRICS", adf_metrics)

        # Load the results and calculate metrics
        results_df = pd.read_sql(f"SELECT * FROM {results_table}", conn)
        metrics = analyze_results(results_df, df)

        # Calculate objective value
        objective_value = metrics['num_exact_couple_matches'] + metrics['num_new_group_couples'] * 1.5

        print(f"Final metrics with meta-optimized params: {metrics}")
        print(f"Final objective value: {objective_value}")

    except Exception as e:
        print(f"Error running {method} with meta-optimized params on {model_type}: {str(e)}")
        success = False

    execution_time = time.time() - start_time

    # Register the experiment
    tracker.register_experiment(
        method=method,
        model_type=model_type,
        dataset_attr=dataset_attr,
        results_table=results_table,
        testdata_table=testdata_table,
        execution_time=execution_time,
        success=success,
        params=best_params,
        objective_value=objective_value,
        used_meta_learning=used_meta_learning
    )

    print(f"Experiment with meta-optimized params completed in {execution_time:.2f} seconds. Success: {success}")


def run_meta_learning_experiments(n_trials_per_method=10):
    """Run experiments with meta-learning enhanced optimization."""
    tracker = EnhancedExperimentTracker()

    try:
        # Define the parameter space to explore
        nb_attributes = [5, 10]  # Can use [5, 10, 15, 20] for complete run
        prop_protected_attr = [0.1, 0.2]  # Can use [0.1, 0.2, 0.5] for complete run
        nb_categories_outcome = [2, 3]  # Can use [2, 3, 4, 6] for complete run
        methods = ['expga', 'sg', 'aequitas', 'adf']
        models = ['rf', 'mlp', 'dt']

        total_configs = len(methods) * len(models) * len(nb_attributes) * len(prop_protected_attr) * len(
            nb_categories_outcome)
        print(f"Planning to run {total_configs} experiment configurations with meta-learning")

        completed = 0

        for method in methods:
            for model in models:
                # Skip ADF for non-MLP models
                if method == 'adf' and model != 'mlp':
                    continue

                for attr in nb_attributes:
                    for protected_attr in prop_protected_attr:
                        for cat in nb_categories_outcome:
                            dataset_attr = {
                                'nb_attributes': attr,
                                'prop_protected_attr': protected_attr,
                                'nb_categories_outcome': cat
                            }

                            # Run experiment with meta-learning
                            run_experiment_with_meta_learning(method, model, dataset_attr, tracker)

                            completed += 1
                            print(f"Progress: {completed}/{total_configs} experiment configurations processed")

        # Final update of meta-models with all accumulated knowledge
        tracker.update_meta_models()

    finally:
        tracker.close()


if __name__ == "__main__":
    # Run with meta-learning optimization
    run_meta_learning_experiments(n_trials_per_method=10)  # Adjust based on time available