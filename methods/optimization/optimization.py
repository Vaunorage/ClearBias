import optuna
import time
import pandas as pd
import sqlite3
import json
import os
from typing import Dict, Any, Optional, Tuple

from data_generator.main import DiscriminationData
from methods.adf.main import run_adf
from methods.exp_ga.algo import run_expga
from methods.aequitas.algo import run_aequitas
from methods.sg.main import run_sg
from path import HERE

DB_PATH = HERE.joinpath("methods/optimization/optimizations.db")


def save_trial_to_db(
        conn: sqlite3.Connection,
        trial_number: int,
        method: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        runtime: float,
        study_name: str,
        extra_data: Optional[Dict[str, Any]] = None,  # Parameter for extra data
        generation_arguments: Optional[Dict[str, Any]] = None  # New parameter for generation arguments
) -> None:
    c = conn.cursor()

    # Convert dictionaries to JSON strings
    params_json = json.dumps(params)
    metrics_json = json.dumps(metrics)
    extra_data_json = json.dumps(extra_data or {})  # Default to empty dict if None
    generation_args_json = json.dumps(generation_arguments or {})  # Convert generation arguments to JSON

    # Get current timestamp
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    # Check if we need to create the table with the required columns
    c.execute("PRAGMA table_info(optimization_trials)")
    columns = [info[1] for info in c.fetchall()]

    # If the extra_data column doesn't exist, add it
    if 'extra_data' not in columns:
        c.execute('ALTER TABLE optimization_trials ADD COLUMN extra_data TEXT')
        conn.commit()

    # If the generation_arguments column doesn't exist, add it
    if 'generation_arguments' not in columns:
        c.execute('ALTER TABLE optimization_trials ADD COLUMN generation_arguments TEXT')
        conn.commit()

    # Insert trial data including generation arguments
    c.execute('''
    INSERT INTO optimization_trials 
    (timestamp, study_name, trial_number, method, parameters, metrics, runtime, extra_data, generation_arguments)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, study_name, trial_number, method, params_json, metrics_json, runtime, extra_data_json,
          generation_args_json))

    conn.commit()


def setup_sqlite_database(db_path: str) -> sqlite3.Connection:
    """
    Set up the SQLite database for storing optimization trials.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        SQLite database connection
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    # Create connection
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create tables if they don't exist with the generation_arguments column
    c.execute('''
    CREATE TABLE IF NOT EXISTS optimization_trials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        study_name TEXT,
        trial_number INTEGER,
        method TEXT,
        parameters TEXT,
        metrics TEXT,
        runtime REAL,
        extra_data TEXT,
        generation_arguments TEXT
    )
    ''')

    conn.commit()
    return conn


def optimize_fairness_testing(
        data: DiscriminationData,
        method: str = 'expga',
        total_timeout: float = 3600,
        n_trials: int = 20,
        fixed_params: Optional[Dict[str, Any]] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        max_runtime_per_trial: Optional[float] = None,
        max_tsn: Optional[int] = None,
        verbose: bool = True,
        use_gpu: bool = False,
        random_seed: int = None,
        sqlite_path: Optional[str] = DB_PATH,
        extra_trial_data: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """
    Optimize fairness testing parameters using Optuna.

    Args:
        data: DiscriminationData object containing the dataset
        method: Fairness testing method ('expga', 'sg', 'aequitas', 'adf')
        total_timeout: Total time budget for optimization in seconds
        n_trials: Number of trials to run
        fixed_params: Fixed parameters that shouldn't be optimized
        study_name: Name of the study
        storage: Storage URL for Optuna
        sampler: Optuna sampler
        pruner: Optuna pruner
        max_runtime_per_trial: Maximum runtime for each trial
        max_tsn: Maximum TSN value
        verbose: Whether to print progress
        use_gpu: Whether to use GPU (only for ExpGA)
        random_seed: Random seed for reproducibility
        sqlite_path: Path to SQLite database for storing results
        extra_trial_data: Additional data to store with trials
        use_cache: Whether to use caching

    Returns:
        Tuple of (best_params, results_df, best_metrics)
    """
    # Set default values for optional parameters
    if fixed_params is None:
        fixed_params = {}

    if extra_trial_data is None:
        extra_trial_data = {}

    if study_name is None:
        study_name = f"{method}_optimization"

    if max_runtime_per_trial is None:
        max_runtime_per_trial = total_timeout // 5

    # Validate method parameter
    supported_methods = ['expga', 'sg', 'aequitas', 'adf']
    if method not in supported_methods:
        raise ValueError(f"Method must be one of {supported_methods}, got {method}")

    # Set up SQLite database connection if path is provided
    conn = None
    if sqlite_path:
        conn = setup_sqlite_database(sqlite_path)

    generation_arguments = getattr(data, 'generation_arguments', None)

    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        trial_start_time = time.time()

        # Define method-specific parameters
        if method == 'expga':
            params = {
                "threshold_rank": trial.suggest_float("threshold_rank", 0.1, 0.9),
                "max_global": trial.suggest_int("max_global", 1000, 50000),
                "max_local": trial.suggest_int("max_local", 10, 500),
                "model_type": trial.suggest_categorical("model_type", ["rf", "dt", "svm", "lr", "mlp"]),
                "cross_rate": trial.suggest_float("cross_rate", 0.5, 0.95),
                "mutation": trial.suggest_float("mutation", 0.01, 0.5),
                "random_seed": trial.suggest_int("random_seed", 10, 200),
                "one_attr_at_a_time": trial.suggest_categorical("one_attr_at_a_time", [True, False])
            }
        elif method == 'sg':
            params = {
                "model_type": trial.suggest_categorical("model_type", ["rf", "dt", "svm", "lr", "mlp"]),
                "cluster_num": trial.suggest_int("cluster_num", 10, 100),
                "one_attr_at_a_time": trial.suggest_categorical("one_attr_at_a_time", [True, False]),
                "random_seed": trial.suggest_int("random_seed", 10, 100)
            }
        elif method == 'aequitas':
            params = {
                "model_type": trial.suggest_categorical("model_type", ["rf", "dt", "svm", "lr", "mlp"]),
                "max_global": trial.suggest_int("max_global", 100, 1000),
                "max_local": trial.suggest_int("max_local", 1000, 10000),
                "step_size": trial.suggest_float("step_size", 0.1, 2.0),
                "init_prob": trial.suggest_float("init_prob", 0.1, 0.9),
                "param_probability_change_size": trial.suggest_float("param_probability_change_size", 0.001, 0.01),
                "direction_probability_change_size": trial.suggest_float("direction_probability_change_size", 0.001,
                                                                         0.01),
                "one_attr_at_a_time": trial.suggest_categorical("one_attr_at_a_time", [True, False]),
                "random_seed": trial.suggest_int("random_seed", 10, 100)
            }
        elif method == 'adf':
            params = {
                "max_global": trial.suggest_int("max_global", 100, 5000),
                "max_local": trial.suggest_int("max_local", 100, 5000),
                "cluster_num": trial.suggest_int("cluster_num", 10, 100),
                "random_seed": trial.suggest_int("random_seed", 10, 100),
                "step_size": trial.suggest_float("step_size", 0.05, 1.0),
                "one_attr_at_a_time": trial.suggest_categorical("one_attr_at_a_time", [True, False])
            }

        # Update params with fixed parameters
        for key, value in fixed_params.items():
            if key == "random_seed" and value is None:
                continue
            params[key] = value

        # Calculate time allocation
        elapsed_time = time.time() - trial_start_time
        remaining_time = total_timeout - elapsed_time

        if remaining_time <= 0:
            raise optuna.exceptions.TrialPruned("Total timeout exceeded")

        run_timeout = min(max_runtime_per_trial, remaining_time)

        # Add runtime parameters
        params["max_runtime_seconds"] = run_timeout
        params["max_tsn"] = max_tsn
        if random_seed:
            params["random_seed"] = random_seed
        params["use_cache"] = use_cache

        if method == 'expga':
            params["use_gpu"] = use_gpu

        if verbose:
            print(f"Trial {trial.number}: Running {method} with parameters: {params}")
            print(f"Time allocation: {run_timeout:.2f} seconds")

        # Run the method
        method_start_time = time.time()
        # try:
        if method == 'expga':
            results_df, metrics = run_expga(data, **params)
        elif method == 'sg':
            results_df, metrics = run_sg(data, **params)
        elif method == 'aequitas':
            results_df, metrics = run_aequitas(data, **params)
        elif method == 'adf':
            results_df, metrics = run_adf(data, **params)
        # except Exception as e:
        #     if verbose:
        #         print(f"Trial {trial.number} failed with error: {str(e)}")
        #     raise optuna.exceptions.TrialPruned()

        method_runtime = time.time() - method_start_time

        # Extract metrics
        dsn = metrics.get("DSN", 0)
        tsn = metrics.get("TSN", 0)
        sur = metrics.get("SUR", 0)

        # Store trial results in SQLite database if available
        if conn:
            save_trial_to_db(
                conn=conn,
                trial_number=trial.number,
                method=method,
                params=params,
                metrics=metrics,
                runtime=method_runtime,
                study_name=study_name,
                extra_data=extra_trial_data,
                generation_arguments=generation_arguments
            )

        if verbose:
            print(f"Trial {trial.number} results: DSN={dsn}, TSN={tsn}, SUR={sur:.4f}")

        return sur

    try:
        # Set up storage if needed
        if storage is None and sqlite_path:
            storage = f"sqlite:///{sqlite_path}"

        # Create Optuna study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler or optuna.samplers.TPESampler(),
            pruner=pruner or optuna.pruners.MedianPruner(),
            direction="maximize",
            load_if_exists=True
        )

        # Run optimization
        start_time = time.time()
        try:
            study.optimize(
                objective,
                n_trials=len(study.trials) + n_trials,
                timeout=total_timeout,
                catch=(Exception,)
            )
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")

        total_time = time.time() - start_time

        if verbose:
            print(f"\nOptimization completed in {total_time:.2f} seconds")
            print(f"Best trial: {study.best_trial.number}")
            print(f"Best value (SUR): {study.best_value:.4f}")
            print(f"Best parameters: {study.best_params}")

        # Get the best parameters
        best_params = study.best_params.copy()

        # Add fixed parameters to best_params
        for key, value in fixed_params.items():
            best_params[key] = value

        # Calculate remaining time for final run
        remaining_time = max(0, total_timeout - total_time)

        # Run final evaluation with best parameters if time remains
        if remaining_time > 0:
            if verbose:
                print(f"\nRunning final evaluation with best parameters and {remaining_time:.2f} seconds")

            # Add required parameters for final run
            best_params["max_runtime_seconds"] = remaining_time
            best_params["max_tsn"] = max_tsn

            if method == 'expga':
                best_params["use_gpu"] = use_gpu

            # Run the selected method with best parameters
            final_start_time = time.time()
            try:
                if method == 'expga':
                    results_df, best_metrics = run_expga(data, **best_params)
                elif method == 'sg':
                    results_df, best_metrics = run_sg(data, **best_params)
                elif method == 'aequitas':
                    results_df, best_metrics = run_aequitas(data, **best_params)
                elif method == 'adf':
                    results_df, best_metrics = run_adf(data, **best_params)
            except Exception as e:
                if verbose:
                    print(f"Final run failed with error: {str(e)}")
                results_df = pd.DataFrame()
                best_metrics = {"SUR": study.best_value}

            final_runtime = time.time() - final_start_time

            # Save final run to database
            if conn:
                save_trial_to_db(
                    conn=conn,
                    trial_number=-1,  # Use -1 to indicate final run
                    method=method,
                    params=best_params,
                    metrics=best_metrics,
                    runtime=final_runtime,
                    study_name=f"{study_name}_final",
                    extra_data=extra_trial_data,
                    generation_arguments=generation_arguments
                )

            if verbose:
                print(f"Final results: DSN={best_metrics.get('DSN', 0)}, "
                      f"TSN={best_metrics.get('TSN', 0)}, SUR={best_metrics.get('SUR', 0):.4f}")
        else:
            results_df = pd.DataFrame()
            best_metrics = {"SUR": study.best_value}

        # Create trials dataframe
        trials_df = study.trials_dataframe()

        return best_params, results_df, best_metrics

    finally:
        # Ensure the database connection is closed
        if conn:
            conn.close()


if __name__ == "__main__":
    from data_generator.main import get_real_data
    for dataset in ['credit', 'bank']:
        for algorithm in ['sg', 'expga', 'adf', 'aequitas']:
            # Get data
            data_obj, schema = get_real_data(dataset, use_cache=True)

            # Fixed parameters that we don't want to optimize
            fixed_params = {
                "db_path": None,
                "analysis_id": None,
            }

            # Additional information to save with each trial
            extra_trial_data = {'dataset': dataset}

            # Run optimization for ExpGA for 10 minutes (600 seconds)
            best_params, results_df, best_metrics = optimize_fairness_testing(
                study_name=f"{dataset}_{algorithm}",
                data=data_obj,
                method=algorithm,  # Choose method: 'expga', 'sg', 'aequitas', 'adf'
                total_timeout=2000,
                n_trials=5,
                fixed_params=fixed_params,
                max_tsn=20000,
                verbose=True,
                random_seed=42,
                extra_trial_data=extra_trial_data  # Pass the extra data dictionary
            )

            print("\nBest parameters found:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")

            print(f"\nBest SUR achieved: {best_metrics['SUR']:.4f}")
