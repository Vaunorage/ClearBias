import random
import time
import optuna
import pandas as pd
import sqlite3
import json
import os
from typing import Dict, Any, Optional, Tuple, Callable

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
        extra_data: Optional[Dict[str, Any]] = None,
        generation_arguments: Optional[Dict[str, Any]] = None
) -> None:
    # [Existing implementation unchanged]
    c = conn.cursor()

    # Convert dictionaries to JSON strings
    params_json = json.dumps(params)
    metrics_json = json.dumps(metrics)
    extra_data_json = json.dumps(extra_data or {})
    generation_args_json = json.dumps(generation_arguments or {})

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
              (timestamp, study_name, trial_number, method, parameters, metrics, runtime, extra_data,
               generation_arguments)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
              ''', (timestamp, study_name, trial_number, method, params_json, metrics_json, runtime, extra_data_json,
                    generation_args_json))

    conn.commit()


def setup_sqlite_database(db_path: str) -> sqlite3.Connection:
    # [Existing implementation unchanged]
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    # Create connection
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create tables if they don't exist with the generation_arguments column
    c.execute('''
              CREATE TABLE IF NOT EXISTS optimization_trials
              (
                  id
                  INTEGER
                  PRIMARY
                  KEY
                  AUTOINCREMENT,
                  timestamp
                  TEXT,
                  study_name
                  TEXT,
                  trial_number
                  INTEGER,
                  method
                  TEXT,
                  parameters
                  TEXT,
                  metrics
                  TEXT,
                  runtime
                  REAL,
                  extra_data
                  TEXT,
                  generation_arguments
                  TEXT
              )
              ''')

    conn.commit()
    return conn


def early_stopping_callback(study: optuna.Study, trial: optuna.Trial, n_trials_without_improvement: int = 10) -> None:
    """
    Callback to stop optimization if no improvement is seen after a certain number of trials.

    Args:
        study: Optuna study object
        trial: Current trial
        n_trials_without_improvement: Number of trials to check for improvement
    """
    if len(study.trials) >= n_trials_without_improvement:
        # Get completed trials among the last n_trials_without_improvement
        completed_trials = [t for t in study.trials[-n_trials_without_improvement:]
                            if t.state == optuna.trial.TrialState.COMPLETE]

        if not completed_trials:
            return

        # For maximization, check if best value hasn't improved
        best_in_window = max([t.value for t in completed_trials])
        if study.best_value <= best_in_window and study.best_trial.number < trial.number - n_trials_without_improvement:
            # No improvement in the last n trials
            raise optuna.exceptions.OptunaError("No improvement in the last trials, stopping.")


def get_parameter_space(method: str, trial: optuna.Trial, exploration_factor: float = 1.0) -> Dict[str, Any]:
    """
    Define parameter space for different methods with adjustable exploration bounds.

    Args:
        method: Fairness testing method ('expga', 'sg', 'aequitas', 'adf')
        trial: Optuna trial object
        exploration_factor: Factor to widen exploration bounds (1.0 = normal, >1.0 = wider)

    Returns:
        Dictionary of parameters
    """
    # Base parameter ranges
    if method == 'expga':
        base_min_threshold = 0.1
        base_max_threshold = 0.9
        base_min_global = 1000
        base_max_global = 50000
        base_min_local = 10
        base_max_local = 500
        base_min_cross = 0.5
        base_max_cross = 0.95
        base_min_mutation = 0.01
        base_max_mutation = 0.5

        # Apply exploration factor to widen ranges
        min_threshold = max(0.01, base_min_threshold - (base_min_threshold * (exploration_factor - 1) * 0.5))
        max_threshold = min(0.99, base_max_threshold + ((1 - base_max_threshold) * (exploration_factor - 1) * 0.5))
        min_global = max(100, int(base_min_global / exploration_factor))
        max_global = int(base_max_global * exploration_factor)
        min_local = max(1, int(base_min_local / exploration_factor))
        max_local = int(base_max_local * exploration_factor)
        min_cross = max(0.1, base_min_cross - ((base_min_cross - 0.1) * (exploration_factor - 1)))
        max_cross = min(0.99, base_max_cross + ((0.99 - base_max_cross) * (exploration_factor - 1)))
        min_mutation = max(0.001, base_min_mutation / exploration_factor)
        max_mutation = min(0.99, base_max_mutation * exploration_factor)

        params = {
            "threshold_rank": trial.suggest_float("threshold_rank", min_threshold, max_threshold),
            "max_global": trial.suggest_int("max_global", min_global, max_global),
            "max_local": trial.suggest_int("max_local", min_local, max_local),
            "model_type": trial.suggest_categorical("model_type", ["rf", "dt", "svm", "lr", "mlp"]),
            "cross_rate": trial.suggest_float("cross_rate", min_cross, max_cross),
            "mutation": trial.suggest_float("mutation", min_mutation, max_mutation),
            "random_seed": trial.suggest_int("random_seed", 10, 200),
            "one_attr_at_a_time": trial.suggest_categorical("one_attr_at_a_time", [True, False])
        }
    elif method == 'sg':
        # Apply exploration factor to cluster_num
        min_clusters = max(2, int(10 / exploration_factor))
        max_clusters = int(100 * exploration_factor)

        params = {
            "model_type": trial.suggest_categorical("model_type", ["rf", "dt", "svm", "lr", "mlp"]),
            "cluster_num": trial.suggest_int("cluster_num", min_clusters, max_clusters),
            "one_attr_at_a_time": trial.suggest_categorical("one_attr_at_a_time", [True, False]),
            "random_seed": trial.suggest_int("random_seed", 10, int(100 * exploration_factor))
        }
    elif method == 'aequitas':
        # Apply exploration factor
        min_global = max(10, int(100 / exploration_factor))
        max_global = int(1000 * exploration_factor)
        min_local = max(100, int(1000 / exploration_factor))
        max_local = int(10000 * exploration_factor)
        min_step = max(0.01, 0.1 / exploration_factor)
        max_step = min(5.0, 2.0 * exploration_factor)

        params = {
            "model_type": trial.suggest_categorical("model_type", ["rf", "dt", "svm", "lr", "mlp"]),
            "max_global": trial.suggest_int("max_global", min_global, max_global),
            "max_local": trial.suggest_int("max_local", min_local, max_local),
            "step_size": trial.suggest_float("step_size", min_step, max_step),
            "init_prob": trial.suggest_float("init_prob", 0.05, 0.95),
            "param_probability_change_size": trial.suggest_float("param_probability_change_size", 0.0001, 0.05),
            "direction_probability_change_size": trial.suggest_float("direction_probability_change_size", 0.0001, 0.05),
            "one_attr_at_a_time": trial.suggest_categorical("one_attr_at_a_time", [True, False]),
            "random_seed": trial.suggest_int("random_seed", 10, int(100 * exploration_factor))
        }
    elif method == 'adf':
        # Apply exploration factor
        min_global = max(10, int(100 / exploration_factor))
        max_global = int(5000 * exploration_factor)
        min_local = max(10, int(100 / exploration_factor))
        max_local = int(5000 * exploration_factor)
        min_clusters = max(2, int(10 / exploration_factor))
        max_clusters = int(100 * exploration_factor)

        params = {
            "max_global": trial.suggest_int("max_global", min_global, max_global),
            "max_local": trial.suggest_int("max_local", min_local, max_local),
            "cluster_num": trial.suggest_int("cluster_num", min_clusters, max_clusters),
            "random_seed": trial.suggest_int("random_seed", 10, int(100 * exploration_factor)),
            "step_size": trial.suggest_float("step_size", 0.01, min(2.0, 1.0 * exploration_factor)),
            "one_attr_at_a_time": trial.suggest_categorical("one_attr_at_a_time", [True, False])
        }
    else:
        raise ValueError(f"Unknown method: {method}")

    return params


def optimize_fairness_testing(
        data: DiscriminationData,
        method: str = 'expga',
        total_timeout: float = 3600,
        n_trials: int = 50,  # Increased from 20 to 50 for better exploration
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
        use_cache: bool = True,
        exploration_factor: float = 1.5,  # New parameter to control exploration
        use_multi_stage_optimization: bool = True,  # Whether to use multi-stage optimization
        enable_early_stopping: bool = True,  # Whether to enable early stopping
        n_startup_trials: int = 15,  # Number of random startup trials for TPE
        n_trials_without_improvement: int = 10,  # Trials before early stopping
        parallel_optimization: bool = False  # Whether to use parallel optimization
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """
    Optimize fairness testing parameters using Optuna with enhanced exploration capabilities.

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
        exploration_factor: Factor to control exploration bounds (>1 = wider bounds)
        use_multi_stage_optimization: Whether to use multi-stage optimization
        enable_early_stopping: Whether to enable early stopping
        n_startup_trials: Number of random trials before using TPE
        n_trials_without_improvement: Number of trials without improvement before early stopping
        parallel_optimization: Whether to use parallel optimization

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

    def objective_factory(current_exploration_factor: float) -> Callable[[optuna.Trial], float]:
        """Create an objective function with specific exploration factor."""

        def objective(trial: optuna.Trial) -> float:
            """Objective function for Optuna optimization."""
            trial_start_time = time.time()

            # Define method-specific parameters with adjusted exploration bounds
            params = get_parameter_space(method, trial, current_exploration_factor)

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
            try:
                if method == 'expga':
                    results_df, metrics = run_expga(data, **params)
                elif method == 'sg':
                    results_df, metrics = run_sg(data, **params)
                elif method == 'aequitas':
                    results_df, metrics = run_aequitas(data, **params)
                elif method == 'adf':
                    results_df, metrics = run_adf(data, **params)
            except Exception as e:
                if verbose:
                    print(f"Trial {trial.number} failed with error: {str(e)}")
                raise optuna.exceptions.TrialPruned()

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

        return objective

    try:
        # Set up storage if needed
        if storage is None and sqlite_path:
            storage = f"sqlite:///{sqlite_path}"

        best_params = {}
        best_value = float('-inf')
        combined_trials = []
        results_df = pd.DataFrame()
        best_metrics = {"SUR": 0.0}

        if use_multi_stage_optimization:
            # Multi-stage optimization approach
            stages = [
                # First stage: Wide exploration with random sampling
                {
                    "name": f"{study_name}_stage1_exploration",
                    "n_trials": n_trials // 3,
                    "sampler": optuna.samplers.RandomSampler(),
                    "pruner": optuna.pruners.NopPruner(),  # No pruning in exploration phase
                    "exploration_factor": exploration_factor * 1.5,  # Extra wide bounds
                },
                # Second stage: Medium exploration with TPE
                {
                    "name": f"{study_name}_stage2_medium",
                    "n_trials": n_trials // 3,
                    "sampler": optuna.samplers.TPESampler(
                        n_startup_trials=n_startup_trials // 2,
                        multivariate=True,
                        constant_liar=True,
                    ),
                    "pruner": optuna.pruners.PercentilePruner(
                        percentile=80.0,
                        n_startup_trials=max(3, n_trials // 10),
                        n_warmup_steps=2
                    ),
                    "exploration_factor": exploration_factor,  # Standard exploration factor
                },
                # Third stage: Focused exploitation with CMA-ES or TPE
                {
                    "name": f"{study_name}_stage3_focused",
                    "n_trials": n_trials // 3,
                    "sampler": optuna.samplers.TPESampler(
                        n_startup_trials=0,  # Use previous trials
                        multivariate=True,
                        constant_liar=True,
                    ),
                    "pruner": optuna.pruners.MedianPruner(
                        n_startup_trials=0,  # Use previous trials
                        n_warmup_steps=0
                    ),
                    "exploration_factor": max(1.0, exploration_factor * 0.7),  # More focused bounds
                }
            ]

            start_time = time.time()
            all_trials = []

            # Run each stage sequentially
            for i, stage in enumerate(stages):
                # Skip if we've run out of time
                elapsed_time = time.time() - start_time
                remaining_time = total_timeout - elapsed_time
                if remaining_time <= 0:
                    if verbose:
                        print(f"Skipping stage {i + 1} due to timeout")
                    continue

                if verbose:
                    print(f"\n=== Starting optimization stage {i + 1}: {stage['name']} ===")
                    print(f"Exploration factor: {stage['exploration_factor']}, Trials: {stage['n_trials']}")

                # Create the study for this stage
                stage_study = optuna.create_study(
                    study_name=stage["name"],
                    storage=storage,
                    sampler=stage["sampler"],
                    pruner=stage["pruner"],
                    direction="maximize",
                    load_if_exists=True
                )

                # Transfer trials from previous stages if needed
                if i > 0 and all_trials:
                    for trial in all_trials:
                        if trial.state == optuna.trial.TrialState.COMPLETE:
                            try:
                                stage_study.add_trial(trial)
                            except Exception:
                                pass  # Skip if trial can't be added

                # Set up callbacks
                callbacks = []
                if enable_early_stopping and i > 0:  # Don't use early stopping in exploration phase
                    es_callback = lambda study, trial: early_stopping_callback(
                        study, trial, n_trials_without_improvement=n_trials_without_improvement
                    )
                    callbacks.append(es_callback)

                # Create objective with current exploration factor
                objective = objective_factory(stage["exploration_factor"])

                # Calculate timeout for this stage
                stage_timeout = min(remaining_time, total_timeout // len(stages))

                # Run optimization for this stage
                try:
                    stage_study.optimize(
                        objective,
                        n_trials=stage["n_trials"],
                        timeout=stage_timeout,
                        callbacks=callbacks,
                        catch=(Exception,),
                        # Use parallel optimization if enabled
                        n_jobs=-1 if parallel_optimization else 1
                    )
                except (KeyboardInterrupt, optuna.exceptions.OptunaError) as e:
                    if verbose:
                        print(f"Stage {i + 1} optimization interrupted: {str(e)}")

                # Save trials from this stage
                all_trials.extend(stage_study.trials)

                # Update best parameters if this stage found better results
                if stage_study.best_value > best_value:
                    best_value = stage_study.best_value
                    best_params = stage_study.best_params.copy()
                    if verbose:
                        print(f"New best value from stage {i + 1}: {best_value:.4f}")

            # Merge all trials for final analysis
            combined_trials = all_trials

        else:
            # Single-stage optimization with enhanced exploration
            if verbose:
                print("\n=== Starting single-stage optimization ===")

            # Setup default sampler with enhanced exploration
            if sampler is None:
                sampler = optuna.samplers.TPESampler(
                    n_startup_trials=n_startup_trials,
                    n_ei_candidates=24,  # Consider more candidates
                    multivariate=True,  # Use multivariate TPE
                    constant_liar=True  # Helps with parallel optimization
                )

            # Setup default pruner with less aggressive pruning
            if pruner is None:
                pruner = optuna.pruners.HyperbandPruner(
                    min_resource=1,
                    max_resource=10,
                    reduction_factor=3
                )

            # Create study
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler,
                pruner=pruner,
                direction="maximize",
                load_if_exists=True
            )

            # Set up callbacks
            callbacks = []
            if enable_early_stopping:
                es_callback = lambda study, trial: early_stopping_callback(
                    study, trial, n_trials_without_improvement=n_trials_without_improvement
                )
                callbacks.append(es_callback)

            # Create objective function with specified exploration factor
            objective = objective_factory(exploration_factor)

            # Run optimization
            start_time = time.time()
            try:
                study.optimize(
                    objective,
                    n_trials=n_trials,
                    timeout=total_timeout,
                    callbacks=callbacks,
                    catch=(Exception,),
                    # Use parallel optimization if enabled
                    n_jobs=-1 if parallel_optimization else 1
                )
            except (KeyboardInterrupt, optuna.exceptions.OptunaError) as e:
                if verbose:
                    print(f"Optimization interrupted: {str(e)}")

            best_params = study.best_params.copy()
            best_value = study.best_value
            combined_trials = study.trials

        # Add fixed parameters to best_params
        for key, value in fixed_params.items():
            best_params[key] = value

        # Calculate total optimization time
        total_time = time.time() - start_time

        if verbose:
            print(f"\nOptimization completed in {total_time:.2f} seconds")
            print(f"Best value (SUR): {best_value:.4f}")
            print(f"Best parameters: {best_params}")

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
                best_metrics = {"SUR": best_value}

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
            best_metrics = {"SUR": best_value}

        # Create trials dataframe - this might need adjustment for multi-stage approach
        trials_df = pd.DataFrame([
            {
                "number": t.number,
                "value": t.value,
                "state": t.state,
                "params": t.params,
                "datetime_start": t.datetime_start,
                "datetime_complete": t.datetime_complete
            }
            for t in combined_trials if hasattr(t, 'value') and t.value is not None
        ])

        return best_params, results_df, best_metrics

    finally:
        # Ensure the database connection is closed
        if conn:
            conn.close()


if __name__ == "__main__":
    from data_generator.main import get_real_data, generate_from_real_data, generate_data
    import numpy as np

    # Configure the optimization settings directly
    datasets = ['adult', 'credit', 'bank']
    algorithms = ['sg', 'expga', 'adf', 'aequitas']
    timeout = 3600  # 1 hour timeout
    n_trials = 30  # Number of trials
    exploration_factor = 1.5  # Wider parameter bounds
    use_multi_stage = True
    use_parallel = False
    visualize_results = False
    compare_strategies = False
    random_seed = 42

    # Create dataset generation functions
    datasets_functions = [
        ('real', get_real_data),
        ('synthetic', generate_from_real_data)
    ] * 10

    pure_synth_datasets = []
    for i in range(100):
        pure_synth_datasets.append(
            (f'pure-synthetic-balanced_{i}', lambda dataset, use_cache=True: (
                generate_data(
                    nb_groups=random.uniform(10, 500),
                    nb_attributes=random.uniform(5, 30),
                    prop_protected_attr=random.uniform(0.1, 0.5),
                    min_similarity=0.3,
                    max_similarity=0.7,
                    use_cache=use_cache
                ), None)
             ))

    datasets_functions.extend(pure_synth_datasets)

    for dataset in datasets:
        for algorithm in algorithms:
            for gen_data in datasets_functions:
                if dataset != 'adult' and 'pure-synthetic' in gen_data[0]:
                    continue

                print(f"\nStarting experiment: {gen_data[0]}_{dataset}_{algorithm}")

                try:
                    # Get data
                    data_obj, schema = gen_data[1](dataset, use_cache=True)

                    # Fixed parameters that we don't want to optimize
                    fixed_params = {
                        "db_path": None,
                        "analysis_id": None,
                    }

                    # Additional information to save with each trial
                    extra_trial_data = {
                        'dataset': dataset,
                        'data_source': gen_data[0],
                        'data_groups': data_obj.nb_groups,
                        'data_attributes': len(data_obj.attributes) if hasattr(data_obj, 'attributes') else 0
                    }

                    unique_timestamp = int(time.time())
                    study_name = f"{gen_data[0]}_{dataset}_{algorithm}_{unique_timestamp}"

                    # Create a fresh storage for each study to prevent parameter conflicts
                    storage = None  # This forces Optuna to create a new in-memory storage

                    # Configure enhanced exploration settings
                    exploration_config = {
                        'exploration_factor': exploration_factor,  # Wider parameter search
                        'use_multi_stage_optimization': use_multi_stage,  # Use multi-stage approach
                        'enable_early_stopping': True,  # Enable early stopping
                        'n_startup_trials': max(5, n_trials // 3),  # More initial random trials
                        'n_trials_without_improvement': n_trials // 3,  # More patient early stopping
                        'parallel_optimization': use_parallel  # Use parallel optimization if possible
                    }

                    # Run optimization with enhanced exploration settings
                    best_params, results_df, best_metrics = optimize_fairness_testing(
                        study_name=study_name,
                        data=data_obj,
                        method=algorithm,
                        total_timeout=timeout,  # Configured timeout
                        n_trials=n_trials,  # Configured number of trials
                        fixed_params=fixed_params,
                        max_tsn=20000,
                        verbose=True,
                        random_seed=random.randint(10, 1000),
                        storage=storage,  # Use fresh storage for each study
                        extra_trial_data=extra_trial_data,
                        exploration_factor=exploration_config['exploration_factor'],
                        use_multi_stage_optimization=exploration_config['use_multi_stage_optimization'],
                        enable_early_stopping=exploration_config['enable_early_stopping'],
                        n_startup_trials=exploration_config['n_startup_trials'],
                        n_trials_without_improvement=exploration_config['n_trials_without_improvement'],
                        parallel_optimization=exploration_config['parallel_optimization']
                    )

                    print(f"\nExperiment completed: {gen_data[0]}_{dataset}_{algorithm}")
                    print("Best parameters found:")
                    for param, value in best_params.items():
                        print(f"  {param}: {value}")
                    print(f"Best SUR achieved: {best_metrics['SUR']:.4f}")

                    # Print additional optimization details
                    print(f"Number of trials executed: {len(results_df) if not results_df.empty else 'N/A'}")
                    if "TSN" in best_metrics and "DSN" in best_metrics:
                        print(f"Discrimination instances found: {best_metrics['DSN']}/{best_metrics['TSN']} tests")

                except Exception as e:
                    print(f"Error in {gen_data[0]}_{dataset}_{algorithm}: {str(e)}")
                    import traceback

                    traceback.print_exc()
                    continue

    print("\nAll optimization experiments completed!")
