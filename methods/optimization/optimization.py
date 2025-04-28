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


def get_best_hyperparameters(
        db_path: str,
        method: str = None,
        dataset: str = None,
        data_source: str = None,
        study_name: str = None,
        top_n: int = 1,
        metric: str = "SUR"
) -> list:
    """
    Get the best hyperparameters from the optimization database.

    Args:
        db_path: Path to the SQLite database
        method: Filter by method name (e.g., 'expga', 'sg', 'aequitas', 'adf')
        dataset: Filter by dataset name (e.g., 'adult', 'credit', 'bank')
        data_source: Filter by data source (e.g., 'real', 'synthetic', 'pure-synthetic-balanced')
        study_name: Filter by study name
        top_n: Number of top results to return
        metric: Metric to optimize for (default: "SUR")

    Returns:
        List of dictionaries containing the best hyperparameters and their metrics
    """
    import json
    import sqlite3

    # Connect to the database
    conn = sqlite3.connect(db_path)

    # Construct the query based on provided filters
    query = """
            SELECT trial_number, \
                   study_name, \
                   method, \
                   parameters, \
                   metrics, \
                   runtime, \
                   extra_data
            FROM optimization_trials
            WHERE 1 = 1 \
            """

    params = []

    if method:
        query += " AND method = ?"
        params.append(method)

    if study_name:
        query += " AND study_name = ?"
        params.append(study_name)

    # Execute the query
    cursor = conn.cursor()
    cursor.execute(query, params)

    # Process the results
    results = []
    for row in cursor.fetchall():
        trial_number, study_name, method, params_json, metrics_json, runtime, extra_data_json = row

        # Parse JSON data
        params_dict = json.loads(params_json)
        metrics_dict = json.loads(metrics_json)
        extra_data_dict = json.loads(extra_data_json) if extra_data_json else {}

        # Apply additional filters on extra_data
        if dataset and ('dataset' not in extra_data_dict or extra_data_dict['dataset'] != dataset):
            continue

        if data_source and ('data_source' not in extra_data_dict or extra_data_dict['data_source'] != data_source):
            continue

        # Get the metric value
        metric_value = metrics_dict.get(metric, 0)

        results.append({
            'trial_number': trial_number,
            'study_name': study_name,
            'method': method,
            'parameters': params_dict,
            'metrics': metrics_dict,
            'runtime': runtime,
            'extra_data': extra_data_dict,
            'metric_value': metric_value
        })

    # Close the connection
    conn.close()

    # Sort results by the specified metric (descending order)
    results.sort(key=lambda x: x['metric_value'], reverse=True)

    # Return top_n results
    return results[:top_n]


if __name__ == "__main__":
    from data_generator.main import get_real_data, generate_from_real_data, generate_data
    import numpy as np
    import time

    # Create dataset generation functions
    datasets_functions = [
        ('real', get_real_data),
    ]

    # Define validation test configurations based on analysis results
    validation_tests = {
        'adf': [
            {
                'name': 'low_max_local',
                'params': {
                    'max_global': 2500,  # Moderate value
                    'max_local': 200,  # Low value
                    'cluster_num': 55,  # Low-mid range (53-62)
                    'step_size': 0.6,  # Higher value
                    'one_attr_at_a_time': True,
                    'random_seed': 42
                }
            },
            {
                'name': 'parameter_interaction',
                'params': {
                    'max_global': 3000,  # Higher value
                    'max_local': 1000,  # Mid value
                    'cluster_num': 55,  # Low-mid range
                    'step_size': 0.4,  # Mid value
                    'one_attr_at_a_time': True,
                    'random_seed': 42
                }
            }
        ],
        'aequitas': [
            {
                'name': 'decision_tree_model',
                'params': {
                    'model_type': 'dt',
                    'max_global': 200,  # Low range (134-294)
                    'max_local': 8000,  # High value
                    'step_size': 0.75,  # Low range (0.7-0.8)
                    'init_prob': 0.65,  # High range (0.6-0.7)
                    'param_probability_change_size': 0.006,
                    'direction_probability_change_size': 0.008,
                    'one_attr_at_a_time': False,
                    'random_seed': 42
                }
            },
            {
                'name': 'model_comparison',
                'params': {
                    'model_type': 'rf',  # Comparison with random forest
                    'max_global': 200,  # Low range
                    'max_local': 8000,  # High value
                    'step_size': 0.75,  # Low value
                    'init_prob': 0.65,  # High value
                    'param_probability_change_size': 0.006,
                    'direction_probability_change_size': 0.008,
                    'one_attr_at_a_time': False,
                    'random_seed': 42
                }
            }
        ],
        'expga': [
            {
                'name': 'optimized_combination',
                'params': {
                    'threshold_rank': 0.38,  # Low value
                    'max_global': 20000,  # Mid value (16860-21851)
                    'max_local': 130,  # Low value (116-145)
                    'model_type': 'dt',  # Best performing
                    'cross_rate': 0.85,  # High value (0.8-0.9)
                    'mutation': 0.4,  # High value (0.4+)
                    'one_attr_at_a_time': False,
                    'random_seed': 42
                }
            },
            {
                'name': 'parameter_interaction',
                'params': {
                    'threshold_rank': 0.4,  # Low-mid value
                    'max_global': 15000,  # Lower value
                    'max_local': 170,  # Mid value (145-175)
                    'model_type': 'dt',  # Best performing
                    'cross_rate': 0.75,  # Mid value
                    'mutation': 0.3,  # Mid value
                    'one_attr_at_a_time': False,
                    'random_seed': 42
                }
            }
        ],
        'sg': [
            {
                'name': 'random_forest_model',
                'params': {
                    'model_type': 'rf',  # Best performing
                    'cluster_num': 20,  # Low value (12-32)
                    'one_attr_at_a_time': False,
                    'random_seed': 42
                }
            },
            {
                'name': 'model_comparison',
                'params': {
                    'model_type': 'dt',  # For comparison
                    'cluster_num': 20,  # Low value (12-32)
                    'one_attr_at_a_time': False,
                    'random_seed': 42
                }
            }
        ]
    }

    # Add dataset-specific tests
    dataset_specific_tests = {
        'adult': {
            'adf': {
                'name': 'adult_optimized',
                'params': {
                    'max_global': 662,
                    'max_local': 4782,
                    'cluster_num': 71,
                    'step_size': 0.0715,
                    'one_attr_at_a_time': True,
                    'random_seed': 42
                }
            },
            'aequitas': {
                'name': 'adult_optimized',
                'params': {
                    'model_type': 'lr',
                    'max_global': 455,
                    'max_local': 6901,
                    'step_size': 1.4701,
                    'init_prob': 0.2874,
                    'param_probability_change_size': 0.0059,
                    'direction_probability_change_size': 0.0080,
                    'one_attr_at_a_time': True,
                    'random_seed': 42
                }
            },
            'expga': {
                'name': 'adult_optimized',
                'params': {
                    'threshold_rank': 0.7227,
                    'max_global': 29418,
                    'max_local': 390,
                    'model_type': 'dt',
                    'cross_rate': 0.7247,
                    'mutation': 0.0498,
                    'one_attr_at_a_time': True,
                    'random_seed': 42
                }
            },
            'sg': {
                'name': 'adult_optimized',
                'params': {
                    'model_type': 'rf',
                    'cluster_num': 70,
                    'one_attr_at_a_time': False,
                    'random_seed': 39
                }
            }
        },
        'bank': {
            'adf': {
                'name': 'bank_optimized',
                'params': {
                    'max_global': 435,
                    'max_local': 164,
                    'cluster_num': 53,
                    'step_size': 0.1667,
                    'one_attr_at_a_time': True,
                    'random_seed': 42
                }
            },
            'aequitas': {
                'name': 'bank_optimized',
                'params': {
                    'model_type': 'rf',
                    'max_global': 968,
                    'max_local': 9209,
                    'step_size': 0.7444,
                    'init_prob': 0.7334,
                    'param_probability_change_size': 0.0099,
                    'direction_probability_change_size': 0.0073,
                    'one_attr_at_a_time': False,
                    'random_seed': 42
                }
            },
            'expga': {
                'name': 'bank_optimized',
                'params': {
                    'threshold_rank': 0.4503,
                    'max_global': 11870,
                    'max_local': 116,
                    'model_type': 'lr',
                    'cross_rate': 0.5881,
                    'mutation': 0.3665,
                    'one_attr_at_a_time': False,
                    'random_seed': 42
                }
            },
            'sg': {
                'name': 'bank_optimized',
                'params': {
                    'model_type': 'rf',
                    'cluster_num': 32,
                    'one_attr_at_a_time': False,
                    'random_seed': 75
                }
            }
        },
        'credit': {
            'adf': {
                'name': 'credit_optimized',
                'params': {
                    'max_global': 3424,
                    'max_local': 2331,
                    'cluster_num': 81,
                    'step_size': 0.6632,
                    'one_attr_at_a_time': False,
                    'random_seed': 42
                }
            },
            'aequitas': {
                'name': 'credit_optimized',
                'params': {
                    'model_type': 'dt',
                    'max_global': 134,
                    'max_local': 7909,
                    'step_size': 0.9233,
                    'init_prob': 0.5581,
                    'param_probability_change_size': 0.0057,
                    'direction_probability_change_size': 0.0086,
                    'one_attr_at_a_time': False,
                    'random_seed': 42
                }
            },
            'expga': {
                'name': 'credit_optimized',
                'params': {
                    'threshold_rank': 0.3825,
                    'max_global': 21851,
                    'max_local': 175,
                    'model_type': 'dt',
                    'cross_rate': 0.9407,
                    'mutation': 0.4252,
                    'one_attr_at_a_time': False,
                    'random_seed': 42
                }
            },
            'sg': {
                'name': 'credit_optimized',
                'params': {
                    'model_type': 'rf',
                    'cluster_num': 12,
                    'one_attr_at_a_time': False,
                    'random_seed': 44
                }
            }
        }
    }

    # Add cross-parameter interaction tests
    interaction_tests = {
        'adf': {
            'name': 'max_local_step_size_interaction',
            'params': [
                # Test different combinations of max_local and step_size
                {'max_local': 200, 'step_size': 0.2},
                {'max_local': 200, 'step_size': 0.6},
                {'max_local': 2000, 'step_size': 0.2},
                {'max_local': 2000, 'step_size': 0.6},
            ],
            'base_params': {
                'max_global': 662,
                'cluster_num': 55,
                'one_attr_at_a_time': True,
                'random_seed': 42
            }
        },
        'expga': {
            'name': 'threshold_mutation_interaction',
            'params': [
                # Test different combinations of threshold_rank and mutation
                {'threshold_rank': 0.3, 'mutation': 0.1},
                {'threshold_rank': 0.3, 'mutation': 0.4},
                {'threshold_rank': 0.7, 'mutation': 0.1},
                {'threshold_rank': 0.7, 'mutation': 0.4}
            ],
            'base_params': {
                'max_global': 20000,
                'max_local': 130,
                'model_type': 'dt',
                'cross_rate': 0.85,
                'one_attr_at_a_time': False,
                'random_seed': 42
            }
        }
    }

    # Run validation tests
    results = {}

    for dataset in ['adult', 'credit', 'bank']:
        results[dataset] = {}

        for algorithm in ['sg', 'expga', 'adf', 'aequitas']:
            algorithm_results = []

            # Get data
            data_obj, schema = get_real_data(dataset, use_cache=True)

            # Fixed parameters that we don't want to optimize
            fixed_params = {
                "db_path": None,
                "analysis_id": None,
                "max_tsn": 20000,
                "max_runtime_seconds": 1200,  # 20 minutes max per test
                "use_cache": True
            }

            if algorithm == 'expga':
                fixed_params["use_gpu"] = False

            # Run general validation tests
            print(f"\n{'=' * 50}")
            print(f"Running validation tests for {dataset} - {algorithm}")
            print(f"{'=' * 50}")

            for test in validation_tests[algorithm]:
                test_name = test['name']
                test_params = test['params'].copy()
                test_params.update(fixed_params)

                print(f"\nRunning test: {test_name}")
                print(f"Parameters: {test_params}")

                try:
                    start_time = time.time()

                    if algorithm == 'expga':
                        results_df, metrics = run_expga(data_obj, **test_params)
                    elif algorithm == 'sg':
                        results_df, metrics = run_sg(data_obj, **test_params)
                    elif algorithm == 'aequitas':
                        results_df, metrics = run_aequitas(data_obj, **test_params)
                    elif algorithm == 'adf':
                        results_df, metrics = run_adf(data_obj, **test_params)

                    runtime = time.time() - start_time

                    test_result = {
                        'test_name': test_name,
                        'parameters': test_params,
                        'metrics': metrics,
                        'runtime': runtime
                    }

                    algorithm_results.append(test_result)

                    print(f"Test completed in {runtime:.2f} seconds")
                    print(
                        f"Results: DSN={metrics.get('DSN', 0)}, TSN={metrics.get('TSN', 0)}, SUR={metrics.get('SUR', 0):.4f}")

                except Exception as e:
                    print(f"Error in test {test_name}: {str(e)}")

            # Run dataset-specific tests
            if dataset in dataset_specific_tests and algorithm in dataset_specific_tests[dataset]:
                test = dataset_specific_tests[dataset][algorithm]
                test_name = test['name']
                test_params = test['params'].copy()
                test_params.update(fixed_params)

                print(f"\nRunning dataset-specific test: {test_name}")
                print(f"Parameters: {test_params}")

                try:
                    start_time = time.time()

                    if algorithm == 'expga':
                        results_df, metrics = run_expga(data_obj, **test_params)
                    elif algorithm == 'sg':
                        results_df, metrics = run_sg(data_obj, **test_params)
                    elif algorithm == 'aequitas':
                        results_df, metrics = run_aequitas(data_obj, **test_params)
                    elif algorithm == 'adf':
                        results_df, metrics = run_adf(data_obj, **test_params)

                    runtime = time.time() - start_time

                    test_result = {
                        'test_name': test_name,
                        'parameters': test_params,
                        'metrics': metrics,
                        'runtime': runtime
                    }

                    algorithm_results.append(test_result)

                    print(f"Test completed in {runtime:.2f} seconds")
                    print(
                        f"Results: DSN={metrics.get('DSN', 0)}, TSN={metrics.get('TSN', 0)}, SUR={metrics.get('SUR', 0):.4f}")

                except Exception as e:
                    print(f"Error in test {test_name}: {str(e)}")

            # Run interaction tests if available
            if algorithm in interaction_tests:
                interaction_test = interaction_tests[algorithm]
                test_name = interaction_test['name']
                base_params = interaction_test['base_params'].copy()
                base_params.update(fixed_params)

                print(f"\nRunning interaction test: {test_name}")

                for i, param_combination in enumerate(interaction_test['params']):
                    test_params = base_params.copy()
                    test_params.update(param_combination)

                    sub_test_name = f"{test_name}_{i + 1}"
                    print(f"\nRunning sub-test: {sub_test_name}")
                    print(f"Parameters: {test_params}")

                    try:
                        start_time = time.time()

                        if algorithm == 'expga':
                            results_df, metrics = run_expga(data_obj, **test_params)
                        elif algorithm == 'sg':
                            results_df, metrics = run_sg(data_obj, **test_params)
                        elif algorithm == 'aequitas':
                            results_df, metrics = run_aequitas(data_obj, **test_params)
                        elif algorithm == 'adf':
                            results_df, metrics = run_adf(data_obj, **test_params)

                        runtime = time.time() - start_time

                        test_result = {
                            'test_name': sub_test_name,
                            'parameters': test_params,
                            'metrics': metrics,
                            'runtime': runtime
                        }

                        algorithm_results.append(test_result)

                        print(f"Test completed in {runtime:.2f} seconds")
                        print(
                            f"Results: DSN={metrics.get('DSN', 0)}, TSN={metrics.get('TSN', 0)}, SUR={metrics.get('SUR', 0):.4f}")

                    except Exception as e:
                        print(f"Error in test {sub_test_name}: {str(e)}")

            results[dataset][algorithm] = algorithm_results

    # Save results to file
    import json

    with open('validation_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary of results
    print("\n\n" + "=" * 80)
    print("VALIDATION TEST RESULTS SUMMARY")
    print("=" * 80)

    for dataset in results:
        print(f"\nDATASET: {dataset}")

        for algorithm in results[dataset]:
            print(f"\n  ALGORITHM: {algorithm}")

            # Sort tests by SUR value
            sorted_tests = sorted(
                results[dataset][algorithm],
                key=lambda x: x['metrics'].get('SUR', 0),
                reverse=True
            )

            for i, test in enumerate(sorted_tests[:3]):  # Show top 3 results
                print(f"    {i + 1}. {test['test_name']}: SUR={test['metrics'].get('SUR', 0):.4f}")
                # Show key parameters
                for param, value in test['parameters'].items():
                    if param not in ['db_path', 'analysis_id', 'max_tsn', 'max_runtime_seconds', 'use_cache',
                                     'use_gpu']:
                        print(f"       {param}: {value}")

    print("\nValidation tests completed successfully!")