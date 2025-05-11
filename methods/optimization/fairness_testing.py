import optuna
from data_generator.main import generate_data
from methods.adf.main import run_adf
from methods.exp_ga.algo import run_expga
from methods.aequitas.algo import run_aequitas
from methods.optimization.analysis import analyze_matching_synthetic_and_result
from methods.sg.main import run_sg
from path import HERE
import time
import os
import sqlite3
import json
from typing import Tuple, Dict, Any, Optional, List, Union
import pandas as pd

DB_PATH = HERE.joinpath("methods/optimization/optimizations.db")


def fetch_data_from_db(
        db_path: str,
        result_id: Union[str, int],
        fetch_synthetic_data: bool = True,
        fetch_results_data: bool = True,
        verbose: bool = False
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    Fetch test results and synthetic data from the database based on result ID.

    Args:
        db_path: Path to the SQLite database
        result_id: ID of the result to fetch (either the integer id or run_id string)
        fetch_synthetic_data: Whether to fetch the synthetic data
        fetch_results_data: Whether to fetch the results data
        verbose: Whether to print progress

    Returns:
        Tuple containing:
            - Synthetic data DataFrame
            - Results DataFrame
            - Metadata dictionary with experiment information
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Determine the type of ID provided
        if isinstance(result_id, int) or (isinstance(result_id, str) and result_id.isdigit()):
            # If ID is an integer, search by primary key
            query = "SELECT * FROM fairness_test_results WHERE id = ?"
            params = (int(result_id),)
        else:
            # If ID is a string, search by run_id
            query = "SELECT * FROM fairness_test_results WHERE run_id = ? LIMIT 1"
            params = (result_id,)

        # Execute the query
        cursor.execute(query, params)
        row = cursor.fetchone()

        if not row:
            if verbose:
                print(f"No result found with ID: {result_id}")
            return None, None, None

        # Get column names
        columns = [description[0] for description in cursor.description]
        result_dict = dict(zip(columns, row))

        # Extract metadata
        experiment_id = result_dict.get('run_id', str(result_dict.get('id', 'unknown')))
        method = result_dict.get('method', 'unknown')
        parameters = json.loads(result_dict.get('parameters', '{}'))
        data_config = json.loads(result_dict.get('data_config', '{}'))
        metrics = json.loads(result_dict.get('metrics', '{}'))

        if verbose:
            print(f"Found result for ID: {result_id}")
            print(f"Run ID: {experiment_id}, Method: {method}")

        metadata = {
            'experiment_id': experiment_id,
            'method': method,
            'parameters': parameters,
            'data_config': data_config,
            'metrics': metrics,
            'timestamp': result_dict.get('timestamp'),
            'runtime': result_dict.get('runtime')
        }

        synthetic_data = None
        results_df = None

        # Check available tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]

        if verbose:
            print(f"Available tables in database: {tables}")

        # Fetch synthetic data
        if fetch_synthetic_data:
            # Check if input_dataframes table exists
            if 'input_dataframes' in tables:
                cursor.execute(
                    "SELECT dataframe_json FROM input_dataframes WHERE run_id = ? LIMIT 1",
                    (experiment_id,)
                )
                row = cursor.fetchone()

                if row and row[0]:
                    try:
                        synthetic_data = pd.read_json(row[0])
                        if verbose:
                            print(
                                f"Successfully loaded synthetic data from input_dataframes table. Shape: {synthetic_data.shape}")
                    except Exception as e:
                        if verbose:
                            print(f"Error parsing synthetic data JSON: {str(e)}")

            # If still no synthetic data, generate it using data_config
            if synthetic_data is None:
                if verbose:
                    print("No synthetic data found in database. Attempting to generate from data_config.")

                # Try to import generate_data function
                try:
                    # This assumes the data_generator module is available in the path
                    from data_generator.main import generate_data
                    synthetic_data = generate_data(**data_config)

                    if verbose:
                        print(f"Successfully generated synthetic data. Shape: {synthetic_data.shape}")
                except ImportError:
                    if verbose:
                        print("Could not import data_generator.main. Please ensure the module is in your path.")
                except Exception as e:
                    if verbose:
                        print(f"Error generating synthetic data: {str(e)}")

        # Fetch results data
        if fetch_results_data:
            # Check if results_dataframes table exists
            if 'results_dataframes' in tables:
                cursor.execute(
                    "SELECT dataframe_json FROM results_dataframes WHERE run_id = ? AND method = ? LIMIT 1",
                    (experiment_id, method)
                )
                row = cursor.fetchone()

                if row and row[0]:
                    try:
                        results_df = pd.read_json(row[0])
                        if verbose:
                            print(
                                f"Successfully loaded results from results_dataframes table. Shape: {results_df.shape}")
                    except Exception as e:
                        if verbose:
                            print(f"Error parsing results JSON: {str(e)}")

            # If no results found in results_dataframes, look for results in metrics
            if results_df is None and 'results' in metrics:
                try:
                    results_json = metrics.get('results')
                    if isinstance(results_json, str):
                        results_df = pd.read_json(results_json)
                        if verbose:
                            print(f"Successfully loaded results from metrics. Shape: {results_df.shape}")
                except Exception as e:
                    if verbose:
                        print(f"Error parsing results from metrics: {str(e)}")

            # If still no results, create an empty DataFrame with expected columns
            if results_df is None:
                results_df = pd.DataFrame(columns=['indv_key', 'couple_key'])
                if verbose:
                    print("Created empty results DataFrame")

        conn.close()

        # If synthetic data is still None, create a synthetic dataset with required columns for testing
        if synthetic_data is None:
            if verbose:
                print("Creating minimal synthetic data with required columns for analysis")

            # Create a minimal synthetic dataset
            synthetic_data = pd.DataFrame({
                'indv_key': [f'indv_{i}' for i in range(10)],
                'group_key': ['group1-group2'] * 5 + ['group3-group4'] * 5,
                'subgroup_key': ['subgroup1'] * 10
            })

            # Add calculated properties
            calculated_properties = [
                'calculated_epistemic_random_forest', 'calculated_aleatoric_random_forest',
                'calculated_aleatoric_entropy', 'calculated_aleatoric_probability_margin',
                'calculated_aleatoric_label_smoothing', 'calculated_epistemic_ensemble',
                'calculated_epistemic_mc_dropout', 'calculated_epistemic_evidential',
                'calculated_epistemic_group', 'calculated_aleatoric_group', 'calculated_magnitude',
                'calculated_group_size', 'calculated_granularity', 'calculated_intersectionality',
                'calculated_uncertainty_group', 'calculated_similarity', 'calculated_subgroup_ratio'
            ]

            for prop in calculated_properties:
                synthetic_data[prop] = 0.5

            # Add some attribute columns for pattern matching
            for i in range(1, 8):
                synthetic_data[f'Attr{i}_T'] = i % 2  # Just some dummy values

            if verbose:
                print(f"Created minimal synthetic data with shape: {synthetic_data.shape}")

        return synthetic_data, results_df, metadata

    except Exception as e:
        if verbose:
            print(f"Error fetching data from database: {str(e)}")
        return None, None, None


def get_best_hyperparameters(
        db_path: str,
        method: str = None,
        dataset: str = None,
        data_source: str = None,
        study_name: str = None,
        run_id: str = None,  # Added run_id parameter
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
        run_id: Filter by run_id
        top_n: Number of top results to return
        metric: Metric to optimize for (default: "SUR")

    Returns:
        List of dictionaries containing the best hyperparameters and their metrics
    """

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

    # Add filter for run_id if provided
    if run_id:
        query += " AND json_extract(extra_data, '$.run_id') = ?"
        params.append(run_id)

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


def setup_sqlite_database(db_path: str) -> sqlite3.Connection:
    """Create or connect to SQLite database for storing results."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    # Create connection
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create tables if they don't exist
    c.execute('''
              CREATE TABLE IF NOT EXISTS fairness_test_results
              (
                  id
                  INTEGER
                  PRIMARY
                  KEY
                  AUTOINCREMENT,
                  timestamp
                  TEXT,
                  run_id
                  TEXT,
                  iteration
                  INTEGER,
                  method
                  TEXT,
                  parameters
                  TEXT,
                  metrics
                  TEXT,
                  runtime
                  REAL,
                  data_config
                  TEXT
              )
              ''')

    # Create optimization_trials table if it doesn't exist
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
                  trial_number
                  INTEGER,
                  study_name
                  TEXT,
                  method
                  TEXT,
                  parameters
                  TEXT,
                  metrics
                  TEXT,
                  runtime
                  REAL,
                  extra_data
                  TEXT
              )
              ''')

    # Create tables for storing dataframes
    c.execute('''
              CREATE TABLE IF NOT EXISTS results_dataframes
              (
                  id
                  INTEGER
                  PRIMARY
                  KEY
                  AUTOINCREMENT,
                  timestamp
                  TEXT,
                  run_id
                  TEXT,
                  iteration
                  INTEGER,
                  method
                  TEXT,
                  dataframe_json
                  TEXT
              )
              ''')

    c.execute('''
              CREATE TABLE IF NOT EXISTS synthetic_dataframes
              (
                  id
                  INTEGER
                  PRIMARY
                  KEY
                  AUTOINCREMENT,
                  timestamp
                  TEXT,
                  run_id
                  TEXT,
                  iteration
                  INTEGER,
                  method
                  TEXT,
                  dataframe_json
                  TEXT
              )
              ''')

    conn.commit()
    return conn


def save_dataframes_to_db(
        conn: sqlite3.Connection,
        results_df: pd.DataFrame,
        data_df: pd.DataFrame,
        run_id: str,
        method: str,
        iteration: int = 0
) -> None:
    """Save results_df and data_obj.dataframe to separate tables in the database."""
    # Create tables if they don't exist
    c = conn.cursor()

    # Create results_dataframes table if it doesn't exist
    c.execute('''
              CREATE TABLE IF NOT EXISTS results_dataframes
              (
                  id
                  INTEGER
                  PRIMARY
                  KEY
                  AUTOINCREMENT,
                  timestamp
                  TEXT,
                  run_id
                  TEXT,
                  iteration
                  INTEGER,
                  method
                  TEXT,
                  dataframe_json
                  TEXT
              )
              ''')

    # Create synthetic_dataframes table if it doesn't exist
    c.execute('''
              CREATE TABLE IF NOT EXISTS synthetic_dataframes
              (
                  id
                  INTEGER
                  PRIMARY
                  KEY
                  AUTOINCREMENT,
                  timestamp
                  TEXT,
                  run_id
                  TEXT,
                  iteration
                  INTEGER,
                  method
                  TEXT,
                  dataframe_json
                  TEXT
              )
              ''')

    # Get current timestamp
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    # Convert dataframes to JSON
    results_json = results_df.to_json(orient='records')
    data_json = data_df.to_json(orient='records')

    # Insert results_df into results_dataframes table
    c.execute('''
              INSERT INTO results_dataframes
                  (timestamp, run_id, iteration, method, dataframe_json)
              VALUES (?, ?, ?, ?, ?)
              ''', (timestamp, run_id, iteration, method, results_json))

    # Insert data_df into synthetic_dataframes table
    c.execute('''
              INSERT INTO synthetic_dataframes
                  (timestamp, run_id, iteration, method, dataframe_json)
              VALUES (?, ?, ?, ?, ?)
              ''', (timestamp, run_id, iteration, method, data_json))

    conn.commit()


def save_result_to_db(
        conn: sqlite3.Connection,
        run_id: str,
        iteration: int,
        method: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        runtime: float,
        data_config: Dict[str, Any]
) -> None:
    """Save test run results to the database."""
    c = conn.cursor()

    # Convert dictionaries to JSON strings
    params_json = json.dumps(params)
    metrics_json = json.dumps(metrics)
    data_config_json = json.dumps(data_config)

    # Get current timestamp
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    # Insert data
    c.execute('''
              INSERT INTO fairness_test_results
              (timestamp, run_id, iteration, method, parameters, metrics, runtime, data_config)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)
              ''', (timestamp, run_id, iteration, method, params_json, metrics_json, runtime, data_config_json))

    conn.commit()


def save_optimization_trial_to_db(
        conn: sqlite3.Connection,
        trial_number: int,
        study_name: str,
        method: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        runtime: float,
        extra_data: Dict[str, Any]
) -> None:
    """Save optimization trial results to the database."""
    c = conn.cursor()

    # Convert dictionaries to JSON strings
    params_json = json.dumps(params)
    metrics_json = json.dumps(metrics)
    extra_data_json = json.dumps(extra_data)

    # Get current timestamp
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    # Insert data
    c.execute('''
              INSERT INTO optimization_trials
              (timestamp, trial_number, study_name, method, parameters, metrics, runtime, extra_data)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)
              ''', (timestamp, trial_number, study_name, method, params_json, metrics_json, runtime, extra_data_json))

    conn.commit()


def get_default_parameters(method: str) -> Dict[str, Any]:
    """Get default parameters for a method if no optimized parameters are found."""
    # Function remains the same
    if method == 'expga':
        return {
            "threshold_rank": 0.5,
            "max_global": 10000,
            "max_local": 100,
            "model_type": "rf",
            "cross_rate": 0.8,
            "mutation": 0.1,
            "one_attr_at_a_time": True,
            "random_seed": 42
        }
    elif method == 'sg':
        return {
            "model_type": "rf",
            "cluster_num": 50,
            "one_attr_at_a_time": True,
            "random_seed": 42
        }
    elif method == 'aequitas':
        return {
            "model_type": "rf",
            "max_global": 500,
            "max_local": 5000,
            "step_size": 0.5,
            "init_prob": 0.5,
            "param_probability_change_size": 0.005,
            "direction_probability_change_size": 0.005,
            "one_attr_at_a_time": True,
            "random_seed": 42
        }
    elif method == 'adf':
        return {
            "max_global": 1000,
            "max_local": 1000,
            "cluster_num": 50,
            "step_size": 0.5,
            "one_attr_at_a_time": True,
            "random_seed": 42
        }
    else:
        return {}


def get_search_space(method: str, trial: optuna.Trial) -> Dict[str, Any]:
    """Define parameter search space for each method."""
    # Function remains the same
    if method == 'expga':
        params = {
            "threshold_rank": trial.suggest_float("threshold_rank", 0.1, 0.9),
            "max_global": trial.suggest_int("max_global", 1000, 20000),
            "max_local": trial.suggest_int("max_local", 50, 500),
            "model_type": "rf",  # Fixed parameter
            "cross_rate": trial.suggest_float("cross_rate", 0.5, 0.95),
            "mutation": trial.suggest_float("mutation", 0.05, 0.3),
            "one_attr_at_a_time": True,  # Fixed parameter
            "random_seed": 42  # Fixed parameter
        }
    elif method == 'sg':
        params = {
            "model_type": "rf",  # Fixed parameter
            "cluster_num": trial.suggest_int("cluster_num", 10, 200),
            "one_attr_at_a_time": True,  # Fixed parameter
            "random_seed": 42  # Fixed parameter
        }
    elif method == 'aequitas':
        params = {
            "model_type": "rf",  # Fixed parameter
            "max_global": trial.suggest_int("max_global", 200, 1000),
            "max_local": trial.suggest_int("max_local", 1000, 10000),
            "step_size": trial.suggest_float("step_size", 0.1, 0.9),
            "init_prob": trial.suggest_float("init_prob", 0.1, 0.9),
            "param_probability_change_size": trial.suggest_float("param_probability_change_size", 0.001, 0.01),
            "direction_probability_change_size": trial.suggest_float("direction_probability_change_size", 0.001, 0.01),
            "one_attr_at_a_time": True,  # Fixed parameter
            "random_seed": 42  # Fixed parameter
        }
    elif method == 'adf':
        params = {
            "max_global": trial.suggest_int("max_global", 500, 2000),
            "max_local": trial.suggest_int("max_local", 500, 2000),
            "cluster_num": trial.suggest_int("cluster_num", 10, 100),
            "step_size": trial.suggest_float("step_size", 0.1, 0.9),
            "one_attr_at_a_time": True,  # Fixed parameter
            "random_seed": 42  # Fixed parameter
        }
    else:
        params = {}

    return params


def run_method_with_params(
        method: str,
        data_obj: Any,
        params: Dict[str, Any],
        db_path: str = None,
        run_id: str = None,
        iteration: int = 0,  # Add iteration parameter
        verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], float]:
    """Run a fairness testing method with given parameters and return results."""
    method_start_time = time.time()

    try:
        if method == 'expga':
            results_df, metrics = run_expga(data_obj, **params)
        elif method == 'sg':
            results_df, metrics = run_sg(data_obj, **params)
        elif method == 'aequitas':
            results_df, metrics = run_aequitas(data_obj, **params)
        elif method == 'adf':
            results_df, metrics = run_adf(data_obj, **params)
        else:
            if verbose:
                print(f"Unknown method: {method}")
            return None, {}, 0

        method_runtime = time.time() - method_start_time

        # Calculate analysis metrics and save to database if db_path is provided
        if db_path and run_id:
            try:
                # Connect to database
                conn = sqlite3.connect(db_path)

                conn.close()

            except Exception as e:
                if verbose:
                    print(f"Error calculating or saving analysis metrics or dataframes: {str(e)}")

        return results_df, data_obj.dataframe, metrics, method_runtime

    except Exception as e:
        if verbose:
            print(f"Error running {method}: {str(e)}")
        return None, None, {}, time.time() - method_start_time


def optimize_hyperparameters(
        method: str,
        data_config: Dict[str, Any],
        n_trials: int = 20,
        max_runtime_per_trial: int = 300,
        max_tsn: int = 10000,
        db_path: str = DB_PATH,
        use_gpu: bool = False,
        verbose: bool = True,
        optimization_metric: str = "SUR",
        run_id: str = None  # Add run_id parameter
) -> Dict[str, Any]:
    """
    Optimize hyperparameters for a fairness testing method using Optuna.

    Args:
        method: Method to optimize ('expga', 'sg', 'aequitas', 'adf')
        data_config: Configuration for data generation
        n_trials: Number of Optuna trials
        max_runtime_per_trial: Maximum runtime for each trial in seconds
        max_tsn: Maximum TSN value
        db_path: Path to SQLite database for storing results
        use_gpu: Whether to use GPU (only for ExpGA)
        verbose: Whether to print progress
        optimization_metric: Metric to optimize (default: "SUR")
        run_id: Unique identifier to link optimization to test runs

    Returns:
        Dictionary with optimized parameters
    """
    # Set up database connection
    conn = setup_sqlite_database(db_path)

    # Generate data using provided configuration
    if verbose:
        print(f"Generating data with config: {data_config}")

    data_obj = generate_data(**data_config)

    # Create a unique study name that includes the run_id if provided
    if run_id:
        study_name = f"{run_id}_{method}_opt"
    else:
        study_name = f"{method}_opt_{int(time.time())}"

    # Define the objective function
    def objective(trial):
        # Get parameters from search space
        params = get_search_space(method, trial)

        # Add runtime parameters
        params["max_runtime_seconds"] = max_runtime_per_trial
        params["max_tsn"] = max_tsn
        params["use_cache"] = True

        if method == 'expga' and use_gpu:
            params["use_gpu"] = use_gpu

        # Run the method
        results_df, synth_df, metrics, runtime = run_method_with_params(
            method=method,
            data_obj=data_obj,
            params=params,
            verbose=verbose
        )

        # Extract metric value (default to 0 if not found or error occurred)
        metric_value = metrics.get(optimization_metric, 0)

        # Save trial results to database
        extra_data = {
            "data_source": "pure-synthetic",
            "dataset": f"synthetic_{data_config.get('nb_groups', 0)}_{data_config.get('nb_attributes', 0)}",
            "run_id": run_id  # Store run_id in extra_data to link optimization trials to test runs
        }

        save_optimization_trial_to_db(
            conn=conn,
            trial_number=trial.number,
            study_name=study_name,
            method=method,
            params=params,
            metrics=metrics,
            runtime=runtime,
            extra_data=extra_data
        )

        if verbose:
            print(f"Trial {trial.number}: {optimization_metric}={metric_value:.4f}, runtime={runtime:.2f}s")

        return metric_value

    try:
        # Create the study and optimize
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        study.optimize(objective, n_trials=n_trials)

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        if verbose:
            print(f"\nOptimization completed for {method}")
            print(f"Best {optimization_metric}: {best_value:.4f}")
            print(f"Best parameters: {best_params}")

        # Add fixed parameters that weren't part of the optimization
        default_params = get_default_parameters(method)
        for key, value in default_params.items():
            if key not in best_params:
                best_params[key] = value

        return best_params

    finally:
        # Close database connection
        conn.close()


def clean_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Clean parameters by removing runtime-specific ones."""
    # Function remains the same
    cleaned_params = params.copy()
    for key in ['max_runtime_seconds', 'max_tsn', 'db_path', 'analysis_id', 'use_cache', 'use_gpu']:
        if key in cleaned_params:
            del cleaned_params[key]
    return cleaned_params


def run_fairness_testing(
        data_config: Dict[str, Any],
        methods: List[str],
        iterations: int,
        max_runtime: int,
        max_tsn: int,
        db_path: str,
        use_optimized_params: bool = True,
        run_optimization: bool = False,
        optimization_trials: int = 20,
        params_db_path: str = DB_PATH,
        use_gpu: bool = False,
        verbose: bool = True,
        run_id: Optional[str] = None,
        optimization_metric: str = "SUR",
        method_params: Dict[str, Any] = None,
):
    """
    Run fairness testing methods on generated data for multiple iterations.
    """
    # Set default run_id if not provided - this is the central run_id generation
    if run_id is None:
        run_id = f"run_{int(time.time())}"

    if verbose:
        print(f"Using run_id: {run_id} for all operations")

    # Set up database connection
    conn = setup_sqlite_database(db_path)

    # Store optimized parameters
    optimized_params = {}

    try:
        # Run optimization if requested
        if run_optimization:
            if verbose:
                print("\n=== Running Hyperparameter Optimization ===")

            for method in methods:
                if verbose:
                    print(f"\nOptimizing parameters for {method}...")

                # Run Optuna optimization WITH the run_id to link it to the test runs
                best_params = optimize_hyperparameters(
                    method=method,
                    data_config=data_config,
                    n_trials=optimization_trials,
                    max_runtime_per_trial=max_runtime // 2,
                    max_tsn=max_tsn // 2,
                    db_path=params_db_path,
                    use_gpu=use_gpu,
                    verbose=verbose,
                    optimization_metric=optimization_metric,
                    run_id=run_id
                )

                # Store optimized parameters
                optimized_params[method] = best_params

        for iteration in range(iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{iterations} ---")

            # Generate data using provided configuration
            if verbose:
                print(f"Generating data with config: {data_config}")

            data_obj = generate_data(**data_config)

            for method in methods:
                if verbose:
                    print(f"\nRunning {method} on iteration {iteration + 1}")

                # Get parameters
                if run_optimization and method in optimized_params:
                    params = optimized_params[method].copy()
                    if verbose:
                        print(f"Using newly optimized parameters for {method}: {params}")
                elif use_optimized_params:
                    try:
                        # Get best parameters for this specific run_id if available
                        best_results = get_best_hyperparameters(
                            db_path=params_db_path,
                            method=method,
                            data_source="pure-synthetic",
                            run_id=run_id,
                            top_n=1
                        )

                        if best_results:
                            params = clean_parameters(best_results[0]['parameters'])
                            if verbose:
                                print(f"Using stored optimized parameters for {method} from run {run_id}: {params}")
                        else:
                            # If no parameters found for this run_id, try without run_id filter
                            best_results = get_best_hyperparameters(
                                db_path=params_db_path,
                                method=method,
                                data_source="pure-synthetic",
                                top_n=1
                            )

                            if best_results:
                                params = clean_parameters(best_results[0]['parameters'])
                                if verbose:
                                    print(f"Using stored optimized parameters for {method} from other runs: {params}")
                            else:
                                params = get_default_parameters(method)
                                if verbose:
                                    print(f"No optimized parameters found for {method}, using defaults: {params}")
                    except Exception as e:
                        if verbose:
                            print(f"Error getting optimized parameters: {str(e)}")
                        params = get_default_parameters(method)
                else:
                    params = get_default_parameters(method)

                # Add runtime parameters
                params["max_runtime_seconds"] = max_runtime
                params["max_tsn"] = max_tsn
                params["use_cache"] = True

                for par in method_params:
                    params[par] = method_params[par]

                if method == 'expga':
                    params["use_gpu"] = use_gpu

                # Create a configuration-specific run_id that includes the master run_id and iteration
                iter_run_id = f"{run_id}_iter_{iteration}"

                # Run the method with db_path for analysis
                results_df, synth_df, metrics, method_runtime = run_method_with_params(
                    method=method,
                    data_obj=data_obj,
                    params=params,
                    db_path=db_path,
                    run_id=iter_run_id,
                    verbose=verbose
                )

                res_with_groups, synth_with_groups = analyze_matching_synthetic_and_result(results_df, synth_df)
                save_dataframes_to_db(
                    conn=conn,
                    results_df=res_with_groups,
                    data_df=synth_with_groups,
                    run_id=iter_run_id,
                    method=method,
                    iteration=iteration,
                )

                # Save results to database (original fairness_test_results table)
                save_result_to_db(
                    conn=conn,
                    run_id=iter_run_id,
                    iteration=iteration,
                    method=method,
                    params=params,
                    metrics=metrics,
                    runtime=method_runtime,
                    data_config=data_obj.generation_arguments
                )


    finally:
        # Close database connection
        conn.close()


if __name__ == "__main__":

    methods = [
        'expga',
        'sg',
        'aequitas',
        'adf'
    ]

    # Set parameters
    iterations = 1
    max_runtime = 200  # seconds
    max_tsn = 40000
    db_path = HERE.joinpath("methods/optimization/fairness_test_results2.db")
    params_db_path = str(DB_PATH.as_posix())
    use_optimized_params = True
    run_optimization = True
    optimization_trials = 3
    use_gpu = False
    verbose = True
    optimization_metric = "SUR"

    method_params = {'one_attr_at_a_time': False}

    # Generate a single master run_id for the entire experiment
    master_run_id = f"experiment_{int(time.time())}"
    print(f"Starting experiment with master run_id: {master_run_id}")

    # Storage for all results
    all_results = []

    # Loop through each data configuration
    data_configs = [
        # Base configuration
        {'nb_groups': 10, 'nb_attributes': 5, 'prop_protected_attr': 0.2, 'min_number_of_classes': 2,
         'max_number_of_classes': 4, 'use_cache': True, 'min_group_size': 100, 'max_group_size': 1000,
         'min_similarity': 0.5},

        # Test different numbers of groups
        {'nb_groups': 2, 'nb_attributes': 5, 'prop_protected_attr': 0.2, 'min_number_of_classes': 2,
         'max_number_of_classes': 4, 'use_cache': True, 'min_group_size': 100, 'max_group_size': 1000,
         'min_similarity': 0.5},
        {'nb_groups': 5, 'nb_attributes': 5, 'prop_protected_attr': 0.2, 'min_number_of_classes': 2,
         'max_number_of_classes': 4, 'use_cache': True, 'min_group_size': 100, 'max_group_size': 1000,
         'min_similarity': 0.5},
        {'nb_groups': 20, 'nb_attributes': 5, 'prop_protected_attr': 0.2, 'min_number_of_classes': 2,
         'max_number_of_classes': 4, 'use_cache': True, 'min_group_size': 100, 'max_group_size': 1000,
         'min_similarity': 0.5},

        # Test different numbers of attributes
        {'nb_groups': 10, 'nb_attributes': 3, 'prop_protected_attr': 0.2, 'min_number_of_classes': 2,
         'max_number_of_classes': 4, 'use_cache': True, 'min_group_size': 100, 'max_group_size': 1000,
         'min_similarity': 0.5},
        {'nb_groups': 10, 'nb_attributes': 8, 'prop_protected_attr': 0.2, 'min_number_of_classes': 2,
         'max_number_of_classes': 4, 'use_cache': True, 'min_group_size': 100, 'max_group_size': 1000,
         'min_similarity': 0.5},
        {'nb_groups': 10, 'nb_attributes': 12, 'prop_protected_attr': 0.2, 'min_number_of_classes': 2,
         'max_number_of_classes': 4, 'use_cache': True, 'min_group_size': 100, 'max_group_size': 1000,
         'min_similarity': 0.5},

        # Test different proportions of protected attributes
        {'nb_groups': 10, 'nb_attributes': 5, 'prop_protected_attr': 0.1, 'min_number_of_classes': 2,
         'max_number_of_classes': 4, 'use_cache': True, 'min_group_size': 100, 'max_group_size': 1000,
         'min_similarity': 0.5},
        {'nb_groups': 10, 'nb_attributes': 5, 'prop_protected_attr': 0.3, 'min_number_of_classes': 2,
         'max_number_of_classes': 4, 'use_cache': True, 'min_group_size': 100, 'max_group_size': 1000,
         'min_similarity': 0.5},
        {'nb_groups': 10, 'nb_attributes': 5, 'prop_protected_attr': 0.5, 'min_number_of_classes': 2,
         'max_number_of_classes': 4, 'use_cache': True, 'min_group_size': 100, 'max_group_size': 1000,
         'min_similarity': 0.5},

        # Test different group sizes
        {'nb_groups': 10, 'nb_attributes': 5, 'prop_protected_attr': 0.2, 'min_number_of_classes': 2,
         'max_number_of_classes': 4, 'use_cache': True, 'min_group_size': 50, 'max_group_size': 200,
         'min_similarity': 0.5},
        {'nb_groups': 10, 'nb_attributes': 5, 'prop_protected_attr': 0.2, 'min_number_of_classes': 2,
         'max_number_of_classes': 4, 'use_cache': True, 'min_group_size': 200, 'max_group_size': 2000,
         'min_similarity': 0.5},
        {'nb_groups': 10, 'nb_attributes': 5, 'prop_protected_attr': 0.2, 'min_number_of_classes': 2,
         'max_number_of_classes': 4, 'use_cache': True, 'min_group_size': 500, 'max_group_size': 5000,
         'min_similarity': 0.5},

        # Test different number of classes
        {'nb_groups': 10, 'nb_attributes': 5, 'prop_protected_attr': 0.2, 'min_number_of_classes': 2,
         'max_number_of_classes': 2, 'use_cache': True, 'min_group_size': 100, 'max_group_size': 1000,
         'min_similarity': 0.5},
        {'nb_groups': 10, 'nb_attributes': 5, 'prop_protected_attr': 0.2, 'min_number_of_classes': 3,
         'max_number_of_classes': 5, 'use_cache': True, 'min_group_size': 100, 'max_group_size': 1000,
         'min_similarity': 0.5},
        {'nb_groups': 10, 'nb_attributes': 5, 'prop_protected_attr': 0.2, 'min_number_of_classes': 4,
         'max_number_of_classes': 8, 'use_cache': True, 'min_group_size': 100, 'max_group_size': 1000,
         'min_similarity': 0.5},

        # Test different similarity values
        {'nb_groups': 10, 'nb_attributes': 5, 'prop_protected_attr': 0.2, 'min_number_of_classes': 2,
         'max_number_of_classes': 4, 'use_cache': True, 'min_group_size': 100, 'max_group_size': 1000,
         'min_similarity': 0.2},
        {'nb_groups': 10, 'nb_attributes': 5, 'prop_protected_attr': 0.2, 'min_number_of_classes': 2,
         'max_number_of_classes': 4, 'use_cache': True, 'min_group_size': 100, 'max_group_size': 1000,
         'min_similarity': 0.7},
        {'nb_groups': 10, 'nb_attributes': 5, 'prop_protected_attr': 0.2, 'min_number_of_classes': 2,
         'max_number_of_classes': 4, 'use_cache': True, 'min_group_size': 100, 'max_group_size': 1000,
         'min_similarity': 0.9},

        # Complex combined variations - Small dataset with many groups
        {'nb_groups': 20, 'nb_attributes': 3, 'prop_protected_attr': 0.4, 'min_number_of_classes': 2,
         'max_number_of_classes': 3, 'use_cache': True, 'min_group_size': 50, 'max_group_size': 200,
         'min_similarity': 0.3},

        # Complex combined variations - Large dataset with few groups
        {'nb_groups': 3, 'nb_attributes': 10, 'prop_protected_attr': 0.3, 'min_number_of_classes': 4,
         'max_number_of_classes': 8, 'use_cache': True, 'min_group_size': 500, 'max_group_size': 5000,
         'min_similarity': 0.8},

        # High dimensional with many classes
        {'nb_groups': 15, 'nb_attributes': 15, 'prop_protected_attr': 0.2, 'min_number_of_classes': 5,
         'max_number_of_classes': 10, 'use_cache': True, 'min_group_size': 200, 'max_group_size': 1000,
         'min_similarity': 0.6},

        # Extreme case - few groups, few attributes, binary classes
        {'nb_groups': 2, 'nb_attributes': 2, 'prop_protected_attr': 0.5, 'min_number_of_classes': 2,
         'max_number_of_classes': 2, 'use_cache': True, 'min_group_size': 1000, 'max_group_size': 2000,
         'min_similarity': 0.9},

        # Extreme case - many groups, many attributes, many classes
        {'nb_groups': 30, 'nb_attributes': 20, 'prop_protected_attr': 0.1, 'min_number_of_classes': 8,
         'max_number_of_classes': 10, 'use_cache': True, 'min_group_size': 50, 'max_group_size': 200,
         'min_similarity': 0.4}
    ]
    for data_index, data_config in enumerate(data_configs):
        print(f"\n===== Testing Data Configuration {data_index + 1}/{len(data_configs)} =====")
        print(f"Parameters: {data_config}")

        # Create a configuration-specific run_id that includes the master run_id
        config_run_id = f"{master_run_id}_config_{data_index}"

        run_fairness_testing(
            data_config=data_config,
            methods=methods,
            iterations=iterations,
            max_runtime=max_runtime,
            max_tsn=max_tsn,
            db_path=db_path,
            use_optimized_params=use_optimized_params,
            run_optimization=run_optimization,
            optimization_trials=optimization_trials,
            params_db_path=params_db_path,
            use_gpu=use_gpu,
            verbose=verbose,
            run_id=config_run_id,
            optimization_metric=optimization_metric,
            method_params=method_params
        )
