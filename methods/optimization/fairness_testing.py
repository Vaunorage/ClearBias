import time
import pandas as pd
import json
import os
import sqlite3

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# Assuming these imports work based on your code structure
from data_generator.main import generate_data
from methods.adf.main import run_adf
from methods.exp_ga.algo import run_expga
from methods.aequitas.algo import run_aequitas
from methods.sg.main import run_sg
from path import HERE

DB_PATH = HERE.joinpath("methods/optimization/optimizations.db")


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

    conn.commit()
    return conn


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


def get_default_parameters(method: str) -> Dict[str, Any]:
    """Get default parameters for a method if no optimized parameters are found."""
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


def clean_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Clean parameters by removing runtime-specific ones."""
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
        params_db_path: str = DB_PATH,
        use_gpu: bool = False,
        verbose: bool = True,
        run_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Run fairness testing methods on generated data for multiple iterations.

    Args:
        data_config: Configuration for data generation
        methods: List of methods to run ('expga', 'sg', 'aequitas', 'adf')
        iterations: Number of iterations to run for each method
        max_runtime: Maximum runtime for each method in seconds
        max_tsn: Maximum TSN value
        db_path: Path to SQLite database for storing results
        use_optimized_params: Whether to use optimized parameters from the database
        params_db_path: Path to the database with optimized parameters
        use_gpu: Whether to use GPU (only for ExpGA)
        verbose: Whether to print progress
        run_id: Unique identifier for this run

    Returns:
        DataFrame with all results
    """
    # Set default run_id if not provided
    if run_id is None:
        run_id = f"run_{int(time.time())}"

    # Set up database connection
    conn = setup_sqlite_database(db_path)

    # Results storage
    all_results = []

    try:
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

                # Get optimized parameters if requested
                if use_optimized_params:
                    try:
                        best_results = get_best_hyperparameters(
                            db_path=params_db_path,
                            method=method,
                            data_source="pure-synthetic",  # This could be more specific
                            top_n=1
                        )

                        if best_results:
                            params = clean_parameters(best_results[0]['parameters'])
                            if verbose:
                                print(f"Using optimized parameters for {method}: {params}")
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

                if method == 'expga':
                    params["use_gpu"] = use_gpu

                # Run the method
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
                        continue

                    method_runtime = time.time() - method_start_time

                    # Extract and display metrics
                    dsn = metrics.get("DSN", 0)
                    tsn = metrics.get("TSN", 0)
                    sur = metrics.get("SUR", 0)

                    if verbose:
                        print(f"{method} results: DSN={dsn}, TSN={tsn}, SUR={sur:.4f}")

                    # Save results to database
                    save_result_to_db(
                        conn=conn,
                        run_id=run_id,
                        iteration=iteration,
                        method=method,
                        params=params,
                        metrics=metrics,
                        runtime=method_runtime,
                        data_config=data_config
                    )

                    # Add to results list
                    result_dict = {
                        'run_id': run_id,
                        'iteration': iteration,
                        'method': method,
                        'runtime': method_runtime,
                        'data_config': str(data_config)
                    }
                    result_dict.update(metrics)
                    all_results.append(result_dict)

                except Exception as e:
                    if verbose:
                        print(f"Error running {method}: {str(e)}")

    finally:
        # Close database connection
        conn.close()

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    return results_df


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize results by method."""
    # Group by method and calculate mean, std, min, max
    summary = results_df.groupby('method')[['DSN', 'TSN', 'SUR', 'runtime']].agg(['mean', 'std', 'min', 'max'])

    return summary


if __name__ == "__main__":
    # Example usage without argument parser

    # Configure methods to run
    methods = ['expga', 'sg', 'aequitas', 'adf']

    # Set parameters
    iterations = 5
    max_runtime = 600  # seconds
    max_tsn = 40000
    db_path = HERE.joinpath("methods/optimization/fairness_test_results.db")
    params_db_path = str(DB_PATH.as_posix())
    use_optimized_params = True
    use_gpu = False
    verbose = True
    output_csv = 'results.csv'

    # Define data generation configurations to test various parameters
    data_configs = [
        # Test different numbers of groups
        {'nb_groups': 50, 'nb_attributes': 20, 'prop_protected_attr': 0.2, 'use_cache': True},
        {'nb_groups': 100, 'nb_attributes': 20, 'prop_protected_attr': 0.2, 'use_cache': True},
        {'nb_groups': 150, 'nb_attributes': 20, 'prop_protected_attr': 0.2, 'use_cache': True},
        {'nb_groups': 200, 'nb_attributes': 20, 'prop_protected_attr': 0.2, 'use_cache': True},
        {'nb_groups': 300, 'nb_attributes': 20, 'prop_protected_attr': 0.2, 'use_cache': True},

        # Test different numbers of attributes
        {'nb_groups': 100, 'nb_attributes': 10, 'prop_protected_attr': 0.2, 'use_cache': True},
        {'nb_groups': 100, 'nb_attributes': 15, 'prop_protected_attr': 0.2, 'use_cache': True},
        {'nb_groups': 100, 'nb_attributes': 25, 'prop_protected_attr': 0.2, 'use_cache': True},
        {'nb_groups': 100, 'nb_attributes': 30, 'prop_protected_attr': 0.2, 'use_cache': True},
        {'nb_groups': 100, 'nb_attributes': 40, 'prop_protected_attr': 0.2, 'use_cache': True},

        # Test different proportions of protected attributes
        {'nb_groups': 100, 'nb_attributes': 20, 'prop_protected_attr': 0.1, 'use_cache': True},
        {'nb_groups': 100, 'nb_attributes': 20, 'prop_protected_attr': 0.2, 'use_cache': True},
        {'nb_groups': 100, 'nb_attributes': 20, 'prop_protected_attr': 0.3, 'use_cache': True},
        {'nb_groups': 100, 'nb_attributes': 20, 'prop_protected_attr': 0.4, 'use_cache': True},
        {'nb_groups': 100, 'nb_attributes': 20, 'prop_protected_attr': 0.5, 'use_cache': True},
        {'nb_groups': 100, 'nb_attributes': 20, 'prop_protected_attr': 0.6, 'use_cache': True},

        # Test different group sizes
        {'nb_groups': 100, 'nb_attributes': 20, 'prop_protected_attr': 0.2, 'min_group_size': 5, 'max_group_size': 50,
         'use_cache': True},
        {'nb_groups': 100, 'nb_attributes': 20, 'prop_protected_attr': 0.2, 'min_group_size': 10, 'max_group_size': 100,
         'use_cache': True},
        {'nb_groups': 100, 'nb_attributes': 20, 'prop_protected_attr': 0.2, 'min_group_size': 20, 'max_group_size': 200,
         'use_cache': True},
        {'nb_groups': 100, 'nb_attributes': 20, 'prop_protected_attr': 0.2, 'min_group_size': 50, 'max_group_size': 500,
         'use_cache': True},

        # Test different number of classes
        {'nb_groups': 100, 'nb_attributes': 20, 'prop_protected_attr': 0.2, 'min_number_of_classes': 2,
         'max_number_of_classes': 3, 'use_cache': True},
        {'nb_groups': 100, 'nb_attributes': 20, 'prop_protected_attr': 0.2, 'min_number_of_classes': 3,
         'max_number_of_classes': 5, 'use_cache': True},
        {'nb_groups': 100, 'nb_attributes': 20, 'prop_protected_attr': 0.2, 'min_number_of_classes': 4,
         'max_number_of_classes': 8, 'use_cache': True},

        # Some combined parameter variations
        {'nb_groups': 150, 'nb_attributes': 30, 'prop_protected_attr': 0.3, 'use_cache': True},
        {'nb_groups': 200, 'nb_attributes': 25, 'prop_protected_attr': 0.4, 'use_cache': True},
        {'nb_groups': 250, 'nb_attributes': 15, 'prop_protected_attr': 0.5, 'use_cache': True},
        {'nb_groups': 75, 'nb_attributes': 35, 'prop_protected_attr': 0.25, 'use_cache': True},

        # Complex variations
        {'nb_groups': 150, 'nb_attributes': 30, 'prop_protected_attr': 0.4, 'min_group_size': 20, 'max_group_size': 200,
         'min_number_of_classes': 3, 'max_number_of_classes': 6, 'use_cache': True},
        {'nb_groups': 200, 'nb_attributes': 25, 'prop_protected_attr': 0.3, 'min_group_size': 30, 'max_group_size': 300,
         'min_number_of_classes': 2, 'max_number_of_classes': 4, 'use_cache': True},
        {'nb_groups': 120, 'nb_attributes': 35, 'prop_protected_attr': 0.5, 'min_group_size': 15, 'max_group_size': 150,
         'min_number_of_classes': 4, 'max_number_of_classes': 8, 'use_cache': True}
    ]

    # Storage for all results
    all_results = []

    # Loop through each data configuration
    for data_index, data_config in enumerate(data_configs):
        print(f"\n===== Testing Data Configuration {data_index + 1}/{len(data_configs)} =====")
        print(f"Parameters: {data_config}")

        # Generate a unique run ID for this configuration
        run_id = f"config_{data_index}_{int(time.time())}"
        results_df = run_fairness_testing(
            data_config=data_config,
            methods=methods,
            iterations=iterations,
            max_runtime=max_runtime,
            max_tsn=max_tsn,
            db_path=db_path,
            use_optimized_params=use_optimized_params,
            params_db_path=params_db_path,
            use_gpu=use_gpu,
            verbose=verbose
        )

        # Add to overall results
        all_results.append(results_df)

        # Display summary for this configuration
        summary = summarize_results(results_df)
        print(f"\nResults Summary for Configuration {data_index + 1}:")
        print(summary)