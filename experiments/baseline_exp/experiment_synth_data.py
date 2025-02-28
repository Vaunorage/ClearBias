import pandas as pd
import numpy as np
import sqlite3
import json
import hashlib
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

from data_generator.main import get_real_data, generate_from_real_data
from data_generator.utils import plot_distribution_comparison
from methods.utils import reformat_discrimination_results, convert_to_non_float_rows, compare_discriminatory_groups

from methods.adf.main1 import adf_fairness_testing
from methods.aequitas.algo import run_aequitas
from methods.exp_ga.algo import run_expga
from methods.sg.main import run_sg
from path import HERE


class ExperimentDatabase:
    """Database manager for fairness testing experiments"""

    def __init__(self, db_path='fairness_experiments.db'):
        """Initialize database connection and create tables if they don't exist"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        """Create necessary tables if they don't exist"""
        cursor = self.conn.cursor()

        # Create experiments table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            method TEXT NOT NULL,
            parameters TEXT NOT NULL,
            results TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Create statistics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS statistics (
            dataset TEXT NOT NULL,
            method TEXT NOT NULL,
            metric TEXT NOT NULL,
            mean REAL,
            std REAL,
            runs INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (dataset, method, metric)
        )
        ''')

        self.conn.commit()

    def generate_experiment_id(self, dataset, method, params):
        """Generate a unique ID for an experiment based on its parameters"""
        # Convert params to sorted string to ensure consistent hashing
        params_str = json.dumps(params, sort_keys=True)
        # Create a string combining all parameters
        experiment_str = f"{dataset}_{method}_{params_str}"
        # Generate a hash
        return hashlib.md5(experiment_str.encode()).hexdigest()

    def experiment_exists(self, experiment_id):
        """Check if an experiment with given ID exists in the database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM experiments WHERE id = ?", (experiment_id,))
        return cursor.fetchone() is not None

    def save_experiment(self, experiment_id, dataset, method, params, results):
        """Save experiment results to database"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO experiments (id, dataset, method, parameters, results) VALUES (?, ?, ?, ?, ?)",
            (experiment_id, dataset, method, json.dumps(params), json.dumps(results))
        )
        self.conn.commit()

    def get_experiment(self, experiment_id):
        """Retrieve experiment results by ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT results FROM experiments WHERE id = ?", (experiment_id,))
        result = cursor.fetchone()
        if result:
            return json.loads(result[0])
        return None

    def update_statistics(self, dataset, method, results_list):
        """Update statistics for a dataset-method combination"""
        cursor = self.conn.cursor()

        # Get all metrics from the results
        all_metrics = set()
        for result in results_list:
            all_metrics.update(result.keys())

        # Calculate and save statistics for each metric
        for metric in all_metrics:
            # Extract values for this metric (if present)
            values = [result.get(metric) for result in results_list if metric in result]

            # Only calculate stats if we have values
            if values:
                if metric in ['unmatched_original_groups'] and isinstance(values[0], list):
                    mean_val = np.mean(list(map(len, values)))
                    std_val = np.std(list(map(len, values)))
                else:
                    mean_val = np.mean(values)
                    std_val = np.std(values)

                # Save to database
                cursor.execute(
                    """INSERT OR REPLACE INTO statistics 
                       (dataset, method, metric, mean, std, runs, timestamp) 
                       VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                    (dataset, method, metric, float(mean_val), float(std_val), len(values))
                )

        self.conn.commit()

    def get_statistics(self, datasets=None, methods=None):
        """Retrieve statistics from the database with optional filtering"""
        cursor = self.conn.cursor()

        query = "SELECT dataset, method, metric, mean, std, runs FROM statistics"
        params = []

        # Add filters if provided
        conditions = []
        if datasets:
            placeholders = ','.join(['?'] * len(datasets))
            conditions.append(f"dataset IN ({placeholders})")
            params.extend(datasets)

        if methods:
            placeholders = ','.join(['?'] * len(methods))
            conditions.append(f"method IN ({placeholders})")
            params.extend(methods)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=['dataset', 'method', 'metric', 'mean', 'std', 'runs'])

        # Pivot for better readability
        if not df.empty:
            # Create a column name that includes the statistic type
            df['metric_stat'] = df.apply(
                lambda row: f"{row['metric']}_mean" if 'mean' in row.name else f"{row['metric']}_std", axis=1)

            # Pivot table
            pivot_df = df.pivot_table(
                index=['dataset', 'method'],
                columns='metric_stat',
                values=['mean', 'std']
            ).reset_index()

            # Flatten multi-index columns
            pivot_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in pivot_df.columns]

            return pivot_df

        return df

    def close(self):
        """Close database connection"""
        self.conn.close()


def get_groups(results_df, data_obj, schema):
    """Extract discriminatory groups from testing results"""
    non_float_df = convert_to_non_float_rows(results_df, schema)
    predefined_groups = reformat_discrimination_results(non_float_df, data_obj.dataframe)
    return predefined_groups


def run_experiment(dataset_name='adult', method='adf', db=None, params={}):
    """Run fairness testing experiment on original and synthetic data with database caching"""

    # Create experiment ID
    experiment_id = db.generate_experiment_id(dataset_name, method, params)

    # Check if experiment exists in database
    if db and db.experiment_exists(experiment_id):
        print(f"Loading cached results for {dataset_name}-{method} (ID: {experiment_id[:8]}...)")
        return db.get_experiment(experiment_id)

    print(f"Running new experiment for {dataset_name}-{method} (ID: {experiment_id[:8]}...)")

    # Load original data
    data_obj, schema = get_real_data(dataset_name)

    # Run fairness testing on original data
    if method == 'adf':
        results_df_origin, metrics_origin = adf_fairness_testing(
            data_obj,
            **params
        )
    elif method == 'aequitas':
        results_df_origin, metrics_origin = run_aequitas(
            discrimination_data=data_obj,
            **params
        )
    elif method == 'expga':
        results_df_origin, metrics_origin = run_expga(
            dataset=data_obj,
            **params
        )
    elif method == 'sg':
        results_df_origin, metrics_origin = run_sg(
            ge=data_obj,
            **params
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Extract discriminatory groups
    predefined_groups_origin = get_groups(results_df_origin, data_obj, schema)

    # Generate synthetic data
    data_obj_synth, schema = generate_from_real_data(
        dataset_name,
        predefined_groups=predefined_groups_origin,
        nb_groups=200
    )

    # Run fairness testing on synthetic data
    if method == 'adf':
        results_df_synth, metrics_synth = adf_fairness_testing(data_obj_synth, **params)
    elif method == 'aequitas':
        results_df_synth, metrics_synth = run_aequitas(discrimination_data=data_obj_synth, **params)
    elif method == 'expga':
        results_df_synth, metrics_synth = run_expga(dataset=data_obj_synth, **params)
    elif method == 'sg':
        results_df_synth, metrics_synth = run_sg(ge=data_obj_synth, **params)

    # Extract discriminatory groups from synthetic data
    predefined_groups_synth = get_groups(results_df_synth, data_obj, schema)

    # Compare discriminatory groups
    comparison_results = compare_discriminatory_groups(predefined_groups_origin, predefined_groups_synth)

    # Prepare results to return
    results = {
        'original_groups_count': len(predefined_groups_origin),
        'synthetic_groups_count': len(predefined_groups_synth)
    }

    # Add comparison metrics
    if isinstance(comparison_results, dict):
        for key, value in comparison_results.items():
            if key not in ['matched_groups', 'unmatched_groups']:
                results[key] = value

    # Save results to database
    if db:
        db.save_experiment(experiment_id, dataset_name, method, params, results)

    return results


def run_multiple_experiments(n_runs=5, methods=None, dataset_names=None, random_seeds=None,
                             db_path='fairness_experiments.db', **kwargs):
    """Run multiple experiments and calculate statistics with database caching"""
    if methods is None:
        methods = ['adf', 'aequitas', 'expga', 'sg']

    if dataset_names is None:
        dataset_names = ['adult']

    if random_seeds is None:
        random_seeds = list(range(42, 42 + n_runs))

    # Initialize database
    db = ExperimentDatabase(db_path)

    # Store all results
    all_results = {dataset: {method: [] for method in methods} for dataset in dataset_names}

    # Method-specific parameters
    method_params = {
        'adf': {'max_global': 5000, 'max_local': 2000, 'max_iter': 5, 'cluster_num': 50, 'max_runtime_seconds': 300},
        'aequitas': {'max_global': 100, 'max_local': 1000, 'step_size': 1.0, 'max_total_iterations': 1000},
        'expga': {'threshold': 0.5, 'threshold_rank': 0.5, 'max_global': 300, 'max_local': 100},
        'sg': {'cluster_num': 50, 'limit': 100, 'iter': 4}
    }

    # Run experiments for each dataset and method
    for dataset_name in dataset_names:
        print(f"\n{'=' * 70}")
        print(f"Running experiments on dataset: {dataset_name}")
        print(f"{'=' * 70}")

        for method in methods:
            print(f"\n{'-' * 50}")
            print(f"Running {method} method ({n_runs} runs)")
            print(f"{'-' * 50}")

            # Get method-specific parameters
            params = method_params.get(method, {})
            params.update(kwargs)  # Add any additional parameters

            # Run multiple times
            for i, seed in enumerate(tqdm(random_seeds, desc=f"{dataset_name}-{method} runs")):
                # Update seed for this run
                run_params = params.copy()
                run_params['random_seed'] = seed

                # Run or retrieve experiment
                result = run_experiment(dataset_name=dataset_name, method=method, db=db, params=run_params)
                all_results[dataset_name][method].append(result)

            # Update statistics in database
            db.update_statistics(dataset_name, method, all_results[dataset_name][method])

    # Get statistics from database
    stats_df = db.get_statistics(dataset_names, methods)

    # Close database connection
    db.close()

    return stats_df, all_results


def visualize_results(stats_df, output_prefix='fairness_testing'):
    """Visualize experiment results"""
    if stats_df.empty:
        print("No data to visualize")
        return

    # Extract metrics for visualization
    metrics = []
    for col in stats_df.columns:
        if col.endswith('_mean_mean') and not col.startswith('dataset') and not col.startswith('method'):
            metrics.append(col.replace('_mean_mean', ''))

    # Create plots for each dataset
    datasets = stats_df['dataset'].unique()

    for dataset in datasets:
        dataset_df = stats_df[stats_df['dataset'] == dataset]

        # Create multi-plot figure
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 6, 6))
        if n_metrics == 1:
            axes = [axes]  # Ensure axes is a list for single metric

        # Plot each metric
        for i, metric in enumerate(metrics):
            # Extract data for this metric
            methods = dataset_df['method'].values
            means = dataset_df[f'{metric}_mean_mean'].values
            stds = dataset_df[f'{metric}_mean_std'].values

            # Create bar chart
            bars = axes[i].bar(methods, means, yerr=stds, capsize=5)

            # Add styling
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.1,
                    f'{height:.2f}',
                    ha='center', va='bottom',
                    rotation=0
                )

        plt.suptitle(f'Fairness Testing Results for {dataset.title()} Dataset', fontsize=16)
        plt.tight_layout()

        # Save figure
        output_path = f'{output_prefix}_{dataset}.png'
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")

    # Show plots
    plt.show()


# Example usage
if __name__ == "__main__":
    # Set database path
    db_path = HERE.joinpath('experiments/baseline_exp/fairness_experiments.db')

    # Run multiple experiments
    n_runs = 3
    stats_df, all_results = run_multiple_experiments(
        n_runs=n_runs,
        methods=['adf'],
        dataset_names=['bank'],
        db_path=db_path
    )

    # Print statistics
    print("\nStatistics across multiple runs:")
    if not stats_df.empty:
        print(stats_df.to_string(index=False))
    else:
        print("No statistics available")

    # Save results to CSV
    if not stats_df.empty:
        stats_df.to_csv('fairness_testing_stats.csv', index=False)
        print("Saved statistics to fairness_testing_stats.csv")

    # Visualize results
    visualize_results(stats_df)

    # Example: load existing database and get statistics without running experiments
    print("\nLoading statistics from database without running experiments:")
    db = ExperimentDatabase(db_path)
    loaded_stats = db.get_statistics()
    if not loaded_stats.empty:
        print(loaded_stats.to_string(index=False))
    else:
        print("No statistics available in database")
    db.close()