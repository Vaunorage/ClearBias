import pandas as pd
import sqlite3
from pathlib import Path
from data_generator.main import get_real_data
from methods.adf.main1 import adf_fairness_testing
from methods.sg.main import run_sg
from methods.exp_ga.algo import run_expga
from methods.aequitas.algo import run_aequitas
import time
from typing import Dict, List
from collections import defaultdict

from path import HERE

DB_PATH = Path(HERE.joinpath("experiments/baseline_exp/exp.db"))


def setup_results_table(conn):
    """Create the results table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS paper_reproduction_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        model TEXT,
        dataset TEXT,
        feature TEXT,
        algorithm TEXT,
        TSN INTEGER,
        DSN INTEGER,
        DSS REAL,
        SUR REAL,
        execution_time REAL,
        UNIQUE(model, dataset, feature, algorithm)
    )
    ''')
    conn.commit()


def get_completed_experiments(conn) -> set:
    """Get a set of already completed experiments."""
    cursor = conn.cursor()
    cursor.execute('''
    SELECT model, dataset, feature, algorithm 
    FROM paper_reproduction_results
    ''')
    return {(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()}


def save_experiment_result(conn, result: dict):
    """Save a single experiment result to the database."""
    cursor = conn.cursor()
    cursor.execute('''
    INSERT OR REPLACE INTO paper_reproduction_results 
    (model, dataset, feature, algorithm, TSN, DSN, DSS, SUR, execution_time)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        result['Model'],
        result['Dataset'],
        result['Feature'],
        result['Algorithm'],
        result['TSN'],
        result['DSN'],
        result['DSS'],
        result['SUR'],
        result['execution_time']
    ))
    conn.commit()


def analyze_discrimination_results(results_df: pd.DataFrame, dataset) -> Dict:
    """
    Analyze discrimination results to get proportions for each protected attribute.
    
    Args:
        results_df: DataFrame containing discriminatory samples
        dataset: DiscriminationData object containing feature information
    
    Returns:
        Dictionary containing metrics for each protected attribute
    """
    metrics = defaultdict(lambda: {'TSN': 0, 'DSN': 0})
    dsn_by_attr_value = defaultdict(lambda: defaultdict(int))

    # Get all protected attributes and their indices
    protected_attrs = dataset.protected_attributes

    # Get pairs of samples from the results DataFrame
    for idx in range(0, len(results_df), 2):
        if idx + 1 >= len(results_df):
            break

        # Get original and modified samples
        original = results_df.iloc[idx]
        modified = results_df.iloc[idx + 1]

        # Extract feature values
        original_features = original[dataset.feature_names].values
        modified_features = modified[dataset.feature_names].values

        # Find which protected attributes were changed
        for attr in protected_attrs:
            idx = dataset.sensitive_indices[attr]
            if original_features[idx] != modified_features[idx]:
                metrics[attr]['DSN'] += 1
                # Track DSN by attribute value
                dsn_by_attr_value[attr][int(original_features[idx])] += 1
                dsn_by_attr_value[attr][int(modified_features[idx])] += 1
            metrics[attr]['TSN'] += 1

    # Calculate final metrics for each attribute
    final_metrics = {}
    for attr, counts in metrics.items():
        tsn = counts['TSN']
        dsn = counts['DSN']
        final_metrics[attr] = {
            'TSN': tsn,
            'DSN': dsn,
            'DSS': round(dsn / tsn, 2) if tsn > 0 else 0,
            'SUR': round((dsn / tsn) * 100, 2) if tsn > 0 else 0,
            'dsn_by_attr_value': dsn_by_attr_value[attr]
        }

    return final_metrics


def run_experiment_for_model(model_type: str, dataset_name: str, sensitive_feature: str, completed_experiments: set) -> \
        List[Dict]:
    """Run experiment for a specific model and dataset combination."""
    print(f"\nRunning experiment for {model_type} on {dataset_name} dataset with {sensitive_feature} feature")

    results = []

    # Get the dataset
    data_obj, schema = get_real_data(dataset_name, use_cache=True)

    # Run ExpGA if not already done
    if (model_type, dataset_name, sensitive_feature, 'ExpGA') not in completed_experiments:
        print("Running ExpGA...")
        start_time = time.time()
        expga_results, expga_metrics = run_expga(
            dataset=data_obj,
            model_type=model_type.lower(),
            threshold=0.5,
            threshold_rank=0.5,
            max_global=5000,
            max_local=2000,
            max_tsn=3000,
            time_limit=10000
        )
        execution_time = time.time() - start_time
        print("EXPGA METRICS", expga_metrics)

        # Use metrics directly from the algorithm's dsn_by_attr_value if available
        if 'dsn_by_attr_value' in expga_metrics:
            for attr, attr_metrics in expga_metrics['dsn_by_attr_value'].items():
                results.append({
                    'Model': model_type,
                    'Dataset': dataset_name,
                    'Feature': attr,
                    'Algorithm': 'ExpGA',
                    'TSN': attr_metrics['tsn'],
                    'DSN': attr_metrics['dsn'],
                    'DSS': round(attr_metrics['dsn'] / attr_metrics['tsn'], 2) if attr_metrics['tsn'] > 0 else 0,
                    'SUR': round((attr_metrics['dsn'] / attr_metrics['tsn']) * 100, 2) if attr_metrics[
                                                                                              'tsn'] > 0 else 0,
                    'execution_time': execution_time
                })
        else:
            # Fallback to analyzing results manually
            metrics = analyze_discrimination_results(expga_results, data_obj)
            for attr, attr_metrics in metrics.items():
                results.append({
                    'Model': model_type,
                    'Dataset': dataset_name,
                    'Feature': attr,
                    'Algorithm': 'ExpGA',
                    'TSN': attr_metrics['TSN'],
                    'DSN': attr_metrics['DSN'],
                    'DSS': attr_metrics['DSS'],
                    'SUR': attr_metrics['SUR'],
                    'execution_time': execution_time
                })

    # Run SG if not already done
    if (model_type, dataset_name, sensitive_feature, 'SG') not in completed_experiments:
        print("Running SG...")
        start_time = time.time()
        sg_results, sg_metrics = run_sg(
            ge=data_obj,
            model_type=model_type.lower(),
            cluster_num=50,
            max_tsn=3000,
            time_limit=10000
        )
        execution_time = time.time() - start_time
        print("SG METRICS", sg_metrics)

        # Use metrics directly from the algorithm's dsn_by_attr_value if available
        if 'dsn_by_attr_value' in sg_metrics:
            for attr, attr_metrics in sg_metrics['dsn_by_attr_value'].items():
                results.append({
                    'Model': model_type,
                    'Dataset': dataset_name,
                    'Feature': attr,
                    'Algorithm': 'SG',
                    'TSN': attr_metrics['tsn'],
                    'DSN': attr_metrics['dsn'],
                    'DSS': round(attr_metrics['dsn'] / attr_metrics['tsn'], 2) if attr_metrics['tsn'] > 0 else 0,
                    'SUR': round((attr_metrics['dsn'] / attr_metrics['tsn']) * 100, 2) if attr_metrics[
                                                                                              'tsn'] > 0 else 0,
                    'execution_time': execution_time
                })
        else:
            # Fallback to analyzing results manually
            metrics = analyze_discrimination_results(sg_results, data_obj)
            for attr, attr_metrics in metrics.items():
                results.append({
                    'Model': model_type,
                    'Dataset': dataset_name,
                    'Feature': attr,
                    'Algorithm': 'SG',
                    'TSN': attr_metrics['TSN'],
                    'DSN': attr_metrics['DSN'],
                    'DSS': attr_metrics['DSS'],
                    'SUR': attr_metrics['SUR'],
                    'execution_time': execution_time
                })

    # Run Aequitas if not already done
    if (model_type, dataset_name, sensitive_feature, 'Aequitas') not in completed_experiments:
        print("Running Aequitas...")
        start_time = time.time()
        aequitas_results, aequitas_metrics = run_aequitas(
            discrimination_data=data_obj,
            model_type=model_type.lower(),
            max_global=5000,
            max_local=2000,
            step_size=1.0,
            random_seed=42,
            max_total_iterations=10000,
            max_tsn=3000,
            time_limit_seconds=10000
        )
        execution_time = time.time() - start_time
        print("AEQUITAS METRICS", aequitas_metrics)

        # Use metrics directly from the algorithm's dsn_by_attr_value if available
        if 'dsn_by_attr_value' in aequitas_metrics:
            for attr, attr_metrics in aequitas_metrics['dsn_by_attr_value'].items():
                results.append({
                    'Model': model_type,
                    'Dataset': dataset_name,
                    'Feature': attr,
                    'Algorithm': 'Aequitas',
                    'TSN': attr_metrics['tsn'],
                    'DSN': attr_metrics['dsn'],
                    'DSS': round(attr_metrics['dsn'] / attr_metrics['tsn'], 2) if attr_metrics['tsn'] > 0 else 0,
                    'SUR': round((attr_metrics['dsn'] / attr_metrics['tsn']) * 100, 2) if attr_metrics[
                                                                                              'tsn'] > 0 else 0,
                    'execution_time': execution_time
                })
        else:
            # Fallback to analyzing results manually
            metrics = analyze_discrimination_results(aequitas_results, data_obj)
            for attr, attr_metrics in metrics.items():
                results.append({
                    'Model': model_type,
                    'Dataset': dataset_name,
                    'Feature': attr,
                    'Algorithm': 'Aequitas',
                    'TSN': attr_metrics['TSN'],
                    'DSN': attr_metrics['DSN'],
                    'DSS': attr_metrics['DSS'],
                    'SUR': attr_metrics['SUR'],
                    'execution_time': execution_time
                })

    # Run ADF (only for MLP) if not already done
    if model_type.lower() == 'mlp' and (
            model_type, dataset_name, sensitive_feature, 'ADF') not in completed_experiments:
        print("Running ADF...")
        start_time = time.time()
        adf_results, adf_metrics = adf_fairness_testing(
            data_obj,
            max_global=10000,
            max_local=5000,
            max_iter=1000,
            cluster_num=100,
            random_seed=42,
            max_tsn=3000,
            max_runtime_seconds=10000
        )
        execution_time = time.time() - start_time
        print("ADF METRICS", adf_metrics)

        # Use metrics directly from the algorithm's dsn_by_attr_value if available
        if 'dsn_by_attr_value' in adf_metrics:
            for attr, attr_metrics in adf_metrics['dsn_by_attr_value'].items():
                results.append({
                    'Model': model_type,
                    'Dataset': dataset_name,
                    'Feature': attr,
                    'Algorithm': 'ADF',
                    'TSN': attr_metrics['tsn'],
                    'DSN': attr_metrics['dsn'],
                    'DSS': round(attr_metrics['dsn'] / attr_metrics['tsn'], 2) if attr_metrics['tsn'] > 0 else 0,
                    'SUR': round((attr_metrics['dsn'] / attr_metrics['tsn']) * 100, 2) if attr_metrics[
                                                                                              'tsn'] > 0 else 0,
                    'execution_time': execution_time
                })
        else:
            # Fallback to analyzing results manually
            metrics = analyze_discrimination_results(adf_results, data_obj)
            for attr, attr_metrics in metrics.items():
                results.append({
                    'Model': model_type,
                    'Dataset': dataset_name,
                    'Feature': attr,
                    'Algorithm': 'ADF',
                    'TSN': attr_metrics['TSN'],
                    'DSN': attr_metrics['DSN'],
                    'DSS': attr_metrics['DSS'],
                    'SUR': attr_metrics['SUR'],
                    'execution_time': execution_time
                })

    return results


def organize_results_by_algorithm(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Organize results into a DataFrame with algorithm-specific columns.
    
    Args:
        results_df: DataFrame with raw results
        
    Returns:
        DataFrame organized by model, dataset, and feature with algorithm-specific columns
    """
    # Initialize an empty list to store reorganized data
    organized_data = []

    # Group results by Model, Dataset, and Feature
    grouped = results_df.groupby(['Model', 'Dataset', 'Feature'])

    for (model, dataset, feature), group in grouped:
        row_data = {
            'Model': model,
            'Dataset': dataset,
            'Feature': feature
        }

        # Add metrics for each algorithm
        for algorithm in ['ExpGA', 'SG', 'ADF']:
            algo_data = group[group['Algorithm'] == algorithm]
            if not algo_data.empty:
                row_data.update({
                    f'{algorithm}_TSN': algo_data.iloc[0]['TSN'],
                    f'{algorithm}_DSN': algo_data.iloc[0]['DSN'],
                    f'{algorithm}_DSS': algo_data.iloc[0]['DSS'],
                    f'{algorithm}_SUR': algo_data.iloc[0]['SUR']
                })
            else:
                # Fill with None if algorithm data is not available
                row_data.update({
                    f'{algorithm}_TSN': None,
                    f'{algorithm}_DSN': None,
                    f'{algorithm}_DSS': None,
                    f'{algorithm}_SUR': None
                })

        organized_data.append(row_data)

    # Create DataFrame and sort by Model, Dataset, Feature
    result_df = pd.DataFrame(organized_data)
    result_df = result_df.sort_values(['Model', 'Dataset', 'Feature'])

    return result_df


def run_all_experiments():
    """Run experiments for all combinations of models, datasets, and features."""
    experiments = [
        ('SVM', 'adult', 'gender'),
        ('SVM', 'adult', 'age'),
        ('SVM', 'adult', 'race'),
        ('SVM', 'credit', 'gender'),
        ('SVM', 'credit', 'age'),
        ('SVM', 'bank', 'age'),
        ('MLP', 'adult', 'gender'),
        ('MLP', 'adult', 'age'),
        ('MLP', 'adult', 'race'),
        ('MLP', 'credit', 'gender'),
        ('MLP', 'credit', 'age'),
        ('MLP', 'bank', 'age'),
        ('RF', 'adult', 'gender'),
        ('RF', 'adult', 'age'),
        ('RF', 'adult', 'race'),
        ('RF', 'credit', 'gender'),
        ('RF', 'credit', 'age'),
        ('RF', 'bank', 'age'),
    ]

    # Connect to database and setup table
    conn = sqlite3.connect(DB_PATH)
    setup_results_table(conn)

    # Get already completed experiments
    completed_experiments = get_completed_experiments(conn)
    print(f"Found {len(completed_experiments)} completed experiments")

    try:
        # Run experiments that haven't been completed
        for model_type, dataset_name, feature in experiments:
            try:
                results = run_experiment_for_model(model_type, dataset_name, feature, completed_experiments)

                # Save results to database
                for result in results:
                    save_experiment_result(conn, result)

            except Exception as e:
                print(f"Error running experiment for {model_type} on {dataset_name} with {feature}: {str(e)}")
    finally:
        # Create final results DataFrame from database
        df = pd.read_sql_query("SELECT * FROM paper_reproduction_results", conn)

        # Organize results by algorithm
        organized_df = organize_results_by_algorithm(df)

        # Save both raw and organized results
        df.to_csv('experiment_results_raw.csv', index=False)
        organized_df.to_csv('experiment_results_organized.csv', index=False)

        print("\nResults saved to experiment_results_raw.csv and experiment_results_organized.csv")
        conn.close()
        return organized_df


if __name__ == "__main__":
    print("Starting experiments...")
    results = run_all_experiments()
    print("\nExperiment Results:")
    print(results)
