import pandas as pd
import sqlite3
import json
from pathlib import Path
from data_generator.main import get_real_data
from experiments.baseline_exp.reproduce_paper.reference_exp import ref_data
from methods.adf.main import run_adf
from methods.sg.main import run_sg
from methods.exp_ga.algo import run_expga
from methods.aequitas.algo import run_aequitas
import time
from typing import Dict, List
from collections import defaultdict

from path import HERE

DB_PATH = Path(HERE.joinpath("experiments/baseline_exp/exp.db"))


def run_experiment_for_model(model_type: str, dataset_name: str, sensitive_feature: str, completed_experiments: set,
                             ref_data: pd.DataFrame, conn) -> List[Dict]:
    """Run experiment for a specific model and dataset combination."""
    print(f"\nRunning experiment for {model_type} on {dataset_name} dataset with {sensitive_feature} feature")

    conv_data_name = {'adult': 'Census', 'credit': 'Credit', 'bank': 'Bank'}

    ref_keys = ref_data[(ref_data['dataset'] == conv_data_name[dataset_name]) & (ref_data['model'] == model_type)]

    # Get the dataset
    data_obj, schema = get_real_data(dataset_name, use_cache=True)

    # Run ExpGA if not already done
    if (model_type, dataset_name, sensitive_feature, 'ExpGA') not in completed_experiments:
        print("Running ExpGA...")

        keys = ref_keys[ref_keys['algorithm'] == 'ExpGA']

        print("MAX TSN", 20000)

        start_time = time.time()
        expga_results, expga_metrics = run_expga(data=data_obj, threshold_rank=0.5, max_global=5000, max_local=2000,
                                                 model_type=model_type.lower(), max_tsn=20000, one_attr_at_a_time=True,
                                                 threshold=0.5, time_limit=1000)
        execution_time = time.time() - start_time
        print("EXPGA METRICS", expga_metrics)

        # Save complete metrics for each protected attribute
        for attr, attr_metrics in expga_metrics['dsn_by_attr_value'].items():
            if attr != 'total':
                result_data = {
                    'Model': model_type,
                    'Dataset': dataset_name,
                    'total_TSN': expga_metrics['TSN'],
                    'total_DSN': expga_metrics['DSN'],
                    'total_DSS': expga_metrics['DSS'],
                    'total_SUR': expga_metrics['SUR'],
                    'Feature': attr,
                    'Algorithm': 'ExpGA',
                    'TSN': attr_metrics['TSN'],
                    'DSN': attr_metrics['DSN'],
                    'DSS': expga_metrics['DSS'],
                    'SUR': attr_metrics['SUR'],
                    'execution_time': execution_time,
                    'total_time': expga_metrics.get('total_time'),
                    'time_limit_reached': expga_metrics.get('time_limit_reached', False),
                    'max_tsn_reached': expga_metrics.get('max_tsn_reached', False),
                    'dsn_by_attr_value': attr_metrics
                }
                save_experiment_result(conn, result_data)

    # Run SG if not already done
    if (model_type, dataset_name, sensitive_feature, 'SG') not in completed_experiments:
        print("Running SG...")

        keys = ref_keys[ref_keys['algorithm'] == 'SG']

        print("MAX TSN", 20000)

        start_time = time.time()
        sg_results, sg_metrics = run_sg(data=data_obj, model_type=model_type.lower(), cluster_num=50, max_tsn=20000,
                                        one_attr_at_a_time=True)
        execution_time = time.time() - start_time
        print("SG METRICS", sg_metrics)

        # Save complete metrics for each protected attribute
        for attr, attr_metrics in sg_metrics['dsn_by_attr_value'].items():
            if attr != 'total':
                result_data = {
                    'Model': model_type,
                    'Dataset': dataset_name,
                    'total_TSN': sg_metrics['TSN'],
                    'total_DSN': sg_metrics['DSN'],
                    'total_DSS': sg_metrics['DSS'],
                    'total_SUR': sg_metrics['SUR'],
                    'Feature': attr,
                    'Algorithm': 'SG',
                    'TSN': attr_metrics['TSN'],
                    'DSN': attr_metrics['DSN'],
                    'DSS': sg_metrics['DSS'],
                    'SUR': attr_metrics['SUR'],
                    'execution_time': execution_time,
                    'total_time': sg_metrics.get('total_time'),
                    'time_limit_reached': sg_metrics.get('time_limit_reached', False),
                    'max_tsn_reached': sg_metrics.get('max_tsn_reached', False),
                    'dsn_by_attr_value': attr_metrics
                }
                save_experiment_result(conn, result_data)

    # Run Aequitas if not already done
    if (model_type, dataset_name, sensitive_feature, 'Aequitas') not in completed_experiments:
        print("Running Aequitas...")

        keys = ref_keys[ref_keys['algorithm'] == 'Aequitas']

        start_time = time.time()
        print("MAX TSN", 20000)
        aequitas_results, aequitas_metrics = run_aequitas(data=data_obj, model_type=model_type.lower(), max_global=5000,
                                                          max_local=2000, step_size=1.0, random_seed=42, max_tsn=20000,
                                                          one_attr_at_a_time=True)
        execution_time = time.time() - start_time
        print("AEQUITAS METRICS", aequitas_metrics)

        # Save complete metrics for each protected attribute
        for attr, attr_metrics in aequitas_metrics['dsn_by_attr_value'].items():
            if attr != 'total':
                result_data = {
                    'Model': model_type,
                    'Dataset': dataset_name,
                    'total_TSN': aequitas_metrics['TSN'],
                    'total_DSN': aequitas_metrics['DSN'],
                    'total_DSS': aequitas_metrics['DSS'],
                    'total_SUR': aequitas_metrics['SUR'],
                    'Feature': attr,
                    'Algorithm': 'Aequitas',
                    'TSN': attr_metrics['TSN'],
                    'DSN': attr_metrics['DSN'],
                    'DSS': aequitas_metrics['DSS'],
                    'SUR': attr_metrics['SUR'],
                    'execution_time': execution_time,
                    'total_time': aequitas_metrics.get('total_time'),
                    'time_limit_reached': aequitas_metrics.get('time_limit_reached', False),
                    'max_tsn_reached': aequitas_metrics.get('max_tsn_reached', False),
                    'dsn_by_attr_value': attr_metrics
                }
                save_experiment_result(conn, result_data)

    # Run ADF (only for MLP) if not already done
    if model_type.lower() == 'mlp' and (
            model_type, dataset_name, sensitive_feature, 'ADF') not in completed_experiments:
        print("Running ADF...")

        keys = ref_keys[ref_keys['algorithm'] == 'ADF']
        print("MAX TSN", 20000)

        start_time = time.time()
        adf_results, adf_metrics = run_adf(data_obj, max_global=2000, max_local=100, cluster_num=100,
                                           random_seed=42, max_runtime_seconds=1000, max_tsn=9000,
                                           one_attr_at_a_time=True)
        execution_time = time.time() - start_time
        print("ADF METRICS", adf_metrics)

        # Save complete metrics for each protected attribute
        for attr, attr_metrics in adf_metrics['dsn_by_attr_value'].items():
            if attr != 'total':
                result_data = {
                    'Model': model_type,
                    'Dataset': dataset_name,
                    'total_TSN': adf_metrics['TSN'],
                    'total_DSN': adf_metrics['DSN'],
                    'total_DSS': adf_metrics['DSS'],
                    'total_SUR': adf_metrics['SUR'],
                    'Feature': attr,
                    'Algorithm': 'ADF',
                    'TSN': attr_metrics['TSN'],
                    'DSN': attr_metrics['DSN'],
                    'DSS': adf_metrics['DSS'],
                    'SUR': attr_metrics['SUR'],
                    'execution_time': execution_time,
                    'total_time': adf_metrics.get('total_time'),
                    'time_limit_reached': adf_metrics.get('time_limit_reached', False),
                    'max_tsn_reached': adf_metrics.get('max_tsn_reached', False),
                    'dsn_by_attr_value': attr_metrics
                }
                save_experiment_result(conn, result_data)


def setup_results_table(conn):
    """Create the results table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS paper_reproduction_results2 (
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
        total_TSN INTEGER,
        total_DSN INTEGER,
        total_DSS REAL,
        total_SUR REAL,
        execution_time REAL,
        total_time REAL,
        time_limit_reached BOOLEAN,
        max_tsn_reached BOOLEAN,
        dsn_by_attr_value TEXT  -- JSON string for attribute value details
    )
    ''')
    conn.commit()


def get_completed_experiments(conn) -> set:
    """Get a set of already completed experiments."""
    cursor = conn.cursor()
    cursor.execute('''
    SELECT model, dataset, feature, algorithm 
    FROM paper_reproduction_results2
    ''')
    return {(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()}


def save_experiment_result(conn, result: dict):
    """Save a single experiment result to the database."""
    cursor = conn.cursor()

    # Convert dsn_by_attr_value dictionary to JSON string if it exists
    dsn_by_attr_value_json = None
    if 'dsn_by_attr_value' in result:
        dsn_by_attr_value_json = json.dumps(result['dsn_by_attr_value'])

    cursor.execute('''
    INSERT INTO paper_reproduction_results2 
    (model, dataset, feature, algorithm, TSN, DSN, DSS, SUR, 
     total_TSN, total_DSN, total_DSS, total_SUR,
     execution_time, total_time, time_limit_reached, max_tsn_reached, dsn_by_attr_value)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        result['Model'],
        result['Dataset'],
        result['Feature'],
        result['Algorithm'],
        result['TSN'],
        result['DSN'],
        result['DSS'],
        result['SUR'],
        result.get('total_TSN', result['TSN']),
        result.get('total_DSN', result['DSN']),
        result.get('total_DSS', result['DSS']),
        result.get('total_SUR', result['SUR']),
        result['execution_time'],
        result.get('total_time'),
        result.get('time_limit_reached', False),
        result.get('max_tsn_reached', False),
        dsn_by_attr_value_json
    ))
    conn.commit()


def analyze_discrimination_results(results_df: pd.DataFrame, dataset) -> Dict:
    """
    Analyze discrimination results to get proportions for each protected attribute.

    Args:
        results_df: DataFrame containing discriminatory samples
        dataset: DiscriminationData object containing feature information

    Returns:
        Dictionary containing metrics for each protected attribute and total metrics
    """
    # Initialize metrics with default values
    metrics = {
        'total': {'TSN': 0, 'DSN': 0, 'DSS': 0, 'SUR': 0}
    }

    # Initialize dsn_by_attr_value for tracking attribute-specific metrics
    dsn_by_attr_value = {'total': {}}

    # Get all protected attributes and their indices
    protected_attrs = dataset.protected_attributes

    for attr in protected_attrs:
        metrics[attr] = {'TSN': 0, 'DSN': 0}
        dsn_by_attr_value[attr] = defaultdict(int)

    # Get pairs of samples from the results DataFrame
    for idx in range(0, len(results_df), 2):
        if idx + 1 >= len(results_df):
            break

        # Increment total metrics
        metrics['total']['TSN'] += 1

        # Get original and modified samples
        original = results_df.iloc[idx]
        modified = results_df.iloc[idx + 1]

        # Check if these samples form a discriminatory pair
        if original['prediction'] != modified['prediction']:
            metrics['total']['DSN'] += 1

        # Extract feature values
        original_features = original[dataset.feature_names].values
        modified_features = modified[dataset.feature_names].values

        # Find which protected attributes were changed
        for attr in protected_attrs:
            idx = dataset.sensitive_indices_dict[attr]
            if original_features[idx] != modified_features[idx]:
                metrics[attr]['TSN'] += 1

                # Check if prediction changed (discriminatory)
                if original['prediction'] != modified['prediction']:
                    metrics[attr]['DSN'] += 1

                    # Track DSN by attribute value
                    dsn_by_attr_value[attr][int(original_features[idx])] += 1
                    dsn_by_attr_value[attr][int(modified_features[idx])] += 1

    # Calculate final metrics for each attribute
    final_metrics = {'dsn_by_attr_value': {}}

    # Calculate total metrics
    total_tsn = metrics['total']['TSN']
    total_dsn = metrics['total']['DSN']

    final_metrics['TSN'] = total_tsn
    final_metrics['DSN'] = total_dsn
    final_metrics['DSS'] = round(total_dsn / total_tsn, 2) if total_tsn > 0 else 0
    final_metrics['SUR'] = round((total_dsn / total_tsn) * 100, 2) if total_tsn > 0 else 0

    # Calculate metrics for each attribute
    for attr, counts in metrics.items():
        if attr != 'total':
            tsn = counts['TSN']
            dsn = counts['DSN']
            dss = round(dsn / tsn, 2) if tsn > 0 else 0
            sur = round((dsn / tsn) * 100, 2) if tsn > 0 else 0

            final_metrics['dsn_by_attr_value'][attr] = {
                'TSN': tsn,
                'DSN': dsn,
                'DSS': dss,
                'SUR': sur,
                'attr_values': dsn_by_attr_value[attr]
            }

    # Add total to dsn_by_attr_value for consistency
    final_metrics['dsn_by_attr_value']['total'] = {
        'TSN': total_tsn,
        'DSN': total_dsn,
        'DSS': final_metrics['DSS'],
        'SUR': final_metrics['SUR']
    }

    return final_metrics


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
    grouped = results_df.groupby(['model', 'dataset', 'feature'])

    for (model, dataset, feature), group in grouped:
        row_data = {
            'model': model,
            'dataset': dataset,
            'feature': feature
        }

        # Add metrics for each algorithm
        for algorithm in ['ExpGA', 'SG', 'ADF', 'Aequitas']:
            algo_data = group[group['algorithm'] == algorithm]
            if not algo_data.empty:
                row_data.update({
                    f'{algorithm}_TSN': algo_data.iloc[0]['TSN'],
                    f'{algorithm}_DSN': algo_data.iloc[0]['DSN'],
                    f'{algorithm}_DSS': algo_data.iloc[0]['DSS'],
                    f'{algorithm}_SUR': algo_data.iloc[0]['SUR'],
                    f'{algorithm}_total_TSN': algo_data.iloc[0]['total_TSN'],
                    f'{algorithm}_total_DSN': algo_data.iloc[0]['total_DSN'],
                    f'{algorithm}_total_DSS': algo_data.iloc[0]['total_DSS'],
                    f'{algorithm}_total_SUR': algo_data.iloc[0]['total_SUR'],
                    f'{algorithm}_time_limit_reached': algo_data.iloc[0]['time_limit_reached'],
                    f'{algorithm}_max_tsn_reached': algo_data.iloc[0]['max_tsn_reached'],
                    f'{algorithm}_total_time': algo_data.iloc[0]['total_time']
                })
            else:
                # Fill with None if algorithm data is not available
                row_data.update({
                    f'{algorithm}_TSN': None,
                    f'{algorithm}_DSN': None,
                    f'{algorithm}_DSS': None,
                    f'{algorithm}_SUR': None,
                    f'{algorithm}_total_TSN': None,
                    f'{algorithm}_total_DSN': None,
                    f'{algorithm}_total_DSS': None,
                    f'{algorithm}_total_SUR': None,
                    f'{algorithm}_time_limit_reached': None,
                    f'{algorithm}_max_tsn_reached': None,
                    f'{algorithm}_total_time': None
                })

        organized_data.append(row_data)

    # Create DataFrame and sort by Model, Dataset, Feature
    result_df = pd.DataFrame(organized_data)
    result_df = result_df.sort_values(['model', 'dataset', 'feature'])

    return result_df


def retrieve_detailed_results(conn):
    """
    Retrieve and parse detailed results from the database, including the dsn_by_attr_value data.

    Args:
        conn: SQLite connection

    Returns:
        DataFrame with parsed detailed results
    """
    cursor = conn.cursor()
    cursor.execute('''
    SELECT model, dataset, feature, algorithm, 
           TSN, DSN, DSS, SUR,
           total_TSN, total_DSN, total_DSS, total_SUR,
           execution_time, total_time, time_limit_reached, max_tsn_reached, dsn_by_attr_value
    FROM paper_reproduction_results2
    ''')

    results = []
    for row in cursor.fetchall():
        result_dict = {
            'model': row[0],
            'dataset': row[1],
            'feature': row[2],
            'algorithm': row[3],
            'TSN': row[4],
            'DSN': row[5],
            'DSS': row[6],
            'SUR': row[7],
            'total_TSN': row[8],
            'total_DSN': row[9],
            'total_DSS': row[10],
            'total_SUR': row[11],
            'execution_time': row[12],
            'total_time': row[13],
            'time_limit_reached': row[14],
            'max_tsn_reached': row[15],
        }

        # Parse the JSON dsn_by_attr_value data if it exists
        if row[16]:
            try:
                dsn_by_attr_value = json.loads(row[16])
                result_dict['dsn_by_attr_value'] = dsn_by_attr_value
            except json.JSONDecodeError:
                result_dict['dsn_by_attr_value'] = None
        else:
            result_dict['dsn_by_attr_value'] = None

        results.append(result_dict)

    return pd.DataFrame(results)


def run_all_experiments():
    """Run experiments for all combinations of models, datasets, and features."""
    experiments = [
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
        # ('SVM', 'adult', 'gender'),
        # ('SVM', 'adult', 'age'),
        # ('SVM', 'adult', 'race'),
        # ('SVM', 'credit', 'gender'),
        # ('SVM', 'credit', 'age'),
        # ('SVM', 'bank', 'age'),
    ]

    # Connect to database and setup table
    conn = sqlite3.connect(DB_PATH)
    setup_results_table(conn)

    # Get already completed experiments
    completed_experiments = get_completed_experiments(conn)
    print(f"Found {len(completed_experiments)} completed experiments")

    # Run experiments that haven't been completed
    for model_type, dataset_name, feature in experiments:
        ref_experiments = ref_data()
        run_experiment_for_model(model_type, dataset_name, feature,
                                 completed_experiments, ref_experiments, conn)


if __name__ == "__main__":
    print("Starting experiments...")
    run_all_experiments()

    # Optionally, retrieve and display the detailed results
    conn = sqlite3.connect(DB_PATH)
    detailed_results = retrieve_detailed_results(conn)
    print("\nDetailed Experiment Results:")
    print(detailed_results.head())

    # Generate organized results for analysis
    organized_results = organize_results_by_algorithm(detailed_results)
    print("\nOrganized Results by Algorithm:")
    print(organized_results.head())

    conn.close()
