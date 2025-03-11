import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from data_generator.main import get_real_data
from methods.adf.main1 import adf_fairness_testing
from methods.sg.main import run_sg
from methods.exp_ga.algo import run_expga
from methods.aequitas.algo import run_aequitas
from methods.utils import get_groups
import time
from typing import Dict, List, Tuple
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
            'SUR': round((dsn / tsn) * 100, 2) if tsn > 0 else 0
        }
    
    return final_metrics

def run_experiment_for_model(model_type: str, dataset_name: str, sensitive_feature: str, completed_experiments: set) -> List[Dict]:
    """Run experiment for a specific model and dataset combination."""
    print(f"\nRunning experiment for {model_type} on {dataset_name} dataset with {sensitive_feature} feature")
    
    results = []
    
    # Get the dataset
    data_obj, schema = get_real_data(dataset_name, use_cache=True)
    
    # Run ExpGA if not already done
    if (model_type, dataset_name, sensitive_feature, 'ExpGA') not in completed_experiments:
        print("Running ExpGA...")
        start_time = time.time()
        expga_results, _ = run_expga(
            dataset=data_obj,
            model_type=model_type.lower(),
            threshold=0.5,
            threshold_rank=0.5,
            max_global=2000,
            max_local=100,
            max_tsn=50000,
            time_limit=10000
        )
        execution_time = time.time() - start_time
        
        # Analyze results for each protected attribute
        metrics = analyze_discrimination_results(expga_results, data_obj)
        
        # Add results for each protected attribute
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
        sg_results, _ = run_sg(
            ge=data_obj,
            model_type=model_type.lower(),
            cluster_num=50,
            max_tsn=50000,
            time_limit=10000
        )
        execution_time = time.time() - start_time
        
        # Analyze results for each protected attribute
        metrics = analyze_discrimination_results(sg_results, data_obj)
        
        # Add results for each protected attribute
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
        aequitas_results, _ = run_aequitas(
            discrimination_data=data_obj,
            model_type=model_type.lower(),
            max_global=10000,
            max_local=5000,
            step_size=1.0,
            random_seed=42,
            max_total_iterations=10000,
            max_tsn=50000,
            time_limit_seconds=10000
        )
        execution_time = time.time() - start_time
        
        # Analyze results for each protected attribute
        metrics = analyze_discrimination_results(aequitas_results, data_obj)
        
        # Add results for each protected attribute
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
    if model_type.lower() == 'mlp' and (model_type, dataset_name, sensitive_feature, 'ADF') not in completed_experiments:
        print("Running ADF...")
        start_time = time.time()
        adf_results, _ = adf_fairness_testing(
            data_obj,
            max_global=5000,
            max_local=2000,
            max_iter=1000,
            cluster_num=100,
            random_seed=42,
            max_tsn=50000,
            max_runtime_seconds=10000
        )
        execution_time = time.time() - start_time
        
        # Analyze results for each protected attribute
        metrics = analyze_discrimination_results(adf_results, data_obj)
        
        # Add results for each protected attribute
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
        ('SVM', 'adult', 'gender'),
        ('SVM', 'adult', 'age'),
        ('SVM', 'adult', 'race'),
        ('SVM', 'credit', 'gender'),
        ('SVM', 'credit', 'age'),
        ('SVM', 'bank', 'age')
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
        df.to_csv('experiment_results.csv', index=False)
        print("\nResults saved to experiment_results.csv")
        conn.close()
        return df

if __name__ == "__main__":
    print("Starting experiments...")
    results = run_all_experiments()
    print("\nExperiment Results:")
    print(results)
