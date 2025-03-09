import pandas as pd
import numpy as np
import sqlite3
from data_generator.main import get_real_data, generate_from_real_data
from methods.adf.main1 import adf_fairness_testing
from methods.aequitas.algo import run_aequitas
from methods.exp_ga.algo import run_expga
from methods.sg.main import run_sg
from methods.utils import compare_discriminatory_groups, get_groups

seed = 42

def run_experiment(method, dataset_name, conn):
    # Get real data
    data_obj, schema = get_real_data(dataset_name, use_cache=True)

    # Run fairness testing on original data
    if method == 'adf':
        results_df_origin, metrics_origin = adf_fairness_testing(
            data_obj,
            max_global=5000,
            max_local=2000,
            max_iter=1000,
            cluster_num=100,
            random_seed=seed,
            max_runtime_seconds=400
        )

    if method == 'aequitas':
        results_df_origin, metrics_origin = run_aequitas(discrimination_data=data_obj,
                                                       model_type='rf', max_global=10000,
                                                       max_local=5000, step_size=1.0, random_seed=42,
                                                       max_total_iterations=10000)
    if method == 'sg':
        results_df_origin, metrics_origin = run_sg(ge=data_obj,
                                                   model_type='rf', cluster_num=50, max_tsn=100, iter=6)
    if method == 'expga':
        results_df_origin, metrics_origin = run_expga(dataset=data_obj,
                                                    threshold=0.5, threshold_rank=0.5, max_global=2000, max_local=100)

    # Get discriminatory groups from original data
    predefined_groups_origin, nb_elements = get_groups(results_df_origin, data_obj, schema)

    # Generate synthetic data with predefined groups
    data_obj_synth, schema = generate_from_real_data(
        dataset_name,
        nb_groups=1,
        predefined_groups=predefined_groups_origin,
        use_cache=True,
        min_alea_uncertainty=0.0,
        max_alea_uncertainty=1.0,
        min_epis_uncertainty=0.0,
        max_epis_uncertainty=1.0
    )

    # Run fairness testing on synthetic data
    if method == 'adf':
        results_df_synth, metrics_synth = adf_fairness_testing(
            data_obj_synth,
            max_global=10000,
            max_local=2000,
            max_iter=2000,
            cluster_num=100,
            random_seed=seed,
            max_runtime_seconds=600
        )

    if method == 'aequitas':
        results_df_synth, metrics_synth = run_aequitas(discrimination_data=data_obj_synth,
                                                       model_type='rf', max_global=10000,
                                                       max_local=5000, step_size=1.0, random_seed=42,
                                                       max_total_iterations=10000)
    if method == 'sg':
        results_df_synth, metrics_synth = run_sg(ge=data_obj_synth,
                                                 model_type='rf', cluster_num=50, max_tsn=100, iter=6)

    if method == 'expga':
        results_df_synth, metrics_synth = run_expga(dataset=data_obj_synth,
                                                    threshold=0.5, threshold_rank=0.5, max_global=2000, max_local=100)

    # Get discriminatory groups from synthetic data
    predefined_groups_synth, nb_elements_synth = get_groups(results_df_synth, data_obj, schema)

    # Compare discriminatory groups
    comparison_results = compare_discriminatory_groups(predefined_groups_origin, predefined_groups_synth)

    result = {
        'seed': seed,
        'method': method,
        'dataset': dataset_name,
        'coverage_ratio': comparison_results['coverage_ratio'],
        'total_groups_matched': comparison_results['total_groups_matched'],
        'total_original_groups': comparison_results['total_original_groups'],
        'total_matched_size': comparison_results['total_matched_size'],
        'total_original_size': comparison_results['total_original_size']
    }

    # Save the result to SQLite
    save_experiment_result(conn, result)

    return result


def save_experiment_result(conn, result):
    """Save an experiment result to the SQLite database."""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO experiment_results 
        (seed, method, dataset, coverage_ratio, total_groups_matched, total_original_groups, 
         total_matched_size, total_original_size)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        result['seed'],
        result['method'],
        result['dataset'],
        result['coverage_ratio'],
        result['total_groups_matched'],
        result['total_original_groups'],
        result['total_matched_size'],
        result['total_original_size']
    ))
    conn.commit()


def setup_database(conn):
    """Initialize the SQLite database with the required table."""
    cursor = conn.cursor()

    # Create the experiment_results table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS experiment_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        seed INTEGER,
        method TEXT,
        dataset TEXT,
        coverage_ratio REAL,
        total_groups_matched INTEGER,
        total_original_groups INTEGER,
        total_matched_size INTEGER,
        total_original_size INTEGER
    )
    ''')

    conn.commit()
    return conn

from path import HERE

num_experiments = 10

# Set fixed random seeds for reproducibility
random_seeds = [42]

# List of datasets to test
datasets = ['credit']

methods = ['sg', 'expga']

conn = sqlite3.connect(HERE.joinpath('experiments/baseline_exp/exp.db'))
setup_database(conn)
# Run experiments
for i in range(num_experiments):
    for method in methods:
        for dataset in datasets:
            results_df = run_experiment(method, dataset, conn)

            # Print results
            print("\nExperiment Results:")
            print(results_df)