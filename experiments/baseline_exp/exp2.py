from data_generator.main import get_real_data, generate_from_real_data
from data_generator.utils import plot_distribution_comparison
from methods.adf.main1 import adf_fairness_testing
from methods.utils import reformat_discrimination_results, convert_to_non_float_rows, compare_discriminatory_groups
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm


def get_groups(results_df_origin, data_obj, schema):
    non_float_df = convert_to_non_float_rows(results_df_origin, schema)
    predefined_groups_origin = reformat_discrimination_results(non_float_df, data_obj.dataframe)
    nb_elements = sum([el.group_size for el in predefined_groups_origin])
    return predefined_groups_origin, nb_elements

def run_experiment(dataset_name, original_params, synth_params):
    # Get original data
    data_obj, schema = get_real_data(dataset_name, use_cache=True)

    # Run fairness testing on original data
    results_df_origin, metrics_origin = adf_fairness_testing(
        data_obj, **original_params
    )
    predefined_groups_origin, nb_elements = get_groups(results_df_origin, data_obj, schema)

    # Generate and test synthetic data
    data_obj_synth, schema = generate_from_real_data(dataset_name, nb_groups=100, use_cache=True)
    results_df_synth, metrics_synth = adf_fairness_testing(
        data_obj_synth, **synth_params
    )
    predefined_groups_synth, nb_elements_synth = get_groups(results_df_synth, data_obj, schema)

    plot_distribution_comparison(schema, data_obj_synth)
    # Compare results
    comparison_results = compare_discriminatory_groups(predefined_groups_origin, predefined_groups_synth)

    return {
        'metrics_origin': metrics_origin,
        'metrics_synth': metrics_synth,
        'comparison_results': comparison_results,
        'nb_elements': nb_elements,
        'nb_elements_synth': nb_elements_synth
    }


def run_multiple_experiments(dataset_name, original_params, synth_params, num_runs=10):
    results_list = []

    for run in tqdm(range(num_runs), desc=f"Running experiments for {dataset_name}"):
        result = run_experiment(dataset_name, original_params, synth_params)
        metrics = {
            'dataset': dataset_name,
            'run': run,
            'coverage_ratio': result['comparison_results']['coverage_ratio'],
            'matched_groups': result['comparison_results']['total_groups_matched'],
            'total_groups': result['comparison_results']['total_original_groups'],
            'matched_size': result['comparison_results']['total_matched_size'],
            'total_size': result['comparison_results']['total_original_size']
        }
        results_list.append(metrics)

    return pd.DataFrame(results_list)


def plot_experiment_results(results_df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Coverage ratio boxplot
    sns.boxplot(data=results_df, x='dataset', y='coverage_ratio', ax=axes[0, 0])
    axes[0, 0].set_title('Coverage Ratio Distribution')

    # Matched groups vs total groups
    for dataset in results_df['dataset'].unique():
        dataset_data = results_df[results_df['dataset'] == dataset]
        axes[0, 1].scatter(dataset_data['total_groups'], dataset_data['matched_groups'],
                           label=dataset, alpha=0.6)
    axes[0, 1].plot([0, results_df['total_groups'].max()], [0, results_df['total_groups'].max()],
                    'k--', alpha=0.3)
    axes[0, 1].set_title('Matched vs Total Groups')
    axes[0, 1].legend()

    # Size comparison
    results_df.groupby('dataset')[['matched_size', 'total_size']].mean().plot(
        kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Average Matched vs Total Size')

    # Run variation
    for dataset in results_df['dataset'].unique():
        dataset_data = results_df[results_df['dataset'] == dataset]
        axes[1, 1].plot(dataset_data['run'], dataset_data['coverage_ratio'],
                        marker='o', label=dataset)
    axes[1, 1].set_title('Coverage Ratio by Run')
    axes[1, 1].legend()

    plt.tight_layout()
    return fig

from path import HERE
import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect(HERE.joinpath('experiments/baseline_exp/experiments.db'))
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset TEXT,
            method_name TEXT,
            run INTEGER,
            coverage_ratio REAL,
            matched_groups INTEGER,
            total_groups INTEGER,
            matched_size INTEGER,
            total_size INTEGER,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    return conn

def save_experiment(conn, dataset, method_name, run, result):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO experiments (
            dataset, method_name, run, coverage_ratio, matched_groups,
            total_groups, matched_size, total_size, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        dataset,
        method_name,
        run,
        result['comparison_results']['coverage_ratio'],
        result['comparison_results']['total_groups_matched'],
        result['comparison_results']['total_original_groups'],
        result['comparison_results']['total_matched_size'],
        result['comparison_results']['total_original_size'],
        str(datetime.now())
    ))
    conn.commit()


#%% Run experiments
datasets = ['bank']
method_name = 'adf'
conn = init_db()
all_results = []

original_params = {'max_global': 5000, 'max_local': 2000, 'max_iter': 5, 'cluster_num': 50, 'max_runtime_seconds': 400}
synth_params = {'max_global': 6000, 'max_local': 1000, 'max_iter': 10, 'cluster_num': 50, 'max_runtime_seconds': 600}

for dataset in datasets:
    for run in tqdm(range(1), desc=f"Running experiments for {dataset}"):
        result = run_experiment(dataset, original_params, synth_params)
        save_experiment(conn, dataset, method_name, run, result)

# results_df = pd.concat(all_results)

