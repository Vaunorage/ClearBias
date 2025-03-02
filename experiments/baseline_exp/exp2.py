import pandas as pd
import numpy as np
from data_generator.main import get_real_data, generate_from_real_data
from methods.adf.main1 import adf_fairness_testing
from methods.utils import compare_discriminatory_groups,  get_groups


def run_experiment(seed, dataset_name):
    # Get real data
    data_obj, schema = get_real_data(dataset_name, use_cache=True)

    # Run fairness testing on original data
    results_df_origin, metrics_origin = adf_fairness_testing(
        data_obj,
        max_global=5000,
        max_local=2000,
        max_iter=1000,
        cluster_num=100,
        random_seed=seed,
        max_runtime_seconds=400
    )

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
    results_df_synth, metrics_synth = adf_fairness_testing(
        data_obj_synth,
        max_global=10000,
        max_local=2000,
        max_iter=2000,
        cluster_num=100,
        random_seed=seed,
        max_runtime_seconds=600
    )

    # Get discriminatory groups from synthetic data
    predefined_groups_synth, nb_elements_synth = get_groups(results_df_synth, data_obj, schema)

    # Compare discriminatory groups
    comparison_results = compare_discriminatory_groups(predefined_groups_origin, predefined_groups_synth)

    return {
        'seed': seed,
        'coverage_ratio': comparison_results['coverage_ratio'],
        'total_groups_matched': comparison_results['total_groups_matched'],
        'total_original_groups': comparison_results['total_original_groups'],
        'total_matched_size': comparison_results['total_matched_size'],
        'total_original_size': comparison_results['total_original_size']
    }


def main(datasets=['bank'], num_experiments=10, random_seeds=None):
    # Set random seeds if not provided
    if random_seeds is None:
        random_seeds = np.random.randint(0, 10000, size=num_experiments)

    # Run experiments and collect results
    results = []
    for dataset in datasets:
        for i, seed in enumerate(random_seeds):
            print(f"Running experiment on dataset {dataset}, run {i + 1}/{num_experiments} with seed {seed}")
            experiment_result = run_experiment(seed, dataset)
            experiment_result['dataset'] = dataset
            results.append(experiment_result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate statistics per dataset
    stats_list = []

    for dataset in results_df['dataset'].unique():
        dataset_results = results_df[results_df['dataset'] == dataset]

        stats = {
            'dataset': dataset,
            'coverage_ratio': {
                'mean': dataset_results['coverage_ratio'].mean(),
                'std': dataset_results['coverage_ratio'].std(),
                'min': dataset_results['coverage_ratio'].min(),
                'max': dataset_results['coverage_ratio'].max()
            },
            'total_groups_matched': {
                'mean': dataset_results['total_groups_matched'].mean(),
                'std': dataset_results['total_groups_matched'].std()
            },
            'total_original_groups': {
                'mean': dataset_results['total_original_groups'].mean(),
                'std': dataset_results['total_original_groups'].std()
            }
        }

        stats_list.append({
            'Dataset': dataset,
            'Metric': 'Coverage Ratio',
            'Mean': stats['coverage_ratio']['mean'],
            'Std Dev': stats['coverage_ratio']['std'],
            'Min': stats['coverage_ratio']['min'],
            'Max': stats['coverage_ratio']['max']
        })

        stats_list.append({
            'Dataset': dataset,
            'Metric': 'Total Groups Matched',
            'Mean': stats['total_groups_matched']['mean'],
            'Std Dev': stats['total_groups_matched']['std'],
            'Min': None,
            'Max': None
        })

        stats_list.append({
            'Dataset': dataset,
            'Metric': 'Total Original Groups',
            'Mean': stats['total_original_groups']['mean'],
            'Std Dev': stats['total_original_groups']['std'],
            'Min': None,
            'Max': None
        })

    stats_df = pd.DataFrame(stats_list)

    return results_df, stats_df

#%% Set number of experiments
num_experiments = 1

# Set fixed random seeds for reproducibility
random_seeds = [42]

# List of datasets to test
datasets = ['bank', 'adult']

# Run experiments
results_df, stats_df = main(datasets, num_experiments, random_seeds)

# Print results
print("\nExperiment Results:")
print(results_df)

print("\nStatistics:")
print(stats_df)

# Save results to CSV
results_df.to_csv("experiment_results.csv", index=False)
stats_df.to_csv("experiment_statistics.csv", index=False)

# Save results by dataset
for dataset in datasets:
    dataset_results = results_df[results_df['dataset'] == dataset]
    dataset_stats = stats_df[stats_df['Dataset'] == dataset]

    dataset_results.to_csv(f"experiment_results_{dataset}.csv", index=False)
    dataset_stats.to_csv(f"experiment_statistics_{dataset}.csv", index=False)

print("\nResults saved to CSV files.")