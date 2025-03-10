from typing import Dict, Any, List
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from data_generator.main import get_real_data, generate_from_real_data
from methods.adf.main1 import adf_fairness_testing
from methods.aequitas.algo import run_aequitas
from methods.exp_ga.algo import run_expga
from methods.sg.main import run_sg
from methods.utils import reformat_discrimination_results, convert_to_non_float_rows, compare_discriminatory_groups


def run_single_experiment(
        dataset_name: str,
        algorithm: str,
        real_params: Dict[str, Any],
        synth_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run a single experiment for a given dataset and algorithm with different params for real and synthetic data
    """
    # Get real data
    data_obj, schema = get_real_data(dataset_name)

    # Run algorithm on real data
    if algorithm == 'adf':
        results_df_origin, metrics_origin = adf_fairness_testing(
            data_obj, **real_params
        )
    elif algorithm == 'aequitas':
        results_df_origin, metrics_origin = run_aequitas(
            discrimination_data=data_obj, **real_params
        )
    elif algorithm == 'expga':
        results_df_origin, metrics_origin = run_expga(
            dataset=data_obj, **real_params
        )
    elif algorithm == 'sg':
        results_df_origin, metrics_origin = run_sg(
            ge=data_obj, **real_params
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Get discriminatory groups from real data
    non_float_df = convert_to_non_float_rows(results_df_origin, schema)
    predefined_groups_origin = reformat_discrimination_results(non_float_df, data_obj.dataframe)

    if not predefined_groups_origin:
        return {
            'dataset': dataset_name,
            'algorithm': algorithm,
            'error': 'No discriminatory groups found in original data'
        }

    # Generate synthetic data
    try:
        data_obj_synth, schema = generate_from_real_data(
            dataset_name,
            predefined_groups=predefined_groups_origin
        )
    except Exception as e:
        return {
            'dataset': dataset_name,
            'algorithm': algorithm,
            'error': f'Error generating synthetic data: {str(e)}'
        }

    # Run algorithm on synthetic data with different parameters
    if algorithm == 'adf':
        results_df_synth, metrics_synth = adf_fairness_testing(
            data_obj_synth, **synth_params
        )
    elif algorithm == 'aequitas':
        results_df_synth, metrics_synth = run_aequitas(
            discrimination_data=data_obj_synth, **synth_params
        )
    elif algorithm == 'expga':
        results_df_synth, metrics_synth = run_expga(
            dataset=data_obj_synth, **synth_params
        )
    elif algorithm == 'sg':
        results_df_synth, metrics_synth = run_sg(
            ge=data_obj_synth, **synth_params
        )

    # Get discriminatory groups from synthetic data
    predefined_groups_synth = reformat_discrimination_results(
        convert_to_non_float_rows(results_df_synth, schema),
        data_obj.dataframe
    )

    # Compare results
    comparison_results = compare_discriminatory_groups(
        predefined_groups_origin,
        predefined_groups_synth
    )

    return {
        'dataset': dataset_name,
        'algorithm': algorithm,
        'original_groups': len(predefined_groups_origin),
        'synthetic_groups': len(predefined_groups_synth),
        'matched_groups': comparison_results['total_groups_matched'],
        'coverage_ratio': comparison_results['coverage_ratio'],
        'total_matched_size': comparison_results['total_matched_size'],
        'total_original_size': comparison_results['total_original_size']
    }


def run_all_experiments(
        datasets: List[str] = ['adult', 'credit', 'bank'],
        algorithms_config: Dict[str, Dict[str, Dict[str, Any]]] = None,
        num_repeats: int = 3
) -> pd.DataFrame:
    """
    Run experiments for all combinations of datasets and algorithms
    """
    if algorithms_config is None:
        algorithms_config = {
            'adf': {
                'real': {
                    'max_global': 5000,
                    'max_local': 2000,
                    'max_iter': 10,
                    'cluster_num': 50,
                    'random_seed': 42
                },
                'synth': {
                    'max_global': 7000,
                    'max_local': 2000,
                    'max_iter': 30,
                    'cluster_num': 50,
                    'random_seed': 42
                }
            },
            'aequitas': {
                'real': {
                    'model_type': 'rf',
                    'max_global': 100,
                    'max_local': 1000,
                    'step_size': 1.0,
                    'random_seed': 42,
                    'max_total_iterations': 1000
                },
                'synth': {
                    'model_type': 'rf',
                    'max_global': 100,
                    'max_local': 1000,
                    'step_size': 1.0,
                    'random_seed': 42,
                    'max_total_iterations': 1000
                }
            },
            'expga': {
                'real': {
                    'threshold': 0.5,
                    'threshold_rank': 0.5,
                    'max_global': 300,
                    'max_local': 100
                },
                'synth': {
                    'threshold': 0.5,
                    'threshold_rank': 0.5,
                    'max_global': 2000,
                    'max_local': 100
                }
            },
            'sg': {
                'real': {
                    'model_type': 'rf',
                    'cluster_num': 50,
                    'limit': 100,
                    'iter': 4
                },
                'synth': {
                    'model_type': 'rf',
                    'cluster_num': 50,
                    'limit': 100,
                    'iter': 6
                }
            }
        }

    results = []

    # Create experiment combinations
    experiments = [
        (dataset, algo, config['real'], config['synth'])
        for dataset in datasets
        for algo, config in algorithms_config.items()
        for _ in range(num_repeats)
    ]

    # Run experiments with progress bar
    for dataset, algo, real_params, synth_params in tqdm(experiments, desc="Running experiments"):
        try:
            result = run_single_experiment(dataset, algo, real_params, synth_params)
            result['timestamp'] = datetime.now().isoformat()
            results.append(result)
        except Exception as e:
            results.append({
                'dataset': dataset,
                'algorithm': algo,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_results.to_csv(f'experiment_results_{timestamp}.csv', index=False)

    return df_results


def create_summary_dataframe(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a clean summary DataFrame with all relevant metrics
    """
    # Remove rows with errors
    df_clean = results_df.dropna(subset=['coverage_ratio'])

    # Calculate metrics for each dataset-algorithm combination
    summary = df_clean.groupby(['dataset', 'algorithm']).agg({
        'coverage_ratio': ['mean', 'std'],
        'matched_groups': ['mean', 'std'],
        'original_groups': 'mean',
        'synthetic_groups': 'mean'
    })

    # Flatten column names
    summary.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0]
                      for col in summary.columns]

    # Reset index for better readability
    summary = summary.reset_index()

    # Round numeric columns
    numeric_cols = summary.select_dtypes(include=['float64']).columns
    summary[numeric_cols] = summary[numeric_cols].round(3)

    # Rename columns for clarity
    summary = summary.rename(columns={
        'coverage_ratio_mean': 'avg_coverage',
        'coverage_ratio_std': 'coverage_std',
        'matched_groups_mean': 'avg_matched_groups',
        'matched_groups_std': 'matched_groups_std',
        'original_groups_mean': 'avg_original_groups',
        'synthetic_groups_mean': 'avg_synthetic_groups'
    })

    return summary


# Example usage:
# if __name__ == "__main__":
#%% Define different configurations for real and synthetic data
algorithms_config = {
    'adf': {
        'real': {
            'max_global': 5000,
            'max_local': 1000,
            'max_iter': 5,
            'cluster_num': 50,
            'random_seed': 42
        },
        'synth': {
            'max_global': 5000,
            'max_local': 1000,
            'max_iter': 20,
            'cluster_num': 50,
            'random_seed': 42
        }
    },
    # 'aequitas': {
    #     'real': {
    #         'model_type': 'rf',
    #         'max_global': 100,
    #         'max_local': 1000,
    #         'step_size': 1.0,
    #         'random_seed': 42,
    #         'max_total_iterations': 1000
    #     },
    #     'synth': {
    #         'model_type': 'rf',
    #         'max_global': 100,
    #         'max_local': 1000,
    #         'step_size': 1.0,
    #         'random_seed': 42,
    #         'max_total_iterations': 1000
    #     }
    # }
}

# Run experiments with custom configurations
results_df = run_all_experiments(
    datasets=['adult'],
    algorithms_config=algorithms_config,
    num_repeats=1
)

# Create and display summary
summary_df = create_summary_dataframe(results_df)
print("\nSummary of Results:")
print(summary_df.to_string(index=False))