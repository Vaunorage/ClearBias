import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from typing import Dict, Tuple
import time
import logging

# Import the functions we need from the provided code
from data_generator.main import generate_data
from methods.adf.main import run_adf
from methods.utils import train_sklearn_model, check_for_error_condition

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('categorical_comparison')


def random_search_fairness_testing(
        discrimination_data,
        max_samples: int = 2000,
        random_seed: int = 42,
        max_runtime_seconds: int = 3600,
        max_tsn: int = None,
        one_attr_at_a_time: bool = False
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    A simple random search baseline for fairness testing.
    Randomly changes attribute values within the valid ranges.

    Args:
        discrimination_data: Data object containing dataset and metadata
        max_samples: Maximum number of samples to test
        random_seed: Random seed for reproducibility
        max_runtime_seconds: Maximum runtime in seconds
        max_tsn: Maximum number of test samples to generate
        one_attr_at_a_time: Whether to vary only one attribute at a time

    Returns:
        Results DataFrame and metrics dictionary
    """
    # Set random seed
    np.random.seed(random_seed)
    random.seed(random_seed)

    logger.info(f"Starting random search fairness testing")

    start_time = time.time()

    dsn_by_attr_value = {e: {'TSN': 0, 'DSN': 0} for e in discrimination_data.protected_attributes}
    dsn_by_attr_value['total'] = 0

    data = discrimination_data.training_dataframe.copy()

    logger.info(f"Dataset shape: {data.shape}")
    logger.info(f"Protected attributes: {discrimination_data.protected_attributes}")

    model, X_train, X_test, y_train, y_test, feature_names = train_sklearn_model(
        data=data,
        model_type='mlp',
        target_col=discrimination_data.outcome_column,
        sensitive_attrs=list(discrimination_data.protected_attributes),
        random_state=random_seed
    )

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    logger.info(f"Model training score: {train_score:.4f}")
    logger.info(f"Model test score: {test_score:.4f}")

    X = discrimination_data.xdf.to_numpy()

    tot_inputs = set()
    all_discriminations = set()
    disc_inputs = set()
    total_all_inputs = []

    # Get the input bounds for each attribute
    input_bounds = discrimination_data.input_bounds

    def should_terminate() -> bool:
        current_runtime = time.time() - start_time
        time_limit_exceeded = current_runtime > max_runtime_seconds
        tsn_threshold_reached = max_tsn is not None and len(tot_inputs) >= max_tsn

        return time_limit_exceeded or tsn_threshold_reached

    # Random search
    for i in range(max_samples):
        if should_terminate():
            break

        # Randomly select a starting instance from the dataset
        idx = np.random.randint(0, len(X))
        instance = X[idx].copy()

        # For categorical attributes, randomly change values within bounds
        for j in range(len(instance)):
            # Only modify with some probability to avoid changing everything
            if np.random.random() < 0.3:  # 30% chance to modify each attribute
                min_val, max_val = input_bounds[j]

                # For categorical attributes, only use integer values
                if discrimination_data.attr_columns[j] in discrimination_data.categorical_columns:
                    instance[j] = np.random.randint(min_val, max_val + 1)
                else:
                    # For continuous attributes, use uniform distribution
                    instance[j] = np.random.uniform(min_val, max_val)

        # Check if this instance exhibits discrimination
        result, result_df, max_discr, org_df, tested_inp = check_for_error_condition(
            logger=logger,
            model=model,
            instance=instance,
            dsn_by_attr_value=dsn_by_attr_value,
            discrimination_data=discrimination_data,
            tot_inputs=tot_inputs,
            all_discriminations=all_discriminations,
            total_all_inputs=total_all_inputs,
            one_attr_at_a_time=one_attr_at_a_time
        )

        if result:
            disc_inputs.add(tuple(instance.astype(float).tolist()))

    # Calculate final results
    end_time = time.time()
    total_time = end_time - start_time

    # Log final results
    tsn = len(tot_inputs)  # Total Sample Number
    dsn = len(all_discriminations)  # Discriminatory Sample Number
    sur = dsn / tsn if tsn > 0 else 0  # Success Rate
    dss = total_time / dsn if dsn > 0 else float('inf')  # Discriminatory Sample Search time

    for k, v in dsn_by_attr_value.items():
        if k != 'total':
            v['SUR'] = v['DSN'] / v['TSN'] if v['TSN'] > 0 else 0
            v['DSS'] = dss

    # Log final results
    logger.info("\nFinal Results:")
    logger.info(f"Total inputs tested: {tsn}")
    logger.info(f"Discriminatory inputs: {len(disc_inputs)}")
    logger.info(f"Total discriminatory pairs: {dsn}")
    logger.info(f"Success rate (SUR): {sur:.4f}")
    logger.info(f"Avg. search time per discriminatory sample (DSS): {dss:.4f} seconds")
    logger.info(f"Total time: {total_time:.2f} seconds")

    # Generate result dataframe
    res_df = []
    case_id = 0
    for org, org_res, counter_org, counter_org_res in all_discriminations:
        indv1 = pd.DataFrame([list(org)], columns=discrimination_data.attr_columns)
        indv2 = pd.DataFrame([list(counter_org)], columns=discrimination_data.attr_columns)

        indv_key1 = "|".join(str(x) for x in indv1[discrimination_data.attr_columns].iloc[0])
        indv_key2 = "|".join(str(x) for x in indv2[discrimination_data.attr_columns].iloc[0])

        # Add the additional columns
        indv1['indv_key'] = indv_key1
        indv1['outcome'] = org_res
        indv2['indv_key'] = indv_key2
        indv2['outcome'] = counter_org_res

        # Create couple_key
        couple_key = f"{indv_key1}-{indv_key2}"
        diff_outcome = abs(indv1['outcome'] - indv2['outcome'])

        df_res = pd.concat([indv1, indv2])
        df_res['couple_key'] = couple_key
        df_res['diff_outcome'] = diff_outcome
        df_res['case_id'] = case_id
        res_df.append(df_res)
        case_id += 1

    if len(res_df) != 0:
        res_df = pd.concat(res_df)
    else:
        res_df = pd.DataFrame([])

    # Add metrics to result dataframe
    res_df['TSN'] = tsn
    res_df['DSN'] = dsn
    res_df['SUR'] = sur
    res_df['DSS'] = dss

    metrics = {
        'TSN': tsn,
        'DSN': dsn,
        'SUR': sur,
        'DSS': dss,
        'total_time': total_time,
        'time_limit_reached': total_time >= max_runtime_seconds,
        'max_tsn_reached': max_tsn is not None and tsn >= max_tsn,
        'dsn_by_attr_value': dsn_by_attr_value
    }

    return res_df, metrics


def run_comparison_experiment():
    """
    Run a comparison experiment between ADF and random search baseline
    across datasets with varying levels of categorical attributes.
    """
    logger.info("Starting comparison experiment between ADF and Random Search")

    # Define experiment parameters
    max_categories_range = [2, 10, 30, 50]  # Number of categories per attribute
    prop_categorical_range = [0.2, 0.6, 1.0]  # Proportion of categorical attributes

    # Fixed parameters
    nb_attributes = 20
    nb_groups = 50
    prop_protected_attr = 0.2
    max_runtime_seconds = 300  # 5 minutes per experiment
    max_tsn = 3000  # Maximum test samples
    random_seeds = [42, 43]  # Multiple seeds for robustness

    # Results storage
    results = []

    # Setup experiment
    for max_categories in max_categories_range:
        for prop_categorical in prop_categorical_range:
            for seed in random_seeds:
                logger.info(f"Running experiment with max_categories={max_categories}, "
                            f"prop_categorical={prop_categorical}, seed={seed}")

                # Generate dataset with specified categorical attributes
                discrimination_data = generate_data(
                    nb_groups=nb_groups,
                    nb_attributes=nb_attributes,
                    min_number_of_classes=2,
                    max_number_of_classes=max_categories,
                    prop_protected_attr=prop_protected_attr,
                    min_group_size=10,
                    max_group_size=100,
                    use_cache=False,  # Don't use cache to ensure fresh data
                    categorical_outcome=True,
                    nb_categories_outcome=2  # Binary outcome for simplicity
                )

                # Ensure the specified proportion of attributes are categorical
                num_categorical = int(nb_attributes * prop_categorical)
                target_categorical_columns = discrimination_data.attr_columns[:num_categorical]
                discrimination_data.categorical_columns = list(
                    set(discrimination_data.categorical_columns).union(set(target_categorical_columns))
                )

                # Run ADF
                logger.info("Running ADF...")
                start_time = time.time()
                _, adf_metrics = run_adf(data=discrimination_data, max_global=200, max_local=200,
                                         cluster_num=20, random_seed=seed,
                                         max_runtime_seconds=max_runtime_seconds, max_tsn=max_tsn,
                                         step_size=1.0)
                adf_runtime = time.time() - start_time

                # Run Random Search
                logger.info("Running Random Search...")
                start_time = time.time()
                _, random_metrics = random_search_fairness_testing(
                    discrimination_data=discrimination_data,
                    max_samples=400,  # Similar number to ADF (max_global + max_local)
                    random_seed=seed,
                    max_runtime_seconds=max_runtime_seconds,
                    max_tsn=max_tsn
                )
                random_runtime = time.time() - start_time

                # Store results
                result = {
                    'max_categories': max_categories,
                    'prop_categorical': prop_categorical,
                    'random_seed': seed,
                    'adf_runtime': adf_runtime,
                    'random_runtime': random_runtime,
                    'adf_TSN': adf_metrics['TSN'],
                    'random_TSN': random_metrics['TSN'],
                    'adf_DSN': adf_metrics['DSN'],
                    'random_DSN': random_metrics['DSN'],
                    'adf_SUR': adf_metrics['SUR'],
                    'random_SUR': random_metrics['SUR'],
                    'adf_DSS': adf_metrics['DSS'],
                    'random_DSS': random_metrics['DSS'],
                    'categorical_attrs': sum(1 for col in discrimination_data.categorical_columns
                                             if col in discrimination_data.attr_columns),
                    'total_attrs': len(discrimination_data.attr_columns)
                }

                results.append(result)

                logger.info(f"ADF: TSN={adf_metrics['TSN']}, DSN={adf_metrics['DSN']}, "
                            f"SUR={adf_metrics['SUR']:.4f}")
                logger.info(f"Random: TSN={random_metrics['TSN']}, DSN={random_metrics['DSN']}, "
                            f"SUR={random_metrics['SUR']:.4f}")

    # Compile results into DataFrame
    results_df = pd.DataFrame(results)

    # Calculate the relative performance
    results_df['relative_SUR'] = results_df['adf_SUR'] / results_df['random_SUR'].replace(0, np.nan)
    results_df['relative_DSN'] = results_df['adf_DSN'] / results_df['random_DSN'].replace(0, np.nan)

    # Save results
    results_df.to_csv('adf_vs_random_comparison_results.csv', index=False)

    # Generate visualizations
    plot_comparison_results(results_df)

    return results_df


def plot_comparison_results(results_df):
    """
    Create visualizations to compare ADF and random search performance.

    Args:
        results_df: DataFrame containing experiment results
    """
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Success Rate (SUR) comparison
    ax = axes[0, 0]

    # Group by max_categories and prop_categorical and average across seeds
    grouped = results_df.groupby(['max_categories', 'prop_categorical']).agg({
        'adf_SUR': 'mean',
        'random_SUR': 'mean'
    }).reset_index()

    # Create x-axis labels
    x_labels = [f"{row['max_categories']}-{row['prop_categorical']}"
                for _, row in grouped.iterrows()]
    x = np.arange(len(x_labels))

    # Plot bars
    width = 0.35
    ax.bar(x - width / 2, grouped['adf_SUR'], width, label='ADF')
    ax.bar(x + width / 2, grouped['random_SUR'], width, label='Random Search')

    ax.set_xlabel('Max Categories - Proportion Categorical')
    ax.set_ylabel('Success Rate (SUR)')
    ax.set_title('ADF vs Random Search: Success Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Relative performance decline with increasing categories
    ax = axes[0, 1]

    for prop, group in results_df.groupby('prop_categorical'):
        rel_perf = group.groupby('max_categories')['relative_SUR'].mean()
        ax.plot(rel_perf.index, rel_perf.values, marker='o', label=f'{prop * 100}% Categorical')

    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Maximum Categories per Attribute')
    ax.set_ylabel('Relative Performance (ADF/Random SUR)')
    ax.set_title('ADF Performance Relative to Random Search')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Discriminatory samples found (DSN)
    ax = axes[1, 0]

    # Same grouping as above
    grouped_dsn = results_df.groupby(['max_categories', 'prop_categorical']).agg({
        'adf_DSN': 'mean',
        'random_DSN': 'mean'
    }).reset_index()

    ax.bar(x - width / 2, grouped_dsn['adf_DSN'], width, label='ADF')
    ax.bar(x + width / 2, grouped_dsn['random_DSN'], width, label='Random Search')

    ax.set_xlabel('Max Categories - Proportion Categorical')
    ax.set_ylabel('Discriminatory Samples (DSN)')
    ax.set_title('ADF vs Random Search: Discriminatory Samples Found')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Relative DSN performance
    ax = axes[1, 1]

    for prop, group in results_df.groupby('prop_categorical'):
        rel_dsn = group.groupby('max_categories')['relative_DSN'].mean()
        ax.plot(rel_dsn.index, rel_dsn.values, marker='o', label=f'{prop * 100}% Categorical')

    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Maximum Categories per Attribute')
    ax.set_ylabel('Relative Performance (ADF/Random DSN)')
    ax.set_title('ADF vs Random: Discriminatory Samples Found')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('adf_vs_random_comparison.png', dpi=300)
    plt.close()

    # Create heatmap of performance ratio (ADF SUR / Random SUR)
    plt.figure(figsize=(10, 8))

    # Average over random seeds
    heatmap_data = results_df.groupby(['max_categories', 'prop_categorical'])['relative_SUR'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='max_categories', columns='prop_categorical', values='relative_SUR')

    plt.imshow(heatmap_pivot, cmap='RdYlGn', aspect='auto', interpolation='nearest', vmin=0.5, vmax=1.5)
    plt.colorbar(label='Relative Performance (ADF/Random)')
    plt.xlabel('Proportion of Categorical Attributes')
    plt.ylabel('Maximum Categories per Attribute')
    plt.title('ADF Performance Relative to Random Search')

    # Set x and y tick labels
    plt.xticks(range(len(heatmap_pivot.columns)), [f'{x:.1f}' for x in heatmap_pivot.columns])
    plt.yticks(range(len(heatmap_pivot.index)), heatmap_pivot.index)

    # Add text annotations
    for i in range(len(heatmap_pivot.index)):
        for j in range(len(heatmap_pivot.columns)):
            plt.text(j, i, f'{heatmap_pivot.iloc[i, j]:.2f}',
                     ha='center', va='center',
                     color='black' if 0.7 < heatmap_pivot.iloc[i, j] < 1.3 else 'white')

    plt.tight_layout()
    plt.savefig('adf_vs_random_heatmap.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    comparison_results = run_comparison_experiment()
    print("Comparison experiment completed. Results saved to adf_vs_random_comparison_results.csv")