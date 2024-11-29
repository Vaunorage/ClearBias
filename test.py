from data_generator.main import DiscriminationData
from path import HERE
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from itertools import combinations
import json
import matplotlib.pyplot as plt
from typing import List, Tuple


def calculate_distances(synth_df: pd.DataFrame, result_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Calculate distances between synthetic and result features."""
    # Ensure both dataframes have the required feature columns
    valid_features = []
    for col in feature_cols:
        if col in synth_df.columns and col in result_df.columns:
            valid_features.append(col)

    if not valid_features:
        raise ValueError(f"No matching feature columns found between synthetic and result data")

    # Use only valid features
    synth_features = synth_df[valid_features].to_numpy()
    result_features = result_df[valid_features].to_numpy()

    distance_matrix = np.linalg.norm(
        synth_features[:, np.newaxis, :] - result_features[np.newaxis, :, :],
        axis=2
    )

    synth_index_map = {idx: i for i, idx in enumerate(synth_df.index)}
    result_index_map = {idx: i for i, idx in enumerate(result_df.index)}

    distance_data = []
    for group_key in synth_df["group_key"].unique():
        group_mask = synth_df["group_key"] == group_key
        synth_indices = synth_df[group_mask].index

        for synth_idx in synth_indices:
            matrix_synth_idx = synth_index_map[synth_idx]

            for result_idx in result_df.index:
                matrix_result_idx = result_index_map[result_idx]
                distance_data.append({
                    "group_key": group_key,
                    "synth_df_couple_key": synth_df.at[synth_idx, "couple_key"],
                    "result_df_couple_key": result_df.at[result_idx, "couple_key"],
                    "distance": distance_matrix[matrix_synth_idx, matrix_result_idx],
                    "experiment_id": synth_df['experiment_id'].iloc[0]
                })

    return pd.DataFrame(distance_data)


def prepare_result_combinations(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Prepare combinations of features for analysis."""
    # Validate feature columns exist in dataframe
    valid_features = [col for col in feature_cols if col in df.columns]
    if not valid_features:
        raise ValueError("No valid feature columns found in data")

    all_combinations = []

    for couple_key in df['couple_key'].unique():
        indivs = couple_key.split('-')
        sorted_indivs = sort_two_strings(indivs[0], indivs[1])
        sorted_couple_key = f"{sorted_indivs[0]}-{sorted_indivs[1]}"

        indiv1_data = df[df['indv_key'] == sorted_indivs[0]]
        indiv2_data = df[df['indv_key'] == sorted_indivs[1]]
        couple_data = pd.concat([indiv1_data, indiv2_data])

        if couple_data.shape[0] == 2:
            continue

        is_part_of_group = couple_data['is_couple_part_of_a_group'].iloc[0] != '0'
        unique_individuals = couple_data[valid_features].drop_duplicates().values
        pairs = list(combinations(range(len(unique_individuals)), 2))

        for i, j in pairs:
            combination = {
                'couple_key': sorted_couple_key,
                'is_part_of_group': is_part_of_group
            }

            indiv1_features = unique_individuals[i]
            indiv2_features = unique_individuals[j]

            for idx, feat in enumerate(valid_features):
                combination[f'{feat}_1'] = indiv1_features[idx]
                combination[f'{feat}_2'] = indiv2_features[idx]

            all_combinations.append(combination)

    return pd.DataFrame(all_combinations)


def load_experiment_data(conn, experiment_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load synthetic and results data for a given experiment."""
    # Load synthetic data
    df_synth = pd.read_sql_query(
        f"SELECT experiment_id, full_data FROM synthetic_data where experiment_id='{experiment_id}'",
        conn
    )
    df_synth = pd.DataFrame(json.loads(df_synth['full_data'].iloc[0]))
    df_synth['experiment_id'] = experiment_id

    # Load results data
    df_result = pd.read_sql_query(
        f"""SELECT * FROM augmented_results ar
        left join main.analysis_metadata am on am.analysis_id=ar.analysis_id
        where experiment_id='{experiment_id}'""",
        conn
    )
    df_result_data = pd.DataFrame(list(df_result['data'].apply(json.loads)))
    df_result = pd.concat([df_result.reset_index(drop=True), df_result_data.reset_index(drop=True)], axis=1)

    # Get all potential feature columns
    synth_features = [col for col in df_synth.columns if 'Attr' in col]
    result_features = [col for col in df_result.columns if 'Attr' in col]

    # Find common features
    feature_cols = list(set(synth_features) & set(result_features))

    return df_synth, df_result, feature_cols


def sort_two_strings(str1: str, str2: str) -> Tuple[str, str]:
    """Sort two strings lexicographically."""
    return (str1, str2) if str1 <= str2 else (str2, str1)


def prepare_result_combinations(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Prepare combinations of features for analysis."""
    # Validate feature columns exist in dataframe
    valid_features = [col for col in feature_cols if col in df.columns]
    if not valid_features:
        raise ValueError("No valid feature columns found in data")

    all_combinations = []

    for couple_key in df['couple_key'].unique():
        indivs = couple_key.split('-')
        sorted_indivs = sort_two_strings(indivs[0], indivs[1])
        sorted_couple_key = f"{sorted_indivs[0]}-{sorted_indivs[1]}"

        indiv1_data = df[df['indv_key'] == sorted_indivs[0]]
        indiv2_data = df[df['indv_key'] == sorted_indivs[1]]
        couple_data = pd.concat([indiv1_data, indiv2_data])

        if couple_data.shape[0] == 2:
            continue

        is_part_of_group = couple_data['is_couple_part_of_a_group'].iloc[0] != '0'
        unique_individuals = couple_data[valid_features].drop_duplicates().values
        pairs = list(combinations(range(len(unique_individuals)), 2))

        for i, j in pairs:
            combination = {
                'couple_key': sorted_couple_key,
                'is_part_of_group': is_part_of_group
            }

            indiv1_features = unique_individuals[i]
            indiv2_features = unique_individuals[j]

            for idx, feat in enumerate(valid_features):
                combination[f'{feat}_1'] = indiv1_features[idx]
                combination[f'{feat}_2'] = indiv2_features[idx]

            all_combinations.append(combination)

    return pd.DataFrame(all_combinations)


def calculate_distances(synth_df: pd.DataFrame, result_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Calculate distances between synthetic and result features."""
    # Ensure both dataframes have the required feature columns
    valid_features = []
    for col in feature_cols:
        if col in synth_df.columns and col in result_df.columns:
            valid_features.append(col)

    if not valid_features:
        raise ValueError(f"No matching feature columns found between synthetic and result data")

    # Use only valid features
    synth_features = synth_df[valid_features].to_numpy()
    result_features = result_df[valid_features].to_numpy()

    distance_matrix = np.linalg.norm(
        synth_features[:, np.newaxis, :] - result_features[np.newaxis, :, :],
        axis=2
    )

    synth_index_map = {idx: i for i, idx in enumerate(synth_df.index)}
    result_index_map = {idx: i for i, idx in enumerate(result_df.index)}

    distance_data = []
    for group_key in synth_df["group_key"].unique():
        group_mask = synth_df["group_key"] == group_key
        synth_indices = synth_df[group_mask].index

        for synth_idx in synth_indices:
            matrix_synth_idx = synth_index_map[synth_idx]

            for result_idx in result_df.index:
                matrix_result_idx = result_index_map[result_idx]
                distance_data.append({
                    "group_key": group_key,
                    "synth_df_couple_key": synth_df.at[synth_idx, "couple_key"],
                    "result_df_couple_key": result_df.at[result_idx, "couple_key"],
                    "distance": distance_matrix[matrix_synth_idx, matrix_result_idx],
                    "experiment_id": synth_df['experiment_id'].iloc[0]
                })

    return pd.DataFrame(distance_data)


def process_experiment(conn, experiment_id: str) -> pd.DataFrame:
    """Process a single experiment and return the correlation results."""
    # Load data
    df_synth, df_result, feature_cols = load_experiment_data(conn, experiment_id)

    if not feature_cols:
        return None

    # Generate combinations
    synth_combinations_df = DiscriminationData.generate_individual_synth_combinations(df_synth) #Assuming DiscriminationData.generate_individual_synth_combinations is not available
    result_combinations_df = prepare_result_combinations(df_result, feature_cols)

    # Calculate distances
    distances_df = calculate_distances(synth_combinations_df, result_combinations_df, feature_cols)

    # Add is_part_of_group if missing
    if "is_part_of_group" not in result_combinations_df.columns:
        result_combinations_df["is_part_of_group"] = False

    # Merge with result data
    distances_df = distances_df.merge(
        result_combinations_df[["couple_key", "is_part_of_group"]].rename(
            columns={"couple_key": "result_df_couple_key"}
        ),
        on="result_df_couple_key",
        how="left"
    )

    # Calculate aggregated statistics
    agg_stats = distances_df.groupby(["group_key", "is_part_of_group", "experiment_id"])["distance"].agg(
        min_distance="min",
        max_distance="max",
        median_distance="median",
        mean_distance="mean"
    ).reset_index()

    # Get calculated columns from synthetic data
    calculated_columns = [col for col in df_synth.columns if col.startswith('calculated')]
    aggregated_metrics = df_synth.groupby('group_key')[calculated_columns].agg(['mean', 'min', 'max', 'median'])
    aggregated_metrics.columns = ['_'.join(col).strip() for col in aggregated_metrics.columns.values]
    aggregated_metrics = aggregated_metrics.reset_index()

    # Merge all data
    final_df = agg_stats.merge(aggregated_metrics, on='group_key', how='left')
    return final_df


def analyze_all_experiments(db_path: str) -> None:
    """Analyze all experiments in the database and create visualization."""
    conn = create_engine(f'sqlite:///{db_path}')

    # Get all experiment IDs
    experiment_ids = pd.read_sql_query("SELECT DISTINCT experiment_id FROM synthetic_data", conn)['experiment_id']

    # Process all experiments
    all_results = []
    for exp_id in experiment_ids:
        exp_results = process_experiment(conn, exp_id)
        if exp_results:
            all_results.append(exp_results)

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Calculate correlations
    distance_columns = ["min_distance", "max_distance", "mean_distance", "median_distance"]
    calculated_columns = [col for col in combined_results.columns if col.startswith("calculated")]
    correlations = combined_results[calculated_columns + distance_columns].corr()
    mean_correlations = correlations.loc[
        list(filter(lambda x: 'mean' in x, correlations.index)),
        ['mean_distance', 'median_distance']
    ]

    # Create visualization
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(mean_correlations.index))
    width = 0.35

    plt.barh(y_pos - width / 2, mean_correlations['mean_distance'], width,
             label='Mean Distance', color='#2196F3', alpha=0.7)
    plt.barh(y_pos + width / 2, mean_correlations['median_distance'], width,
             label='Median Distance', color='#FF5722', alpha=0.7)

    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Correlation Value', fontsize=12)
    plt.ylabel('Metrics', fontsize=12)
    plt.title('Distance Correlation Comparison Across All Experiments', fontsize=14, pad=20)
    plt.yticks(y_pos, mean_correlations.index, fontsize=10)
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


DB_PATH = HERE.joinpath('experiments/discrimination_detection_results5.db')
analyze_all_experiments(DB_PATH)