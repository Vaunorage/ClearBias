from data_generator.main import DiscriminationData
import pandas as pd
from typing import List, Dict
from sqlalchemy import create_engine
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm.auto import tqdm
import numpy as np


def calculate_correlation_matrices(db_path: str, experiment_ids: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Create connection pool for thread safety
    engine = create_engine(f'sqlite:///{db_path}', pool_size=len(experiment_ids), max_overflow=0)

    def sort_couple_key(couple_key: str) -> str:
        """Vectorized couple key sorting"""
        indiv1, indiv2 = couple_key.split('-')
        return f"{min(indiv1, indiv2)}-{max(indiv1, indiv2)}"

    def prepare_result_combinations(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        # Vectorized operations for result combinations
        df['sorted_couple_key'] = df['couple_key'].apply(sort_couple_key)

        # Group by couple key and process all at once
        grouped = df.groupby('sorted_couple_key')
        valid_couples = grouped.filter(lambda x: len(x) == 2)

        # Prepare feature combinations using vectorized operations
        combinations_list = []
        for _, couple_data in valid_couples.groupby('sorted_couple_key'):
            if couple_data.shape[0] != 2:
                continue

            is_part_of_group = couple_data['is_couple_part_of_a_group'].iloc[0] != '0'
            features = couple_data[feature_cols].values

            combination = {
                'couple_key': couple_data['sorted_couple_key'].iloc[0],
                'is_part_of_group': is_part_of_group
            }

            for idx, feat in enumerate(feature_cols):
                combination[f'{feat}_1'] = features[0][idx]
                combination[f'{feat}_2'] = features[1][idx]

            combinations_list.append(combination)

        return pd.DataFrame(combinations_list)

    def compute_distances(synth_features: np.ndarray, result_features: np.ndarray,
                          batch_size: int = 5000) -> np.ndarray:
        """Optimized distance computation with larger batches and vectorized operations"""
        n_synth = len(synth_features)
        n_result = len(result_features)
        distance_matrix = np.zeros((n_synth, n_result))

        for i in range(0, n_synth, batch_size):
            batch_end = min(i + batch_size, n_synth)
            batch = synth_features[i:batch_end]

            # Vectorized distance calculation
            distances = np.sqrt(((batch[:, np.newaxis, :] - result_features) ** 2).sum(axis=2))
            distance_matrix[i:batch_end] = distances

        return distance_matrix

    def calculate_distance_stats(synth_df: pd.DataFrame, distance_matrix: np.ndarray) -> pd.DataFrame:
        """Vectorized distance statistics calculation"""
        unique_groups = synth_df['group_key'].unique()
        group_indices = [synth_df['group_key'] == group for group in unique_groups]

        stats = pd.DataFrame({
            'group_key': unique_groups,
            'min_distance': [np.min(distance_matrix[idx]) for idx in group_indices],
            'max_distance': [np.max(distance_matrix[idx]) for idx in group_indices],
            'mean_distance': [np.mean(distance_matrix[idx]) for idx in group_indices],
            'median_distance': [np.median(distance_matrix[idx]) for idx in group_indices]
        }).set_index('group_key')

        return stats

    def process_experiment(experiment_id: str, engine, pbar: tqdm) -> tuple[str, pd.DataFrame, pd.DataFrame]:
        """Process a single experiment with progress tracking"""
        try:
            pbar.set_description(f"Processing {experiment_id}")

            # Load data efficiently
            with engine.connect() as conn:
                df_synth = pd.read_sql_query(
                    f"SELECT full_data FROM synthetic_data WHERE experiment_id='{experiment_id}'",
                    conn
                )
                if df_synth.empty:
                    pbar.update(1)
                    return experiment_id, pd.DataFrame(), pd.DataFrame()

                df_synth = pd.DataFrame(json.loads(df_synth['full_data'].iloc[0]))
                df_synth['calculated_epistemic'] = df_synth['epis_uncertainty']
                df_synth['calculated_aleatoric'] = df_synth['alea_uncertainty']
                df_synth['experiment_id'] = experiment_id

                df_result = pd.read_sql_query(
                    f"""SELECT * FROM augmented_results ar
                LEFT JOIN main.analysis_metadata am ON am.analysis_id=ar.analysis_id
                WHERE experiment_id='{experiment_id}'""", conn)

                if df_result.empty:
                    pbar.update(1)
                    return experiment_id, pd.DataFrame(), pd.DataFrame()

            # Process data
            feature_cols = [col for col in df_synth.columns if 'Attr' in col]
            synth_combinations = DiscriminationData.generate_individual_synth_combinations(df_synth)

            df_result_data = pd.DataFrame(list(df_result['data'].apply(json.loads)))
            df_result = pd.concat([df_result.reset_index(drop=True), df_result_data.reset_index(drop=True)], axis=1)
            result_combinations = prepare_result_combinations(df_result, feature_cols)

            # Calculate distances
            comb_feature_cols = [col for col in synth_combinations.columns if 'Attr' in col]
            synth_features = synth_combinations[comb_feature_cols].to_numpy()
            result_features = result_combinations[comb_feature_cols].to_numpy()

            distance_matrix = compute_distances(synth_features, result_features)
            distance_stats = calculate_distance_stats(synth_combinations, distance_matrix)

            # Merge and calculate correlations
            merged_df = df_synth.join(distance_stats, on='group_key', how='left')
            calc_cols = [col for col in merged_df.columns if col.startswith("calculated")]
            dist_cols = [col for col in merged_df.columns if col.endswith("distance")]

            correlation_data = merged_df[['group_key'] + calc_cols + dist_cols].drop_duplicates()
            correlations = np.corrcoef(correlation_data[calc_cols].T, correlation_data[dist_cols].T)

            # Extract relevant correlation coefficients
            n_calc = len(calc_cols)
            correlation_matrix = correlations[:n_calc, n_calc:]

            pbar.update(1)
            return experiment_id, pd.DataFrame(correlation_matrix, index=calc_cols, columns=dist_cols), correlation_data

        except Exception as e:
            print(f"Error processing experiment {experiment_id}: {str(e)}")
            pbar.update(1)
            return experiment_id, pd.DataFrame(), pd.DataFrame()

    # Create progress bar
    pbar = tqdm(total=len(experiment_ids), desc="Overall Progress")

    # Process experiments in parallel with progress tracking
    with ThreadPoolExecutor(max_workers=min(len(experiment_ids), 4)) as executor:
        results = list(executor.map(partial(process_experiment, engine=engine, pbar=pbar), experiment_ids))

    pbar.close()

    # Combine results
    matrices = {exp_id: matrix for exp_id, matrix, merged_dfs in results if not matrix.empty}
    metrics_df = pd.concat(matrices, axis=0, names=['experiment_id', 'calculated_metric'])

    merged_dfs = {exp_id: matrix for exp_id, matrix, merged_dfs in results if not matrix.empty}
    distances_df = pd.concat(merged_dfs, axis=0, names=['experiment_id', 'distances'])

    return metrics_df, distances_df

# Example usage:
from path import HERE

DB_PATH = HERE.joinpath('experiments/discrimination_detection_results7.db')
conn = create_engine(f'sqlite:///{DB_PATH}')

experiment_ids = pd.read_sql_query(f"SELECT experiment_id FROM synthetic_data", conn)

experiment_ids = experiment_ids['experiment_id'].to_list()[:1]

correlation_matrices_df, distances_df = calculate_correlation_matrices(db_path=DB_PATH, experiment_ids=experiment_ids)

print('dddd')