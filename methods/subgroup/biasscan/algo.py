from typing import TypedDict, Tuple
import time

import pandas as pd
from itertools import product

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from data_generator.old.main3 import DiscriminationData
from methods.subgroup.biasscan.mdss_detector import bias_scan
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from methods.utils import train_sklearn_model


class ResultRow(TypedDict, total=False):
    group_id: int
    outcome: int
    diff_outcome: int
    indv_key: str
    couple_key: str


ResultDF = DataFrame


def make_products_df(data: DiscriminationData, subsets, all_attributes, chunk_size=10000):
    """
    Create product DataFrame in memory-efficient chunks
    """

    def process_subset_chunk(subset, outcome, chunk_start, chunk_size):
        # Get all possible combinations
        product_lists = list(product(*subset.values()))
        total_products = len(product_lists)

        # Process only the current chunk
        chunk_end = min(chunk_start + chunk_size, total_products)
        chunk_products = product_lists[chunk_start:chunk_end]

        if not chunk_products:
            return pd.DataFrame(columns=all_attributes + [data.outcome_column])

        # Create DataFrame for this chunk
        chunk_df = pd.DataFrame(chunk_products, columns=subset.keys())

        # Add missing attributes
        for attr in all_attributes:
            if attr not in chunk_df.columns:
                chunk_df[attr] = None

        chunk_df[data.outcome_column] = outcome

        # Clean up the chunk
        chunk_df = chunk_df.drop_duplicates()
        chunk_df = chunk_df.dropna()

        return chunk_df

    all_dfs = []

    for subset, outcome in subsets:
        # Calculate total number of combinations
        total_combinations = 1
        for values in subset.values():
            total_combinations *= len(values)

        # Process in chunks
        for chunk_start in range(0, total_combinations, chunk_size):
            chunk_df = process_subset_chunk(subset, outcome, chunk_start, chunk_size)
            if not chunk_df.empty:
                all_dfs.append(chunk_df)

    if not all_dfs:
        return pd.DataFrame(columns=all_attributes + [data.outcome_column])

    return pd.concat(all_dfs, ignore_index=True)


def run_bias_scan(data, random_state=42, bias_scan_num_iters=50,
                  bias_scan_scoring='Poisson', bias_scan_favorable_value='high',
                  bias_scan_mode='ordinal', max_runtime_seconds=None, use_cache=True) -> Tuple[pd.DataFrame, dict]:
    start_time = time.time()

    # Split the data
    model, X_train, X_test, y_train, y_test, feature_names, metrics = train_sklearn_model(
        data=data.training_dataframe.copy(),
        model_type='rf',
        target_col=data.outcome_column,
        sensitive_attrs=list(data.protected_attributes),
        random_state=random_state,
        use_cache=use_cache
    )

    # Prepare data for bias scan
    observations = pd.Series(data.dataframe[data.outcome_column].to_numpy().squeeze())
    expectations = pd.Series(model.predict(data.dataframe[list(data.attributes)].values))

    # Perform bias scan
    _, _, subsets1 = bias_scan(data=data.dataframe[list(data.attributes)],
                               observations=observations,
                               expectations=expectations,
                               verbose=True,
                               num_iters=bias_scan_num_iters,
                               scoring=bias_scan_scoring,
                               favorable_value=bias_scan_favorable_value,
                               overpredicted=True,
                               mode=bias_scan_mode)

    _, _, subsets2 = bias_scan(data=data.dataframe[list(data.attributes)],
                               observations=observations,
                               expectations=expectations,
                               verbose=True,
                               num_iters=bias_scan_num_iters,
                               scoring=bias_scan_scoring,
                               favorable_value=bias_scan_favorable_value,
                               overpredicted=False,
                               mode=bias_scan_mode)

    product_dfs1 = make_products_df(data, subsets1, list(data.attributes))
    product_dfs2 = make_products_df(data, subsets2, list(data.attributes))

    result_df = pd.concat([product_dfs1, product_dfs2], ignore_index=True)
    result_df.drop_duplicates(inplace=True)
    
    # Check if result_df is empty
    if result_df.empty:
        # Return an empty DataFrame with the required columns
        empty_result = pd.DataFrame(columns=list(data.attributes) + [data.outcome_column, 'subgroup_key', 'diff_outcome'])
        metrics = {
            'TSN': len(data.dataframe),
            'DSN': 0,
            'DSR': 0,
            'DSS': float('inf')
        }
        return empty_result, metrics

    # Calculate outcome for the generated subgroups
    feature_cols = list(data.attributes)
    result_df_filled = result_df[feature_cols].copy()

    # Handle potential None values by filling with median from training data
    for col in feature_cols:
        if result_df_filled[col].isnull().any():
            median_val = data.training_dataframe[col].median()
            result_df_filled[col].fillna(median_val, inplace=True)

    # Predict outcomes using the trained model
    predicted_outcomes = model.predict(result_df_filled[feature_names])
    result_df[data.outcome_column] = predicted_outcomes

    # Add individual key
    result_df['subgroup_key'] = result_df.apply(
        lambda row: '|'.join(str(int(row[col])) if row[col] is not None else '*' for col in list(data.attributes)),
        axis=1
    )

    # Calculate outcome differences from the mean
    mean_outcome = result_df[data.outcome_column].mean()
    result_df['diff_outcome'] = (result_df[data.outcome_column] - mean_outcome).abs()

    # Calculate metrics
    total_time = time.time() - start_time
    tsn = len(data.dataframe)
    dsn = result_df['subgroup_key'].nunique() if 'subgroup_key' in result_df.columns else 0
    dsr = dsn / tsn if tsn > 0 else 0
    dss = total_time / dsn if dsn > 0 else float('inf')

    metrics = {
        'TSN': tsn,
        'DSN': dsn,
        'DSR': dsr,
        'DSS': dss
    }

    return result_df, metrics
