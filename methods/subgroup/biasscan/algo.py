from typing import TypedDict, Tuple

import pandas as pd
from itertools import product

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from methods.subgroup.biasscan.mdss_detector import bias_scan
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


class ResultRow(TypedDict, total=False):
    group_id: int
    outcome: int
    diff_outcome: int
    indv_key: str
    couple_key: str


ResultDF = DataFrame


def format_mdss_results(subsets1, subsets2, all_attributes, ge) -> pd.DataFrame:
    """
    Format MDSS results with memory-efficient processing
    """

    def make_products_df(subsets, all_attributes, chunk_size=10000):
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
                return pd.DataFrame(columns=all_attributes + [ge.outcome_column])

            # Create DataFrame for this chunk
            chunk_df = pd.DataFrame(chunk_products, columns=subset.keys())

            # Add missing attributes
            for attr in all_attributes:
                if attr not in chunk_df.columns:
                    chunk_df[attr] = None

            chunk_df[ge.outcome_column] = outcome

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
            return pd.DataFrame(columns=all_attributes + [ge.outcome_column])

        return pd.concat(all_dfs, ignore_index=True)

    # Process each subset group separately
    product_dfs1 = make_products_df(subsets1, all_attributes)
    product_dfs2 = make_products_df(subsets2, all_attributes)

    if product_dfs1.empty and product_dfs2.empty:
        print("Warning: Both product_dfs1 and product_dfs2 are empty")
        return pd.DataFrame(columns=all_attributes + [ge.outcome_column])

    product_dfs = pd.concat([product_dfs1, product_dfs2], ignore_index=True)

    def select_min_max(group):
        if len(group) < 2:
            return pd.DataFrame()

        min_val = group[ge.outcome_column].min()
        max_val = group[ge.outcome_column].max()

        if min_val == max_val:
            return pd.DataFrame()

        min_rows = group[group[ge.outcome_column] == min_val]
        max_rows = group[group[ge.outcome_column] == max_val]
        return pd.concat([min_rows.iloc[0:1], max_rows.iloc[0:1]])

    # Group and process results
    grouping_cols = [k for k, v in ge.attributes.items() if not v]
    result = product_dfs.groupby(grouping_cols, group_keys=True).apply(select_min_max).reset_index(drop=True)

    if result.empty:
        print("Warning: result DataFrame is empty")
        return pd.DataFrame(columns=[
            'group_id', *grouping_cols,
            *[col for col in all_attributes if col not in grouping_cols],
            ge.outcome_column, 'indv_key', 'couple_key', 'diff_outcome'
        ])

    # Add required columns
    result['group_id'] = result.groupby(grouping_cols).ngroup()
    other_cols = [col for col in result.columns
                  if col not in grouping_cols + ['group_id', ge.outcome_column]]
    result = result[['group_id'] + grouping_cols + other_cols + [ge.outcome_column]]

    # Sort and process pairs
    result = result.sort_values(['group_id', ge.outcome_column]).reset_index(drop=True)
    result['indv_key'] = result.apply(
        lambda row: '|'.join(str(int(row[col])) for col in list(ge.attributes)),
        axis=1
    )
    result['couple_key'] = result.groupby(result.index // 2)['indv_key'].transform('-'.join)

    # Calculate outcome differences
    result['diff_outcome'] = result.groupby('couple_key')[ge.outcome_column].transform(
        lambda x: abs(x.diff().iloc[-1])
    )

    # Set correct types
    result = result.astype({
        'group_id': 'int',
        'outcome': 'int',
        'diff_outcome': 'int',
        'indv_key': 'str',
        'couple_key': 'str'
    })

    # Set types for attribute columns
    for attr in ge.attributes:
        if attr in result.columns:
            result[attr] = result[attr].astype('float')

    return result


def run_bias_scan(ge,
                  test_size=0.2,
                  random_state=42,
                  n_estimators=100,
                  bias_scan_num_iters=50,
                  bias_scan_scoring='Poisson',
                  bias_scan_favorable_value='high',
                  bias_scan_mode='ordinal') -> Tuple[pd.DataFrame, dict]:
    # Split the data
    train_df, test_df = train_test_split(ge.dataframe, test_size=test_size, random_state=random_state)

    X_train = train_df[list(ge.attributes)].values
    y_train = train_df[ge.outcome_column]
    X_test = test_df[list(ge.attributes)].values
    y_test = test_df[ge.outcome_column]

    # Train the model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Prepare data for bias scan
    observations = pd.Series(ge.dataframe[ge.outcome_column].to_numpy().squeeze())
    expectations = pd.Series(model.predict(ge.dataframe[list(ge.attributes)].values))

    # Perform bias scan
    l1, l2, subsets1 = bias_scan(data=ge.dataframe[list(ge.attributes)],
                                 observations=observations,
                                 expectations=expectations,
                                 verbose=True,
                                 num_iters=bias_scan_num_iters,
                                 scoring=bias_scan_scoring,
                                 favorable_value=bias_scan_favorable_value,
                                 overpredicted=True,
                                 mode=bias_scan_mode)

    d1, d2, subsets2 = bias_scan(data=ge.dataframe[list(ge.attributes)],
                                 observations=observations,
                                 expectations=expectations,
                                 verbose=True,
                                 num_iters=bias_scan_num_iters,
                                 scoring=bias_scan_scoring,
                                 favorable_value=bias_scan_favorable_value,
                                 overpredicted=False,
                                 mode=bias_scan_mode)

    # Format results
    result_df = format_mdss_results(subsets1, subsets2, list(ge.attributes), ge)

    return result_df, {'accuracy': accuracy, 'report': report}
