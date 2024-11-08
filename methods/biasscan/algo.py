from typing import TypedDict, Union, Tuple, Dict, Any

import pandas as pd
from itertools import product

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from typing_extensions import NotRequired

from methods.biasscan.mdss_detector import bias_scan
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


class ResultRow(TypedDict, total=False):
    group_id: int
    val: float
    outcome: int
    diff_outcome: int
    indv_key: str
    couple_key: str


ResultDF = DataFrame


def run_bias_scan(ge,
                  test_size=0.2,
                  random_state=42,
                  n_estimators=100,
                  bias_scan_num_iters=50,
                  bias_scan_scoring='Poisson',
                  bias_scan_favorable_value='high',
                  bias_scan_mode='ordinal') -> Tuple[pd.DataFrame, dict]:

    def format_mdss_results(subsets1, subsets2, all_attributes) -> pd.DataFrame:
        def make_products_df(subsets):
            product_dfs = []
            for subset, val in subsets:
                product_lists = product(*subset.values())
                columns = subset.keys()
                product_df = pd.DataFrame(product_lists, columns=columns)
                print(f"Shape after creating DataFrame: {product_df.shape}")
                for attr in all_attributes:
                    if attr not in product_df.columns:
                        product_df[attr] = None
                    product_df['val'] = val
                    product_dfs.append(product_df)

                if not product_dfs:
                    print("Warning: product_dfs is empty")
                    return pd.DataFrame(columns=all_attributes + ['val'])

                product_dfs = pd.concat(product_dfs, axis=0)
                print(f"Shape after concatenation: {product_dfs.shape}")

                product_dfs = product_dfs.drop_duplicates()
                print(f"Shape after drop_duplicates: {product_dfs.shape}")

                product_dfs = product_dfs.dropna()
                print(f"Shape after dropna: {product_dfs.shape}")

                if product_dfs.empty:
                    print("Warning: product_dfs is empty after preprocessing")
                    return pd.DataFrame(columns=all_attributes + ['val'])  # Ensure consistent columns

                product_dfs[ge.outcome_column] = model.predict(product_dfs[list(ge.attributes)])
                return product_dfs

        product_dfs1 = make_products_df(subsets1)
        product_dfs2 = make_products_df(subsets2)

        if product_dfs1.empty and product_dfs2.empty:
            print("Warning: Both product_dfs1 and product_dfs2 are empty")
            return pd.DataFrame(columns=all_attributes + ['val', ge.outcome_column])

        product_dfs = pd.concat([product_dfs1, product_dfs2])

        def select_min_max(group):
            min_val = group[ge.outcome_column].min()
            max_val = group[ge.outcome_column].max()
            if min_val != max_val:
                min_rows = group[group[ge.outcome_column] == min_val]
                max_rows = group[group[ge.outcome_column] == max_val]
                min_row = min_rows.sample(n=1)
                max_row = max_rows.sample(n=1)
                return pd.concat([min_row, max_row])
            else:
                return pd.DataFrame()

        product_dfs[ge.outcome_column] = model.predict(product_dfs[list(ge.attributes)])
        grouping_cols = [k for k, v in ge.attributes.items() if not v]
        grouped = product_dfs.groupby(grouping_cols)
        filtered_groups = grouped.filter(lambda x: len(x) >= 2)
        result = filtered_groups.groupby(grouping_cols).apply(select_min_max).reset_index(drop=True)

        if result.empty:
            print("Warning: result DataFrame is empty")
            return pd.DataFrame(columns=[
                'group_id', *grouping_cols, *[col for col in all_attributes if col not in grouping_cols],
                'val', ge.outcome_column, 'indv_key', 'couple_key', 'diff_outcome'
            ])

        result['group_id'] = result.groupby(grouping_cols).ngroup()
        other_cols = [col for col in result.columns if
                      col not in grouping_cols and col != 'group_id' and col != ge.outcome_column]
        result = result[['group_id'] + grouping_cols + other_cols + [ge.outcome_column]]
        result = result.sort_values(['group_id', ge.outcome_column])
        result = result.reset_index(drop=True)

        result['indv_key'] = result.apply(lambda row: '|'.join(str(int(row[col])) for col in list(ge.attributes)),
                                          axis=1)
        result['couple_key'] = result.groupby(result.index // 2)['indv_key'].transform('-'.join)

        # Add a check to ensure we have pairs
        if any(result.groupby('couple_key').size() != 2):
            print("Warning: Not all couple_keys have exactly two individuals")

        result['diff_outcome'] = result.groupby('couple_key')['outcome'].transform(lambda x: abs(x.diff().iloc[-1]))

        # Ensure the result matches the ResultDF type
        result: ResultDF = result.astype({
            'group_id': 'int',
            'val': 'float',  # or 'int' depending on actual type
            'outcome': 'int',
            'diff_outcome': 'int',
            'indv_key': 'str',
            'couple_key': 'str'
        })

        # Dynamically set types for attribute columns
        for attr in ge.attributes:
            if attr in result.columns:
                result[attr] = result[attr].astype('float')  # Adjust based on your data

        return result

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
    result_df = format_mdss_results(subsets1, subsets2, list(ge.attributes))

    return result_df, {'accuracy': accuracy, 'report': report}
