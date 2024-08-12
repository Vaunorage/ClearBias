import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from methods.biasscan.mdss_detector import bias_scan
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_generator.main import generate_data

ge = generate_data(min_number_of_classes=2, max_number_of_classes=6, nb_attributes=6,
                   prop_protected_attr=0.3, nb_groups=500, max_group_size=50, hiddenlayers_depth=3,
                   min_similarity=0.0, max_similarity=1.0, min_alea_uncertainty=0.0,
                   max_alea_uncertainty=1.0, min_epis_uncertainty=0.0, max_epis_uncertainty=1.0,
                   min_magnitude=0.0, max_magnitude=1.0, min_frequency=0.0, max_frequency=1.0,
                   categorical_outcome=True, nb_categories_outcome=4)

# %
train_df, test_df = train_test_split(ge.dataframe, test_size=0.2, random_state=42)

X_train = train_df[list(ge.attributes)].values
y_train = train_df[ge.outcome_column]
X_test = test_df[list(ge.attributes)].values
y_test = test_df[ge.outcome_column]

model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# %%
observations = pd.Series(ge.dataframe[ge.outcome_column].to_numpy().squeeze())
expectations = pd.Series(model.predict(ge.dataframe[list(ge.attributes)].values))

# Perform the bias scan
_, _, subsets1 = bias_scan(data=ge.dataframe[list(ge.attributes)], observations=observations, expectations=expectations,
                           verbose=True, num_iters=50, scoring='Poisson', favorable_value='high',
                           overpredicted=True, mode='ordinal')

_, _, subsets2 = bias_scan(data=ge.dataframe[list(ge.attributes)], observations=observations, expectations=expectations,
                           verbose=True, num_iters=50, scoring='Poisson', favorable_value='high',
                           overpredicted=False, mode='ordinal')


def format_mdss_results(subsets1, subsets2, all_attributes):
    def make_products_df(subsets):
        product_dfs = []
        for subset, val in subsets:
            product_lists = product(*subset.values())

            columns = subset.keys()
            product_df = pd.DataFrame(product_lists, columns=columns)

            for attr in all_attributes:
                if attr not in product_df.columns:
                    product_df[attr] = None

            product_df['val'] = val
            product_dfs.append(product_df)

        product_dfs = pd.concat(product_dfs, axis=0)

        product_dfs = product_dfs.drop_duplicates().dropna()

        product_dfs[ge.outcome_column] = model.predict(product_dfs[list(ge.attributes)])

        return product_dfs

    product_dfs1 = make_products_df(subsets1)
    product_dfs2 = make_products_df(subsets2)

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

    # The rest of the code remains the same
    product_dfs[ge.outcome_column] = model.predict(product_dfs[list(ge.attributes)])
    grouping_cols = [k for k, v in ge.attributes.items() if not v]
    grouped = product_dfs.groupby(grouping_cols)
    filtered_groups = grouped.filter(lambda x: len(x) >= 2)
    result = filtered_groups.groupby(grouping_cols).apply(select_min_max).reset_index(drop=True)
    result['group_id'] = result.groupby(grouping_cols).ngroup()
    other_cols = [col for col in result.columns if
                  col not in grouping_cols and col != 'group_id' and col != ge.outcome_column]
    result = result[['group_id'] + grouping_cols + other_cols + [ge.outcome_column]]
    result = result.sort_values(['group_id', ge.outcome_column])
    result = result.reset_index(drop=True)

    result['ind_key'] = result.apply(lambda row: '|'.join(str(int(row[col])) for col in list(ge.attributes)), axis=1)
    result['couple_key'] = result.groupby(result.index // 2)['ind_key'].transform('*'.join)

    return result


# Run the formatting function
result_df = format_mdss_results(subsets1, subsets2, list(ge.attributes))

print(result_df)

# %%
