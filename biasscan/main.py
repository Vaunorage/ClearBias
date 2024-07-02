import pandas as pd
from itertools import product
from biasscan.mdss_detector import bias_scan
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from data_generator.main import generate_data

df, protected_attr = generate_data(min_number_of_classes=2, max_number_of_classes=6, nb_attributes=6,
                                   prop_protected_attr=0.3, nb_groups=500, max_group_size=50, hiddenlayers_depth=3,
                                   min_similarity=0.0,
                                   max_similarity=1.0, min_alea_uncertainty=0.0, max_alea_uncertainty=1.0,
                                   min_epis_uncertainty=0.0, max_epis_uncertainty=1.0,
                                   min_magnitude=0.0, max_magnitude=1.0, min_frequency=0.0, max_frequency=1.0,
                                   categorical_outcome=True, nb_categories_outcome=4)

# %
train_df, test_df = df.random_split(p=0.2)

X_train = train_df.xdf()
y_train = train_df.ydf()
X_test = test_df.xdf()
y_test = test_df.ydf()

model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# %%
observations = pd.Series(df.ydf().to_numpy().squeeze())
expectations = pd.Series(model.predict(df.xdf()))

# Perform the bias scan
_, _, subsets1 = bias_scan(data=df.xdf(), observations=observations, expectations=expectations,
                           verbose=True, num_iters=10, scoring='Poisson', favorable_value='high',
                           overpredicted=True, mode='ordinal')

_, _, subsets2 = bias_scan(data=df.xdf(), observations=observations, expectations=expectations,
                           verbose=True, num_iters=10, scoring='Poisson', favorable_value='high',
                           overpredicted=False, mode='ordinal')


def format_mdss_results(subsets, all_attributes):
    # Generating the product of all attribute values
    product_dfs = []
    for subset, val in subsets:
        product_lists = product(*subset.values())

        # Creating a DataFrame from the product of lists
        columns = subset.keys()
        product_df = pd.DataFrame(product_lists, columns=columns)

        for attr in all_attributes:
            if attr not in product_df.columns:
                product_df[attr] = None

        product_df['val'] = val
        product_dfs.append(product_df)

    product_dfs = pd.concat(product_dfs, axis=0)

    product_dfs = product_dfs.drop_duplicates()

    adf_aggregated = df.adf().drop_duplicates().groupby(df.x_cols).mean().reset_index()
    result = pd.merge(product_dfs, adf_aggregated, on=df.x_cols, how='left')

    mask = result[df.y_col[0]].isna()
    if mask.shape[0] > 0:
        predictions = model.predict(result.loc[mask, df.x_cols])
        result.loc[mask, df.y_col[0]] = predictions

    return result


# Run the formatting function
result_df1 = format_mdss_results(subsets1, df.x_cols)
result_df2 = format_mdss_results(subsets2, df.x_cols)

print(result_df2)

# %%
