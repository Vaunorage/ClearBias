"""
This script demonstrates how to use the FlipTest method on the 'adult' dataset
to identify potential discrimination. It loads the data, prepares it for FlipTest,
runs the optimal transport optimization, and prints the results.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import distance
from sklearn import preprocessing

# Add project root to path to allow imports from other modules
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from data_generator.main import get_real_data, DiscriminationData, DataSchema
from methods.individual.fliptest.exactot import optimize_gurobi as optimize
from methods.individual.fliptest.exactot import util

def run_fliptest_on_dataset(
    discrimination_data: DiscriminationData,
    data_schema: DataSchema,
    protected_attribute: str,
    group1_val=0,
    group2_val=1
):
    """
    Runs the FlipTest method on a given dataset to find discrimination.

    Args:
        discrimination_data: The dataset object containing the dataframe.
        data_schema: The schema of the dataset.
        protected_attribute: The name of the protected attribute to analyze.
        group1_val: The value representing the first group in the protected attribute column.
        group2_val: The value representing the second group in the protected attribute column.
    """
    df = discrimination_data.dataframe
    print(f"Splitting data based on protected attribute: '{protected_attribute}'")

    if protected_attribute not in data_schema.attr_names:
        raise ValueError(f"Protected attribute '{protected_attribute}' not found in dataset columns: {data_schema.attr_names}")

    group1_df = df[df[protected_attribute] == group1_val]
    group2_df = df[df[protected_attribute] == group2_val]

    if len(group1_df) == 0 or len(group2_df) == 0:
        print(f"Warning: One or both groups for attribute '{protected_attribute}' are empty. Skipping.")
        return None

    print(f"Group 1 ('{protected_attribute}' = {group1_val}) size: {len(group1_df)}")
    print(f"Group 2 ('{protected_attribute}' = {group2_val}) size: {len(group2_df)}")

    feature_columns = [col for col in data_schema.attr_names if col != protected_attribute]
    outcome_column = 'outcome'

    X1 = group1_df[feature_columns].values.astype(np.float64)
    y1 = group1_df[outcome_column].values

    X2 = group2_df[feature_columns].values.astype(np.float64)
    y2 = group2_df[outcome_column].values

    print("Scaling features...")
    X_combined_scaled = preprocessing.scale(np.vstack([X1, X2]))
    X1_scaled = X_combined_scaled[:len(X1)]
    X2_scaled = X_combined_scaled[len(X1):]

    print("Calculating distance matrix...")
    dists = distance.cdist(X1_scaled, X2_scaled, metric='cityblock')

    print('Solving for the exact optimal transport mapping...')
    forward, reverse = optimize.optimize(X1_scaled, X2_scaled, dists)
    forward, reverse = util.get_index_arrays(forward, reverse)

    mean_dist = util.get_mean_dist(X1_scaled, X2_scaled, forward)
    print(f'Mean L1 distance for {protected_attribute}: {mean_dist:.4f}')

    results = {
        "X1": X1_scaled, "X2": X2_scaled,
        "y1": y1, "y2": y2,
        "columns": feature_columns,
        "forward_mapping": forward,
        "reverse_mapping": reverse,
        "mean_distance": mean_dist
    }
    return results

if __name__ == "__main__":
    print("Loading 'adult' dataset...")
    adult_data, adult_schema = get_real_data('adult', use_cache=True)

    # Get the anonymized names of the protected attributes from the data schema
    protected_attribute_names = [
        name for name, is_protected
        in zip(adult_schema.attr_names, adult_schema.protected_attr)
        if is_protected
    ]

    print(f"Found protected attributes: {protected_attribute_names}")

    for protected_attribute in protected_attribute_names:
        print("\n" + "="*50)
        print(f"Running FlipTest for '{protected_attribute}' attribute")
        print("="*50)

        # For multi-category attributes, we need to select two groups to compare.
        # We will compare the first two unique values found in the column.
        unique_values = np.sort(adult_data.dataframe[protected_attribute].unique())
        if len(unique_values) < 2:
            print(f"Skipping attribute '{protected_attribute}' as it does not have at least two unique values.")
            continue

        group1_val = unique_values[0]
        group2_val = unique_values[1]

        try:
            results = run_fliptest_on_dataset(
                adult_data,
                adult_schema,
                protected_attribute=protected_attribute,
                group1_val=group1_val,
                group2_val=group2_val
            )
            if results:
                 print(f"FlipTest for '{protected_attribute}' completed successfully.")

        except (ImportError, NameError) as e:
            print(f"\nError: Could not run optimizer. Gurobi might not be installed or licensed.")
            print(f"Please see: https://www.gurobi.com/documentation/")
            print(f"Original error: {e}")
            # Stop if Gurobi is not found
            break
        except Exception as e:
            print(f"An unexpected error occurred while processing '{protected_attribute}': {e}")
