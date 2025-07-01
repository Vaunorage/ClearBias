import os
import sys
import pandas as pd

# Add the project root to the Python path to allow for absolute imports
# This is a common practice in projects to avoid relative import issues.
# We assume the script is run from within the 'fair_naive_bayes' directory or the project root.
try:
    # Assuming the script is in methods/group/fair_naive_bayes
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
except NameError:
    # Fallback for interactive environments
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_generator.main import get_real_data
# We need these functions to manually calculate the NB parameters from a DataFrame
from methods.group.fair_naive_bayes.parameter_learner.data_processor import (
    get_params_dict,
    maximum_likelihood_from_data,
    convert_result_to_parameters
)
from methods.group.fair_naive_bayes.pattern_finder.pattern_finder import PatternFinder


def find_discrimination_example():
    """
    An example script that loads the 'adult' dataset using get_real_data,
    binarizes the data, calculates Naive Bayes parameters, and then runs the
    PatternFinder to find discriminating patterns.
    """
    print("--- Running Discrimination Finder Example with get_real_data ---")

    # --- 1. Load data using get_real_data ---
    dataset_name = 'adult'
    print(f"Loading '{dataset_name}' dataset using get_real_data...")
    try:
        data, schema = get_real_data(dataset_name, use_cache=True)
        df = data.dataframe
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure the data can be fetched or is cached correctly.")
        return

    print("Dataset loaded. Binarizing data for the Naive Bayes model...")

    # --- 2. Binarize the data ---
    # The PatternFinder's underlying model requires all features to be binary (0 or 1).
    # We will perform a simple binarization for this example.
    binarized_df = pd.DataFrame()
    target_name = 'outcome' # As defined in the data_generator
    binarized_df[target_name] = df[target_name]

    for attr_name in schema.attr_names:
        if df[attr_name].nunique() <= 2:
            # If the attribute is already binary, just copy it.
            binarized_df[attr_name] = df[attr_name]
        else:
            # For non-binary columns, we binarize at the 75th percentile (top quartile).
            # This may help find patterns in the upper range of an attribute.
            quantile_val = df[attr_name].quantile(0.75)
            binarized_df[attr_name] = (df[attr_name] > quantile_val).astype(int)
            print(f"  - Binarized '{attr_name}' by splitting at its 75th percentile value ({quantile_val:.2f})")

    # --- 3. Manually construct metadata needed for parameter learning ---
    # This information was previously read from a .net.txt file in the old approach.
    feature_names = schema.attr_names
    bn_dict = {i: name for i, name in enumerate(feature_names)}
    sensitive_var_ids = [i for i, name in enumerate(feature_names) if schema.protected_attr[schema.attr_names.index(name)]]
    target_value = 1  # The unfavorable outcome (e.g., income > 50K)

    sensitive_names = [bn_dict[i] for i in sensitive_var_ids]
    print(f"Sensitive attributes identified: {sensitive_names}")

    # --- 4. Calculate Naive Bayes parameters from the binarized data ---
    print("\nCalculating Naive Bayes parameters from binarized data...")
    params_feature_names = list(feature_names)
    
    # Step 4a: Count co-occurrences of features and the target variable.
    params_dict = get_params_dict(binarized_df, params_feature_names, target_name)
    
    # Step 4b: Calculate probabilities (Maximum Likelihood Estimation).
    prob_dict = maximum_likelihood_from_data(params_dict, target_name)
    
    # Step 4c: Convert probabilities into the list format required by PatternFinder.
    root_params, leaf_params = convert_result_to_parameters(prob_dict, sensitive_var_ids, bn_dict, target_name)

    # --- 5. Find Discriminating Patterns ---
    delta = 0.01  # Fairness threshold
    k = 5        # Number of patterns to find
    print(f"\nSearching for the top {k} discriminating patterns with a threshold of {delta}...")
    
    pf = PatternFinder(root_params, leaf_params, target_value, sensitive_var_ids)
    raw_patterns = pf.get_discriminating_patterns(delta, k)

    # --- 6. Process and Print Results ---
    if not raw_patterns:
        print("\nNo discriminating patterns found with the given threshold.")
    else:
        print(f"\n--- Top {len(raw_patterns)} Discriminating Patterns Found ---")
        for i, pattern in enumerate(raw_patterns):
            base_features = [f"{bn_dict.get(fid, f'ID:{fid}')} = {val}" for fid, val in pattern.base]
            sens_features = [f"{bn_dict.get(fid, f'ID:{fid}')} = {val}" for fid, val in pattern.sens]

            print(f"\nPattern #{i+1}:")
            print(f"  - Subgroup (non-sensitive): {', '.join(base_features) or 'N/A'}")
            print(f"  - Subgroup (sensitive):     {', '.join(sens_features)}")
            print(f"  - Discrimination Score:     {pattern.score:.4f}")
            print(f"  - Details:")
            print(f"    - P(unfavorable | subgroup, sensitive group): {pattern.pDXY:.4f}")
            print(f"    - P(unfavorable | subgroup, other groups):  {pattern.pD_XY:.4f}")

    print(f"\nSearch visited {pf.num_visited} nodes.")
    print("\n--- Example Finished ---")


if __name__ == '__main__':
    find_discrimination_example()
