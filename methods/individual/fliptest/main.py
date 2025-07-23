import numpy as np
import pandas as pd
import time
from scipy.spatial import distance
from sklearn import preprocessing
from data_generator.main import DiscriminationData, generate_optimal_discrimination_data
from methods.individual.fliptest import optimize_gurobi as optimize
from methods.individual.fliptest import util


def run_fliptest_on_dataset(
        discrimination_data: DiscriminationData,
        protected_attribute: str,
        group1_val=0,
        group2_val=1
):
    """
    Runs the FlipTest method on a given dataset to find discrimination.

    Args:
        discrimination_data: The dataset object containing the dataframe.
        protected_attribute: The name of the protected attribute to analyze.
        group1_val: The value representing the first group in the protected attribute column.
        group2_val: The value representing the second group in the protected attribute column.
        
    Returns:
        tuple: (results_df, metrics_dict) where:
            - results_df: A pandas DataFrame containing all identified discriminatory pairs
            - metrics_dict: A dictionary with summary statistics about the analysis
    """
    df = discrimination_data.dataframe
    print(f"Splitting data based on protected attribute: '{protected_attribute}'")

    if protected_attribute not in discrimination_data.attr_columns:
        raise ValueError(
            f"Protected attribute '{protected_attribute}' not found in dataset columns: {discrimination_data.attr_columns}")

    group1_df = df[df[protected_attribute] == group1_val]
    group2_df = df[df[protected_attribute] == group2_val]

    if len(group1_df) == 0 or len(group2_df) == 0:
        print(f"Warning: One or both groups for attribute '{protected_attribute}' are empty. Skipping.")
        return None

    print(f"Group 1 ('{protected_attribute}' = {group1_val}) size: {len(group1_df)}")
    print(f"Group 2 ('{protected_attribute}' = {group2_val}) size: {len(group2_df)}")

    feature_columns = [col for col in discrimination_data.attr_columns if col != protected_attribute]
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
    try:
        forward, reverse = optimize.optimize(X1_scaled, X2_scaled, dists)
        forward, reverse = util.get_index_arrays(forward, reverse)

        mean_dist = util.get_mean_dist(X1_scaled, X2_scaled, forward)
        print(f'Mean L1 distance for {protected_attribute}: {mean_dist:.4f}')
    except Exception as e:
        print(f"Error in optimal transport calculation: {e}")
        print(f"This may be due to significant class imbalance in '{protected_attribute}' attribute.")
        print(f"Group sizes: {len(X1_scaled)} vs {len(X2_scaled)} (ratio: {len(X1_scaled) / len(X2_scaled):.2f})")
        return None

    # Start timing for metrics
    start_time = time.time()
    
    # Create a results DataFrame with discriminatory pairs
    pairs_data = []
    discriminatory_count = 0
    total_count = 0
    
    # For each individual in group 1, find its match in group 2
    for i in range(len(X1_scaled)):
        if forward[i] >= 0:  # Only consider valid mappings
            total_count += 1
            j = forward[i]
            
            # Check if there's a difference in outcome
            outcome_diff = abs(y1[i] - y2[j])
            
            if outcome_diff > 0:  # This is a discriminatory pair
                discriminatory_count += 1
                
                # Extract attribute values for both individuals
                attr_values1 = group1_df.iloc[i][discrimination_data.attr_columns].values
                attr_values2 = group2_df.iloc[j][discrimination_data.attr_columns].values
                
                # Create keys using the format from utils.py
                indv_key1 = "|".join(str(x) for x in attr_values1)
                indv_key2 = "|".join(str(x) for x in attr_values2)
                
                # Create couple_key
                couple_key = f"{indv_key1}-{indv_key2}"
                
                # Create new dataframes with just the necessary data
                # For individual 1
                indv1_data = {}
                for idx, col in enumerate(discrimination_data.attr_columns):
                    indv1_data[col] = attr_values1[idx]
                indv1_data['indv_key'] = indv_key1
                indv1_data['outcome'] = y1[i]
                indv1_data['couple_key'] = couple_key
                indv1_data['diff_outcome'] = outcome_diff
                indv1_data['case_id'] = discriminatory_count - 1
                
                # For individual 2
                indv2_data = {}
                for idx, col in enumerate(discrimination_data.attr_columns):
                    indv2_data[col] = attr_values2[idx]
                indv2_data['indv_key'] = indv_key2
                indv2_data['outcome'] = y2[j]
                indv2_data['couple_key'] = couple_key
                indv2_data['diff_outcome'] = outcome_diff
                indv2_data['case_id'] = discriminatory_count - 1
                
                # Add to pairs data
                pairs_data.append(pd.Series(indv1_data))
                pairs_data.append(pd.Series(indv2_data))
    
    # Calculate metrics
    end_time = time.time()
    total_time = end_time - start_time
    success_rate = discriminatory_count / total_count if total_count > 0 else 0
    avg_search_time = total_time / discriminatory_count if discriminatory_count > 0 else 0
    
    # Create the results DataFrame
    if pairs_data:
        results_df = pd.DataFrame(pairs_data)
        
        # Add global metrics to each row
        results_df['TSN'] = total_count
        results_df['DSN'] = discriminatory_count
        results_df['SUR'] = success_rate
        results_df['DSS'] = avg_search_time
    else:
        # Create an empty DataFrame with the right columns if no pairs found
        results_df = pd.DataFrame(columns=list(df.columns) + 
                                ['indv_key', 'outcome', 'couple_key', 'diff_outcome', 'case_id',
                                 'TSN', 'DSN', 'SUR', 'DSS'])
    
    # Create metrics dictionary
    metrics = {
        "TSN": total_count,  # Total Sample Number
        "DSN": discriminatory_count,  # Discriminatory Sample Number
        "SUR": success_rate,  # Success Rate
        "DSS": avg_search_time,  # Discriminatory Sample Search time
        "total_time": total_time,
        "mean_distance": mean_dist,
        "protected_attribute": protected_attribute,
        "group1_val": group1_val,
        "group2_val": group2_val,
        "raw_results": {
            "X1": X1_scaled, "X2": X2_scaled,
            "y1": y1, "y2": y2,
            "columns": feature_columns,
            "forward_mapping": forward,
            "reverse_mapping": reverse
        }
    }
    
    return results_df, metrics


def run_fliptest(data: DiscriminationData, max_runs: int = None, max_runtime_seconds: int = None):
    """
    Run FlipTest on a dataset for multiple protected attributes.
    
    Args:
        data: The dataset object containing the dataframe.
        max_runs: Maximum number of protected attributes to check. If None, all will be checked.
        
    Returns:
        tuple: (combined_results_df, metrics_dict) where:
            - combined_results_df: A pandas DataFrame containing all discriminatory pairs across all attributes
            - metrics_dict: A dictionary with summary statistics about the analysis
    """
    # Get all protected attributes
    protected_attrs = data.protected_attributes
    print(f"Found {len(protected_attrs)} protected attributes: {protected_attrs}")
    
    # Calculate class balance for each protected attribute
    protected_attrs_to_check = []
    for attr in protected_attrs:
        unique_values = data.dataframe[attr].unique()
        if len(unique_values) < 2:
            print(f"Skipping attribute '{attr}' as it does not have at least two unique values.")
            continue
            
        # Get counts for each value
        value_counts = data.dataframe[attr].value_counts()
        smallest_group = value_counts.min()
        largest_group = value_counts.max()
        
        # Calculate balance ratio (smaller is worse)
        balance_ratio = smallest_group / largest_group if largest_group > 0 else 0
        protected_attrs_to_check.append((attr, balance_ratio))
        print(f"Attribute '{attr}' has balance ratio: {balance_ratio:.4f}")
    
    # Sort by balance ratio (most balanced first)
    protected_attrs_to_check.sort(key=lambda x: x[1], reverse=True)
    
    # Limit number of attributes to check if max_runs is specified
    limit = max_runs if max_runs is not None else len(protected_attrs_to_check)
    attributes_to_check = [attr for attr, ratio in protected_attrs_to_check[:limit]]
    print(f"Will check the following protected attributes: {attributes_to_check}")
    print(f"Running {len(attributes_to_check)} out of {len(protected_attrs_to_check)} available protected attributes")

    # Lists to store results for each protected attribute
    all_results_dfs = []
    all_metrics = {}
    successful_runs = 0
    total_start_time = time.time()
    
    for protected_attribute in attributes_to_check:
        # Check if runtime has exceeded the limit
        if max_runtime_seconds is not None and (time.time() - total_start_time) > max_runtime_seconds:
            print(f"\nStopping FlipTest as runtime has exceeded the {max_runtime_seconds} second limit.")
            break

        print("\n" + "=" * 50)
        print(f"Running FlipTest for '{protected_attribute}' attribute")
        print("=" * 50)

        unique_values = np.sort(data.dataframe[protected_attribute].unique())
        if len(unique_values) < 2:
            print(f"Skipping attribute '{protected_attribute}' as it does not have at least two unique values.")
            continue

        result = run_fliptest_on_dataset(
            data,
            protected_attribute=protected_attribute,
            group1_val=unique_values[0],
            group2_val=unique_values[1]
        )
        
        if result:
            results_df, metrics = result  # Unpack the tuple
            print(f"FlipTest for '{protected_attribute}' completed successfully.")
            print(f"Found {metrics['DSN']} discriminatory pairs out of {metrics['TSN']} total pairs.")
            print(f"Success rate: {metrics['SUR']:.4f}, Mean L1 distance: {metrics['mean_distance']:.4f}")
            
            # Add protected attribute name to the results DataFrame
            results_df['protected_attribute'] = protected_attribute
            
            # Store results
            all_results_dfs.append(results_df)
            all_metrics[protected_attribute] = metrics
            successful_runs += 1

    # Calculate overall metrics
    total_time = time.time() - total_start_time
    
    # Combine all results DataFrames
    if all_results_dfs:
        combined_results_df = pd.concat(all_results_dfs, ignore_index=True)
    else:
        # Create an empty DataFrame with the right columns if no results
        combined_results_df = pd.DataFrame(columns=['indv_key', 'outcome', 'couple_key', 'diff_outcome', 
                                                  'case_id', 'TSN', 'DSN', 'SUR', 'DSS', 'protected_attribute'])
    
    # Create overall metrics dictionary
    # Aggregate metrics across all attributes
    total_tsn = sum(m['TSN'] for m in all_metrics.values())
    total_dsn = sum(m['DSN'] for m in all_metrics.values())
    
    # Calculate weighted average of DSS
    total_dss_numerator = sum(m['DSS'] * m['DSN'] for m in all_metrics.values())
    if total_dsn > 0:
        overall_dss = total_dss_numerator / total_dsn
    else:
        overall_dss = 0

    # Calculate overall SUR
    if total_tsn > 0:
        overall_sur = total_dsn / total_tsn
    else:
        overall_sur = 0

    # Final metrics dictionary in the desired format
    metrics = {
        'DSN': total_dsn,
        'TSN': total_tsn,
        'DSS': overall_dss,
        'SUR': overall_sur
    }

    print(f"\nCompleted {successful_runs} successful FlipTest runs out of {len(attributes_to_check)} attempted.")
    print(f"Total execution time: {total_time:.2f} seconds")

    return combined_results_df, metrics


if __name__ == "__main__":
    print("\nFlipTest: Individual Fairness Testing using Optimal Transport")
    print("=" * 70)
    print("This algorithm identifies discriminatory pairs by finding the optimal transport")
    print("mapping between individuals in different protected groups.")
    print("=" * 70)
    
    # data_obj, schema = get_real_data('adult', use_cache=True)
    data_obj = generate_optimal_discrimination_data(nb_groups=100,
                                                    nb_attributes=15,
                                                    prop_protected_attr=0.3,
                                                    nb_categories_outcome=1,
                                                    use_cache=True)

    # Example: limit to 2 runs and a 60-second runtime
    results_df, metrics = run_fliptest(data_obj, max_runs=2, max_runtime_seconds=60)
    
    # Print summary of results
    print(metrics)

