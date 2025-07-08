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

    if protected_attribute not in data_obj.attr_columns:
        raise ValueError(
            f"Protected attribute '{protected_attribute}' not found in dataset columns: {data_obj.attr_columns}")

    group1_df = df[df[protected_attribute] == group1_val]
    group2_df = df[df[protected_attribute] == group2_val]

    if len(group1_df) == 0 or len(group2_df) == 0:
        print(f"Warning: One or both groups for attribute '{protected_attribute}' are empty. Skipping.")
        return None

    print(f"Group 1 ('{protected_attribute}' = {group1_val}) size: {len(group1_df)}")
    print(f"Group 2 ('{protected_attribute}' = {group2_val}) size: {len(group2_df)}")

    feature_columns = [col for col in data_obj.attr_columns if col != protected_attribute]
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


def run_fliptest(data_obj: DiscriminationData, max_runs: int = None):
    """
    Run FlipTest on a dataset for multiple protected attributes.
    
    Args:
        data_obj: The dataset object containing the dataframe.
        max_runs: Maximum number of protected attributes to check. If None, all will be checked.
        
    Returns:
        tuple: (combined_results_df, metrics_dict) where:
            - combined_results_df: A pandas DataFrame containing all discriminatory pairs across all attributes
            - metrics_dict: A dictionary with summary statistics about the analysis
    """
    # Get all protected attributes
    protected_attrs = data_obj.protected_attributes
    print(f"Found {len(protected_attrs)} protected attributes: {protected_attrs}")
    
    # Calculate class balance for each protected attribute
    protected_attrs_to_check = []
    for attr in protected_attrs:
        unique_values = data_obj.dataframe[attr].unique()
        if len(unique_values) < 2:
            print(f"Skipping attribute '{attr}' as it does not have at least two unique values.")
            continue
            
        # Get counts for each value
        value_counts = data_obj.dataframe[attr].value_counts()
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
        print("\n" + "=" * 50)
        print(f"Running FlipTest for '{protected_attribute}' attribute")
        print("=" * 50)

        unique_values = np.sort(data_obj.dataframe[protected_attribute].unique())
        if len(unique_values) < 2:
            print(f"Skipping attribute '{protected_attribute}' as it does not have at least two unique values.")
            continue

        result = run_fliptest_on_dataset(
            data_obj,
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
    overall_metrics = {
        "total_runs": len(attributes_to_check),
        "successful_runs": successful_runs,
        "total_time": total_time,
        "attribute_metrics": all_metrics
    }
    
    print(f"\nCompleted {successful_runs} successful FlipTest runs out of {len(attributes_to_check)} attempted.")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    return combined_results_df, overall_metrics


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

    # Example: limit to 2 runs
    results_df, metrics = run_fliptest(data_obj, max_runs=2)
    
    # Print summary of results
    print("\n## Output")
    print("\nThe algorithm returns two main outputs:")
    
    print("\n### Results DataFrame (results_df)")
    print("A pandas DataFrame containing all identified discriminatory pairs with the following columns:")
    print("- All original feature columns from the dataset")
    print("- `indv_key`: A unique identifier for each individual instance")
    print("- `outcome`: The predicted outcome for the instance")
    print("- `couple_key`: A key linking two instances that form a discriminatory pair")
    print("- `diff_outcome`: The absolute difference in outcomes between the pair")
    print("- `case_id`: A unique identifier for each discriminatory case")
    print("- `TSN`: Total Sample Number - total number of input pairs tested")
    print("- `DSN`: Discriminatory Sample Number - number of discriminatory pairs found")
    print("- `SUR`: Success Rate - ratio of DSN to TSN")
    print("- `DSS`: Discriminatory Sample Search time - average time to find a discriminatory sample")
    print("- `protected_attribute`: The protected attribute used for this pair")
    
    print("\n### Metrics Dictionary (metrics)")
    print("A dictionary containing summary statistics:")
    print("- `total_runs`: Total number of protected attributes tested")
    print("- `successful_runs`: Number of protected attributes that completed successfully")
    print("- `total_time`: Total execution time")
    print("- `attribute_metrics`: Detailed metrics for each protected attribute")
    
    # Print summary statistics
    print("\n## Summary Statistics")
    print("-" * 60)
    print(f"Total runs attempted: {metrics['total_runs']}")
    print(f"Successful runs: {metrics['successful_runs']}")
    print(f"Total execution time: {metrics['total_time']:.2f} seconds")
    print("-" * 60)
    
    # Print details for each attribute
    for attr, attr_metrics in metrics['attribute_metrics'].items():
        print(f"Protected Attribute: {attr}")
        print(f"  Total pairs analyzed: {attr_metrics['TSN']}")
        print(f"  Discriminatory pairs found: {attr_metrics['DSN']}")
        print(f"  Success rate: {attr_metrics['SUR']:.4f}")
        print(f"  Mean L1 Distance: {attr_metrics['mean_distance']:.4f}")
        print("-" * 60)
    
    # Print sample of the results DataFrame
    if not results_df.empty:
        print("\n## Example Output DataFrame")
        print("\nHere's a sample of the output DataFrame:")
        print("\n", results_df.head().to_string())
        print(f"\nTotal discriminatory pairs found: {len(results_df) // 2}")
        print("(Each pair consists of two rows in the DataFrame)")
    else:
        print("\nNo discriminatory pairs found.")

