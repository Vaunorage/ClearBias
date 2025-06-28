import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Set, Any
from dataclasses import dataclass, field
import math

from data_generator.main import DiscriminationData

# Group Fairness (Statistical Parity)

def run_group_fairness(data, threshold: float = 0.05, 
                      min_group_size: int = 20, random_seed: int = 42, 
                      max_runtime_seconds: int = 3600) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Implements group fairness (statistical parity) testing to find discrimination in a dataset.
    
    Group fairness is satisfied when subjects in both protected and unprotected groups have
    equal probability of being assigned to the positive predicted class.
    
    Formula: P(d = 1|G = m) = P(d = 1|G = f)
    
    Args:
        data: Data object containing dataset and metadata
        threshold: Maximum acceptable difference in positive outcome rates between groups
        min_group_size: Minimum size of groups to consider for analysis
        random_seed: Random seed for reproducibility
        max_runtime_seconds: Maximum runtime in seconds before early termination
        
    Returns:
        Results DataFrame and metrics dictionary
    """
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    logger = logging.getLogger("GroupFairness")
    
    start_time = time.time()
    
    # Handle case where data is a tuple (data, schema) returned by get_real_data
    if isinstance(data, tuple) and len(data) == 2:
        data = data[0]  # Extract the DiscriminationData object
    
    # Ensure we have a DiscriminationData object
    if not isinstance(data, DiscriminationData):
        raise TypeError(f"Expected DiscriminationData object, got {type(data)}")
    
    # Log basic information
    logger.info(f"Dataset shape: {data.dataframe.shape}")
    logger.info(f"Protected attributes: {data.protected_attributes}")
    logger.info(f"Time limit: {max_runtime_seconds} seconds")
    
    # Get the outcome column (predicted decisions)
    outcome_col = data.outcome_column
    
    # Initialize results storage
    discriminatory_groups = []
    metrics = {}
    
    # Function to check if we should terminate due to time constraints
    def should_terminate() -> bool:
        current_runtime = time.time() - start_time
        return current_runtime > max_runtime_seconds
    
    # Calculate baseline positive outcome rate for the entire dataset
    baseline_positive_rate = data.dataframe[outcome_col].mean()
    metrics['baseline_positive_rate'] = baseline_positive_rate
    logger.info(f"Baseline positive outcome rate: {baseline_positive_rate:.4f}")
    
    # Analyze each protected attribute individually
    for attr in data.protected_attributes:
        if should_terminate():
            logger.info("Time limit reached. Terminating early.")
            break
            
        logger.info(f"Analyzing protected attribute: {attr}")
        
        # Get unique values for this attribute
        unique_values = data.dataframe[attr].unique()
        
        # Calculate positive outcome rate for each value of the attribute
        for value in unique_values:
            # Get the subset of data for this attribute value
            group_data = data.dataframe[data.dataframe[attr] == value]
            
            # Skip groups that are too small
            if len(group_data) < min_group_size:
                logger.info(f"Skipping group {attr}={value} (size {len(group_data)} < {min_group_size})")
                continue
                
            # Calculate positive outcome rate for this group
            group_positive_rate = group_data[outcome_col].mean()
            
            # Calculate the difference from baseline
            rate_difference = abs(group_positive_rate - baseline_positive_rate)
            
            # Check if this group is discriminated against
            is_discriminated = rate_difference > threshold
            
            if is_discriminated:
                # Create a record for this discriminatory group
                group_record = {
                    'attribute': attr,
                    'value': value,
                    'group_size': len(group_data),
                    'group_positive_rate': group_positive_rate,
                    'baseline_positive_rate': baseline_positive_rate,
                    'rate_difference': rate_difference,
                    'is_discriminated': is_discriminated
                }
                
                discriminatory_groups.append(group_record)
                logger.info(f"Found discriminatory group: {attr}={value} with rate difference {rate_difference:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(discriminatory_groups)
    
    # Calculate metrics
    metrics['total_groups_analyzed'] = len(unique_values) * len(data.protected_attributes)
    metrics['discriminatory_groups_found'] = len(discriminatory_groups)
    metrics['max_rate_difference'] = results_df['rate_difference'].max() if not results_df.empty else 0
    metrics['runtime_seconds'] = time.time() - start_time
    
    logger.info(f"Analysis complete. Found {metrics['discriminatory_groups_found']} discriminatory groups.")
    logger.info(f"Runtime: {metrics['runtime_seconds']:.2f} seconds")
    
    return results_df, metrics


def analyze_intersectional_fairness(data, threshold: float = 0.05,
                                  min_group_size: int = 20, max_attributes: int = 2,
                                  random_seed: int = 42, max_runtime_seconds: int = 3600) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Extends group fairness to analyze intersectional discrimination across multiple protected attributes.
    
    Args:
        data: Data object containing dataset and metadata
        threshold: Maximum acceptable difference in positive outcome rates between groups
        min_group_size: Minimum size of groups to consider for analysis
        max_attributes: Maximum number of attributes to consider in intersectional analysis
        random_seed: Random seed for reproducibility
        max_runtime_seconds: Maximum runtime in seconds before early termination
        
    Returns:
        Results DataFrame and metrics dictionary
    """
    import itertools
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    logger = logging.getLogger("IntersectionalFairness")
    
    start_time = time.time()
    
    # Handle case where data is a tuple (data, schema) returned by get_real_data
    if isinstance(data, tuple) and len(data) == 2:
        data = data[0]  # Extract the DiscriminationData object
    
    # Ensure we have a DiscriminationData object
    if not isinstance(data, DiscriminationData):
        raise TypeError(f"Expected DiscriminationData object, got {type(data)}")
    
    # Log basic information
    logger.info(f"Dataset shape: {data.dataframe.shape}")
    logger.info(f"Protected attributes: {data.protected_attributes}")
    logger.info(f"Max attributes for intersectional analysis: {max_attributes}")
    logger.info(f"Time limit: {max_runtime_seconds} seconds")
    
    # Get the outcome column (predicted decisions)
    outcome_col = data.outcome_column
    
    # Initialize results storage
    discriminatory_groups = []
    metrics = {}
    
    # Function to check if we should terminate due to time constraints
    def should_terminate() -> bool:
        current_runtime = time.time() - start_time
        return current_runtime > max_runtime_seconds
    
    # Calculate baseline positive outcome rate for the entire dataset
    baseline_positive_rate = data.dataframe[outcome_col].mean()
    metrics['baseline_positive_rate'] = baseline_positive_rate
    logger.info(f"Baseline positive outcome rate: {baseline_positive_rate:.4f}")
    
    # Generate all possible combinations of protected attributes up to max_attributes
    total_combinations = 0
    analyzed_combinations = 0
    
    for k in range(1, min(max_attributes + 1, len(data.protected_attributes) + 1)):
        for attr_combo in itertools.combinations(data.protected_attributes, k):
            total_combinations += 1
            
            if should_terminate():
                logger.info("Time limit reached. Terminating early.")
                break
                
            logger.info(f"Analyzing attribute combination: {attr_combo}")
            analyzed_combinations += 1
            
            # Get all possible value combinations for these attributes
            value_ranges = [data.dataframe[attr].unique() for attr in attr_combo]
            
            for value_combo in itertools.product(*value_ranges):
                # Create a filter for this specific combination of attribute values
                filter_condition = pd.Series(True, index=data.dataframe.index)
                for attr, value in zip(attr_combo, value_combo):
                    filter_condition = filter_condition & (data.dataframe[attr] == value)
                
                # Get the subset of data for this combination
                group_data = data.dataframe[filter_condition]
                
                # Skip groups that are too small
                if len(group_data) < min_group_size:
                    continue
                    
                # Calculate positive outcome rate for this group
                group_positive_rate = group_data[outcome_col].mean()
                
                # Calculate the difference from baseline
                rate_difference = abs(group_positive_rate - baseline_positive_rate)
                
                # Check if this group is discriminated against
                is_discriminated = rate_difference > threshold
                
                if is_discriminated:
                    # Create a record for this discriminatory group
                    group_record = {
                        'attributes': '+'.join(attr_combo),
                        'values': '+'.join(str(v) for v in value_combo),
                        'group_size': len(group_data),
                        'group_positive_rate': group_positive_rate,
                        'baseline_positive_rate': baseline_positive_rate,
                        'rate_difference': rate_difference,
                        'is_discriminated': is_discriminated
                    }
                    
                    discriminatory_groups.append(group_record)
                    logger.info(f"Found discriminatory group: {group_record['attributes']}={group_record['values']} with rate difference {rate_difference:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(discriminatory_groups)
    
    # Calculate metrics
    metrics['total_combinations_possible'] = total_combinations
    metrics['combinations_analyzed'] = analyzed_combinations
    metrics['discriminatory_groups_found'] = len(discriminatory_groups)
    metrics['max_rate_difference'] = results_df['rate_difference'].max() if not results_df.empty else 0
    metrics['runtime_seconds'] = time.time() - start_time
    
    logger.info(f"Analysis complete. Found {metrics['discriminatory_groups_found']} discriminatory groups.")
    logger.info(f"Runtime: {metrics['runtime_seconds']:.2f} seconds")
    
    return results_df, metrics


# For testing purposes
if __name__ == "__main__":
    from data_generator.main import get_real_data
    
    # Get sample data
    data_obj, _ = get_real_data("adult", use_cache=True)  # Unpack the tuple properly
    
    # Run group fairness analysis
    results_df, metrics = run_group_fairness(data_obj)
    
    # Print results
    print("\nGroup Fairness Results:")
    print(f"Found {metrics['discriminatory_groups_found']} discriminatory groups")
    print(f"Max rate difference: {metrics['max_rate_difference']:.4f}")
    print("\nDiscriminatory Groups:")
    print(results_df)
