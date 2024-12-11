from data_generator.main import generate_valid_correlation_matrix, generate_data

# %%
nb_attributes = 20
correlation_matrix = generate_valid_correlation_matrix(nb_attributes)

data = generate_data(
    nb_attributes=nb_attributes,
    correlation_matrix=correlation_matrix,
    min_number_of_classes=2,
    max_number_of_classes=9,
    prop_protected_attr=0.4,
    nb_groups=100,
    max_group_size=100,
    categorical_outcome=True,
    nb_categories_outcome=4,
    corr_matrix_randomness=1)

print(f"Generated {len(data.dataframe)} samples in {data.nb_groups} groups")
# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def unique_individuals_ratio(data: pd.DataFrame, individual_col: str, group_col: str,
                             attr_possible_values: Dict[str, List[int]]) -> Tuple[pd.DataFrame, float, int, int]:
    """
    Calculate unique individuals ratio both overall and per group.

    Parameters:
    data (pd.DataFrame): Input DataFrame
    individual_col (str): Name of the column containing individual IDs
    group_col (str): Name of the column containing group IDs
    attr_possible_values (Dict[str, List[int]]): Dictionary of possible values for each attribute

    Returns:
    Tuple containing:
    - DataFrame with group-level statistics
    - Overall unique ratio
    - Total duplicate count
    - Total number of individuals
    """
    # Overall statistics
    unique_individuals_count = data[individual_col].nunique()
    total_individuals = data.shape[0]
    duplicates_count = total_individuals - unique_individuals_count
    overall_ratio = unique_individuals_count / total_individuals if total_individuals > 0 else 0

    # Group-level statistics
    group_stats = pd.DataFrame({
        'total_rows': data.groupby(group_col).size(),
        'unique_individuals': data.groupby(group_col)[individual_col].nunique()
    }).reset_index()

    # Calculate ratios and percentages for each group
    group_stats['unique_ratio'] = group_stats['unique_individuals'] / group_stats['total_rows']
    group_stats['duplicate_count'] = group_stats['total_rows'] - group_stats['unique_individuals']
    group_stats['pct_unique'] = (group_stats['unique_ratio'] * 100).round(2)

    # Sort by total rows in descending order
    group_stats = group_stats.sort_values('total_rows', ascending=False)

    # Rename columns for clarity
    group_stats.columns = [group_col, 'Total Rows', 'Unique Individuals',
                           'Unique Ratio', 'Duplicates', '% Unique']

    return group_stats, overall_ratio, duplicates_count, total_individuals


def group_summary_table(data: pd.DataFrame, individual_col: str, group_col: str) -> pd.DataFrame:
    """
    Creates a summary table showing the total number of rows and unique individuals per group.
    """
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'total_rows': data.groupby(group_col).size(),
        'unique_individuals': data.groupby(group_col)[individual_col].nunique()
    }).reset_index()

    # Sort by total_rows in descending order
    summary = summary.sort_values('total_rows', ascending=False)

    # Add percentage of unique individuals
    summary['pct_unique'] = (summary['unique_individuals'] / summary['total_rows'] * 100).round(2)

    # Rename columns for clarity
    summary.columns = [group_col, 'Total Rows', 'Unique Individuals', '% Unique']

    return summary


def individuals_in_multiple_groups(data: pd.DataFrame, individual_col: str, group_col: str) -> None:
    """
    Analyzes and visualizes the distribution of individuals across multiple groups.
    """
    # Display the summary table first
    print("\nGroup Summary Table:")
    summary_table = group_summary_table(data, individual_col, group_col)
    print(summary_table.to_string(index=False))
    print("\nDistribution of Individuals Across Groups:")

    # Create histogram
    group_counts = data.groupby(individual_col)[group_col].nunique()
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(group_counts, bins=range(1, group_counts.max() + 2),
                                     edgecolor='black', align='left')

    # Add text annotations on top of each bar
    for count, patch in zip(counts, patches):
        plt.text(patch.get_x() + patch.get_width() / 2, count,
                 f'{int(count)}', ha='center', va='bottom')

    plt.title('Histogram of Individuals Belonging to Multiple Groups with Counts')
    plt.xlabel('Number of Groups')
    plt.ylabel('Number of Individuals')
    plt.xticks(range(1, group_counts.max() + 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


# Example usage:
individual_col = 'indv_key'
group_col = 'group_key'

# Get both group-level and overall statistics
group_stats, overall_ratio, duplicates_count, total = unique_individuals_ratio(
    data.dataframe, individual_col, group_col, data.attr_possible_values
)

# Display results
print("\nGroup-level Statistics:")
print(group_stats.to_string(index=False))
print(f"\nOverall Statistics:")
print(f"Overall Unique Individuals Ratio: {overall_ratio:.4f}")
print(f"Total Duplicates: {duplicates_count}")
print(f"Total Individuals: {total}")

# Generate visualization
individuals_in_multiple_groups(data.dataframe, individual_col, group_col)
