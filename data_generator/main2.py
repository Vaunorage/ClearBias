import numpy as np
import pandas as pd
import random
import itertools
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import copy


@dataclass
class SubgroupProfile:
    """Class to store a generated subgroup's profile."""
    attr_values: List[int]  # Values for each attribute (-1 for non-relevant)
    attributes_used: List[int]  # Indices of attributes used to define this subgroup
    protected_degree: float  # How much this group is defined by protected attributes (0.0-1.0)


def estimate_subgroups_needed(nb_groups: int) -> int:
    """
    Estimate the number of subgroups needed to generate the desired number of groups.

    For n groups, we need approximately sqrt(2n) subgroups for enough pairwise combinations.
    """
    # Formula: n_subgroups * (n_subgroups - 1) / 2 >= nb_groups
    # Solving for n_subgroups gives: n_subgroups >= (1 + sqrt(1 + 8 * nb_groups)) / 2
    n_subgroups = math.ceil((1 + math.sqrt(1 + 8 * nb_groups)) / 2)

    # Add a small buffer to ensure we have enough combinations
    return n_subgroups + 2


def generate_diverse_subgroups(
        num_subgroups: int,
        attr_categories: List[List[int]],
        protected_attr: List[bool],
        attr_names: List[str],
        categorical_distribution: Dict[str, List[float]] = None
) -> List[SubgroupProfile]:
    """
    Generate diverse subgroups by strategically varying the attributes used.

    Args:
        num_subgroups: Number of subgroups to generate
        attr_categories: List of possible values for each attribute
        protected_attr: Boolean list indicating which attributes are protected
        attr_names: Names of the attributes
        categorical_distribution: Distribution for categorical attributes

    Returns:
        List of SubgroupProfile objects
    """
    num_attributes = len(attr_categories)
    subgroups = []

    # Ensure we have variation in attributes used
    # Strategy: Vary the number and type of attributes used to define each subgroup

    # Create segments of the parameter space to ensure diversity
    protected_indices = [i for i, is_protected in enumerate(protected_attr) if is_protected]
    non_protected_indices = [i for i, is_protected in enumerate(protected_attr) if not is_protected]

    for i in range(num_subgroups):
        # Determine how many attributes to use for this subgroup (between 1 and num_attributes/2)
        num_attrs_to_use = random.randint(1, max(1, num_attributes // 2))

        # Decide on protected vs non-protected balance
        # Use a range of ratios to ensure diversity
        protected_ratio = i / max(1, num_subgroups - 1)  # 0.0 to 1.0

        # Calculate how many protected and non-protected attributes to use
        num_protected = min(len(protected_indices),
                            max(1, int(num_attrs_to_use * protected_ratio)))
        num_non_protected = min(len(non_protected_indices),
                                max(1, num_attrs_to_use - num_protected))

        # Select random attributes of each type
        selected_protected = random.sample(protected_indices, num_protected) if protected_indices else []
        selected_non_protected = random.sample(non_protected_indices,
                                               num_non_protected) if non_protected_indices else []

        # Combine selected attributes
        selected_attributes = selected_protected + selected_non_protected

        # Generate attribute values
        attr_values = [-1] * num_attributes  # -1 indicates attribute not used

        for attr_idx in selected_attributes:
            # Get valid categories (excluding -1 which represents missing values)
            valid_categories = [cat for cat in attr_categories[attr_idx] if cat != -1]

            if valid_categories:
                # Either use categorical distribution if provided, or uniform selection
                if (categorical_distribution and
                        attr_names[attr_idx] in categorical_distribution and
                        len(categorical_distribution[attr_names[attr_idx]]) == len(valid_categories)):

                    # Use provided distribution
                    probs = categorical_distribution[attr_names[attr_idx]]
                    attr_values[attr_idx] = random.choices(valid_categories, weights=probs)[0]
                else:
                    # Uniform selection
                    attr_values[attr_idx] = random.choice(valid_categories)

        # Calculate protected degree
        if not selected_attributes:
            protected_degree = 0.0
        else:
            protected_degree = len(selected_protected) / len(selected_attributes)

        # Create subgroup profile
        subgroup = SubgroupProfile(
            attr_values=attr_values,
            attributes_used=selected_attributes,
            protected_degree=protected_degree
        )

        subgroups.append(subgroup)

    return subgroups


def are_subgroups_compatible(sg1: SubgroupProfile, sg2: SubgroupProfile) -> bool:
    """
    Check if two subgroups can be combined to form a valid group.

    Criteria:
    1. They must differ in at least one common attribute
    2. They shouldn't have conflicting values for non-differing attributes
    """
    # Find attributes used by both subgroups
    common_attributes = set(sg1.attributes_used).intersection(set(sg2.attributes_used))

    # If no common attributes, they're trivially different
    if not common_attributes:
        return True

    # Check for at least one differing attribute
    has_difference = False
    for attr_idx in common_attributes:
        if sg1.attr_values[attr_idx] != sg2.attr_values[attr_idx]:
            has_difference = True
            break

    return has_difference


def generate_subgroup_combination_data(
        subgroup1: SubgroupProfile,
        subgroup2: SubgroupProfile,
        group_id: int,
        subgroup_size1: int,
        subgroup_size2: int,
        subgroup_bias: float,
        attr_names: List[str],
        attr_categories: List[List[int]]
) -> pd.DataFrame:
    """
    Generate synthetic data for a pair of subgroups.

    Args:
        subgroup1, subgroup2: SubgroupProfile objects for the two subgroups
        group_id: Identifier for this group
        subgroup_size1, subgroup_size2: Size of each subgroup
        subgroup_bias: Bias to apply to subgroup2 outcome
        attr_names: Names of the attributes
        attr_categories: List of possible values for each attribute

    Returns:
        DataFrame with generated data
    """
    # Generate data for subgroup 1
    data1 = []
    for _ in range(subgroup_size1):
        row = generate_individual_row(subgroup1.attr_values, attr_categories)
        # Add outcome (no bias for subgroup 1)
        outcome = random.random() > 0.5
        data1.append(row + [outcome, 1, group_id])  # [attributes..., outcome, subgroup, group_id]

    # Generate data for subgroup 2
    data2 = []
    for _ in range(subgroup_size2):
        row = generate_individual_row(subgroup2.attr_values, attr_categories)
        # Add outcome with bias
        outcome = random.random() > (0.5 + subgroup_bias)  # Add bias
        data2.append(row + [outcome, 2, group_id])  # [attributes..., outcome, subgroup, group_id]

    # Combine data
    all_data = data1 + data2

    # Create DataFrame
    columns = attr_names + ['outcome', 'subgroup', 'group_id']
    df = pd.DataFrame(all_data, columns=columns)

    # Add metadata columns
    df['similarity'] = calculate_similarity(subgroup1, subgroup2)
    df['protected_degree'] = (subgroup1.protected_degree + subgroup2.protected_degree) / 2
    df['subgroup_bias'] = subgroup_bias

    return df


def generate_individual_row(base_values: List[int], attr_categories: List[List[int]]) -> List:
    """
    Generate a row for an individual based on subgroup base values.

    For attributes defined in the subgroup (not -1), use the specified value.
    For attributes not defined, randomly select from possible values.
    """
    row = []
    for i, base_val in enumerate(base_values):
        if base_val != -1:
            # Use the subgroup's defined value
            row.append(base_val)
        else:
            # Randomly select from possible categories (excluding -1 which is for missing)
            valid_cats = [cat for cat in attr_categories[i] if cat != -1]
            if valid_cats:
                row.append(random.choice(valid_cats))
            else:
                row.append(-1)  # Fallback if no valid categories

    return row


def calculate_similarity(sg1: SubgroupProfile, sg2: SubgroupProfile) -> float:
    """Calculate similarity between two subgroups."""
    # Find all attributes used by either subgroup
    all_used_attrs = set(sg1.attributes_used).union(set(sg2.attributes_used))

    if not all_used_attrs:
        return 0.0

    # Count attributes with the same value
    same_value_count = sum(1 for idx in all_used_attrs
                           if idx in sg1.attributes_used and idx in sg2.attributes_used and
                           sg1.attr_values[idx] == sg2.attr_values[idx] and
                           sg1.attr_values[idx] != -1)

    # Calculate similarity as the ratio of same-valued attributes to all used attributes
    return same_value_count / len(all_used_attrs)


def generate_subgroup_based_discrimination_data(
        nb_groups: int,
        nb_attributes: int,
        attr_categories: List[List[int]],
        protected_attr: List[bool],
        attr_names: List[str],
        min_group_size: int = 10,
        max_group_size: int = 100,
        min_subgroup_bias: float = 0.1,
        max_subgroup_bias: float = 0.5,
        categorical_distribution: Dict[str, List[float]] = None
) -> pd.DataFrame:
    """
    Generate discrimination data using the subgroup-first approach.

    Args:
        nb_groups: Number of groups to generate
        nb_attributes: Number of attributes
        attr_categories: List of possible values for each attribute
        protected_attr: Boolean list indicating which attributes are protected
        attr_names: Names of the attributes
        min_group_size, max_group_size: Range for group sizes
        min_subgroup_bias, max_subgroup_bias: Range for subgroup bias
        categorical_distribution: Distribution for categorical attributes

    Returns:
        DataFrame with generated discrimination data
    """
    # Estimate number of subgroups needed
    num_subgroups = estimate_subgroups_needed(nb_groups)
    print(f"Generating {num_subgroups} subgroups to create {nb_groups} groups")

    # Generate diverse subgroups
    subgroups = generate_diverse_subgroups(
        num_subgroups, attr_categories, protected_attr, attr_names, categorical_distribution
    )

    # Find compatible subgroup pairs
    compatible_pairs = []
    for i in range(len(subgroups)):
        for j in range(i + 1, len(subgroups)):
            if are_subgroups_compatible(subgroups[i], subgroups[j]):
                compatible_pairs.append((i, j))

    print(f"Found {len(compatible_pairs)} compatible subgroup pairs")

    # If we don't have enough compatible pairs, generate more subgroups
    if len(compatible_pairs) < nb_groups:
        additional_subgroups = generate_diverse_subgroups(
            nb_groups - len(compatible_pairs) + 10,  # Add buffer
            attr_categories, protected_attr, attr_names, categorical_distribution
        )

        # Find additional compatible pairs
        original_count = len(subgroups)
        subgroups.extend(additional_subgroups)

        for i in range(original_count):
            for j in range(original_count, len(subgroups)):
                if are_subgroups_compatible(subgroups[i], subgroups[j]):
                    compatible_pairs.append((i, j))

        for i in range(original_count, len(subgroups)):
            for j in range(i + 1, len(subgroups)):
                if are_subgroups_compatible(subgroups[i], subgroups[j]):
                    compatible_pairs.append((i, j))

        print(f"After adding subgroups, found {len(compatible_pairs)} compatible pairs")

    # Generate groups from compatible pairs
    all_data = []
    used_group_count = 0

    # Use a random sample of pairs if we have more than needed
    if len(compatible_pairs) > nb_groups:
        compatible_pairs = random.sample(compatible_pairs, nb_groups)

    for i, (sg1_idx, sg2_idx) in enumerate(compatible_pairs):
        if used_group_count >= nb_groups:
            break

        # Determine group size and subgroup sizes
        group_size = random.randint(min_group_size, max_group_size)

        # Add some variation to subgroup sizes (40-60% split, up to 30-70%)
        split_variation = random.uniform(0.4, 0.6)
        subgroup_size1 = max(2, int(group_size * split_variation))
        subgroup_size2 = max(2, group_size - subgroup_size1)

        # Generate random bias
        subgroup_bias = random.uniform(min_subgroup_bias, max_subgroup_bias)

        # Generate data for this group
        group_data = generate_subgroup_combination_data(
            subgroups[sg1_idx], subgroups[sg2_idx],
            used_group_count, subgroup_size1, subgroup_size2,
            subgroup_bias, attr_names, attr_categories
        )

        all_data.append(group_data)
        used_group_count += 1

    # Combine all group data
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df
    else:
        # Return empty DataFrame with correct structure if no groups were created
        columns = attr_names + ['outcome', 'subgroup', 'group_id',
                                'similarity', 'protected_degree', 'subgroup_bias']
        return pd.DataFrame(columns=columns)


# Example usage:
def example():
    # Example configuration
    nb_groups = 50
    nb_attributes = 10

    # Define attribute categories (values each attribute can take)
    attr_categories = [list(range(-1, 3)) for _ in range(nb_attributes)]  # -1 plus 0,1,2 for each

    # Define which attributes are protected
    protected_attr = [i % 3 == 0 for i in range(nb_attributes)]  # Every 3rd attribute is protected

    # Define attribute names
    attr_names = [f"Attr{i + 1}_{'T' if protected_attr[i] else 'X'}" for i in range(nb_attributes)]

    # Generate the data
    df = generate_subgroup_based_discrimination_data(
        nb_groups=nb_groups,
        nb_attributes=nb_attributes,
        attr_categories=attr_categories,
        protected_attr=protected_attr,
        attr_names=attr_names,
        min_group_size=20,
        max_group_size=100,
        min_subgroup_bias=0.1,
        max_subgroup_bias=0.4
    )

    print(f"Generated dataset with {len(df)} rows and {len(df['group_id'].unique())} groups")

    # Display sample
    print(df.head())

    # Summary statistics
    group_stats = df.groupby('group_id').agg({
        'similarity': 'first',
        'protected_degree': 'first',
        'subgroup_bias': 'first',
        'outcome': lambda x: (x == True).mean()
    }).rename(columns={'outcome': 'positive_outcome_rate'})

    print("\nGroup Statistics:")
    print(group_stats.describe())

    return df


if __name__ == "__main__":
    example()