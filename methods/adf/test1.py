import pandas as pd
import numpy as np
from collections import defaultdict


def analyze_group_discrimination(df):
    """
    Analyze discrimination between pairs of groups based on protected attributes.
    Returns the gaps in outcomes between groups with different protected attributes.
    """
    # Get protected attribute columns
    protected_cols = [col for col in df.columns if col.endswith('_T')]

    # Initialize storage for group comparisons
    group_comparisons = []

    # Find pairs using couple_key
    paired_data = df[df['couple_key'] != -1].copy()

    # For each couple_key, compare the protected attributes
    for couple_key in paired_data['couple_key'].unique():
        pair = paired_data[paired_data['couple_key'] == couple_key].copy()

        if len(pair) != 2:
            continue

        # Sort by outcome to ensure consistent ordering (lower outcome first)
        pair = pair.sort_values('outcome')

        # Get protected attributes for both individuals
        group1_attrs = {col: pair.iloc[0][col] for col in protected_cols}
        group2_attrs = {col: pair.iloc[1][col] for col in protected_cols}

        # Calculate outcome gap
        outcome_gap = pair.iloc[1]['outcome'] - pair.iloc[0]['outcome']

        # Store comparison
        comparison = {
            'group1_attributes': group1_attrs,
            'group2_attributes': group2_attrs,
            'outcome_gap': outcome_gap,
            'couple_key': couple_key
        }
        group_comparisons.append(comparison)

    # Convert to DataFrame
    comparisons_df = pd.DataFrame(group_comparisons)

    # Aggregate results by group pairs
    aggregated_results = []

    # Create a mapping of all unique protected attribute combinations
    unique_combinations = defaultdict(lambda: defaultdict(list))

    for _, row in comparisons_df.iterrows():
        group1_key = tuple(sorted(row['group1_attributes'].items()))
        group2_key = tuple(sorted(row['group2_attributes'].items()))

        # Store gap for this combination
        unique_combinations[group1_key][group2_key].append(row['outcome_gap'])

    # Calculate average gaps for each unique combination
    for group1, group2_dict in unique_combinations.items():
        for group2, gaps in group2_dict.items():
            group1_dict = dict(group1)
            group2_dict = dict(group2)

            result = {
                'group1_protected_attributes': group1_dict,
                'group2_protected_attributes': group2_dict,
                'average_gap': np.mean(gaps),
                'num_cases': len(gaps),
                'std_gap': np.std(gaps),
                'max_gap': max(gaps),
                'min_gap': min(gaps)
            }
            aggregated_results.append(result)

    # Convert to DataFrame and sort by average gap
    results_df = pd.DataFrame(aggregated_results)
    results_df = results_df.sort_values('average_gap', ascending=False)

    return results_df


def format_results(results_df):
    """
    Format the results into a more readable format
    """
    formatted_results = []

    for _, row in results_df.iterrows():
        group1_attrs = [f"{k}={v}" for k, v in row['group1_protected_attributes'].items()]
        group2_attrs = [f"{k}={v}" for k, v in row['group2_protected_attributes'].items()]

        formatted_result = {
            'group1': " & ".join(group1_attrs),
            'group2': " & ".join(group2_attrs),
            'average_gap': round(row['average_gap'], 3),
            'num_cases': row['num_cases'],
            'std_gap': round(row['std_gap'], 3),
            'max_gap': round(row['max_gap'], 3),
            'min_gap': round(row['min_gap'], 3)
        }
        formatted_results.append(formatted_result)

    return pd.DataFrame(formatted_results)


# Example usage with sample data
def create_sample_data(n_samples=1000):
    """
    Create sample data for demonstration
    """
    np.random.seed(42)

    # Generate sample data
    data = {
        'Attr7_T': np.random.choice(['A', 'B'], n_samples),
        'Attr8_T': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'outcome': np.random.binomial(1, 0.5, n_samples),
        'couple_key': [-1] * n_samples
    }

    df = pd.DataFrame(data)

    # Create some paired cases
    for i in range(0, n_samples, 2):
        if i + 1 < n_samples:
            couple_key = i // 2
            df.loc[i:i + 1, 'couple_key'] = couple_key

            # Ensure some discrimination patterns
            if df.loc[i, 'Attr7_T'] == 'A' and df.loc[i, 'Attr8_T'] == 'X':
                df.loc[i, 'outcome'] = 1
                df.loc[i + 1, 'outcome'] = 0

    return df


# %% Create sample data
df = create_sample_data()

# Analyze discrimination
results = analyze_group_discrimination(df)

# Format results
formatted_results = format_results(results)

print("\nDiscrimination Analysis Results:")
print(formatted_results)

# %%

predefined_groups = [
    {
        'granularity': 2,
        'intersectionality': 1,
        'group_size': 50,
        'subgroup_bias': 0.3,
        'similarity': 0.7,
        'alea_uncertainty': 0.2,
        'epis_uncertainty': 0.1,
        'frequency': 0.8,
        'avg_diff_outcome': 1,
        'diff_subgroup_size': 0.2,
        'subgroup1': {'Attr1_T' : 3, 'Attr2_T': 1, 'Attr3_X': 3},
        'subgroup2': {'Attr1_T' : 2, 'Attr2_T': 2, 'Attr3_X': 2}
    },
    # ... other groups
]
