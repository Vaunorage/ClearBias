import json
import sqlite3
import pandas as pd
from path import HERE
import re
from functools import lru_cache

DB_PATH = HERE.joinpath("methods/optimization/fairness_test_results2.db")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()


@lru_cache(maxsize=2048)
def matches_pattern(pattern_string, test_string, data_schema):
    @lru_cache(maxsize=2048)
    def _compile_pattern(pattern_string, data_schema):
        res_pattern = []
        for k, v in zip(pattern_string.split('|'), data_schema.split('|')):
            if k == '*':
                res_pattern.append(f"[{v}]")
            else:
                res_pattern.append(k)
        res_pattern = "\|".join(res_pattern)
        return res_pattern

    if '-' in pattern_string:
        if '-' not in data_schema:
            data_schema = f"{data_schema}-{data_schema}"
        result_pat = []
        for pat1, pat2 in zip(pattern_string.split('-'), data_schema.split('-')):
            result_pat.append(_compile_pattern(pat1, pat2))

        pattern = re.compile('-'.join(result_pat))
        res = bool(pattern.match(test_string))
        return res
    else:
        pattern = re.compile(_compile_pattern(pattern_string, data_schema))
        return bool(pattern.match(test_string))


def identify_group_membership(synthetic_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    # Make copies to avoid modifying the original dataframes
    synthetic_df = synthetic_df.copy()
    results_df = results_df.copy()

    # Ensure keys are strings
    synthetic_df['indv_key'] = synthetic_df['indv_key'].astype(str)
    synthetic_df['group_key'] = synthetic_df['group_key'].astype(str)
    results_df['indv_key'] = results_df['indv_key'].astype(str)
    results_df['couple_key'] = results_df['couple_key'].astype(str)

    # Get unique groups and create a dictionary of individuals for each group
    unique_groups = synthetic_df['group_key'].unique()
    group_individuals = {
        group_key: set(synthetic_df[synthetic_df['group_key'] == group_key]['indv_key'].unique())
        for group_key in unique_groups
    }

    # Identify attribute columns (those starting with 'Attr' and ending with '_T')
    attr_columns = [col for col in synthetic_df.columns if col.startswith('Attr')]
    possible_values_per_column = [set(synthetic_df[col].unique()) for col in attr_columns]
    possible_values_per_column = '|'.join([''.join(list(map(str, e))) for e in possible_values_per_column])

    # Pre-compute group patterns
    group_patterns = {group_key: group_key.split('-') for group_key in unique_groups}
    subgroup_patterns = {group_key for group_key in synthetic_df['subgroup_key'].unique()}

    group_key_list = synthetic_df['group_key'].unique().tolist()

    def get_matching_subgroup_for_indv_key(indv_key):
        pattern_matches = [
            subgroup for subgroup in subgroup_patterns if
            matches_pattern(subgroup, indv_key, possible_values_per_column)
        ]
        return pattern_matches

    # Function to check if an individual belongs to a group based on pattern matching
    def get_matching_groups_for_indv_key(indv_key):
        # Direct membership check
        direct_matches = [
            group_key for group_key, indv_set in group_individuals.items()
            if indv_key in indv_set
        ]

        # Pattern-based membership check
        pattern_matches = [
            group_key for group_key, patterns in group_patterns.items()
            if (len(patterns) >= 2 and
                (matches_pattern(patterns[0], indv_key, possible_values_per_column) or
                 matches_pattern(patterns[1], indv_key, possible_values_per_column)))
        ]

        # Combine unique matches
        return list(set(direct_matches + pattern_matches))

    def get_matching_groups_for_couple_key(couple_key):
        res = []

        couple_key_elems = couple_key.split('-')
        if len(couple_key_elems) != 2:
            print(f"Warning: Unexpected couple key format: {couple_key}")
            return res

        opt1 = f"{couple_key_elems[0]}-{couple_key_elems[1]}"
        opt2 = f"{couple_key_elems[1]}-{couple_key_elems[0]}"

        for grp_key in group_key_list:
            if matches_pattern(grp_key, opt1, possible_values_per_column) or matches_pattern(grp_key, opt2,
                                                                                             possible_values_per_column):
                res.append(grp_key)
        return res

    results_df['indv_matching_subgroups'] = results_df['indv_key'].apply(get_matching_subgroup_for_indv_key)
    results_df['indv_matching_groups'] = results_df['indv_key'].apply(get_matching_groups_for_indv_key)
    results_df['couple_matching_groups'] = results_df['couple_key'].apply(get_matching_groups_for_couple_key)

    return results_df[
        ['indv_key', 'couple_key', 'indv_matching_subgroups', 'indv_matching_groups', 'couple_matching_groups']]


def calculate_match_counts_from_groups(res_with_groups, synth_df):
    """
    Calculate match counts using the already identified group memberships in res_with_groups

    Args:
        res_with_groups: DataFrame with individual and couple matching groups information
        synth_df: The synthetic data DataFrame

    Returns:
        DataFrame with match counts for each group
    """
    # Get unique groups from synthetic data
    unique_groups = synth_df['group_key'].unique()

    # Initialize results dictionary
    group_matches = {
        group_key: {
            'indv_matches': 0,
            'couple_matches': 0,
            'unique_indv_matches': 0,
            'unique_couple_matches': 0
        }
        for group_key in unique_groups
    }

    # Count individual matches by exploding the indv_matching_groups column
    indv_match_counts = (
        res_with_groups
        .explode('indv_matching_groups')
        .groupby('indv_matching_groups')
        .size()
        .reset_index(name='indv_matches')
        .rename(columns={'indv_matching_groups': 'group_key'})
    )

    # Count unique individual matches
    unique_indv_match_counts = (
        res_with_groups
        .explode('indv_matching_groups')
        .drop_duplicates(['indv_key', 'indv_matching_groups'])
        .groupby('indv_matching_groups')
        .size()
        .reset_index(name='unique_indv_matches')
        .rename(columns={'indv_matching_groups': 'group_key'})
    )

    # Count couple matches by exploding the couple_matching_groups column
    couple_match_counts = (
        res_with_groups
        .explode('couple_matching_groups')
        .groupby('couple_matching_groups')
        .size()
        .reset_index(name='couple_matches')
        .rename(columns={'couple_matching_groups': 'group_key'})
    )

    # Count unique couple matches
    unique_couple_match_counts = (
        res_with_groups
        .explode('couple_matching_groups')
        .drop_duplicates(['couple_key', 'couple_matching_groups'])
        .groupby('couple_matching_groups')
        .size()
        .reset_index(name='unique_couple_matches')
        .rename(columns={'couple_matching_groups': 'group_key'})
    )

    # Merge all counts into a single DataFrame
    match_counts = (
        pd.DataFrame({'group_key': unique_groups})
        .merge(indv_match_counts, on='group_key', how='left')
        .merge(unique_indv_match_counts, on='group_key', how='left')
        .merge(couple_match_counts, on='group_key', how='left')
        .merge(unique_couple_match_counts, on='group_key', how='left')
        .fillna(0)
    )

    # Convert count columns to integers
    count_columns = ['indv_matches', 'unique_indv_matches', 'couple_matches', 'unique_couple_matches']
    match_counts[count_columns] = match_counts[count_columns].astype(int)

    return match_counts


# %%

def analyze_matching_synthetic_and_result(res, synth):
    # Identify group membership first
    mapping_df = identify_group_membership(synth, res)
    res_with_groups = res.merge(mapping_df, on=['indv_key', 'couple_key'], how='left')

    # Calculate match counts using the group information in res_with_groups
    match_counts = calculate_match_counts_from_groups(res_with_groups, synth)
    synth_with_groups = synth.merge(match_counts, on=['group_key'], how='left')

    return res_with_groups, synth_with_groups


def get_result_and_synthetic_matching_res(experiment_id):
    df_res = pd.read_sql_query(f"SELECT * FROM results_dataframes where run_id = '{experiment_id}'", conn)
    df_synth = pd.read_sql_query(f"SELECT * FROM synthetic_dataframes where run_id = '{experiment_id}'", conn)
    res_with_groups = pd.DataFrame(json.loads(df_res['dataframe_json'][0]))
    synth_with_groups = pd.DataFrame(json.loads(df_synth['dataframe_json'][0]))

    return res_with_groups, synth_with_groups


# %%
if __name__ == '__main__':
    res_with_groups, synth_with_groups = get_result_and_synthetic_matching_res(
            experiment_id='experiment_1746970529_config_0_iter_0')
    ll = res_with_groups[['couple_key', 'indv_key', 'indv_matching_subgroups',
                          'indv_matching_groups', 'couple_matching_groups']]
    print('dddd')
