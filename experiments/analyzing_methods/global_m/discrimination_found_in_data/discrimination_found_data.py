import pandas as pd
from data_generator.main import generate_data
from path import HERE
import sqlite3
from functools import lru_cache

data_obj = generate_data()

DB_PATH = HERE.joinpath("experiments/analyzing_methods/global/global_testing_res.db")

conn = sqlite3.connect(DB_PATH)

result_table = "main.aequitas_dt_1744527159_results"
test_table = "main.aequitas_dt_1744527159_testdata"

# Load data from database
results_df = pd.read_sql(f"SELECT * FROM {result_table}", conn)
synthetic_data = pd.read_sql(f"SELECT * FROM {test_table}", conn)


# %%
@lru_cache(maxsize=4096)
def matches_pattern(pattern: str, value: str) -> bool:
    for sub_pat, sub_pat_val in zip(pattern.split("-"), value.split("-")):
        for el1, el2 in zip(sub_pat.split('|'), sub_pat_val.split('|')):
            if el1 == '*':
                continue
            elif el1 != el2:
                return False
    return True


def is_individual_part_of_the_original_indv(indv_key, indv_key_list):
    return indv_key in indv_key_list


def is_couple_part_of_a_group(couple_key, group_key_list):
    res = []

    couple_key_elems = couple_key.split('-')
    if len(couple_key_elems) != 2:
        print(f"Warning: Unexpected couple key format: {couple_key}")
        return res

    opt1 = f"{couple_key_elems[0]}-{couple_key_elems[1]}"
    opt2 = f"{couple_key_elems[1]}-{couple_key_elems[0]}"

    for grp_key in group_key_list:
        if matches_pattern(grp_key, opt1) or matches_pattern(grp_key, opt2):
            res.append(grp_key)
    return res


# Convert to string type once
synthetic_data = synthetic_data.astype({
    'indv_key': str,
    'group_key': str,
    'subgroup_key': str
})

results_df = results_df.astype({
    'indv_key': str,
    'couple_key': str
})

# Get unique groups
unique_groups = synthetic_data['group_key'].unique()


def analyze_group(group_key: str) -> pd.DataFrame:
    # Get individuals in this group from synthetic data
    group_synthetic_indv = set(
        synthetic_data[synthetic_data['group_key'] == group_key]['indv_key']
    )

    # Split group key into its two patterns
    group_patterns = group_key.split('-')
    subgroup1_pattern, subgroup2_pattern = group_patterns[0], group_patterns[1]

    # Find exact individual matches in results
    exact_indv_matches = [
        key for key in results_df['indv_key'].unique()
        if is_individual_part_of_the_original_indv(key, group_synthetic_indv)
    ]

    # For couples, we need to check if both individuals in the couple are exact matches
    exact_couple_matches = []
    for couple_key in results_df['couple_key'].unique():
        indv1, indv2 = couple_key.split('-')
        if (is_individual_part_of_the_original_indv(indv1, group_synthetic_indv) and
                is_individual_part_of_the_original_indv(indv2, group_synthetic_indv)):
            exact_couple_matches.append(couple_key)

    # Find new individuals matching either group pattern but not in original data
    subgroup1_new_group_indv = []
    for key in results_df['indv_key'].unique():
        if matches_pattern(subgroup1_pattern, key):
            subgroup1_new_group_indv.append(key)

    subgroup2_new_group_indv = []
    for key in results_df['indv_key'].unique():
        if matches_pattern(subgroup2_pattern, key):
            subgroup2_new_group_indv.append(key)

    new_group_indv = subgroup1_new_group_indv + subgroup2_new_group_indv

    # Find new couples matching group pattern but not in original data
    new_group_couples = []
    for key in results_df['couple_key'].unique():
        if is_couple_part_of_a_group(key, [group_key]) and key not in exact_couple_matches:
            new_group_couples.append(key)

    subgroup1_analysis = pd.Series({
        'group_key': group_key,
        'subgroup_key': subgroup1_pattern,

        'individuals_part_of_original_data': exact_indv_matches,
        'couples_part_of_original_data': exact_couple_matches,
        'new_individuals_part_of_a_group_regex': new_group_indv,
        'new_individuals_part_of_a_subgroup_regex': subgroup1_new_group_indv,
        'new_couples_part_of_a_group_regex': new_group_couples,

        'num_exact_individual_matches': len(exact_indv_matches),
        'num_exact_couple_matches': len(exact_couple_matches),
        'num_new_group_individuals': len(new_group_indv),
        'num_new_group_couples': len(new_group_couples)
    })

    subgroup2_analysis = pd.Series({
        'group_key': group_key,
        'subgroup_key': subgroup2_pattern,

        'individuals_part_of_original_data': exact_indv_matches,
        'couples_part_of_original_data': exact_couple_matches,
        'new_individuals_part_of_a_group_regex': new_group_indv,
        'new_individuals_part_of_a_subgroup_regex': subgroup2_new_group_indv,
        'new_couples_part_of_a_group_regex': new_group_couples,

        'num_exact_individual_matches': len(exact_indv_matches),
        'num_exact_couple_matches': len(exact_couple_matches),
        'num_new_group_individuals': len(new_group_indv),
        'num_new_group_couples': len(new_group_couples)
    })

    res = pd.concat([subgroup1_analysis, subgroup2_analysis], axis=1).T

    return res


# Create group analysis DataFrame
analyzed_groups = []
for group_key in unique_groups:
    analyzed_groups.append(analyze_group(group_key))

group_analysis_df = pd.concat(analyzed_groups)

# %%
print('ddd')
