import json
import re
from typing import List, Tuple
from methods.exp_ga.algo import run_expga
from methods.biasscan.algo import run_bias_scan
from methods.ml_check.algo import run_mlcheck
from methods.aequitas.algo import run_aequitas
from scipy.stats import stats
from sqlalchemy import create_engine
from data_generator.main2 import generate_data
import sqlite3
from datetime import datetime
import pandas as pd
from scipy import stats

from path import HERE

# %%

DB_PATH = HERE.joinpath("experiments/discrimination_detection_results2.db")


def init_database(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS discrimination_detection_results2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Algorithm TEXT,
            Num_Attributes INTEGER,
            Prop_Protected_Attr REAL,
            Num_Groups INTEGER,
            Max_Group_Size INTEGER,
            Correct_Couple_Detection_Rate REAL,
            Total_Couples_Detected INTEGER,
            True_Positives INTEGER,
            False_Positives INTEGER,
            Prop_Original_Individuals REAL,
            Prop_New_Individuals REAL,
            Prop_Original_in_Group REAL,
            Prop_New_in_Group REAL,
            Prop_Groups_Detected REAL,
            date TEXT
        )
    ''')
    conn.commit()


def get_completed_experiments(conn: sqlite3.Connection) -> List[Tuple[str, str, str]]:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT algorithm, dataset, protected_attribute 
        FROM discrimination_detection_results2
    """)
    return cursor.fetchall()


def experiment_completed(conn, algorithm, nb_attributes, prop_protected_attr, nb_groups, max_group_size):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 1 FROM discrimination_detection_results2
        WHERE Algorithm = ? AND Num_Attributes = ? AND Prop_Protected_Attr = ? 
        AND Num_Groups = ? AND Max_Group_Size = ?
        LIMIT 1
    """, (algorithm, nb_attributes, prop_protected_attr, nb_groups, max_group_size))
    return cursor.fetchone() is not None


def save_results_to_sqlite(group_details, results_table, attribute_trends_table):
    def stringify_lists(value):
        if isinstance(value, list):
            return json.dumps(value)
        return value

    conn = sqlite3.connect(DB_PATH)

    def save_table(df, table_name):
        if not df.empty:
            # Get existing columns from the database table
            try:
                existing_df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 0", conn)
                existing_columns = existing_df.columns.tolist()

                # Filter the DataFrame to include only existing columns
                df_filtered = df[df.columns.intersection(existing_columns)]

                if not df_filtered.empty:
                    df_filtered.applymap(stringify_lists).to_sql(table_name, conn, if_exists='append', index=False)
                    print(f"Data saved successfully to {table_name}.")
                else:
                    print(f"Warning: No matching columns found for {table_name}. Skipping this table.")
                    print(f"DataFrame columns: {df.columns.tolist()}")
                    print(f"Existing table columns: {existing_columns}")
            except Exception as e:
                print(f"Error reading existing table {table_name}: {str(e)}")
                print("Creating new table with all columns.")
                df.applymap(stringify_lists).to_sql(table_name, conn, if_exists='append', index=False)

    save_table(group_details, 'group_details')
    save_table(results_table, 'overall_results')
    save_table(attribute_trends_table, 'attribute_trends')

    print("Data saving process completed.")


def matches_pattern(pattern_string: str, test_string: str) -> bool:
    regex_pattern = pattern_string.replace("*", r"\d").replace("|", r"\|").replace("-", r"\-")
    pattern = re.compile(f"^{regex_pattern}$")
    return bool(pattern.match(test_string))


def is_individual_part_of_the_original_indv(indv_key, indv_key_list):
    return indv_key in indv_key_list


def is_individual_part_of_a_group(indv_key, group_key_list):
    res = []
    for grp in group_key_list:
        opt1, opt2 = grp.split('-')
        if matches_pattern(opt1, indv_key) or matches_pattern(opt2, indv_key):
            res.append(grp)
    return res


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


def count_individuals_per_group(aequitas_results: pd.DataFrame, all_group_keys: set) -> dict:
    group_counts = {group: 0 for group in all_group_keys}

    for _, row in aequitas_results.iterrows():
        groups = row['couple_part_of_a_group']
        for group in groups:
            if group in group_counts:
                group_counts[group] += 1

    return group_counts


def create_group_individual_table(aequitas_results: pd.DataFrame, synthetic_data: pd.DataFrame,
                                  all_group_keys: set) -> pd.DataFrame:
    calculated_properties = [
        'calculated_epistemic', 'calculated_aleatoric', 'relevance',
        'calculated_magnitude', 'calculated_group_size', 'calculated_granularity',
        'calculated_intersectionality', 'calculated_uncertainty',
        'calculated_similarity', 'calculated_subgroup_ratio'
    ]

    # Create a dictionary of groups with their individuals, metrics, and couples
    group_data = {group: {'individuals': set(), 'metrics': [], 'couples': set()} for group in all_group_keys}

    # Process synthetic data
    for _, row in synthetic_data.iterrows():
        group = row['group_key']
        if group in group_data:
            group_data[group]['individuals'].add(row['indv_key'])
            group_data[group]['metrics'].append({prop: row[prop] for prop in calculated_properties})

    # Process Aequitas results
    detected_groups = set()
    for _, row in aequitas_results.iterrows():
        for group in row['couple_part_of_a_group']:
            if group in group_data:
                detected_groups.add(group)
                indv1, indv2 = row['couple_key'].split('-')
                group_data[group]['individuals'].update([indv1, indv2])
                group_data[group]['couples'].add(row['couple_key'])

    # Create result data
    result_data = []
    for group, data in group_data.items():
        base_row = {
            'group_key': group,
            'individuals': list(data['individuals']),
            'num_individuals': len(data['individuals']),
            'detected': group in detected_groups,
            'couple_keys': list(data['couples'])
        }

        if data['metrics']:
            for metrics in data['metrics']:
                row = base_row.copy()
                row.update(metrics)
                result_data.append(row)
        else:
            row = base_row.copy()
            row.update({prop: None for prop in calculated_properties})
            result_data.append(row)

    return pd.DataFrame(result_data)


def evaluate_discrimination_detection(synthetic_data: pd.DataFrame, results_df: pd.DataFrame) -> tuple:
    if results_df.empty:
        return pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames

    synthetic_data['indv_key'] = synthetic_data['indv_key'].astype(str)
    synthetic_data['group_key'] = synthetic_data['group_key'].astype(str)
    synthetic_data['subgroup_key'] = synthetic_data['subgroup_key'].astype(str)

    results_df['indv_key'] = results_df['indv_key'].astype(str)
    results_df['couple_key'] = results_df['couple_key'].astype(str)

    all_indv_keys = set(synthetic_data['indv_key'])
    all_group_keys = set(synthetic_data['group_key'])

    results_df['indv_in_original_indv'] = results_df['indv_key'].apply(
        lambda x: is_individual_part_of_the_original_indv(x, all_indv_keys))

    # Check if 'indv_part_of_group' column exists, if not, create it
    if 'indv_part_of_group' not in results_df.columns:
        results_df['indv_part_of_group'] = results_df['indv_key'].apply(
            lambda x: is_individual_part_of_a_group(x, all_group_keys))

    results_df['couple_part_of_a_group'] = results_df['couple_key'].apply(
        lambda x: is_couple_part_of_a_group(x, all_group_keys))

    results_df['correct_detection'] = results_df['couple_part_of_a_group'].apply(lambda x: len(x) > 0)

    # Calculate evaluation metrics
    total_couples = len(results_df)
    true_positives = results_df['correct_detection'].sum()
    false_positives = total_couples - true_positives

    # Proportion of Original Individuals
    total_individuals = len(set(results_df['indv_key']))
    original_individuals = sum(results_df['indv_in_original_indv'])
    p_original = original_individuals / total_individuals if total_individuals > 0 else 0

    # Proportion of New Individuals
    p_new = 1 - p_original

    # Proportion of original individuals that belong to a group
    original_in_group = sum(results_df[results_df['indv_in_original_indv']]['indv_part_of_group'].apply(len) > 0)
    p_group_original = original_in_group / original_individuals if original_individuals > 0 else 0

    # Proportion of new individuals that belong to a group
    new_individuals = total_individuals - original_individuals
    new_in_group = sum(
        results_df[~results_df['indv_in_original_indv']]['indv_part_of_group'].apply(len) > 0)
    p_group_new = new_in_group / new_individuals if new_individuals > 0 else 0

    # Correct Couple Detection Rate (Precision)
    r_correct = true_positives / total_couples if total_couples > 0 else 0

    # Proportion of Groups Detected
    detected_groups = set()
    for groups in results_df['couple_part_of_a_group']:
        detected_groups.update(groups)
    p_groups = len(detected_groups) / len(all_group_keys) if len(all_group_keys) > 0 else 0

    # Calculate average values for metrics from synthetic data
    calculated_properties = [
        'calculated_epistemic', 'calculated_aleatoric', 'relevance',
        'calculated_magnitude', 'calculated_group_size', 'calculated_granularity',
        'calculated_intersectionality', 'calculated_uncertainty',
        'calculated_similarity', 'calculated_subgroup_ratio'
    ]
    avg_metrics = synthetic_data[calculated_properties].mean().to_dict()
    avg_metrics = {f"Average {k.replace('calculated_', '').capitalize()}": v for k, v in avg_metrics.items()}

    # Prepare the evaluation metrics
    metrics = {
        "Proportion of Original Individuals": p_original,
        "Proportion of New Individuals": p_new,
        "Proportion of original individuals that belong to a group": p_group_original,
        "Proportion of new individuals that belong to a group": p_group_new,
        "Correct Couple Detection Rate": r_correct,
        "Total Couples Detected": total_couples,
        "True Positives": true_positives,
        "False Positives": false_positives,
        "Proportion of Groups Detected": p_groups,
        **avg_metrics
    }

    # Add metrics to results_df
    for key, value in metrics.items():
        results_df[key] = value

    # Create group-individual table
    group_individual_table = create_group_individual_table(results_df, synthetic_data, all_group_keys)

    return results_df, group_individual_table


# %%
# aequitas_results, group_details = evaluate_discrimination_detection(ge.dataframe, results_df)


# %%


def run_test_suite():
    conn = sqlite3.connect(DB_PATH)  # Replace with your actual database file path
    init_database(conn)
    test_configurations = [
        # Varying number of attributes
        {"nb_attributes": 3, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 5, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 20, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 50, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},

        # Varying proportion of protected attributes
        {"nb_attributes": 10, "prop_protected_attr": 0.05, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.1, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.3, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.5, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.8, "nb_groups": 50, "max_group_size": 100},

        # Varying number of discriminatory groups
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 10, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 25, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 100, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 200, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 500, "max_group_size": 100},

        # Varying max group size (to test different magnitudes of discrimination)
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 20},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 50},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 200},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 500},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 1000},
    ]

    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Algorithm-specific parameters
    aequitas_params = {
        "model_type": "RandomForest",
        "perturbation_unit": 1,
        "threshold": 0,
        "global_iteration_limit": 100,
        "local_iteration_limit": 10
    }

    bias_scan_params = {
        "test_size": 0.3,
        "random_state": 42,
        "n_estimators": 200,
        "bias_scan_num_iters": 100,
        "bias_scan_scoring": 'Poisson',
        "bias_scan_favorable_value": 'high',
        "bias_scan_mode": 'ordinal'
    }

    expga_params = {
        "threshold": 0.5,
        "threshold_rank": 0.5,
        "max_global": 50,
        "max_local": 50
    }

    mlcheck_params = {
        "iteration_no": 1
    }

    algorithms = [
        ("Aequitas", run_aequitas, aequitas_params),
        # ("BiasScan", run_bias_scan, bias_scan_params),
        ("ExpGA", run_expga, expga_params),
        ("MLCheck", run_mlcheck, mlcheck_params)
    ]

    results = []
    attribute_trends = []
    all_group_details = []

    calculated_properties = [
        'calculated_epistemic', 'calculated_aleatoric', 'relevance',
        'calculated_magnitude', 'calculated_group_size', 'calculated_granularity',
        'calculated_intersectionality', 'calculated_uncertainty',
        'calculated_similarity', 'calculated_subgroup_ratio'
    ]

    for config in test_configurations:
        for algo_name, algo_func, algo_params in algorithms:
            if experiment_completed(conn, algo_name, config["nb_attributes"], config["prop_protected_attr"],
                                    config["nb_groups"], config["max_group_size"]):
                print(f"Skipping experiment: {algo_name} with {config} (already completed)")
                continue

            ge = generate_data(
                nb_attributes=config["nb_attributes"],
                min_number_of_classes=2,
                max_number_of_classes=6,
                prop_protected_attr=config["prop_protected_attr"],
                nb_groups=config["nb_groups"],
                max_group_size=config["max_group_size"],
                categorical_outcome=True,
                nb_categories_outcome=4
            )

            if algo_name == "Aequitas":
                algo_results, _ = algo_func(
                    ge.training_dataframe,
                    col_to_be_predicted=ge.outcome_column,
                    sensitive_param_name_list=ge.protected_attributes,
                    **algo_params
                )
            else:
                algo_results, _ = algo_func(ge, **algo_params)

            if algo_results.empty:
                print(f"Warning: {algo_name} produced empty results for the current configuration.")
                continue

            eval_results, group_details = evaluate_discrimination_detection(ge.dataframe, algo_results)

            if eval_results.empty or group_details.empty:
                print(f"Warning: Evaluation produced empty results for {algo_name} with the current configuration.")
                continue

            group_details['Algorithm'] = algo_name
            for key, value in config.items():
                group_details[key] = value
            for key, value in algo_params.items():
                group_details[f'algo_{key}'] = value

            all_group_details.append(group_details)

            results.append({
                "Algorithm": algo_name,
                "Num_Attributes": config["nb_attributes"],
                "Prop_Protected_Attr": config["prop_protected_attr"],
                "Num_Groups": config["nb_groups"],
                "Max_Group_Size": config["max_group_size"],
                "Correct_Couple_Detection_Rate": eval_results["Correct Couple Detection Rate"].iloc[0],
                "Total_Couples_Detected": eval_results["Total Couples Detected"].iloc[0],
                "True_Positives": eval_results["True Positives"].iloc[0],
                "False_Positives": eval_results["False Positives"].iloc[0],
                "Prop_Original_Individuals": eval_results["Proportion of Original Individuals"].iloc[0],
                "Prop_New_Individuals": eval_results["Proportion of New Individuals"].iloc[0],
                "Prop_Original_in_Group":
                    eval_results["Proportion of original individuals that belong to a group"].iloc[0],
                "Prop_New_in_Group": eval_results["Proportion of new individuals that belong to a group"].iloc[0],
                "Prop_Groups_Detected": eval_results["Proportion of Groups Detected"].iloc[0]
            })

            for attr in calculated_properties:
                detected_groups = group_details[group_details['detected'] == True]
                undetected_groups = group_details[group_details['detected'] == False]

                if not detected_groups.empty and not undetected_groups.empty:
                    t_stat, p_value = stats.ttest_ind(detected_groups[attr], undetected_groups[attr])

                    attribute_trends.append({
                        "Algorithm": algo_name,
                        "Attribute": attr,
                        "Num_Attributes": config["nb_attributes"],
                        "Prop_Protected_Attr": config["prop_protected_attr"],
                        "Num_Groups": config["nb_groups"],
                        "Max_Group_Size": config["max_group_size"],
                        "Detected_Mean": detected_groups[attr].mean(),
                        "Undetected_Mean": undetected_groups[attr].mean(),
                        "T_Statistic": t_stat,
                        "P_Value": p_value
                    })

            results_df = pd.DataFrame(results[-1:])
            attribute_trends_df = pd.DataFrame(attribute_trends[-1:])
            all_group_details_df = pd.concat(all_group_details[-1:], ignore_index=True)

            results_df['date'] = current_datetime
            attribute_trends_df['date'] = current_datetime
            all_group_details_df['date'] = current_datetime

            save_results_to_sqlite(all_group_details_df, results_df, attribute_trends_df)

    conn.close()

    results_df = pd.DataFrame(results)
    attribute_trends_df = pd.DataFrame(attribute_trends)
    all_group_details_df = pd.concat(all_group_details, ignore_index=True)

    results_df['date'] = current_datetime
    attribute_trends_df['date'] = current_datetime
    all_group_details_df['date'] = current_datetime

    return results_df, attribute_trends_df, all_group_details_df


# %% Run the test suite and display results
results_df, attribute_trends_df, all_group_details_df = run_test_suite()

# %%

import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt

# Connect to the SQLite database
db_path = 'methods/aequitas/discrimination_detection_results.db'
conn = sqlite3.connect(db_path)

# Load data from the database into pandas DataFrames
overall_results = pd.read_sql_query("SELECT * FROM overall_results", conn)
group_details = pd.read_sql_query("SELECT * FROM group_details", conn)

# Close the connection after loading the data
conn.close()

# ------------------- Step 1: Summary Tables ------------------- #

# Correct column names according to the provided data
# Summary table 1: General performance metrics (from overall_results)
performance_summary = overall_results[[
    'Algorithm', 'Correct Couple Detection Rate', 'Prop Original Individuals',
    'Prop New Individuals', 'Prop Groups Detected', 'True Positives',
    'False Positives', 'Total Couples Detected'
]].groupby('Algorithm').mean()

# Display the performance summary table
print("Performance Summary:")
print(performance_summary.to_string())

# Summary table 2: Group property statistics for detected and undetected groups
group_stats = group_details.groupby('detected')[[
    'calculated_epistemic', 'calculated_aleatoric', 'relevance', 'calculated_magnitude',
    'calculated_group_size', 'calculated_granularity', 'calculated_intersectionality',
    'calculated_uncertainty', 'calculated_similarity', 'calculated_subgroup_ratio'
]].mean()

print("\nGroup Property Statistics (Detected vs Undetected):")
print(group_stats.to_string())

# ------------------- Step 2: Visualizations ------------------- #

# Visualization 1: Bar plot of Correct Couple Detection Rate for each algorithm
plt.figure(figsize=(10, 6))
sns.barplot(x='Algorithm', y='Correct Couple Detection Rate', data=overall_results)
plt.title('Correct Couple Detection Rate by Algorithm')
plt.ylabel('Correct Couple Detection Rate')
plt.xlabel('Algorithm')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization 2: Heatmap of group property correlations with detection likelihood
correlation_data = group_details[['detected', 'calculated_epistemic', 'calculated_aleatoric',
                                  'relevance', 'calculated_magnitude', 'calculated_group_size',
                                  'calculated_granularity', 'calculated_intersectionality',
                                  'calculated_uncertainty', 'calculated_similarity',
                                  'calculated_subgroup_ratio']]

# Convert 'detected' column from boolean to integer for correlation calculation
correlation_data['detected'] = correlation_data['detected'].astype(int)

# Compute correlation matrix
correlation_matrix = correlation_data.corr()

# Generate the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation between Group Properties and Detection Likelihood')
plt.tight_layout()
plt.show()

# Visualization 3: Pair plot of group properties, colored by detection status
plt.figure(figsize=(12, 10))
sns.pairplot(group_details, hue='detected', vars=[
    'calculated_epistemic', 'calculated_aleatoric', 'relevance', 'calculated_magnitude',
    'calculated_group_size', 'calculated_granularity', 'calculated_intersectionality',
    'calculated_uncertainty', 'calculated_similarity', 'calculated_subgroup_ratio'
])
plt.suptitle('Pair Plot of Group Properties by Detection Status', y=1.02)
plt.tight_layout()
plt.show()

# ------------------- Step 3: Statistical Analysis for Results ------------------- #

from scipy.stats import ttest_ind

# Perform t-tests to check if there is a statistically significant difference between
# detected and undetected groups for each property

ttest_results = {}
for column in ['calculated_epistemic', 'calculated_aleatoric', 'relevance', 'calculated_magnitude',
               'calculated_group_size', 'calculated_granularity', 'calculated_intersectionality',
               'calculated_uncertainty', 'calculated_similarity', 'calculated_subgroup_ratio']:
    detected_groups = group_details[group_details['detected'] == True][column]
    undetected_groups = group_details[group_details['detected'] == False][column]

    t_stat, p_value = ttest_ind(detected_groups, undetected_groups, equal_var=False)
    ttest_results[column] = {'t-statistic': t_stat, 'p-value': p_value}

# Convert the t-test results to a DataFrame for better readability
ttest_df = pd.DataFrame(ttest_results).T
print("\nT-test Results (Detected vs Undetected Groups):")
print(ttest_df.to_string())

# Highlight properties with statistically significant differences (p-value < 0.05)
significant_differences = ttest_df[ttest_df['p-value'] < 0.05]
print("\nSignificant Differences Between Detected and Undetected Groups:")
print(significant_differences.to_string())
