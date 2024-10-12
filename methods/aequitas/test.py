from data_generator.main2 import generate_data
from methods.aequitas.algo import run_aequitas
import pandas as pd
from typing import Tuple, Dict, List

nb_attributes = 5

ge = generate_data(
    nb_attributes=nb_attributes,
    min_number_of_classes=8,
    max_number_of_classes=10,
    prop_protected_attr=0.4,
    nb_groups=20,
    max_group_size=100,
    categorical_outcome=True,
    nb_categories_outcome=4)

# %%
global_iteration_limit = 1000
local_iteration_limit = 10
model_type = "RandomForest"
results_df, model_scores = run_aequitas(ge.training_dataframe, col_to_be_predicted=ge.outcome_column,
                                        sensitive_param_name_list=ge.protected_attributes,
                                        perturbation_unit=1, model_type=model_type, threshold=0,
                                        global_iteration_limit=global_iteration_limit,
                                        local_iteration_limit=local_iteration_limit)


# %%


def evaluate_discrimination_detection(synthetic_data: pd.DataFrame, aequitas_results: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate how well the Aequitas algorithm finds discrimination cases within the synthetic data.

    :param synthetic_data: DataFrame containing the synthetic data with group_key, subgroup_key, and indv_key
    :param aequitas_results: DataFrame containing the results from the Aequitas algorithm
    :return: Tuple containing evaluation metrics, a DataFrame with detailed results, and a DataFrame with group information
    """
    # Prepare the synthetic data
    synthetic_data['indv_key'] = synthetic_data['indv_key'].astype(str)
    synthetic_data['group_key'] = synthetic_data['group_key'].astype(str)
    synthetic_data['subgroup_key'] = synthetic_data['subgroup_key'].astype(str)

    # Prepare the Aequitas results
    aequitas_results['indv_key'] = aequitas_results['indv_key'].astype(str)
    aequitas_results['couple_key'] = aequitas_results['couple_key'].astype(str)

    # Create a set of all indv_keys in the synthetic data for faster lookup
    synthetic_indv_keys = set(synthetic_data['indv_key'])

    # Create a set of all group_keys in the synthetic data
    all_group_keys = set(synthetic_data['group_key'])

    # Function to get the common group for a couple, ensuring they're from different subgroups
    def get_couple_group(couple_key: str) -> Tuple[List[str], bool, bool]:
        indv1, indv2 = couple_key.split('*')

        # Check if both indv_keys exist in synthetic data
        indv1_exists = indv1 in synthetic_indv_keys
        indv2_exists = indv2 in synthetic_indv_keys

        if not (indv1_exists and indv2_exists):
            return [], indv1_exists, indv2_exists

        group_subgroup1 = synthetic_data[synthetic_data['indv_key'] == indv1][['group_key', 'subgroup_key']]
        group_subgroup2 = synthetic_data[synthetic_data['indv_key'] == indv2][['group_key', 'subgroup_key']]

        common_groups = []
        for _, row1 in group_subgroup1.iterrows():
            for _, row2 in group_subgroup2.iterrows():
                if row1['group_key'] == row2['group_key'] and row1['subgroup_key'] != row2['subgroup_key']:
                    common_groups.append(row1['group_key'])

        return list(set(common_groups)), indv1_exists, indv2_exists  # Remove duplicates if any

    # Evaluate each couple in the Aequitas results
    aequitas_results['couple_evaluation'] = aequitas_results['couple_key'].apply(get_couple_group)
    aequitas_results['common_groups'] = aequitas_results['couple_evaluation'].apply(lambda x: x[0])
    aequitas_results['indv1_exists'] = aequitas_results['couple_evaluation'].apply(lambda x: x[1])
    aequitas_results['indv2_exists'] = aequitas_results['couple_evaluation'].apply(lambda x: x[2])
    aequitas_results['correct_detection'] = aequitas_results['common_groups'].apply(lambda x: len(x) > 0)

    # Calculate evaluation metrics
    total_couples = len(aequitas_results)
    correct_detections = aequitas_results['correct_detection'].sum()
    incorrect_detections = total_couples - correct_detections

    accuracy = correct_detections / total_couples if total_couples > 0 else 0

    # Calculate precision, recall, and F1 score
    true_positives = correct_detections
    false_positives = incorrect_detections
    false_negatives = len(all_group_keys) - true_positives  # Assuming each group should have been detected

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate the number of couples with missing individuals
    couples_with_missing_indv = \
        aequitas_results[~(aequitas_results['indv1_exists'] & aequitas_results['indv2_exists'])].shape[0]

    # Prepare the evaluation metrics

    aequitas_results["Accuracy"] = accuracy
    aequitas_results["Precision"] = precision
    aequitas_results["Recall"] = recall
    aequitas_results["F1 Score"] = f1_score
    aequitas_results["Total Couples Detected"] = total_couples
    aequitas_results["Correct Detections"] = correct_detections
    aequitas_results["Incorrect Detections"] = incorrect_detections
    aequitas_results["Couples with Missing Individuals"] = couples_with_missing_indv

    return aequitas_results


# %%
detailed_results = evaluate_discrimination_detection(ge.dataframe, results_df)
print(detailed_results)
