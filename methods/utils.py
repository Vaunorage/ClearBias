from collections import defaultdict
from typing import List

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from pandas import DataFrame

from data_generator.main import DataSchema
from dataclasses import dataclass


@dataclass
class GroupDefinition:
    group_size: int
    subgroup_bias: float
    similarity: float
    alea_uncertainty: float
    epis_uncertainty: float
    frequency: float
    avg_diff_outcome: int
    diff_subgroup_size: float
    subgroup1: dict  # {'Attr1_T': 3, 'Attr2_T': 1, 'Attr3_X': 3}
    subgroup2: dict  # {'Attr1_T': 2, 'Attr2_T': 2, 'Attr3_X': 2}


def train_sklearn_model(data, model_type='rf', model_params=None, target_col='class', sensitive_attrs=None,
                        test_size=0.2, random_state=42):
    """
    Train a sklearn model on the given data

    Parameters:
    -----------
    data : pandas DataFrame
        The input data containing features and target variable
    model_type : str, optional (default='rf')
        Type of model to train. Options:
        - 'rf': Random Forest
        - 'svm': Support Vector Machine
        - 'lr': Logistic Regression
        - 'dt': Decision Tree
    model_params : dict, optional
        Parameters for the model. If None, default parameters will be used
    target_col : str, optional (default='class')
        Name of the target column in data
    sensitive_attrs : list, optional
        List of sensitive attribute columns to exclude from features
    test_size : float, optional (default=0.2)
        Proportion of dataset to include in the test split
    random_state : int, optional (default=42)
        Random state for reproducibility

    Returns:
    --------
    model : sklearn model
        Trained model
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training target
    y_test : array-like
        Test target
    feature_names : list
        List of feature names in order
    """

    # Default parameters for each model type
    np.random.seed(random_state)
    sklearn.utils.check_random_state(random_state)

    # If using parallel processing (especially for RandomForest), set n_jobs=1
    default_params = {
        'rf': {
            'n_estimators': 100,
            'random_state': random_state,
            'n_jobs': 1  # Add this to ensure deterministic behavior
        },
        'svm': {'kernel': 'rbf', 'random_state': random_state},
        'lr': {'max_iter': 1000, 'random_state': random_state},
        'dt': {'random_state': random_state}
    }
    # Select model parameters
    params = model_params if model_params is not None else default_params[model_type]

    # Initialize model based on type
    model_map = {
        'rf': RandomForestClassifier,
        'svm': SVC,
        'lr': LogisticRegression,
        'dt': DecisionTreeClassifier
    }

    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types are: {list(model_map.keys())}")

    # Prepare features and target
    drop_cols = [target_col]
    X = data.drop(drop_cols, axis=1)
    feature_names = list(X.columns)  # Store feature names
    y = data[target_col]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Create and train model
    model = model_map[model_type](**params)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, feature_names


def reformat_discrimination_results(non_float_df, original_df) -> List[GroupDefinition]:
    """
    Reformats the discrimination results into the desired format with GroupDefinition objects
    for all combinations of protected attributes ending with '_T' present in the dataset.
    Ensures each unique subgroup pair is only included once.
    """
    # Initialize empty list for GroupDefinition objects and set for tracking unique pairs
    group_definitions = []
    seen_pairs = set()
    group_size = {}

    # Get all protected attribute columns ending with '_T'
    protected_attrs = [col for col in non_float_df.columns if col.endswith('_T')]

    for el in non_float_df['case_id'].unique():
        pair_df = non_float_df[non_float_df['case_id'] == el]
        if pair_df.shape[0] != 2:
            continue

        subgroup1_protected_attr = pair_df[protected_attrs].iloc[0]
        subgroup2_protected_attr = pair_df[protected_attrs].iloc[1]

        # Create a hashable representation of the subgroup pair (sorted to ensure consistent ordering)
        pair_key = tuple(sorted([
            tuple(subgroup1_protected_attr.items()),
            tuple(subgroup2_protected_attr.items())
        ]))

        # Skip if we've already processed this pair
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        subgroup1_data = original_df[(original_df[protected_attrs] == subgroup1_protected_attr.tolist()).all(axis=1)]
        subgroup2_data = original_df[(original_df[protected_attrs] == subgroup2_protected_attr.tolist()).all(axis=1)]

        if len(subgroup1_data) == 0 or len(subgroup2_data) == 0:
            continue

        subgroup1_outcomes = subgroup1_data['outcome'].values
        subgroup2_outcomes = subgroup2_data['outcome'].values

        avg_diff_outcome = abs(float(np.mean(subgroup2_outcomes) - np.mean(subgroup1_outcomes)))
        if avg_diff_outcome == 0:  # Skip if no discrimination
            continue

        if tuple(subgroup1_protected_attr.items()) in group_size:
            group_size[tuple(subgroup1_protected_attr.items())]['nb'] += 1
        else:
            group_size[tuple(subgroup1_protected_attr.items())] = {'nb': 1, 'size': len(subgroup1_data)}

        if tuple(subgroup2_protected_attr.items()) in group_size:
            group_size[tuple(subgroup2_protected_attr.items())]['nb'] += 1
        else:
            group_size[tuple(subgroup2_protected_attr.items())] = {'nb': 1, 'size': len(subgroup2_data)}

        # Create GroupDefinition object
        group_def = {
            'group_size': (len(subgroup1_data) + len(subgroup2_data)) / len(original_df),
            'subgroup_bias': abs(avg_diff_outcome),
            'similarity': np.mean([
                len(set(subgroup1_data[attr].unique()) & set(subgroup2_data[attr].unique())) /
                len(set(subgroup1_data[attr].unique()) | set(subgroup2_data[attr].unique()))
                for attr in original_df.columns if not attr.endswith('_T') and attr != 'outcome'
            ]),
            'alea_uncertainty': 0, 'epis_uncertainty': 0, 'frequency': 1,
            'avg_diff_outcome': avg_diff_outcome,
            'diff_subgroup_size': abs(len(subgroup1_data) - len(subgroup2_data)) / (
                    len(subgroup1_data) + len(subgroup2_data)),
            'subgroup1': subgroup1_protected_attr,
            'subgroup2': subgroup2_protected_attr
        }

        group_definitions.append(group_def)

    group_definitions_res = []
    for gr in group_definitions:
        subgroup1_key_pair = tuple(gr['subgroup1'].items())
        subgroup2_key_pair = tuple(gr['subgroup2'].items())

        subgroup1_size = group_size[subgroup1_key_pair]['size'] / group_size[subgroup1_key_pair]['nb']
        subgroup2_size = group_size[subgroup2_key_pair]['size'] / group_size[subgroup2_key_pair]['nb']

        gr['group_size'] = subgroup1_size + subgroup2_size
        group_definitions_res.append(GroupDefinition(**gr))

    return group_definitions_res


def convert_to_non_float_rows(df: pd.DataFrame, schema: DataSchema):
    df_copy = df[schema.attr_names].copy().astype(int)
    df_res = df.copy()
    df_res[schema.attr_names] = df_copy
    return df_res


def compare_discriminatory_groups(original_groups, synthetic_groups):
    """
    Compare discriminatory groups between original and synthetic data based on protected attribute values.
    
    Args:
        original_groups: List of discriminatory groups from original data
        synthetic_groups: List of discriminatory groups from synthetic data
        
    Returns:
        dict: Dictionary containing comparison metrics
    """
    matched_groups = []
    unmatched_original = []

    # Helper function to create a hashable representation of subgroups
    def get_subgroups_hash(group):
        subgroup1_items = tuple(sorted(group.subgroup1.items()))
        subgroup2_items = tuple(sorted(group.subgroup2.items()))
        return tuple(sorted([subgroup1_items, subgroup2_items]))

    # Create dictionaries for faster lookup
    synth_groups_dict = {get_subgroups_hash(group): group for group in synthetic_groups}

    # Compare each original group with synthetic groups
    for orig_group in original_groups:
        orig_hash = get_subgroups_hash(orig_group)
        if orig_hash in synth_groups_dict:
            matched_groups.append((orig_group, synth_groups_dict[orig_hash]))
        else:
            unmatched_original.append(orig_group)

    # Calculate metrics
    total_original_size = sum(group.group_size for group in original_groups)
    total_matched_size = sum(group.group_size for group, _ in matched_groups)

    return {
        'matched_groups': matched_groups,
        'unmatched_original_groups': unmatched_original,
        'total_groups_matched': len(matched_groups),
        'total_original_groups': len(original_groups),
        'coverage_ratio': total_matched_size / total_original_size if total_original_size > 0 else 0,
        'total_matched_size': total_matched_size,
        'total_original_size': total_original_size
    }
