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
    protected_attrs = [col for col in non_float_df.columns if col.endswith('_T')]
    group_definitions = []
    seen_pairs = set()

    # Pre-compute case pairs
    case_groups = non_float_df.groupby('case_id')
    valid_cases = [group for name, group in case_groups if len(group) == 2]

    # Pre-compute attribute combinations for original_df
    attr_combinations = {}
    for attrs in non_float_df[protected_attrs].drop_duplicates().itertuples(index=False):
        mask = (original_df[protected_attrs] == attrs).all(axis=1)
        attr_combinations[tuple(attrs)] = {
            'data': original_df[mask],
            'outcomes': original_df[mask]['outcome'].values
        }

    # Pre-compute similarity attributes
    similarity_attrs = [attr for attr in original_df.columns
                        if not attr.endswith('_T') and attr != 'outcome']

    for pair_df in valid_cases:
        subgroup1_attrs = tuple(pair_df[protected_attrs].iloc[0])
        subgroup2_attrs = tuple(pair_df[protected_attrs].iloc[1])

        pair_key = tuple(sorted([
            tuple(zip(protected_attrs, subgroup1_attrs)),
            tuple(zip(protected_attrs, subgroup2_attrs))
        ]))

        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        subgroup1 = attr_combinations[subgroup1_attrs]
        subgroup2 = attr_combinations[subgroup2_attrs]

        if len(subgroup1['data']) == 0 or len(subgroup2['data']) == 0:
            continue

        avg_diff_outcome = abs(np.mean(subgroup2['outcomes']) - np.mean(subgroup1['outcomes']))
        if avg_diff_outcome == 0:
            continue

        # Calculate similarity using vectorized operations
        similarity = np.mean([
            len(set(subgroup1['data'][attr].unique()) &
                set(subgroup2['data'][attr].unique())) /
            len(set(subgroup1['data'][attr].unique()) |
                set(subgroup2['data'][attr].unique()))
            for attr in similarity_attrs
        ])

        total_len = len(original_df)
        subgroup1_len = len(subgroup1['data'])
        subgroup2_len = len(subgroup2['data'])

        group_def = {
            'group_size': (subgroup1_len + subgroup2_len) / total_len,
            'subgroup_bias': avg_diff_outcome,
            'similarity': similarity,
            'alea_uncertainty': 0,
            'epis_uncertainty': 0,
            'frequency': 1,
            'avg_diff_outcome': avg_diff_outcome,
            'diff_subgroup_size': abs(subgroup1_len - subgroup2_len) /
                                  (subgroup1_len + subgroup2_len),
            'subgroup1': pd.Series(subgroup1_attrs, index=protected_attrs),
            'subgroup2': pd.Series(subgroup2_attrs, index=protected_attrs)
        }

        group_definitions.append(GroupDefinition(**group_def))

    return group_definitions


def convert_to_non_float_rows(df: pd.DataFrame, schema: DataSchema):
    df_copy = df[schema.attr_names].copy().astype(int)
    df_res = df.copy()
    df_res[schema.attr_names] = df_copy
    return df_res


def get_subgroups_hash(group):
    # Pre-sort subgroup items once
    sg1 = tuple(sorted(group.subgroup1.items()))
    sg2 = tuple(sorted(group.subgroup2.items()))
    # Compare tuples directly instead of sorting again
    return (sg1, sg2) if sg1 < sg2 else (sg2, sg1)


def compare_discriminatory_groups(original_groups, synthetic_groups):
    # Pre-compute hashes for synthetic groups
    synth_hashes = {get_subgroups_hash(group): group for group in synthetic_groups}

    # Pre-allocate lists with known sizes
    matched_pairs = []
    matched_pairs_size = 0
    total_original_size = 0

    # Single pass through original groups
    for orig_group in original_groups:
        total_original_size += orig_group.group_size
        orig_hash = get_subgroups_hash(orig_group)
        if orig_hash in synth_hashes:
            matched_pairs.append((orig_group, synth_hashes[orig_hash]))
            matched_pairs_size += orig_group.group_size

    return {
        'matched_groups': matched_pairs,
        'total_groups_matched': len(matched_pairs),
        'total_original_groups': len(original_groups),
        'coverage_ratio': matched_pairs_size / total_original_size if total_original_size > 0 else 0,
        'total_matched_size': matched_pairs_size,
        'total_original_size': total_original_size
    }
