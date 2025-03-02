from typing import List

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_generator.main import GroupDefinition, DataSchema


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
    print("Pre-computing attribute combinations...")
    attr_combinations = {}
    for attrs in tqdm(non_float_df[protected_attrs].drop_duplicates().itertuples(index=False),
                      desc="Attribute combinations"):
        mask = (original_df[protected_attrs] == attrs).all(axis=1)
        attr_combinations[tuple(attrs)] = {
            'data': original_df[mask],
            'outcomes': original_df[mask]['outcome'].values
        }

    # Pre-compute similarity attributes
    similarity_attrs = [attr for attr in original_df.columns
                        if not attr.endswith('_T') and attr != 'outcome']

    subgroups_infos = {}

    print("Processing valid cases...")
    for pair_df in tqdm(valid_cases, desc="Processing case pairs"):
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

        if subgroup1_attrs not in subgroups_infos:
            subgroups_infos[subgroup1_attrs] = {'size': subgroup1_len, 'nb': 1}
        else:
            subgroups_infos[subgroup1_attrs]['nb'] += 1

        if subgroup2_attrs not in subgroups_infos:
            subgroups_infos[subgroup2_attrs] = {'size': subgroup2_len, 'nb': 1}
        else:
            subgroups_infos[subgroup2_attrs]['nb'] += 1

        group_def = {
            'subgroup_bias': avg_diff_outcome,
            'similarity': similarity,
            'alea_uncertainty': 0,
            'epis_uncertainty': 0,
            'frequency': 1,
            'avg_diff_outcome': avg_diff_outcome,
            'diff_subgroup_size': abs(subgroup1_len - subgroup2_len) /
                                  (subgroup1_len + subgroup2_len),
            'subgroup1': {k: v for k, v in zip(protected_attrs, subgroup1_attrs)},
            'subgroup2': {k: v for k, v in zip(protected_attrs, subgroup2_attrs)}
        }

        group_definitions.append((group_def, subgroup1_attrs, subgroup2_attrs))

    n_group_definitions = []
    for group_def, subgroup1_attrs, subgroup2_attrs in group_definitions:
        subgroup1_len = subgroups_infos[subgroup1_attrs]['size'] / subgroups_infos[subgroup1_attrs]['nb']
        subgroup2_len = subgroups_infos[subgroup2_attrs]['size'] / subgroups_infos[subgroup2_attrs]['nb']
        group_def['group_size'] = int(subgroup1_len + subgroup2_len)
        if group_def['group_size'] >= 2:
            # print(group_def['group_size'])
            n_group_definitions.append(GroupDefinition(**group_def))

    return n_group_definitions


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


def check_groups_in_synthetic_data(data_obj_synth, predefined_groups_origin):
    """
    Check if the predefined groups from original data are present in the synthetic dataframe.
    A group is considered present only if both its subgroups exist independently in the synthetic data.

    Args:
        data_obj_synth: The synthetic data object containing the dataframe
        predefined_groups_origin: List of predefined groups from original data

    Returns:
        dict: Dictionary containing results of which groups are present and which are missing
    """
    results = {
        'present_groups': [],
        'missing_groups': [],
        'total_groups': len(predefined_groups_origin)
    }

    synth_df = data_obj_synth.dataframe

    for group in predefined_groups_origin:
        # Check if subgroup1 exists
        subgroup1_mask = None
        for feature, value in group.subgroup1.items():
            condition = (synth_df[feature] == value)
            if subgroup1_mask is None:
                subgroup1_mask = condition
            else:
                subgroup1_mask = subgroup1_mask & condition

        # Check if subgroup2 exists
        subgroup2_mask = None
        for feature, value in group.subgroup2.items():
            condition = (synth_df[feature] == value)
            if subgroup2_mask is None:
                subgroup2_mask = condition
            else:
                subgroup2_mask = subgroup2_mask & condition

        # A group is present only if both subgroups are present in the data
        if subgroup1_mask.any() and subgroup2_mask.any():
            results['present_groups'].append({
                'subgroup1_features': group.subgroup1,
                'subgroup2_features': group.subgroup2,
                'original_size': group.group_size,
                'subgroup1_size': subgroup1_mask.sum(),
                'subgroup2_size': subgroup2_mask.sum()
            })
        else:
            missing_reason = []
            if not subgroup1_mask.any():
                missing_reason.append("subgroup1 not found")
            if not subgroup2_mask.any():
                missing_reason.append("subgroup2 not found")

            results['missing_groups'].append({
                'subgroup1_features': group.subgroup1,
                'subgroup2_features': group.subgroup2,
                'original_size': group.group_size,
                'reason': ', '.join(missing_reason)
            })

    results['groups_found'] = len(results['present_groups'])
    results['groups_missing'] = len(results['missing_groups'])
    results['coverage_percentage'] = (len(results['present_groups']) / len(predefined_groups_origin)) * 100

    return results
