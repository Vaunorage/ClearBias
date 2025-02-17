from collections import defaultdict

import numpy as np
import pandas as pd
from pandas import DataFrame


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
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    # Default parameters for each model type
    default_params = {
        'rf': {'n_estimators': 100, 'random_state': random_state},
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


def reformat_discrimination_results(df):
    """
    Reformats the discrimination results into the desired format with group and subgroup columns
    """
    # Initialize empty lists for the new format
    reformatted_data = []

    # Group counter to assign group numbers
    group_counter = 0

    # Get pairs using couple_key
    paired_data = df[df['couple_key'] != -1].copy()

    # Process each unique pair
    for couple_key in paired_data['couple_key'].unique():
        pair = paired_data[paired_data['couple_key'] == couple_key].copy()

        if len(pair) != 2:
            continue

        # Sort by outcome to ensure consistent ordering
        pair = pair.sort_values('outcome')

        # Extract protected attributes
        row_data = {
            'group': group_counter,
            'subgroup': 0,  # First individual in pair
            'Attr7_T': pair.iloc[0]['Attr7_T'],
            'Attr8_T': pair.iloc[0]['Attr8_T'],
            'average_gap': pair.iloc[1]['outcome'] - pair.iloc[0]['outcome'],
            'num_cases': 1,  # Each pair is one case
            'std_gap': 0,  # For individual pairs, std is 0
            'max_gap': pair.iloc[1]['outcome'] - pair.iloc[0]['outcome'],
            'min_gap': pair.iloc[1]['outcome'] - pair.iloc[0]['outcome']
        }
        reformatted_data.append(row_data)

        # Add second individual of the pair
        row_data = row_data.copy()
        row_data['subgroup'] = 1  # Second individual in pair
        row_data['Attr7_T'] = pair.iloc[1]['Attr7_T']
        row_data['Attr8_T'] = pair.iloc[1]['Attr8_T']
        reformatted_data.append(row_data)

        group_counter += 1

    # Convert to DataFrame
    result_df = pd.DataFrame(reformatted_data)

    # Ensure all columns are present and in the correct order
    columns = ['group', 'subgroup', 'Attr7_T', 'Attr8_T', 'average_gap',
               'num_cases', 'std_gap', 'max_gap', 'min_gap']
    result_df = result_df[columns]

    return result_df