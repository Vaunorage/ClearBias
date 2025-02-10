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
