import time
import sqlite3
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from path import HERE

DB_PATH = HERE.joinpath("experiments/analyzing_methods/global/global_testing_res.db")


class MetaLearner:
    """
    Meta-learning system that learns across experiments to predict good hyperparameters
    based on dataset characteristics and model type.
    """

    def __init__(self, db_path=DB_PATH):
        """Initialize the meta-learner with a database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.meta_models = {}  # Dictionary to store trained meta-models for each method
        self._ensure_meta_learning_table_exists()

    def _ensure_meta_learning_table_exists(self):
        """Create a table to store meta-learning models if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS meta_learning_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            method TEXT NOT NULL,
            model_type TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            model_blob BLOB NOT NULL,
            meta_features TEXT NOT NULL,
            performance_metric REAL NOT NULL
        )
        ''')
        self.conn.commit()

    def collect_meta_data(self, method: str = None):
        """
        Collect data from previous experiments to use for meta-learning.

        Args:
            method: Optional method name to filter data

        Returns:
            DataFrame with dataset characteristics, hyperparameters, and performance metrics
        """
        # Query to join experiment_tracking and optuna_trials tables
        query = """
        SELECT 
            e.method, e.model_type, e.nb_attributes, e.prop_protected_attr, 
            e.nb_categories_outcome, e.params, 
            o.num_exact_couple_matches, o.num_new_group_couples, o.objective_value
        FROM 
            experiment_tracking e
        JOIN 
            optuna_trials o
        ON 
            e.method = o.method AND e.model_type = o.model_type AND
            e.nb_attributes = o.nb_attributes AND e.prop_protected_attr = o.prop_protected_attr AND
            e.nb_categories_outcome = o.nb_categories_outcome AND e.params = o.params
        WHERE 
            e.success = 1
        """

        if method:
            query += f" AND e.method = '{method}'"

        meta_data = pd.read_sql(query, self.conn)

        if len(meta_data) == 0:
            print("No meta-data available from previous experiments")
            return None

        # Parse the JSON parameter strings
        meta_data['params_dict'] = meta_data['params'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else {})

        # Extract individual parameters as columns
        for param_name in ['threshold_rank', 'max_global', 'max_local', 'threshold',
                           'step_size', 'cluster_num']:
            meta_data[param_name] = meta_data['params_dict'].apply(
                lambda x: x.get(param_name, np.nan))

        # Create combined feature for method and model_type
        meta_data['method_model'] = meta_data['method'] + '_' + meta_data['model_type']

        return meta_data

    def train_meta_model(self, method: str):
        """
        Train a meta-model for a specific method to predict good hyperparameters.

        Args:
            method: Method name to train model for

        Returns:
            Boolean indicating if training was successful
        """
        # Get data for this method
        meta_data = self.collect_meta_data(method)

        if meta_data is None or len(meta_data) < 10:  # Need sufficient data to train
            print(f"Insufficient data to train meta-model for {method}")
            return False

        # Identify features and target
        # Dataset characteristics and model type are features
        features = ['nb_attributes', 'prop_protected_attr', 'nb_categories_outcome']

        # One-hot encode model_type
        model_dummies = pd.get_dummies(meta_data['model_type'], prefix='model')
        meta_data = pd.concat([meta_data, model_dummies], axis=1)
        features.extend(model_dummies.columns)

        # Method-specific hyperparameters that we want to predict
        target_params = []

        if method == 'expga':
            target_params = ['threshold_rank', 'max_global', 'max_local', 'threshold']
        elif method == 'aequitas':
            target_params = ['max_global', 'max_local', 'step_size']
        elif method == 'adf':
            target_params = ['max_global', 'max_local', 'cluster_num', 'step_size']
        elif method == 'sg':
            target_params = ['cluster_num']

        if not target_params:
            print(f"No target parameters defined for method {method}")
            return False

        # Prepare the dataset
        X = meta_data[features].copy()
        y = meta_data[['objective_value'] + target_params].copy()

        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train models for each parameter
        self.meta_models[method] = {}
        self.meta_models[method]['features'] = features
        self.meta_models[method]['scaler'] = scaler
        self.meta_models[method]['target_params'] = target_params
        self.meta_models[method]['param_models'] = {}

        # For each parameter, train a Random Forest regressor
        for param in target_params:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y[param])
            self.meta_models[method]['param_models'][param] = model

        # Train a model to predict overall objective value
        objective_model = RandomForestRegressor(n_estimators=100, random_state=42)
        objective_model.fit(X_scaled, y['objective_value'])
        self.meta_models[method]['objective_model'] = objective_model

        print(f"Successfully trained meta-model for {method}")
        return True

    def predict_good_params(self, method: str, model_type: str, dataset_attr: dict) -> Optional[Dict]:
        """
        Predict good hyperparameters for a new experiment using meta-learning.

        Args:
            method: Method name
            model_type: Model type
            dataset_attr: Dictionary with dataset attributes

        Returns:
            Dictionary of predicted good parameters or None if prediction not possible
        """
        # Check if we have a trained meta-model for this method
        if method not in self.meta_models:
            # Try to train one
            success = self.train_meta_model(method)
            if not success:
                return None

        # Prepare input features
        features = self.meta_models[method]['features']
        scaler = self.meta_models[method]['scaler']

        # Create a new dataframe with the features
        X_new = pd.DataFrame([dataset_attr])

        # Add model_type one-hot encoding
        for feature in features:
            if feature.startswith('model_'):
                model_name = feature.split('_')[1]
                X_new[feature] = 1 if model_type == model_name else 0

        # Handle missing features
        for feature in features:
            if feature not in X_new.columns:
                X_new[feature] = 0

        # Ensure all features are present in the correct order
        X_new = X_new[features]

        # Scale features
        X_new_scaled = scaler.transform(X_new)

        # Make predictions for each parameter
        predicted_params = {}
        for param in self.meta_models[method]['target_params']:
            value = self.meta_models[method]['param_models'][param].predict(X_new_scaled)[0]

            # Apply constraints based on parameter type
            if param in ['threshold_rank', 'threshold', 'step_size']:
                value = max(0.01, min(0.99, value))  # Constrain to [0.01, 0.99]
            elif param in ['max_global', 'max_local', 'cluster_num']:
                value = max(10, int(value))  # Ensure positive integers

            predicted_params[param] = value

        # Add fixed parameters
        predicted_params['max_runtime_seconds'] = 800

        # Predict expected objective value
        expected_objective = self.meta_models[method]['objective_model'].predict(X_new_scaled)[0]
        print(f"Predicted objective value: {expected_objective:.4f}")

        return predicted_params

    def save_meta_model(self, method: str):
        """
        Save the trained meta-model to the database.

        Args:
            method: Method name
        """
        if method not in self.meta_models:
            print(f"No meta-model available for {method}")
            return

        # Serialize the meta-model
        import pickle
        model_blob = pickle.dumps(self.meta_models[method])

        # Get meta-features used
        meta_features = json.dumps(self.meta_models[method]['features'])

        # Calculate performance metric (can be refined)
        performance_metric = 0.0  # Placeholder

        # Get unique model types
        meta_data = self.collect_meta_data(method)
        model_types = meta_data['model_type'].unique() if meta_data is not None else []

        # Save a record for each model type
        timestamp = int(time.time())
        cursor = self.conn.cursor()

        for model_type in model_types:
            cursor.execute('''
            INSERT INTO meta_learning_models 
            (method, model_type, timestamp, model_blob, meta_features, performance_metric)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                method,
                model_type,
                timestamp,
                model_blob,
                meta_features,
                performance_metric
            ))

        self.conn.commit()
        print(f"Saved meta-model for {method} to database")

    def load_meta_model(self, method: str):
        """
        Load a meta-model from the database.

        Args:
            method: Method name

        Returns:
            Boolean indicating if loading was successful
        """
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT model_blob FROM meta_learning_models 
        WHERE method = ? 
        ORDER BY timestamp DESC LIMIT 1
        ''', (method,))

        result = cursor.fetchone()
        if result:
            # Deserialize the meta-model
            import pickle
            self.meta_models[method] = pickle.loads(result[0])
            print(f"Loaded meta-model for {method} from database")
            return True

        print(f"No meta-model found for {method} in database")
        return False

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()


class MetaOptuna:
    """
    Enhanced Optuna study that incorporates meta-learning for better starting points
    and more efficient hyperparameter optimization.
    """

    def __init__(self, method: str, model_type: str, dataset_attr: dict,
                 meta_learner: MetaLearner, db_path=DB_PATH):
        """
        Initialize a meta-learning enhanced Optuna study.

        Args:
            method: Method name
            model_type: Model type
            dataset_attr: Dictionary with dataset attributes
            meta_learner: MetaLearner instance
            db_path: Path to the database
        """
        self.method = method
        self.model_type = model_type
        self.dataset_attr = dataset_attr
        self.meta_learner = meta_learner
        self.db_path = db_path

        # Base parameters
        self.base_params = {
            'expga': {'threshold_rank': 0.5, 'max_global': 3000, 'max_local': 1000, 'threshold': 0.5,
                      'max_runtime_seconds': 800},
            'aequitas': {'max_global': 100, 'max_local': 1000, 'step_size': 1.0,
                         'max_runtime_seconds': 800},
            'adf': {'max_global': 20000, 'max_local': 100, 'cluster_num': 50, 'max_runtime_seconds': 800,
                    'step_size': 0.05},
            'sg': {'cluster_num': 50, 'max_runtime_seconds': 800}
        }

        # Parameter search spaces
        self.param_search_spaces = {
            'expga': {
                'threshold_rank': (0.1, 0.9),
                'max_global': (1000, 5000),
                'max_local': (500, 2000),
                'threshold': (0.1, 0.9)
            },
            'aequitas': {
                'max_global': (50, 300),
                'max_local': (500, 2000),
                'step_size': (0.5, 2.0)
            },
            'adf': {
                'max_global': (10000, 30000),
                'max_local': (50, 200),
                'cluster_num': (20, 100),
                'step_size': (0.01, 0.1)
            },
            'sg': {
                'cluster_num': (20, 100)
            }
        }

        # Get predicted good parameters from meta-learner
        self.meta_params = meta_learner.predict_good_params(method, model_type, dataset_attr)

        # Create Optuna study
        self.study = optuna.create_study(direction='maximize')

    def get_param_bounds(self, param_name):
        """Get the search bounds for a parameter."""
        method_search_space = self.param_search_spaces.get(self.method, {})
        return method_search_space.get(param_name, (0, 1))  # Default bounds

    def suggest_params(self, trial):
        """
        Suggest parameters for a trial, using meta-learning knowledge when available.

        Args:
            trial: Optuna trial

        Returns:
            Dictionary of suggested parameters
        """
        # Start with base parameters
        params = self.base_params[self.method].copy()

        # Get method-specific search space
        search_space = self.param_search_spaces.get(self.method, {})

        # Use meta-learning to guide search if available
        if self.meta_params:
            print(f"Using meta-learning to guide parameter search for trial {trial.number}")

            for param_name, param_range in search_space.items():
                meta_value = self.meta_params.get(param_name)

                if meta_value is not None:
                    # Use the meta-learned value as a starting point
                    if isinstance(param_range[0], int):
                        # For integer parameters, set appropriate bounds around meta value
                        lower = max(param_range[0], int(meta_value * 0.7))
                        upper = min(param_range[1], int(meta_value * 1.3))
                        params[param_name] = trial.suggest_int(param_name, lower, upper)
                    else:
                        # For float parameters, use meta value as mean of distributions
                        std_dev = (param_range[1] - param_range[0]) / 4
                        lower = max(param_range[0], meta_value - std_dev)
                        upper = min(param_range[1], meta_value + std_dev)
                        params[param_name] = trial.suggest_float(param_name, lower, upper)
                else:
                    # Fall back to regular search space
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
        else:
            # No meta-learning available, use regular parameter search
            for param_name, param_range in search_space.items():
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])

        return params

    def run_optimization(self, objective_func, n_trials=30):
        """
        Run the optimization with meta-learning guidance.

        Args:
            objective_func: Function that evaluates parameters
            n_trials: Number of trials to run

        Returns:
            Dictionary of best parameters found
        """

        # Define the meta-learning enhanced objective function
        def meta_objective(trial):
            # Get suggested parameters using meta-learning
            params = self.suggest_params(trial)

            # Evaluate the parameters
            return objective_func(params, trial.number)

        # Run optimization
        self.study.optimize(meta_objective, n_trials=n_trials)

        # Return the best parameters
        return self.study.best_params


def integrate_meta_learning(experiment_runner):
    """
    Modify an existing experiment runner to use meta-learning.

    Args:
        experiment_runner: Original experiment runner module
    """
    # Create a meta-learner instance
    meta_learner = MetaLearner()

    # Function to create a meta-Optuna optimizer
    def create_meta_optimizer(method, model_type, dataset_attr, objective_func, n_trials=30):
        """
        Create and run a meta-learning enhanced optimizer.

        Returns:
            Dictionary of best parameters
        """
        meta_optuna = MetaOptuna(method, model_type, dataset_attr, meta_learner)
        return meta_optuna.run_optimization(objective_func, n_trials)

    # Replace the original run_optimization function
    experiment_runner.run_optimization = create_meta_optimizer

    # Ensure meta-learning is used for parameter suggestions
    original_get_method_params = experiment_runner.get_method_params

    def meta_enhanced_get_params(method, model_type, dataset_attr=None):
        """Get parameters with meta-learning if available."""
        if dataset_attr:
            # Try to get parameters from meta-learner first
            meta_params = meta_learner.predict_good_params(method, model_type, dataset_attr)
            if meta_params:
                return meta_params

        # Fall back to original method
        return original_get_method_params(method, model_type)

    # Replace the parameter getter
    experiment_runner.get_method_params = meta_enhanced_get_params

    # Close meta-learner when done
    meta_learner.close()


if __name__ == "__main__":
    # Demo of meta-learning
    meta_learner = MetaLearner()

    # Attempt to train meta-models for each method
    for method in ['expga', 'sg', 'aequitas', 'adf']:
        print(f"\nTraining meta-model for {method}...")
        meta_learner.train_meta_model(method)

        # Try to predict good parameters for a sample configuration
        dataset_attr = {
            'nb_attributes': 10,
            'prop_protected_attr': 0.2,
            'nb_categories_outcome': 3
        }

        for model_type in ['rf', 'mlp', 'dt']:
            predicted_params = meta_learner.predict_good_params(method, model_type, dataset_attr)

            if predicted_params:
                print(f"Predicted good parameters for {method} on {model_type}:")
                for param, value in predicted_params.items():
                    print(f"  {param}: {value}")

    meta_learner.close()