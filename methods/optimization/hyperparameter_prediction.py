import pandas as pd
import numpy as np
import json
import sqlite3
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any, Optional, Tuple, List
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from path import HERE


class HyperparameterPredictor:
    """Class to predict optimal hyperparameters based on past optimization results."""

    def __init__(self, db_path: str = None):
        """Initialize the predictor."""
        if db_path is None:
            self.db_path = HERE.joinpath("methods/optimizations.db")
        else:
            self.db_path = db_path

        self.models = {}  # Store models for different hyperparameters
        self.scalers = {}  # Store scalers for continuous variables
        self.encoders = {}  # Store encoders for categorical variables
        self.feature_columns = []
        self.categorical_params = ['model_type', 'one_attr_at_a_time']

    def load_optimization_data(self, method: str = None) -> pd.DataFrame:
        """Load optimization results from SQLite database."""
        conn = sqlite3.connect(self.db_path)

        # Build query
        query = """
        SELECT 
            trial_number,
            method,
            parameters,
            metrics,
            generation_arguments,
            runtime
        FROM optimization_trials
        """

        if method:
            query += f" WHERE method = '{method}'"

        df = pd.read_sql_query(query, conn)
        conn.close()

        # Parse JSON columns
        df['parameters'] = df['parameters'].apply(json.loads)
        df['metrics'] = df['metrics'].apply(json.loads)
        df['generation_arguments'] = df['generation_arguments'].apply(json.loads)

        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from generation arguments and other data."""
        features = []

        for _, row in df.iterrows():
            # Extract generation arguments
            gen_args = row['generation_arguments'] or {}

            # Extract metrics
            metrics = row['metrics']

            # Create feature dictionary
            feature_dict = {
                # Basic trial info
                'method': row['method'],
                'runtime': row['runtime'],
                'SUR': metrics.get('SUR', 0),
                'DSN': metrics.get('DSN', 0),
                'TSN': metrics.get('TSN', 0),

                # Generation arguments
                'nb_groups': gen_args.get('nb_groups', 0),
                'nb_attributes': gen_args.get('nb_attributes', 0),
                'min_number_of_classes': gen_args.get('min_number_of_classes', 0),
                'max_number_of_classes': gen_args.get('max_number_of_classes', 0),
                'prop_protected_attr': gen_args.get('prop_protected_attr', 0),
                'min_group_size': gen_args.get('min_group_size', 0),
                'max_group_size': gen_args.get('max_group_size', 0),
                'min_similarity': gen_args.get('min_similarity', 0),
                'max_similarity': gen_args.get('max_similarity', 0),
                'min_alea_uncertainty': gen_args.get('min_alea_uncertainty', 0),
                'max_alea_uncertainty': gen_args.get('max_alea_uncertainty', 0),
                'min_epis_uncertainty': gen_args.get('min_epis_uncertainty', 0),
                'max_epis_uncertainty': gen_args.get('max_epis_uncertainty', 0),
                'min_frequency': gen_args.get('min_frequency', 0),
                'max_frequency': gen_args.get('max_frequency', 0),
                'min_diff_subgroup_size': gen_args.get('min_diff_subgroup_size', 0),
                'max_diff_subgroup_size': gen_args.get('max_diff_subgroup_size', 0),
                'min_granularity': gen_args.get('min_granularity', 0),
                'max_granularity': gen_args.get('max_granularity', 0),
                'min_intersectionality': gen_args.get('min_intersectionality', 0),
                'max_intersectionality': gen_args.get('max_intersectionality', 0),
                'categorical_outcome': gen_args.get('categorical_outcome', False),
                'nb_categories_outcome': gen_args.get('nb_categories_outcome', 0),
                'corr_matrix_randomness': gen_args.get('corr_matrix_randomness', 0),
                'categorical_distribution': gen_args.get('categorical_distribution', 'balanced'),
                'categorical_influence': gen_args.get('categorical_influence', 0),
            }

            # Add hyperparameters
            params = row['parameters']
            for param, value in params.items():
                if param not in ['max_runtime_seconds', 'max_tsn', 'use_cache', 'db_path', 'analysis_id']:
                    feature_dict[f'param_{param}'] = value

            features.append(feature_dict)

        return pd.DataFrame(features)

    def prepare_training_data(self, method: str) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Prepare training data for a specific method."""
        # Load data
        df = self.load_optimization_data(method)
        if len(df) == 0:
            raise ValueError(f"No data found for method {method}")

        # Prepare features
        features_df = self.prepare_features(df)

        # Identify hyperparameters for this method
        param_columns = [col for col in features_df.columns if col.startswith('param_')]

        # Separate features (X) and targets (Y)
        feature_cols = [col for col in features_df.columns if
                        not col.startswith('param_') and col not in ['SUR', 'DSN', 'TSN']]
        X = features_df[feature_cols].copy()

        # Create target for each hyperparameter
        Y = {}
        for param_col in param_columns:
            param_name = param_col.replace('param_', '')
            Y[param_name] = features_df[param_col]

        # Encode categorical features
        for col in X.columns:
            if X[col].dtype == 'object' or isinstance(X[col].iloc[0], str):
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col])

        # Scale continuous features
        continuous_cols = X.select_dtypes(include=['float64', 'int64']).columns
        self.scalers['features'] = StandardScaler()
        X[continuous_cols] = self.scalers['features'].fit_transform(X[continuous_cols])

        self.feature_columns = X.columns.tolist()

        return X, Y

    def train_models(self, method: str, test_size: float = 0.2) -> Dict[str, float]:
        """Train random forest models for each hyperparameter."""
        # Prepare data
        X, Y = self.prepare_training_data(method)

        # Split data
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)

        scores = {}

        # Train a model for each hyperparameter
        for param_name, y in Y.items():
            # Determine if parameter is categorical or continuous
            if param_name in self.categorical_params:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                # Encode categorical targets
                self.encoders[f'target_{param_name}'] = LabelEncoder()
                y_encoded = self.encoders[f'target_{param_name}'].fit_transform(y)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                y_encoded = y

            # Split targets
            y_train, y_test = y_encoded[X_train.index], y_encoded[X_test.index]

            # Train model
            model.fit(X_train, y_train)

            # Evaluate
            if param_name in self.categorical_params:
                score = model.score(X_test, y_test)  # Accuracy
            else:
                predictions = model.predict(X_test)
                score = r2_score(y_test, predictions)

            scores[param_name] = score
            self.models[f'{method}_{param_name}'] = model

        return scores

    def predict_hyperparameters(self, method: str, generation_args: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal hyperparameters for a new configuration."""
        # Create feature dictionary
        feature_dict = {
            'method': method,
            'runtime': 0,  # Placeholder
            'nb_groups': generation_args.get('nb_groups', 0),
            'nb_attributes': generation_args.get('nb_attributes', 0),
            'min_number_of_classes': generation_args.get('min_number_of_classes', 0),
            'max_number_of_classes': generation_args.get('max_number_of_classes', 0),
            'prop_protected_attr': generation_args.get('prop_protected_attr', 0),
            'min_group_size': generation_args.get('min_group_size', 0),
            'max_group_size': generation_args.get('max_group_size', 0),
            'min_similarity': generation_args.get('min_similarity', 0),
            'max_similarity': generation_args.get('max_similarity', 0),
            'min_alea_uncertainty': generation_args.get('min_alea_uncertainty', 0),
            'max_alea_uncertainty': generation_args.get('max_alea_uncertainty', 0),
            'min_epis_uncertainty': generation_args.get('min_epis_uncertainty', 0),
            'max_epis_uncertainty': generation_args.get('max_epis_uncertainty', 0),
            'min_frequency': generation_args.get('min_frequency', 0),
            'max_frequency': generation_args.get('max_frequency', 0),
            'min_diff_subgroup_size': generation_args.get('min_diff_subgroup_size', 0),
            'max_diff_subgroup_size': generation_args.get('max_diff_subgroup_size', 0),
            'min_granularity': generation_args.get('min_granularity', 0),
            'max_granularity': generation_args.get('max_granularity', 0),
            'min_intersectionality': generation_args.get('min_intersectionality', 0),
            'max_intersectionality': generation_args.get('max_intersectionality', 0),
            'categorical_outcome': generation_args.get('categorical_outcome', False),
            'nb_categories_outcome': generation_args.get('nb_categories_outcome', 0),
            'corr_matrix_randomness': generation_args.get('corr_matrix_randomness', 0),
            'categorical_distribution': generation_args.get('categorical_distribution', 'balanced'),
            'categorical_influence': generation_args.get('categorical_influence', 0),
        }

        # Create DataFrame
        X = pd.DataFrame([feature_dict])

        # Encode categorical features
        for col in X.columns:
            if col in self.encoders:
                X[col] = self.encoders[col].transform(X[col])

        # Scale continuous features
        continuous_cols = X.select_dtypes(include=['float64', 'int64']).columns
        if 'features' in self.scalers:
            X[continuous_cols] = self.scalers['features'].transform(X[continuous_cols])

        # Ensure we have all required columns
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_columns]

        # Predict each hyperparameter
        predictions = {}
        for model_key, model in self.models.items():
            if model_key.startswith(f'{method}_'):
                param_name = model_key.replace(f'{method}_', '')
                pred = model.predict(X)[0]

                # Decode if categorical
                if param_name in self.categorical_params and f'target_{param_name}' in self.encoders:
                    pred = self.encoders[f'target_{param_name}'].inverse_transform([int(pred)])[0]

                predictions[param_name] = pred

        return predictions

    def save_models(self, filepath: str):
        """Save trained models to disk."""
        data = {
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'categorical_params': self.categorical_params
        }
        joblib.dump(data, filepath)

    def load_models(self, filepath: str):
        """Load trained models from disk."""
        data = joblib.load(filepath)
        self.models = data['models']
        self.scalers = data['scalers']
        self.encoders = data['encoders']
        self.feature_columns = data['feature_columns']
        self.categorical_params = data['categorical_params']

    def plot_feature_importance(self, method: str) -> None:
        """Plot feature importance for each hyperparameter model."""
        for model_key, model in self.models.items():
            if model_key.startswith(f'{method}_'):
                param_name = model_key.replace(f'{method}_', '')

                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = self.feature_columns

                    # Create plot
                    plt.figure(figsize=(10, 6))
                    sorted_idx = np.argsort(importances)[-15:]  # Top 15 features
                    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
                    plt.xlabel('Feature Importance')
                    plt.title(f'Feature Importance for {param_name} ({method})')
                    plt.tight_layout()
                    plt.show()


def predict_best_hyperparameters(
        method: str,
        generation_args: Dict[str, Any],
        db_path: Optional[str] = None,
        retrain: bool = False,
        model_save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predict the best hyperparameters for a given method and generation arguments.

    Args:
        method: Fairness testing method ('expga', 'sg', 'aequitas', 'adf')
        generation_args: Dictionary of generation arguments
        db_path: Path to SQLite database with optimization results
        retrain: Whether to retrain the model even if a saved model exists
        model_save_path: Path to save/load the trained model

    Returns:
        Dictionary of predicted hyperparameters
    """
    # Initialize predictor
    predictor = HyperparameterPredictor(db_path)

    # Set default model save path
    if model_save_path is None:
        model_save_path = HERE.joinpath(f"methods/{method}_predictor.joblib")

    # Load or train model
    if not retrain and model_save_path.exists():
        predictor.load_models(model_save_path)
    else:
        # Train models
        scores = predictor.train_models(method)
        print(f"Model scores for {method}:")
        for param, score in scores.items():
            print(f"  {param}: {score:.3f}")

        # Save model
        predictor.save_models(model_save_path)

    # Predict hyperparameters
    predictions = predictor.predict_hyperparameters(method, generation_args)

    return predictions


# Example usage
if __name__ == "__main__":
    # Example generation arguments
    example_gen_args = {
        'gen_order': 1,
        'nb_groups': 10,
        'nb_attributes': 5,
        'min_number_of_classes': 2,
        'max_number_of_classes': 5,
        'prop_protected_attr': 0.3,
        'min_group_size': 100,
        'max_group_size': 1000,
        'min_similarity': 0.1,
        'max_similarity': 0.9,
        'min_alea_uncertainty': 0.1,
        'max_alea_uncertainty': 0.5,
        'min_epis_uncertainty': 0.1,
        'max_epis_uncertainty': 0.5,
        'min_frequency': 0.1,
        'max_frequency': 0.9,
        'min_diff_subgroup_size': 0.1,
        'max_diff_subgroup_size': 0.5,
        'min_granularity': 0.1,
        'max_granularity': 0.5,
        'min_intersectionality': 0.1,
        'max_intersectionality': 0.5,
        'categorical_outcome': False,
        'nb_categories_outcome': 2,
        'corr_matrix_randomness': 0.3,
        'categorical_distribution': 'balanced',
        'categorical_influence': 0.5,
    }

    # Predict optimal hyperparameters
    predicted_params = predict_best_hyperparameters(
        method='expga',
        generation_args=example_gen_args,
        retrain=True  # Train the model
    )

    print("\nPredicted optimal hyperparameters:")
    for param, value in predicted_params.items():
        print(f"  {param}: {value}")

    # You can also visualize feature importance
    predictor = HyperparameterPredictor()
    predictor.load_models(HERE.joinpath("methods/expga_predictor.joblib"))
    predictor.plot_feature_importance('expga')