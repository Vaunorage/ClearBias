import numpy as np
import tensorflow as tf
import sys, os
import pandas as pd
from typing import List, Tuple, Optional, Dict
from sklearn.model_selection import train_test_split
from tensorflow.python.platform import flags
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from adf_model.tutorial_models import dnn
from adf_utils.utils_tf import model_prediction, model_argmax
from adf_tutorial.utils import cluster, gradient_graph

FLAGS = flags.FLAGS

class DataSchema:
    """Schema for the dataset including protected attributes"""
    def __init__(self, attr_names: List[str], protected_attr: List[bool]):
        self.attr_names = attr_names
        self.protected_attr = protected_attr

class DiscriminationData:
    """Container for discrimination-related data"""
    def __init__(self, dataframe: pd.DataFrame, categorical_columns: List[str],
                 attributes: Dict[str, bool], outcome_column: str):
        self.dataframe = dataframe
        self.categorical_columns = categorical_columns
        self.attributes = attributes
        self.outcome_column = outcome_column

def generate_schema_from_dataframe(
    df: pd.DataFrame,
    protected_columns: List[str],
    outcome_column: str,
    use_attr_naming_pattern: bool = True
) -> Tuple[DataSchema, pd.DataFrame]:
    """
    Generate schema and encode categorical variables
    """
    # Create copy to avoid modifying original
    df = df.copy()
    
    # Encode categorical columns
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Create schema
    attr_names = [col for col in df.columns if col != outcome_column]
    protected_attr = [col in protected_columns for col in attr_names]
    
    # Rename columns if requested
    if use_attr_naming_pattern:
        column_mapping = {name: f'Attribute{i+1}' for i, name in enumerate(attr_names)}
        column_mapping[outcome_column] = 'outcome'
        df = df.rename(columns=column_mapping)
        attr_names = [column_mapping[name] for name in attr_names]
        protected_columns = [column_mapping[name] for name in protected_columns]
    
    schema = DataSchema(attr_names, protected_attr)
    
    return schema, df

def get_dataset(
    dataset_name: str,
    protected_columns: Optional[List[str]] = None,
    outcome_column: Optional[str] = None,
) -> Tuple[DiscriminationData, DataSchema]:
    """
    Fetch and process UCI datasets
    """
    dataset_configs = {
        'adult': {
            'id': 2,
            'protected_columns': ['race', 'sex'] if protected_columns is None else protected_columns,
            'outcome_column': 'income' if outcome_column is None else outcome_column,
            'drop_columns': ['fnlwgt']
        },
        'credit': {
            'id': 144,
            'protected_columns': ['Attribute8', 'Attribute12'] if protected_columns is None else protected_columns,
            'outcome_column': 'Attribute20' if outcome_column is None else outcome_column,
            'drop_columns': []
        }
    }

    if dataset_name not in dataset_configs:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(dataset_configs.keys())}")

    config = dataset_configs[dataset_name]
    
    # Fetch dataset
    dataset = fetch_ucirepo(id=config['id'])
    df = dataset['data']['original']
    
    if config['drop_columns']:
        df = df.drop(columns=config['drop_columns'])
    
    # Generate schema and encode data
    schema, enc_df = generate_schema_from_dataframe(
        df,
        protected_columns=config['protected_columns'],
        outcome_column=config['outcome_column']
    )
    
    # Create discrimination data object
    data = DiscriminationData(
        dataframe=enc_df,
        categorical_columns=list(schema.attr_names) + ['outcome'],
        attributes={k:v for k,v in zip(schema.attr_names, schema.protected_attr)},
        outcome_column='outcome'
    )
    
    return data, schema

def train_model(X, y, input_shape, nb_classes=2):
    """
    Train DNN model on dataset
    """
    model = dnn(input_shape=input_shape, nb_classes=nb_classes)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)
    
    return model, sess

def custom_dataset_fair_testing(
    dataset_name: str,
    protected_columns: Optional[List[str]] = None,
    outcome_column: Optional[str] = None,
    model_path='./models/',
    cluster_num=4,
    max_global=1000,
    max_local=1000,
    max_iter=10
):
    """
    Main function for fairness testing
    """
    # Load and preprocess dataset
    data, schema = get_dataset(
        dataset_name,
        protected_columns,
        outcome_column
    )
    
    # Prepare data for training
    X = data.dataframe.drop(columns=[data.outcome_column])
    y = data.dataframe[data.outcome_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    input_shape = (None, X.shape[1])
    model, sess = train_model(X_train, y_train, input_shape)
    
    # Get placeholders and predictions
    x = model.layers[0].input
    preds = model.layers[-1].output
    
    # Create gradient graph
    grad_0 = gradient_graph(x, preds)
    
    # Cluster the testing data
    clusters = cluster(X_test.values, cluster_num)
    
    # Clean up
    sess.close()

if __name__ == '__main__':
    flags.DEFINE_string('dataset', 'adult', 'Name of the dataset (adult or credit)')
    flags.DEFINE_list('protected_columns', None, 'Names of protected attribute columns')
    flags.DEFINE_string('outcome_column', None, 'Name of the outcome column')
    flags.DEFINE_string('model_path', './models/', 'Path to save/load the model')
    flags.DEFINE_integer('cluster_num', 4, 'Number of clusters for testing')
    flags.DEFINE_integer('max_global', 1000, 'Maximum samples for global search')
    flags.DEFINE_integer('max_local', 1000, 'Maximum samples for local search')
    flags.DEFINE_integer('max_iter', 10, 'Maximum iterations for global perturbation')
    
    # Example usage:
    # python custom_dataset_tutorial.py --dataset=adult 
    #                                  --protected_columns=race,sex
    #                                  --outcome_column=income
    
    tf.app.run(lambda _: custom_dataset_fair_testing(
        FLAGS.dataset,
        FLAGS.protected_columns,
        FLAGS.outcome_column,
        FLAGS.model_path,
        FLAGS.cluster_num,
        FLAGS.max_global,
        FLAGS.max_local,
        FLAGS.max_iter
    ))
