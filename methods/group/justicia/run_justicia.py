import os
import pandas as pd
from data_generator.main import get_real_data
from methods.utils import train_sklearn_model
from justicia.metrics import Metric
import justicia.utils


def find_discrimination_with_justicia(dataset_name='adult', model_type='rf'):
    """
    Loads a dataset, trains a model, and uses Justicia to find discrimination.

    :param dataset_name: The name of the dataset to use (e.g., 'adult').
    :param model_type: The type of model to train ('rf', 'lr', etc.).
    """

    discrimination_data, data_schema = get_real_data(dataset_name, use_cache=True)

    model, _, X_test, _, y_test, _, metrics = train_sklearn_model(
        data=discrimination_data.training_dataframe,
        model_type=model_type,
        target_col=discrimination_data.outcome_column,
        sensitive_attrs=discrimination_data.protected_attributes
    )
    accuracy = metrics['accuracy']

    # Justicia works with the test set
    test_df = X_test.copy()
    test_df[discrimination_data.outcome_column] = y_test

    # Justicia expects the outcome column to be named 'target'
    test_df.rename(columns={discrimination_data.outcome_column: 'target'}, inplace=True)

    # Define the sensitive attributes for Justicia
    sensitive_attributes = discrimination_data.protected_attributes

    # Run Justicia
    metric = Metric(model=model, data=test_df, sensitive_attributes=sensitive_attributes, verbose=False)
    res = metric.compute()
    print(res)


if __name__ == '__main__':
    find_discrimination_with_justicia(dataset_name='adult', model_type='dt')
