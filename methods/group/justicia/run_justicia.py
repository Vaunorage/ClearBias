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

    # Justicia works with the test set
    test_df = X_test.copy()
    test_df[discrimination_data.outcome_column] = y_test

    # Justicia expects the outcome column to be named 'target'
    test_df.rename(columns={discrimination_data.outcome_column: 'target'}, inplace=True)

    # Define the sensitive attributes for Justicia
    sensitive_attributes = discrimination_data.protected_attributes

    # Run Justicia
    metric = Metric(model=model, data=test_df, sensitive_attributes=sensitive_attributes, verbose=False)
    metric.compute()

    # --- New code to format the output ---
    most_favored = metric.most_favored_group
    least_favored = metric.least_favored_group

    subgroups_data = []
    if most_favored:
        subgroups_data.append({'Group Type': 'Most Favored', **most_favored})
    if least_favored:
        subgroups_data.append({'Group Type': 'Least Favored', **least_favored})

    if subgroups_data:
        # Decode the subgroup values from numerical back to original categories
        decoded_subgroups = []
        for subgroup in subgroups_data:
            decoded_subgroup = {'Group Type': subgroup['Group Type']}
            # Create a reverse map for easier lookup
            reverse_maps = {attr: {v: k for k, v in mapping.items()} for attr, mapping in data_schema.category_maps.items()}

            for key, value in subgroup.items():
                if key in reverse_maps:
                    decoded_subgroup[key] = reverse_maps[key].get(value, value) # Get decoded value, or original if not found
                elif key != 'Group Type':
                     decoded_subgroup[key] = value
            decoded_subgroups.append(decoded_subgroup)

        subgroups_df = pd.DataFrame(decoded_subgroups).set_index('Group Type')
        # Fill NaN values for better display
        subgroups_df = subgroups_df.fillna('')
        print("Justicia identified the following subgroups with the largest difference in outcomes:")
        print(subgroups_df.to_string())
    else:
        print("Justicia did not identify any discriminated subgroups.")


if __name__ == '__main__':
    find_discrimination_with_justicia(dataset_name='adult', model_type='dt')
