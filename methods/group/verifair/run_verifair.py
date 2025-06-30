import os
import pandas as pd
import numpy as np

from itertools import combinations
from data_generator.main import get_real_data
from methods.utils import train_sklearn_model
from methods.group.verifair.verify.verify import verify
from methods.group.verifair.util.log import log, setCurOutput, INFO


class SklearnModelSampler:
    """A wrapper for a trained scikit-learn model to make it compatible with VeriFair's verify function."""

    def __init__(self, model, data_group):
        """
        Initializes the sampler.

        :param model: A trained scikit-learn classifier.
        :param data_group: A pandas DataFrame containing the data for a specific group.
        """
        self.model = model
        self.data = data_group
        self.n_total_samples = 0

    def sample(self, n_samples):
        """
        Generates predictions for a random sample of the data group.

        :param n_samples: The number of samples to generate.
        :return: A numpy array of predictions (0s and 1s).
        """
        if self.data.empty:
            return np.array([0] * n_samples)  # Return neutral outcome if group is empty

        # Sample with replacement from the data for the specific group
        sample_indices = np.random.choice(self.data.index, size=n_samples, replace=True)
        data_sample = self.data.loc[sample_indices]

        # Get predictions from the scikit-learn model
        predictions = self.model.predict(data_sample)
        self.n_total_samples += n_samples
        return predictions


def _run_single_verifair_analysis(model, X_test, analysis_attribute, group_a_val, group_b_val, c, Delta, delta,
                                  n_samples, n_max, is_causal, log_iters):
    """
    Runs a single VeriFair analysis and returns the results as a dictionary.
    """
    log(f"\n--- Running VeriFair for: {analysis_attribute} ({group_a_val} vs {group_b_val}) ---", INFO)

    group_a_data = X_test[X_test[analysis_attribute] == group_a_val]
    group_b_data = X_test[X_test[analysis_attribute] == group_b_val]

    sampler_a = SklearnModelSampler(model, group_a_data)
    sampler_b = SklearnModelSampler(model, group_b_data)

    result = verify(sampler_a, sampler_b, c, Delta, delta, n_samples, n_max, is_causal, log_iters)

    if result is None:
        log('VeriFair analysis failed to converge!', INFO)
        return None

    is_fair, is_ambiguous, n_successful_samples, E = result
    n_total_samples = sampler_a.n_total_samples + sampler_b.n_total_samples

    log('Pr[fair = {}] >= 1.0 - {}'.format(is_fair, 2.0 * delta), INFO)
    log('E[ratio] = {}'.format(E), INFO)
    log('Is fair: {}'.format(bool(is_fair)), INFO)
    log('Is ambiguous: {}'.format(bool(is_ambiguous)), INFO)
    log('Successful samples: {}, Attempted samples: {}'.format(n_successful_samples, n_total_samples), INFO)
    log('--- Analysis Finished ---', INFO)

    return {
        'attribute': analysis_attribute,
        'group_a': group_a_val,
        'group_b': group_b_val,
        'is_fair': bool(is_fair),
        'is_ambiguous': bool(is_ambiguous),
        'estimated_ratio': E,
        'p_value': 2.0 * delta,
        'successful_samples': n_successful_samples,
        'total_samples': n_total_samples
    }


# --- Main Analysis Function ---
def find_discrimination_with_verifair(dataset_name='adult',
                                      analysis_attribute=None, group_a_value=None, group_b_value=None,
                                      analyze_all_combinations=False,
                                      c=0.15, Delta=0.0, delta=0.5 * 1e-10,
                                      n_samples=1, n_max=100000, is_causal=False, log_iters=1000):
    """
    Loads a dataset, trains a model, and uses VeriFair to find discrimination.

    :param dataset_name: The name of the dataset to use (e.g., 'adult').
    :param analysis_attribute: str, The attribute to analyze for discrimination. Defaults to the first sensitive attribute.
    :param group_a_value: The value for the first group to compare. Defaults to the first unique value of the attribute.
    :param group_b_value: The value for the second group to compare. Defaults to the second unique value of the attribute.
    :param analyze_all_combinations: bool, if True, automatically runs analysis for all combinations of protected attributes.
    :param c: float (minimum probability ratio to be fair)
    :param Delta: float (threshold on inequalities)
    :param delta: float (parameter delta)
    :param n_samples: int (number of samples per iteration)
    :param n_max: int (maximum number of iterations)
    :param is_causal: bool (whether to use the causal specification)
    :param log_iters: int (log every N iterations)
    :return: A pandas DataFrame summarizing the analysis results.
    """
    # Step 1: Load the data
    log(f"Loading dataset: {dataset_name}", INFO)
    discrimination_data, data_schema = get_real_data(dataset_name, use_cache=True)

    # Step 2: Train the model
    log("Training a RandomForest model...", INFO)
    model, _, X_test, _, y_test, _, metrics = train_sklearn_model(
        data=discrimination_data.training_dataframe,
        model_type='rf',
        target_col=discrimination_data.outcome_column,
        sensitive_attrs=discrimination_data.protected_attributes
    )
    accuracy = metrics['accuracy']
    log(f"Model trained with accuracy: {accuracy:.2f}", INFO)

    # Step 3: Run VeriFair analysis
    test_df = X_test.copy()
    test_df[discrimination_data.outcome_column] = y_test

    results_list = []

    if analyze_all_combinations:
        log("\n--- Analyzing all combinations of protected attributes ---", INFO)
        for attribute in discrimination_data.protected_attributes:
            unique_values = test_df[attribute].unique()
            if len(unique_values) < 2:
                log(f"Skipping attribute '{attribute}': has fewer than 2 unique values.", INFO)
                continue

            for group_a_val, group_b_val in combinations(unique_values, 2):
                result = _run_single_verifair_analysis(model, X_test, attribute, group_a_val, group_b_val, c, Delta,
                                                       delta,
                                                       n_samples, n_max, is_causal, log_iters)
                if result:
                    results_list.append(result)
    else:
        if analysis_attribute is None:
            analysis_attribute = discrimination_data.protected_attributes[0]
            log(f"No analysis attribute specified. Using the first sensitive attribute: '{analysis_attribute}'", INFO)
        elif analysis_attribute not in test_df.columns:
            log(f"Error: The specified analysis attribute '{analysis_attribute}' is not in the dataset.", INFO)
            return pd.DataFrame()

        unique_values = test_df[analysis_attribute].unique()

        if group_a_value is None or group_b_value is None:
            if len(unique_values) < 2:
                log(f"Attribute '{analysis_attribute}' has fewer than 2 unique values. Cannot perform comparison.",
                    INFO)
                return pd.DataFrame()
            group_a_val_to_use, group_b_val_to_use = unique_values[0], unique_values[1]
            log(f"No group values specified. Using the first two unique values for '{analysis_attribute}': {group_a_val_to_use} and {group_b_val_to_use}",
                INFO)
        else:
            if group_a_value not in unique_values:
                log(f"Error: Group A value '{group_a_value}' not found for attribute '{analysis_attribute}'.", INFO)
                return pd.DataFrame()
            if group_b_value not in unique_values:
                log(f"Error: Group B value '{group_b_value}' not found for attribute '{analysis_attribute}'.", INFO)
                return pd.DataFrame()
            group_a_val_to_use, group_b_val_to_use = group_a_value, group_b_value

        result = _run_single_verifair_analysis(model, X_test, analysis_attribute, group_a_val_to_use,
                                               group_b_val_to_use, c, Delta,
                                               delta, n_samples, n_max, is_causal, log_iters)
        if result:
            results_list.append(result)

    return pd.DataFrame(results_list)


if __name__ == '__main__':
    log('Starting VeriFair analysis script...', INFO)

    # To automatically analyze all combinations of protected attributes:
    summary_df = find_discrimination_with_verifair(
        dataset_name='adult',
        analyze_all_combinations=True
    )

    log('Script finished.', INFO)

    if not summary_df.empty:
        print("\n--- Analysis Summary ---")
        print(summary_df.to_string())
        print("----------------------")
