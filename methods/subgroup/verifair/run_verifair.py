import time
import pandas as pd
import numpy as np

from itertools import combinations
from data_generator.main import get_real_data, DiscriminationData, generate_optimal_discrimination_data
from methods.utils import train_sklearn_model
from methods.subgroup.verifair.verify.verify import verify
from methods.subgroup.verifair.util.log import log, setCurOutput, INFO
from sklearn.metrics import mutual_info_score


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
                                  n_samples, n_max, is_causal, log_iters, max_runtime_seconds=None):
    """
    Runs a single VeriFair analysis and returns the results as a dictionary.
    """
    log(f"\n--- Running VeriFair for: {analysis_attribute} ({group_a_val} vs {group_b_val}) ---", INFO)

    group_a_data = X_test[X_test[analysis_attribute] == group_a_val]
    group_b_data = X_test[X_test[analysis_attribute] == group_b_val]

    sampler_a = SklearnModelSampler(model, group_a_data)
    sampler_b = SklearnModelSampler(model, group_b_data)

    result = verify(sampler_a, sampler_b, c, Delta, delta, n_samples, n_max, is_causal, log_iters,
                    max_runtime_seconds=max_runtime_seconds)

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


def run_verifair(data: DiscriminationData, c=0.15, Delta=0.0,
                 delta=0.5 * 1e-10, n_samples=1, n_max=100000, is_causal=False, log_iters=1000,
                 max_runtime_seconds=None):
    """
    Trains a model and uses VeriFair to find discrimination on the given data.
    By default, this will test all combinations of protected attributes, starting with the one
    most likely to have discrimination based on mutual information score.

    :param data: A DiscriminationData object containing the dataset.
    :param c: float (confidence parameter)
    :param Delta: float (threshold on inequalities)
    :param delta: float (acceptable error probability)
    :param n_samples: int (number of samples to draw in each iteration)
    :param n_max: int, The maximum number of samples to draw from the model. Default is 100,000.
    :param is_causal: bool, Whether to use the causal version of VeriFair. Default is False.
    :param log_iters: int, The number of iterations after which to log progress.
    :param max_runtime_seconds: int, The maximum time in seconds to run the analysis for. Default is None (no limit).
    :return: A tuple containing a pandas DataFrame with the analysis results and a dictionary of summary metrics.
    """
    start_time = time.time()

    # Step 1: Train the model
    model, X_train, X_test, y_train, y_test, feature_names, metrics = train_sklearn_model(
        data=data.training_dataframe,
        target_col=data.outcome_column,
        sensitive_attrs=data.protected_attributes
    )
    accuracy = metrics['accuracy']
    log(f"Model trained with accuracy: {accuracy:.2f}", INFO)

    # Step 2: Determine order of attributes to analyze based on mutual information
    log("\n--- Determining order of attributes to analyze... ---", INFO)
    df = data.training_dataframe
    outcome_col = data.outcome_column
    mutual_info_scores = {
        attr: mutual_info_score(df[attr], df[outcome_col])
        for attr in data.protected_attributes
    }
    sorted_attributes = sorted(mutual_info_scores, key=mutual_info_scores.get, reverse=True)
    for attr in sorted_attributes:
        log(f"  - {attr} (Mutual Information: {mutual_info_scores[attr]:.4f})", INFO)

    # Step 3: Run VeriFair analysis for all combinations
    test_df = X_test.copy()
    test_df[data.outcome_column] = y_test
    results_list = []

    log("\n--- Analyzing all combinations of protected attributes ---", INFO)
    for attribute in sorted_attributes:
        if max_runtime_seconds and (time.time() - start_time) > max_runtime_seconds:
            log(f'Global timeout of {max_runtime_seconds} seconds reached. Halting analysis.', INFO)
            break

        unique_values = test_df[attribute].unique()
        if len(unique_values) < 2:
            log(f"Skipping attribute '{attribute}': has fewer than 2 unique values.", INFO)
            continue

        for group_a_val, group_b_val in combinations(unique_values, 2):
            if max_runtime_seconds and (time.time() - start_time) > max_runtime_seconds:
                log(f'Global timeout of {max_runtime_seconds} seconds reached. Halting analysis.', INFO)
                break

            # Calculate remaining time for this specific analysis run
            remaining_time = None
            if max_runtime_seconds:
                remaining_time = max_runtime_seconds - (time.time() - start_time)
                if remaining_time <= 0:
                    continue  # Skip if no time left

            result = _run_single_verifair_analysis(model, X_test, attribute, group_a_val, group_b_val, c, Delta,
                                                   delta,
                                                   n_samples, n_max, is_causal, log_iters, remaining_time)
            if result:
                results_list.append(result)
        else:
            continue  # only executed when inner loop is not broken
        break  # only executed when inner loop is broken

    res_df = pd.DataFrame(results_list)

    new_res_df = []
    for row in results_list:
        el1 = {e: None for e in data.attr_columns}
        el2 = {e: None for e in data.attr_columns}

        el1[row['attribute']] = row['group_a']
        el2[row['attribute']] = row['group_b']

        diff_outcome = data.dataframe[data.dataframe[row['attribute']] == row['group_a']]['outcome'].mean() - \
                       data.dataframe[data.dataframe[row['attribute']] == row['group_b']]['outcome'].mean()

        el1['indv_key'] = '|'.join([str(row['group_a']) if e == row['attribute'] else "*" for e in data.attr_columns])
        el2['indv_key'] = '|'.join([str(row['group_b']) if e == row['attribute'] else "*" for e in data.attr_columns])

        el1['diff_outcome'] = diff_outcome
        el2['diff_outcome'] = diff_outcome

        el1['couple_key'] = f"{el1['indv_key']}-{el2['indv_key']}"
        el2['couple_key'] = f"{el1['indv_key']}-{el2['indv_key']}"

        new_res_df.extend([el1, el2])

    new_res_df = pd.DataFrame.from_records(new_res_df)
    # --- 7. Calculate Final Metrics ---
    end_time = time.time()
    total_time = end_time - start_time

    if not new_res_df.empty:
        tsn = res_df['total_samples'].sum()
        dsn = res_df[res_df['is_fair'] == False].shape[0]
    else:
        tsn = 0
        dsn = 0

    sur = dsn / tsn if tsn > 0 else 0  # Success Rate
    dss = total_time / dsn if dsn > 0 else float('inf')  # Discriminatory Sample Search time

    metrics = {
        'TSN': tsn,
        'DSN': dsn,
        'SUR': sur,
        'DSS': dss,
        'total_time': total_time,
        'nodes_visited': tsn,  # Using total samples as a proxy for nodes visited
    }

    # Add metrics to result dataframe for consistency with other methods
    if not new_res_df.empty:
        for key, value in metrics.items():
            new_res_df[key] = value

    return new_res_df, metrics


if __name__ == '__main__':
    log('Starting VeriFair analysis script...', INFO)

    data = generate_optimal_discrimination_data(use_cache=True)

    summary_df, metrics = run_verifair(
        data=data,
        c=0.15, Delta=0.0,
        delta=0.5 * 1e-10, n_samples=1, n_max=1000, is_causal=False, log_iters=1000,
        max_runtime_seconds=300
    )

    log('Script finished.', INFO)

    if not summary_df.empty:
        print("\n--- Analysis Summary ---")
        print(summary_df.to_string())

    print(f"\n--- Summary Metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")
