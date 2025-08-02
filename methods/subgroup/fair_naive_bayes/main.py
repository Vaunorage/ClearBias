import os
import sys
import time
import logging
import pandas as pd

from methods.utils import train_sklearn_model

# Add the project root to the Python path
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
except NameError:
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_generator.main import generate_optimal_discrimination_data, DiscriminationData, DataSchema
from methods.subgroup.fair_naive_bayes.parameter_learner.data_processor import (
    get_params_dict,
    maximum_likelihood_from_data,
    convert_result_to_parameters
)
from methods.subgroup.fair_naive_bayes.pattern_finder.pattern_finder import PatternFinder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('FairNaiveBayes')


def run_naive_bayes(data: DiscriminationData, delta=0.01, k=5, max_runtime_seconds: int = None,
                    random_state=42, use_cache=True):
    """
    An example script that binarizes data, calculates Naive Bayes parameters, and then runs the
    PatternFinder to find discriminating patterns.

    Returns:
        pd.DataFrame: A dataframe containing the details of discriminating patterns.
        dict: A dictionary containing summary metrics.
    """
    start_time = time.time()
    if max_runtime_seconds is not None and max_runtime_seconds <= 0:
        logger.warning("max_runtime_seconds must be positive. Ignoring the time limit.")
        max_runtime_seconds = None
    logger.info("--- Running Discrimination Finder with Fair Naive Bayes ---")

    df = data.dataframe

    model, X_train, X_test, y_train, y_test, feature_names, metrics = train_sklearn_model(
        data=data.training_dataframe.copy(),
        model_type='rf',
        target_col=data.outcome_column,
        sensitive_attrs=list(data.protected_attributes),
        random_state=random_state,
        use_cache=use_cache
    )

    logger.info("Dataset loaded. Binarizing data for the Naive Bayes model...")

    # --- 2. Binarize the data ---
    binarized_df = pd.DataFrame()
    target_name = 'outcome'

    # Binarize all attributes and the target column
    all_columns = data.attr_columns + [target_name]
    for attr_name in all_columns:
        if df[attr_name].nunique() <= 2:
            binarized_df[attr_name] = df[attr_name]
        else:
            median = df[attr_name].median()
            binarized_df[attr_name] = (df[attr_name] > median).astype(int)
            logger.info(f"  - Binarized '{attr_name}' by splitting at its median value ({median:.2f})")

    # --- 3. Manually construct metadata needed for parameter learning ---
    feature_names = data.attr_columns
    bn_dict = {i: name for i, name in enumerate(feature_names)}
    target_value = 1

    sensitive_names = data.protected_attributes
    logger.info(f"Sensitive attributes identified: {sensitive_names}")

    # --- 4. Calculate Naive Bayes parameters from the binarized data ---
    logger.info("\nCalculating Naive Bayes parameters from binarized data...")
    params_feature_names = list(feature_names)
    params_dict = get_params_dict(binarized_df, params_feature_names, target_name)
    prob_dict = maximum_likelihood_from_data(params_dict, target_name)
    root_params, leaf_params = convert_result_to_parameters(prob_dict, data.sensitive_indices, bn_dict, target_name)

    # --- 5. Find Discriminating Patterns ---
    logger.info(f"\nSearching for the top {k} discriminating patterns with a threshold of {delta}...")

    pf = PatternFinder(root_params, leaf_params, target_value, data.sensitive_indices, start_time=start_time,
                       max_runtime_seconds=max_runtime_seconds)
    raw_patterns = pf.get_discriminating_patterns(delta, k)

    # --- 6. Process and Format Results ---
    pattern_results = []
    if not raw_patterns:
        logger.info("\nNo discriminating patterns found with the given threshold.")
    else:
        logger.info(f"\n--- Top {len(raw_patterns)} Discriminating Patterns Found ---")
        for i, pattern in enumerate(raw_patterns):
            base_features = {bn_dict.get(fid): val for fid, val in pattern.base}
            base_features = {**base_features, **{e: None for e in data.attr_columns if e not in base_features}}
            base_features['nature'] = 'base'

            sens_features = {bn_dict.get(fid): val for fid, val in pattern.sens}
            sens_features = {**sens_features, **{e: None for e in data.attr_columns if e not in sens_features}}
            sens_features['nature'] = 'sensitive'

            pattern_info = {
                'case_id': i + 1,
                'discrimination_score': pattern.score,
                'p_unfavorable_sensitive': pattern.pDXY,
                'p_unfavorable_others': pattern.pD_XY,
            }

            pattern_results.append({**pattern_info, **base_features})
            pattern_results.append({**pattern_info, **sens_features})

    res_df = pd.DataFrame(pattern_results)

    res_df.drop_duplicates(inplace=True)

    # Calculate outcome for the generated subgroups
    feature_cols = list(data.attributes)
    result_df_filled = res_df[feature_cols].copy()

    # Handle potential None values by filling with median from training data
    for col in feature_cols:
        if result_df_filled[col].isnull().any():
            median_val = data.training_dataframe[col].median()
            result_df_filled[col].fillna(median_val, inplace=True)

    # Predict outcomes using the trained model
    predicted_outcomes = model.predict(result_df_filled[feature_names])
    res_df[data.outcome_column] = predicted_outcomes

    # Add individual key
    res_df['subgroup_key'] = res_df.apply(
        lambda row: '|'.join(str(int(row[col])) if not pd.isna(row[col]) else '*' for col in list(data.attributes)),
        axis=1
    )

    # Calculate outcome differences from the mean
    mean_outcome = res_df[data.outcome_column].mean()
    res_df['diff_outcome'] = (res_df[data.outcome_column] - mean_outcome).abs()

    # --- 7. Calculate Final Metrics ---
    end_time = time.time()
    total_time = end_time - start_time
    tsn = pf.num_visited  # Total Searched Nodes as a proxy for Total Sample Number
    dsn = len(raw_patterns)  # Discriminatory Sample Number
    sur = dsn / tsn if tsn > 0 else 0  # Success Rate
    dss = total_time / dsn if dsn > 0 else float('inf')  # Discriminatory Sample Search time

    metrics = {
        'TSN': tsn,
        'DSN': dsn,
        'SUR': sur,
        'DSS': dss,
        'total_time': total_time,
        'nodes_visited': pf.num_visited,
    }

    logger.info("\n--- Final Metrics ---")
    logger.info(f"Total nodes visited (TSN proxy): {tsn}")
    logger.info(f"Total discriminating patterns (DSN): {dsn}")
    logger.info(f"Success rate (SUR): {sur:.4f}")
    logger.info(f"Avg. search time per discriminatory pattern (DSS): {dss:.4f} seconds")
    logger.info(f"Total time: {total_time:.2f} seconds")

    # Add metrics to result dataframe for consistency with other methods
    if not res_df.empty:
        for key, value in metrics.items():
            res_df[key] = value

    return res_df, metrics


if __name__ == '__main__':
    data = generate_optimal_discrimination_data(use_cache=True)
    res_df, metrics = run_naive_bayes(data, max_runtime_seconds=60)

    if not res_df.empty:
        print("\n--- Discrimination Results ---")
        print(res_df.to_string())

    print(f"\n--- Summary Metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")
