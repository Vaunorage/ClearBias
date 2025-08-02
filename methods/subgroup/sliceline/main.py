import pandas as pd
import time
import logging
from methods.subgroup.sliceline.src.sliceline import Slicefinder
from data_generator.main import DiscriminationData, get_real_data
from methods.utils import train_sklearn_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_sliceline(data: DiscriminationData, K=5, alpha=0.95, max_l=3, max_runtime_seconds=60, random_state=42,
                  use_cache=True, logger=None):
    """
    Finds top K slices with high FPR and FNR using Sliceline.
    """
    start_time = time.time()
    df = data.training_dataframe_with_ypred.copy()
    X = df[data.attr_columns]
    y_true = df[data.outcome_column]
    y_pred = df[data.y_pred_col]

    model, X_train, X_test, y_train, y_test, feature_names, metrics = train_sklearn_model(
        data=data.training_dataframe.copy(),
        model_type='rf',
        target_col=data.outcome_column,
        sensitive_attrs=list(data.protected_attributes),
        random_state=random_state,
        use_cache=use_cache
    )

    # Calculate False Positive and False Negative errors
    errors_fp = ((y_true == 0) & (y_pred == 1)).astype(int)
    errors_fn = ((y_true == 1) & (y_pred == 0)).astype(int)

    all_results = []

    # Find slices for False Positives
    slice_finder_fp = Slicefinder(k=K, alpha=alpha, max_l=max_l, max_time=max_runtime_seconds)
    slice_finder_fp.fit(X, errors_fp)

    check_1_fp = getattr(slice_finder_fp, 'top_slices_', None) is not None
    check_2_fp = slice_finder_fp.top_slices_.any()
    if check_1_fp and check_2_fp:
        top_k_fpr = pd.DataFrame(slice_finder_fp.top_slices_, columns=data.attr_columns)
        top_k_fpr = pd.concat([top_k_fpr, pd.DataFrame(slice_finder_fp.top_slices_statistics_)])
        top_k_fpr['metric'] = 'fpr'
        all_results.append(top_k_fpr)

    # Find slices for False Negatives
    slice_finder_fn = Slicefinder(k=K, alpha=alpha, max_l=max_l, max_time=max_runtime_seconds)
    slice_finder_fn.fit(X, errors_fn)

    check_1_fn = getattr(slice_finder_fn, 'top_slices_', None) is not None
    check_2_fn = slice_finder_fn.top_slices_.any()

    if check_1_fn and check_2_fn:
        top_k_fnr = pd.DataFrame(slice_finder_fn.top_slices_, columns=data.attr_columns)
        top_k_fnr = pd.concat([top_k_fnr, pd.DataFrame(slice_finder_fn.top_slices_statistics_)])
        top_k_fnr['metric'] = 'fnr'
        all_results.append(top_k_fnr)

    if len(all_results) == 0:
        print("Sliceline did not find any significant slices.")
        return pd.DataFrame(), {}

    df_final = pd.concat(all_results, ignore_index=True)

    tsn = X.shape[0]
    dsn = df_final.shape[0]

    df_final['subgroup_key'] = df_final[data.attr_columns].fillna('*').apply(lambda x: "|".join(x.astype(str)), axis=1)

    feature_cols = list(data.attributes)
    result_df_filled = df_final[feature_cols].copy()

    # Handle potential None values by filling with median from training data
    for col in feature_cols:
        if result_df_filled[col].isnull().any():
            median_val = data.training_dataframe[col].median()
            result_df_filled[col].fillna(median_val, inplace=True)

    # Predict outcomes using the trained model
    predicted_outcomes = model.predict(result_df_filled[feature_names])
    df_final[data.outcome_column] = predicted_outcomes

    df_final['diff_outcome'] = df_final[data.outcome_column] - df_final[data.outcome_column].mean()

    runtime = time.time() - start_time
    if df_final.empty:
        return pd.DataFrame(), {"runtime": runtime, "TSN": tsn, "DSN": 0, "SUR": 0, "DSS": float('inf')}

    # Format the dataframe
    df_final['slice'] = df_final.apply(
        lambda row: ', '.join([f"{col}={row[col]}" for col in row.index if
                               row[col] is not None and col not in ['slice_size', 'slice_mean', 'effect_size',
                                                                    'metric', 'size']]), axis=1)

    # Calculate metrics
    sur = dsn / tsn if tsn > 0 else 0
    dss = runtime / dsn if dsn > 0 else float('inf')

    metrics = {
        "RUNTIME": runtime,
        "TSN": tsn,
        "DSN": dsn,
        "SUR": sur,
        "DSS": dss
    }

    if logger:
        logger.info(f"Sliceline completed in {runtime:.2f} seconds.")
        logger.info(f"Total inputs tested (TSN): {tsn}")
        logger.info(f"Discriminatory samples in slices (DSN): {dsn}")
        logger.info(f"Success Rate (SUR): {sur:.4f}")
        logger.info(f"Discriminatory Sample Search time (DSS): {dss:.4f}")

    return df_final, metrics


if __name__ == '__main__':
    # Load data
    data_obj, schema = get_real_data('adult', use_cache=False)
    # Run sliceline
    res, metrics = run_sliceline(data_obj, K=2, max_runtime_seconds=30, logger=logger)

    if not res.empty:
        print("Top Slices found by Sliceline:")
        print(res)
        print(f"\nMetrics: {metrics}")
