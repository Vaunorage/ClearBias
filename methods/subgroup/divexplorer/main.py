import pandas as pd
from pandas import json_normalize
import signal
from contextlib import contextmanager
import time
import logging

from data_generator.main import generate_optimal_discrimination_data, get_real_data, DiscriminationData
from methods.subgroup.divexplorer.divexplorer.FP_Divergence import FP_Divergence
from methods.subgroup.divexplorer.divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
from methods.utils import make_final_metrics_and_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, lambda signum, frame: exec('raise TimeoutError'))
    # Schedule the signal to be sent after ``time`` seconds.
    signal.alarm(time)

    try:
        yield
    finally:
        # Unschedule the signal so it won't be triggered
        # if the timeout is not reached.
        signal.alarm(0)


def run_divexploer(data_obj: DiscriminationData, K=5, max_runtime_seconds=60, min_support=0.05):
    start_time = time.time()
    all_discriminations = []
    dsn_by_attr_value = {}

    try:
        with timeout(max_runtime_seconds):
            decoded_df = data_obj.training_dataframe_with_ypred

            fp_diver = FP_DivergenceExplorer(decoded_df,
                                             true_class_name=data_obj.outcome_column,
                                             predicted_class_name=data_obj.y_pred_col)

            FP_fm = fp_diver.getFrequentPatternDivergence(min_support=min_support)

            fp_divergence_fpr = FP_Divergence(FP_fm, "d_fpr")
            top_k_fpr = fp_divergence_fpr.getDivergenceTopKDf(K=K, th_redundancy=0)
            for _, row in top_k_fpr.iterrows():
                all_discriminations.append(row['itemsets'])

            fp_divergence_fnr = FP_Divergence(FP_fm, "d_fnr")
            top_k_fnr = fp_divergence_fnr.getDivergenceTopKDf(K=K, th_redundancy=0)
            for _, row in top_k_fnr.iterrows():
                all_discriminations.append(row['itemsets'])

    except TimeoutError:
        logger.info("Divexplorer timed out")

    # Mocking some values that are not directly available from divexplorer
    tot_inputs = set()
    for i in range(len(data_obj.training_dataframe)):
        tot_inputs.add(tuple(data_obj.training_dataframe.iloc[i]))

    res_df, metrics = make_final_metrics_and_dataframe(data_obj, tot_inputs, all_discriminations, dsn_by_attr_value,
                                                       start_time, logger=logger)

    return res_df, metrics


if __name__ == '__main__':
    # data_obj = generate_optimal_discrimination_data(nb_groups=100,
    #                                                 nb_attributes=15,
    #                                                 prop_protected_attr=0.3,
    #                                                 nb_categories_outcome=1,
    #                                                 use_cache=True)
    data_obj, schema = get_real_data('adult', use_cache=False)

    res, metrics = run_divexploer(data_obj, max_runtime_seconds=60)
    print(res)
    print(metrics)
