import pandas as pd
from pandas import json_normalize
import signal
from contextlib import contextmanager

from data_generator.main import generate_optimal_discrimination_data, get_real_data, DiscriminationData
from methods.subgroup.divexplorer.divexplorer.FP_Divergence import FP_Divergence
from methods.subgroup.divexplorer.divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer


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


def run_divexploer(data_obj: DiscriminationData, K=5, max_runtime_seconds=60):
    def parse_itemset(itemset_str):
        """Convert itemset string to dictionary"""
        items = {e.split('=')[0]: e.split('=')[1] for e in itemset_str}

        for attribute in data_obj.attributes:
            if attribute not in items:
                items[attribute] = None

        return items

    try:
        with timeout(max_runtime_seconds):
            decoded_df = data_obj.training_dataframe_with_ypred

            fp_diver = FP_DivergenceExplorer(decoded_df,
                                             true_class_name=data_obj.outcome_column,
                                             predicted_class_name=data_obj.y_pred_col)

            FP_fm = fp_diver.getFrequentPatternDivergence(min_support=0.05)

            fp_divergence_fpr = FP_Divergence(FP_fm, "d_fpr")
            fp_divergence_fnr = FP_Divergence(FP_fm, "d_fnr")

            top_k_fpr = fp_divergence_fpr.getDivergenceTopKDf(K=K, th_redundancy=0)
            top_k_fpr['metric'] = 'fpr'
            top_k_fpr.rename(columns={'d_fpr': 'value'}, inplace=True)

            top_k_fnr = fp_divergence_fnr.getDivergenceTopKDf(K=K, th_redundancy=0)
            top_k_fnr['metric'] = 'fnr'
            top_k_fnr.rename(columns={'d_fnr': 'value'}, inplace=True)

            top_k_df = pd.concat([top_k_fpr, top_k_fnr])

            top_k_df['items'] = top_k_df['itemsets'].apply(parse_itemset)

            df_final = pd.concat([top_k_df.drop(columns=['itemsets', 'items'], axis=1), json_normalize(top_k_df['items'])],
                                 axis=1)

            return df_final
    except TimeoutError:
        return pd.DataFrame() # Return empty dataframe if timeout is reached


if __name__ == '__main__':
    # data_obj = generate_optimal_discrimination_data(nb_groups=100,
    #                                                 nb_attributes=15,
    #                                                 prop_protected_attr=0.3,
    #                                                 nb_categories_outcome=1,
    #                                                 use_cache=True)
    data_obj, schema = get_real_data('adult', use_cache=False)

    res = run_divexploer(data_obj, max_runtime_seconds=60)
