import pandas as pd
import numpy as np
import re
from pandas import json_normalize
from sliceline.slicefinder import Slicefinder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from data_generator.main import DiscriminationData, get_real_data

def parse_sliceline_itemset(itemset_str, all_attributes):
    """Convert sliceline itemset string to dictionary"""
    items = {}
    # Regex to find all 'column_name == value' or 'column_name <= value' etc.
    pattern = re.compile(r'(`?)([^`=><!]+)\1\s*([=><!]+)\s*(\S+)')
    matches = pattern.findall(itemset_str)
    
    for _, col, op, val in matches:
        col = col.strip()
        # Try to convert value to numeric, otherwise keep as string
        try:
            items[col] = pd.to_numeric(val)
        except (ValueError, TypeError):
            items[col] = val.strip("'\"") # remove quotes

    for attribute in all_attributes:
        if attribute not in items:
            items[attribute] = None
    return items


def run_sliceline(data_obj: DiscriminationData, K=5, alpha=0.95, max_l=3):
    """
    Finds top K slices with high FPR and FNR using Sliceline.
    """
    df = data_obj.training_dataframe_with_ypred.copy()
    X = df[data_obj.attr_columns]
    y_true = df[data_obj.outcome_column]
    y_pred = df[data_obj.y_pred_col]

    # Calculate False Positive and False Negative errors
    errors_fp = ((y_true == 0) & (y_pred == 1)).astype(int)
    errors_fn = ((y_true == 1) & (y_pred == 0)).astype(int)

    all_results = []

    # Find slices for False Positives
    slice_finder_fp = Slicefinder(k=K, alpha=alpha, max_l=max_l)
    slice_finder_fp.fit(X, errors_fp)
    
    if getattr(slice_finder_fp, 'top_slices_').any():
        top_k_fpr = pd.DataFrame(slice_finder_fp.top_slices_, columns=data_obj.attr_columns)
        top_k_fpr = pd.concat([top_k_fpr, pd.DataFrame(slice_finder_fp.top_slices_statistics_)])
        top_k_fpr['metric'] = 'fpr'
        all_results.append(top_k_fpr)

    # Find slices for False Negatives
    slice_finder_fn = Slicefinder(k=K, alpha=alpha, max_l=max_l)
    slice_finder_fn.fit(X, errors_fn)

    if getattr(slice_finder_fn, 'top_slices_').any():
        top_k_fnr = pd.DataFrame(slice_finder_fn.top_slices_, columns=data_obj.attr_columns)
        top_k_fnr = pd.concat([top_k_fnr, pd.DataFrame(slice_finder_fn.top_slices_statistics_)])
        top_k_fnr['metric'] = 'fnr'
        all_results.append(top_k_fnr)
        
    if not all_results:
        print("Sliceline did not find any significant slices.")
        return pd.DataFrame()

    df_final = pd.concat(all_results, ignore_index=True)

    return df_final


if __name__ == '__main__':
    # Load data
    data_obj, schema = get_real_data('adult', use_cache=False)
    # Run sliceline
    res = run_sliceline(data_obj, K=2)
    
    if not res.empty:
        print("Top Slices found by Sliceline:")
        print(res)
