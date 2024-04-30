import sqlite3
import pandas as pd

from aequitas_algo.algo import run_aequitas
from data_generator.main2 import generate_data

# %%

df = generate_data(min_number_of_classes=2, max_number_of_classes=6, nb_attributes=7, prop_protected_attr=0.4,
                   nb_elems=1000, hiddenlayers_depth=3)

results_df = run_aequitas(df, col_to_be_predicted="LeaveOrNot",
                          sensitive_param_name_list=["Education", "Age"],
                          perturbation_unit=1, model_type="DecisionTree", threshold=0)
