import pandas as pd

from aequitas_algo.algo import run_aequitas
from paths import HERE

df = pd.read_csv(HERE.joinpath("aequitas_algo/Employee.csv").as_posix())
results_df = run_aequitas(df, col_to_be_predicted="LeaveOrNot",
                          sensitive_param_name_list=["Education", "Age"],
                          perturbation_unit=1, model_type="DecisionTree", threshold=0)

# %%
index_col = ['Sensitive Attribute', 'Total Inputs', 'Discriminatory Inputs', 'Percentage Discriminatory Inputs']
ll = results_df.set_index(index_col).apply(lambda x: x.apply(pd.Series).stack()).reset_index()
