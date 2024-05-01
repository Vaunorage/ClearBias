import numpy as np
import pandas as pd

from aequitas_algo.algo import run_aequitas
from data_generator.main2 import generate_data
from data_generator.utils import scale_dataframe, visualize_df

df, protected_attr = generate_data(min_number_of_classes=2, max_number_of_classes=6, nb_attributes=6,
                                   prop_protected_attr=0.1, nb_elems=100, hiddenlayers_depth=3, min_similarity=0.0,
                                   max_similarity=1.0, min_alea_uncertainty=0.0, max_alea_uncertainty=1.0,
                                   min_epis_uncertainty=0.0, max_epis_uncertainty=1.0,
                                   min_magnitude=0.0, max_magnitude=1.0, min_frequency=0.0, max_frequency=1.0,
                                   categorical_outcome=True, nb_categories_outcome=4)

visualize_df(df, ['granularity', 'alea_uncertainty', 'epis_uncertainty', 'magnitude', 'diff_outcome'],
             'diff_outcome', 'figure4.png')

# %%
dff = df[[e for e in protected_attr] + ['outcome']]
# scaled_df, min_values, max_values = scale_dataframe(dff)

# %%
results_df = run_aequitas(dff, col_to_be_predicted="outcome",
                          sensitive_param_name_list=[k for k, e in protected_attr.items() if e],
                          perturbation_unit=1, model_type="DecisionTree", threshold=0)
# %%

org_found = results_df[[e for e in protected_attr]].drop_duplicates().values.tolist()
org1_found = dff[[e for e in protected_attr]].drop_duplicates().values.tolist()
percent_of_org_in_found = sum([1 for e in org_found if e in org1_found])/len(org_found)
percent_found = sum([1 for e in org_found if e in org1_found])/len(org1_found)

# il faut sassurer de trouver deux individus pour dire quil ya de la discrimination
# il faut regarder si la magnitude est differente

# %%
print('sss')
# original_df = scale_dataframe(scaled_df, reverse=True, min_values=min_values, max_values=max_values)
