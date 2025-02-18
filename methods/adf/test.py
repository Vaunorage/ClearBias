from data_generator.main import get_real_data, generate_from_real_data, generate_data
from methods.adf.main1 import adf_fairness_testing
from methods.utils import reformat_discrimination_results, convert_to_non_float_rows

# %%
data_obj, schema = get_real_data('adult')

# Run fairness testing
results_df_orgin, metrics_orgin = adf_fairness_testing(data_obj, max_global=5000, max_local=2000, max_iter=1,
                                                       cluster_num=50)

# %%
predefined_groups_origin = reformat_discrimination_results(convert_to_non_float_rows(results_df_orgin, schema))
nb_elements = sum([el.group_size for el in predefined_groups_origin])

# %%
data_obj, schema = generate_from_real_data('adult', predefined_groups=predefined_groups_origin,
                                           additional_random_rows=data_obj.dataframe.shape[0] - nb_elements)
# Run fairness testing
results_df_synth, metrics_synth = adf_fairness_testing(data_obj, max_global=5000, max_local=2000, max_iter=2,
                                                       cluster_num=50)

# %%
predefined_groups_synth = reformat_discrimination_results(convert_to_non_float_rows(results_df_orgin, schema))
