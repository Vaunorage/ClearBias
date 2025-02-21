from matplotlib import pyplot as plt

from data_generator.main import get_real_data, generate_from_real_data
from data_generator.utils import plot_distribution_comparison
from methods.adf.main1 import adf_fairness_testing
from methods.utils import reformat_discrimination_results, convert_to_non_float_rows, compare_discriminatory_groups

# %%
data_obj, schema = get_real_data('adult')

results_df_origin, metrics_origin = adf_fairness_testing(data_obj, max_global=5000, max_local=2000, max_iter=10,
                                                         cluster_num=50, random_seed=42)

# %%
non_float_df = convert_to_non_float_rows(results_df_origin, schema)
predefined_groups_origin = reformat_discrimination_results(non_float_df, data_obj.dataframe)
nb_elements = sum([el.group_size for el in predefined_groups_origin])

# %%
data_obj_synth, schema = generate_from_real_data('adult', predefined_groups=predefined_groups_origin, extra_rows=1000)

#%%
fig = plot_distribution_comparison(schema, data_obj_synth)
plt.show()

#%% Run fairness testing
results_df_synth, metrics_synth = adf_fairness_testing(data_obj_synth, max_global=7000, max_local=2000, max_iter=30,
                                                       cluster_num=50, random_seed=42)

# %%
predefined_groups_synth = reformat_discrimination_results(convert_to_non_float_rows(results_df_synth, schema),
                                                          data_obj.dataframe)

#%%
comparison_results = compare_discriminatory_groups(predefined_groups_origin, predefined_groups_synth)

print("Dddd")
