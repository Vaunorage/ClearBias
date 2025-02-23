from methods.sg.main import run_sg
from matplotlib import pyplot as plt

from data_generator.main import get_real_data, generate_from_real_data
from data_generator.utils import plot_distribution_comparison
from methods.utils import reformat_discrimination_results, convert_to_non_float_rows, compare_discriminatory_groups

# %%
data_obj, schema = get_real_data('adult')
results_df_origin, metrics = run_sg(ge=data_obj,
                                    model_type='rf', cluster_num=50, limit=100, iter=4)

# %%
non_float_df = convert_to_non_float_rows(results_df_origin, schema)
predefined_groups_origin = reformat_discrimination_results(non_float_df, data_obj.dataframe)
nb_elements = sum([el.group_size for el in predefined_groups_origin])

# %%
data_obj_synth, schema = generate_from_real_data('adult',
                                                 predefined_groups=predefined_groups_origin,
                                                 extra_rows=1000)

# %%
fig = plot_distribution_comparison(schema, data_obj_synth)
plt.show()

# %% Run fairness testing
results_df_synth, metrics_synth = run_sg(ge=data_obj_synth,
                                    model_type='rf', cluster_num=50, limit=100, iter=6)
# %%
predefined_groups_synth = reformat_discrimination_results(convert_to_non_float_rows(results_df_synth, schema),
                                                          data_obj.dataframe)

# %%
comparison_results = compare_discriminatory_groups(predefined_groups_origin, predefined_groups_synth)

# %%
