from methods.individual.sg.main import run_sg
from matplotlib import pyplot as plt

from data_generator.main import get_real_data, generate_from_real_data
from data_generator.utils import plot_distribution_comparison
from methods.utils import reformat_discrimination_results, convert_to_non_float_rows, compare_discriminatory_groups


# %%
data_obj, schema = get_real_data('adult', use_cache=True)
results_df_origin, metrics = run_sg(data=data_obj, model_type='rf', cluster_num=50, max_tsn=1000,
                                    one_attr_at_a_time=True)

# %%
non_float_df = convert_to_non_float_rows(results_df_origin, schema)
predefined_groups_origin = reformat_discrimination_results(non_float_df, data_obj.dataframe)

# %%
data_obj_synth, schema = generate_from_real_data('adult',
                                                 predefined_groups=predefined_groups_origin, nb_groups=1)

# %%
fig = plot_distribution_comparison(schema, data_obj_synth)
plt.show()

# %% Run fairness testing
results_df_synth, metrics_synth = run_sg(data=data_obj_synth, model_type='rf', cluster_num=50, max_tsn=4000,
                                         one_attr_at_a_time=True)

# %%
predefined_groups_synth = reformat_discrimination_results(convert_to_non_float_rows(results_df_synth, schema),
                                                          data_obj.dataframe)

# %%
comparison_results = compare_discriminatory_groups(predefined_groups_origin, predefined_groups_synth)

# %%
