from methods.exp_ga.algo import run_expga
from matplotlib import pyplot as plt
from data_generator.main import get_real_data, generate_from_real_data
from data_generator.utils import plot_distribution_comparison
from methods.utils import reformat_discrimination_results, convert_to_non_float_rows, compare_discriminatory_groups

# %%
data_obj, schema = get_real_data('adult', use_cache=True)
results_df_origin, metrics = run_expga(data=data_obj, threshold_rank=0.5, max_global=3000, max_local=1000,
                                       max_tsn=50000, threshold=0.5)

# %%
non_float_df = convert_to_non_float_rows(results_df_origin, schema)
predefined_groups_origin = reformat_discrimination_results(non_float_df, data_obj.dataframe)
nb_elements = sum([el.group_size for el in predefined_groups_origin])

# %%
data_obj_synth, schema = generate_from_real_data('adult',
                                                 predefined_groups=predefined_groups_origin)

# %%
fig = plot_distribution_comparison(schema, data_obj_synth)
plt.show()

# %% Run fairness testing
results_df_synth, metrics_synth = run_expga(data=data_obj_synth, threshold_rank=0.5, max_global=2000, max_local=100,
                                            max_tsn=50000, threshold=0.5)

# %%
predefined_groups_synth = reformat_discrimination_results(convert_to_non_float_rows(results_df_synth, schema),
                                                          data_obj.dataframe)

# %%
comparison_results = compare_discriminatory_groups(predefined_groups_origin, predefined_groups_synth)

# %%
