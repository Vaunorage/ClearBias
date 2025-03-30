from matplotlib import pyplot as plt

from data_generator.main import get_real_data, generate_from_real_data
from data_generator.utils import plot_distribution_comparison
from methods.adf.main import adf_fairness_testing
from methods.utils import compare_discriminatory_groups, check_groups_in_synthetic_data, get_groups

# %%
data_obj, schema = get_real_data('adult', use_cache=True)
results_df_origin, metrics_origin = adf_fairness_testing(data_obj, max_global=20000, max_local=100,
                                                         cluster_num=50, random_seed=42, max_runtime_seconds=400,
                                                         max_tsn=3000, step_size=0.05)

# %%
predefined_groups_origin, nb_elements = get_groups(results_df_origin, data_obj, schema)

# %%
data_obj_synth, schema = generate_from_real_data('adult', nb_groups=1, predefined_groups=predefined_groups_origin,
                                                 use_cache=True)

# %%
group_check_results = check_groups_in_synthetic_data(data_obj_synth, predefined_groups_origin)
print(f"Found {group_check_results['groups_found']} out of {group_check_results['total_groups']} groups")
print(f"Coverage: {group_check_results['coverage_percentage']:.2f}%")

# %% Run fairness testing
results_df_synth, metrics_synth = adf_fairness_testing(data_obj_synth, max_global=20000, max_local=100,
                                                       cluster_num=50, random_seed=42, max_runtime_seconds=600,
                                                       step_size=0.05, max_tsn=3000)

# %%
predefined_groups_synth, nb_elements_synth = get_groups(results_df_synth, data_obj, schema)

# %%
comparison_results = compare_discriminatory_groups(predefined_groups_origin, predefined_groups_synth)

# print(comparison_results)
# %%
fig = plot_distribution_comparison(schema, data_obj_synth)
plt.show()
