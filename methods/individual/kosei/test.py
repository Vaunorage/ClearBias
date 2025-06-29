from matplotlib import pyplot as plt

from data_generator.main import get_real_data, generate_from_real_data
from data_generator.utils import plot_distribution_comparison
from methods.individual.kosei.main import run_kosei
from methods.utils import compare_discriminatory_groups, check_groups_in_synthetic_data, get_groups

# %% Get real data and run KOSEI on it
data_obj, schema = get_real_data('adult', use_cache=True)
results_df_origin, metrics_origin = run_kosei(
    discrimination_data=data_obj,
    num_samples=200,
    local_search_limit=100,
    random_seed=42,
    max_runtime_seconds=400
)

# %% Extract discriminatory groups from the results
predefined_groups_origin, nb_elements = get_groups(results_df_origin, data_obj, schema)

# %% Generate synthetic data based on the discriminatory groups found
data_obj_synth, schema = generate_from_real_data(
    'adult',
    nb_groups=1,
    predefined_groups=predefined_groups_origin,
    # use_cache=True
)

# %% Check if the synthetic data contains the discriminatory groups
group_check_results = check_groups_in_synthetic_data(data_obj_synth, predefined_groups_origin)
print(f"Found {group_check_results['groups_found']} out of {group_check_results['total_groups']} groups")
print(f"Coverage: {group_check_results['coverage_percentage']:.2f}%")

# %% Run KOSEI on the synthetic data
results_df_synth, metrics_synth = run_kosei(
    discrimination_data=data_obj_synth,
    num_samples=200,
    local_search_limit=100,
    random_seed=42,
    max_runtime_seconds=600
)

# %% Extract discriminatory groups from the synthetic data results
predefined_groups_synth, nb_elements_synth = get_groups(results_df_synth, data_obj, schema)

# %% Compare the discriminatory groups from the original and synthetic data
comparison_results = compare_discriminatory_groups(predefined_groups_origin, predefined_groups_synth)

# Print comparison results
print("\nComparison of discriminatory groups:")
print(f"Original groups: {len(predefined_groups_origin)}")
print(f"Synthetic groups: {len(predefined_groups_synth)}")
print(f"Common groups: {comparison_results['common_groups']}")
print(f"Similarity score: {comparison_results['similarity_score']:.2f}")

# %% Plot distribution comparison
fig = plot_distribution_comparison(schema, data_obj_synth)
plt.show()
