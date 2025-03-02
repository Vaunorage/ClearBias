
from data_generator.main import generate_data, get_real_data, generate_data_schema, GroupDefinition
import matplotlib.pyplot as plt

from data_generator.utils import plot_distribution_comparison, print_distribution_stats, visualize_df, \
    create_parallel_coordinates_plot, plot_and_print_metric_distributions, unique_individuals_ratio, \
    individuals_in_multiple_groups, plot_correlation_matrices

# data, schema = generate_from_real_data('bank')
data, schema = get_real_data('adult')

# %%
nb_attributes = 20

# schema = generate_data_schema(min_number_of_classes=2, max_number_of_classes=9, prop_protected_attr=0.4,
#                               nb_attributes=nb_attributes)

predefined_groups = [
    GroupDefinition(
        group_size=50, subgroup_bias=0.3, similarity=0.8, alea_uncertainty=0.2, epis_uncertainty=0.3,
        frequency=0.7, avg_diff_outcome=2, diff_subgroup_size=0.2,
        subgroup1={'Attr7_T': 3, 'Attr8_T': 1},
        subgroup2={'Attr7_T': 2, 'Attr8_T': 2}
    )
]

#%%
data = generate_data(
    nb_attributes=nb_attributes,
    nb_groups=100,
    max_group_size=100,
    categorical_outcome=True,
    nb_categories_outcome=4,
    corr_matrix_randomness=1,
    categorical_influence=1,
    data_schema=schema,
    use_cache=False,
    predefined_groups=predefined_groups,
    # additional_random_rows=30000
)

print(f"Generated {len(data.dataframe)} samples in {data.nb_groups} groups")

#%%
# df = pd.concat([data.xdf, data.ydf],axis=1)
# fig = visualize_df(df, data.attr_columns, data.outcome_column, HERE.joinpath('ll.png'))
# fig.show()
# %%
create_parallel_coordinates_plot(data.dataframe)
plt.show()

# Print statistics
print_distribution_stats(schema, data)

# %%

plot_and_print_metric_distributions(data.dataframe)

#%%
# Example usage:
individual_col = 'indv_key'
group_col = 'group_key'

unique_ratio, duplicates_count, total = unique_individuals_ratio(data.dataframe, 'indv_key', data.attr_possible_values)
individuals_in_multiple_groups_count = individuals_in_multiple_groups(data.dataframe, individual_col, group_col)

print(f"Unique Individuals Ratio: {unique_ratio}, duplicate : {duplicates_count}, total: {total}")

#%%
plot_correlation_matrices(schema.correlation_matrix, data)

