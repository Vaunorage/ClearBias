from data_generator.main import generate_data, get_real_data, GroupDefinition
import matplotlib.pyplot as plt

from data_generator.utils import plot_distribution_comparison, print_distribution_stats

# data, schema = generate_from_real_data('bank')
data, schema = get_real_data('adult')

# %%
nb_attributes = 20

# schema = generate_data_schema(min_number_of_classes=2, max_number_of_classes=9, prop_protected_attr=0.4,
#                               nb_attributes=nb_attributes)

# predefined_groups = [
#     GroupDefinition(
#         group_size=50, subgroup_bias=0.3, similarity=0.8, alea_uncertainty=0.2, epis_uncertainty=0.3,
#         frequency=0.7, avg_diff_outcome=2, diff_subgroup_size=0.2,
#         subgroup1={'Attr7_T': 3, 'Attr8_T': 1},
#         subgroup2={'Attr7_T': 2, 'Attr8_T': 2}
#     )
# ]

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
    # predefined_groups=predefined_groups,
    # additional_random_rows=30000
)

print(f"Generated {len(data.dataframe)} samples in {data.nb_groups} groups")

# %%

fig = plot_distribution_comparison(schema, data)
plt.show()

# Print statistics
print_distribution_stats(schema, data)
