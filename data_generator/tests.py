from data_generator.main import generate_valid_correlation_matrix, generate_data

# %%
nb_attributes = 20
correlation_matrix = generate_valid_correlation_matrix(nb_attributes)

data = generate_data(
    nb_attributes=nb_attributes,
    correlation_matrix=correlation_matrix,
    min_number_of_classes=2,
    max_number_of_classes=9,
    prop_protected_attr=0.4,
    nb_groups=100,
    max_group_size=100,
    categorical_outcome=True,
    nb_categories_outcome=4)

print(f"Generated {len(data.dataframe)} samples in {data.nb_groups} groups")