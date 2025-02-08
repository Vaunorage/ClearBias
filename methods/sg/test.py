from data_generator.main import generate_data, get_real_data, generate_from_real_data
from methods.aequitas.algo import run_aequitas
from methods.sg.main import symbolic_generation

# ge = generate_data(
#     nb_attributes=6,
#     min_number_of_classes=2,
#     max_number_of_classes=6,
#     prop_protected_attr=0.3,
#     nb_groups=100,
#     max_group_size=100,
#     categorical_outcome=True,
#     nb_categories_outcome=4,
#     use_cache=True)

ge, schema = get_real_data('adult')

# Then run symbolic generation with the model type
symbolic_generation(
    dataset=ge.dataframe,  # Pass the entire ge object, not just the dataframe
    model_type='lr',  # Choose from: 'lr', 'rf', 'svm', 'mlp'
    cluster_num=5,    # or whatever number of clusters you want
    limit=1000,       # maximum number of test cases
    iter=1            # iteration number
)

print("Sdsdd")