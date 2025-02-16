from data_generator.main import generate_data, get_real_data, generate_from_real_data
from methods.aequitas.algo import run_aequitas

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

# %%

results, global_cases = run_aequitas(
    discrimination_data=ge,
    model_type='rf', max_global=200, max_local=100, step_size=1.0
)
print('helo')
