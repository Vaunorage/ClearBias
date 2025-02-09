from data_generator.main import generate_data, get_real_data, generate_from_real_data
from methods.sg.main import run_sg

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

ge, ge_schema = get_real_data('adult')

res = run_sg(ge)

print(res)
print("Sdsdd")