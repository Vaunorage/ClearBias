from data_generator.main import generate_data
from methods.exp_ga.algo import run_expga

ge = generate_data(
    nb_attributes=6,
    min_number_of_classes=2,
    max_number_of_classes=4,
    prop_protected_attr=0.3,
    nb_groups=100,
    max_group_size=100,
    categorical_outcome=True,
    nb_categories_outcome=4)

#%%
result_df, report = run_expga(ge, threshold=0.5, threshold_rank=0.5, max_global=50, max_local=50)
print(result_df)
