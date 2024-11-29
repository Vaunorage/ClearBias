from data_generator.main import generate_data
from methods.ml_check.algo import run_mlcheck

ge = generate_data(
    nb_attributes=4,
    min_number_of_classes=2,
    max_number_of_classes=2,
    prop_protected_attr=0.2,
    # min_granularity=1,
    # max_granularity=4,
    nb_groups=5,
    max_group_size=100,
    categorical_outcome=True,
    nb_categories_outcome=4)

#%%
result_df, report = run_mlcheck(ge, iteration_no=1)
print(result_df)
print(report)
