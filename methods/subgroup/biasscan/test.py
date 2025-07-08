from data_generator.main import generate_data
from methods.subgroup.biasscan.algo import run_bias_scan

ge = generate_data(
    nb_attributes=6,
    min_number_of_classes=2,
    max_number_of_classes=4,
    prop_protected_attr=0.1,
    nb_groups=100,
    max_group_size=100,
    categorical_outcome=True,
    nb_categories_outcome=4,
    use_cache=True
)

result_df, report = run_bias_scan(
    ge,
    test_size=0.3,
    random_state=42,
    n_estimators=200,
    bias_scan_num_iters=100,
    bias_scan_scoring='Poisson',
    bias_scan_favorable_value='high',
    bias_scan_mode='ordinal'
)

# %%
print(result_df)
print(f"Classification Report:\n{report}")
