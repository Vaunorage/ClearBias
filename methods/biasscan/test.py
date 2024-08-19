from data_generator.main import generate_data
from methods.biasscan.algo import perform_bias_scan

ge = generate_data(min_number_of_classes=2, max_number_of_classes=6, nb_attributes=6,
                   prop_protected_attr=0.3, nb_groups=500, max_group_size=50, hiddenlayers_depth=3,
                   min_similarity=0.0, max_similarity=1.0, min_alea_uncertainty=0.0,
                   max_alea_uncertainty=1.0, min_epis_uncertainty=0.0, max_epis_uncertainty=1.0,
                   min_magnitude=0.0, max_magnitude=1.0, min_frequency=0.0, max_frequency=1.0,
                   categorical_outcome=True, nb_categories_outcome=4)

result_df, accuracy, report = perform_bias_scan(
    ge,
    test_size=0.3,
    random_state=42,
    n_estimators=200,
    bias_scan_num_iters=100,
    bias_scan_scoring='Poisson',  # Changed from default 'Poisson'
    bias_scan_favorable_value='high',  # Changed from default 'high'
    bias_scan_mode='ordinal'  # Changed from default 'ordinal'
)

print(result_df)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")