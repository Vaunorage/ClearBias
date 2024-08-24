from data_generator.main import generate_data
from methods.aequitas.algo import run_aequitas

model_type = "DecisionTree"
ge = generate_data(min_number_of_classes=2, max_number_of_classes=6, nb_attributes=6,
                   prop_protected_attr=0.3, nb_groups=500, max_group_size=50, hiddenlayers_depth=3,
                   min_similarity=0.0, max_similarity=1.0, min_alea_uncertainty=0.0,
                   max_alea_uncertainty=1.0, min_epis_uncertainty=0.0, max_epis_uncertainty=1.0,
                   min_magnitude=0.0, max_magnitude=1.0, min_frequency=0.0, max_frequency=1.0,
                   categorical_outcome=True, nb_categories_outcome=4)
global_iteration_limit = 100
local_iteration_limit = 10
results_df, model_scores = run_aequitas(ge.training_dataframe, col_to_be_predicted=ge.outcome_column,
                                        sensitive_param_name_list=ge.protected_attributes,
                                        perturbation_unit=1, model_type=model_type, threshold=0,
                                        global_iteration_limit=global_iteration_limit,
                                        local_iteration_limit=local_iteration_limit)

print(results_df)
