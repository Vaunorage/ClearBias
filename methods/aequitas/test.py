from data_generator.main import generate_data
from methods.aequitas.algo import run_aequitas

ge = generate_data(
    nb_attributes=6,
    min_number_of_classes=2,
    max_number_of_classes=6,
    prop_protected_attr=0.3,
    nb_groups=100,
    max_group_size=100,
    categorical_outcome=True,
    nb_categories_outcome=4)

# %%
global_iteration_limit = 100
local_iteration_limit = 10
model_type = "RandomForest"
results_df, model_scores = run_aequitas(ge.training_dataframe, col_to_be_predicted=ge.outcome_column,
                                        sensitive_param_name_list=ge.protected_attributes,
                                        perturbation_unit=1, model_type=model_type, threshold=0,
                                        global_iteration_limit=global_iteration_limit,
                                        local_iteration_limit=local_iteration_limit)
