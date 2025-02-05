params = 13

sensitive_param = 7 # Starts at 1.

input_bounds = [
    (17, 90),    # age
    (0, 8),      # workclass
    (0, 16),     # education
    (1, 16),     # education-num
    (0, 6),      # marital-status
    (0, 14),     # occupation
    (0, 6),      # relationship
    (0, 4),      # race
    (0, 1),      # sex
    (0, 99999),  # capital-gain
    (0, 4356),   # capital-loss
    (1, 99),     # hours-per-week
    (0, 41)      # native-country
]

classifier_name = 'Decision_tree_standard_unfair.pkl'

threshold = 0

perturbation_unit = 1

retraining_inputs = "Retrain_Example_File.txt"