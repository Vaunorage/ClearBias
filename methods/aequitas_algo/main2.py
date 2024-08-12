from methods.aequitas_algo.main import run_algo

k = 1000

min_max_labels = ["similarity", "alea_uncertainty", "epis_uncertainty",
                      "magnitude", "frequency", "number_of_classes"]
params = {'model_type': 1, 'nb_attributes': 6, 'prop_protected_attr': 0.30063730416661205, 'nb_groups': 169,
          'max_group_size': 53, 'hiddenlayers_depth': 3, 'categorical_outcome': True, 'nb_categories_outcome': 3,
          'global_iteration_limit': 1005, 'local_iteration_limit': 138, 'min_similarity': 0.39509481658888956,
          'max_similarity': 0.7086540299373023, 'min_alea_uncertainty': 0.30201095664426236,
          'max_alea_uncertainty': 0.5689560957598876, 'min_epis_uncertainty': 0.9990630066520138,
          'max_epis_uncertainty': 0.9991732310883257, 'min_magnitude': 0.44157047582420417,
          'max_magnitude': 0.8475542709444843, 'min_frequency': 0.2361866069995815,
          'max_frequency': 0.5010885392176213, 'min_number_of_classes': 3, 'max_number_of_classes': 4}

print(params)
results_df, couple_tpr, couple_fpr, model_scores, couple_df, protected_attr = run_algo(**params)
print(results_df)
