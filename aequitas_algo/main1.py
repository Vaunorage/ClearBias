#!/usr/bin/env python
# coding: utf-8

from sqlalchemy import create_engine

from aequitas_algo.algo import run_aequitas
from data_generator.main import generate_data
import random

from paths import HERE

DB_PATH = "/home/vaunorage/PycharmProjects/clear/ClearBias/experiment_results.db"
engine = create_engine(f'sqlite:///{DB_PATH}')


def run_algo(model_type, min_number_of_classes=2, max_number_of_classes=6, nb_attributes=6,
             prop_protected_attr=0.3, nb_groups=500, max_group_size=50, hiddenlayers_depth=3, min_similarity=0.0,
             max_similarity=1.0, min_alea_uncertainty=0.0, max_alea_uncertainty=1.0,
             min_epis_uncertainty=0.0, max_epis_uncertainty=1.0,
             min_magnitude=0.0, max_magnitude=1.0, min_frequency=0.0, max_frequency=1.0,
             categorical_outcome=True, nb_categories_outcome=4,
             global_iteration_limit=1000, local_iteration_limit=100):
    modelst = ["DecisionTree", "MLPC", "SVM", "RandomForest"]

    model_type = modelst[model_type]
    df, protected_attr = generate_data(min_number_of_classes=min_number_of_classes, max_group_size=max_group_size,
                                       max_number_of_classes=max_number_of_classes, nb_attributes=nb_attributes,
                                       prop_protected_attr=prop_protected_attr, nb_groups=nb_groups,
                                       hiddenlayers_depth=hiddenlayers_depth, min_similarity=min_similarity,
                                       max_similarity=max_similarity, min_alea_uncertainty=min_alea_uncertainty,
                                       max_alea_uncertainty=max_alea_uncertainty,
                                       min_epis_uncertainty=min_epis_uncertainty,
                                       max_epis_uncertainty=max_epis_uncertainty,
                                       min_magnitude=min_magnitude, max_magnitude=max_magnitude,
                                       min_frequency=min_frequency, max_frequency=max_frequency,
                                       categorical_outcome=categorical_outcome,
                                       nb_categories_outcome=nb_categories_outcome)

    dff = df[[e for e in protected_attr] + ['outcome']]
    results_df, model_scores = run_aequitas(dff, col_to_be_predicted="outcome",
                                            sensitive_param_name_list=[k for k, e in protected_attr.items() if e],
                                            perturbation_unit=1, model_type=model_type, threshold=0,
                                            global_iteration_limit=global_iteration_limit,
                                            local_iteration_limit=local_iteration_limit)

    print("Merge results")
    df_true_pos = df.drop(columns=list(protected_attr)).merge(results_df.drop(columns=list(protected_attr)), how='left',
                                                              on='couple_key')
    df_false_pos = results_df.merge(df, how='left', on='couple_key')

    print("Calculate metrics")
    unique_couple_keys_in_df = df['couple_key'].nunique()
    unique_couple_keys_in_results_df = results_df['couple_key'].nunique()

    couple_tpr, couple_fpr = 0, 0

    if unique_couple_keys_in_df > 0:
        couple_tpr = df_true_pos[~df_true_pos['subgroup_id'].isnull()]['couple_key'].unique().shape[
                         0] / unique_couple_keys_in_df

    if unique_couple_keys_in_results_df > 0:
        couple_fpr = df_false_pos[~df_false_pos['collisions'].isnull()]['couple_key'].unique().shape[
                         0] / unique_couple_keys_in_results_df

    return couple_tpr, couple_fpr, model_scores, df_true_pos, protected_attr


def random_scale_boundary(boundary, min_scale=0.0, max_scale=1.0):
    original_min, original_max = boundary
    scale_min = random.uniform(min_scale, max_scale)
    scale_max = random.uniform(scale_min, max_scale)

    if isinstance(original_min, int) and isinstance(original_max, int):
        return (int(scale_min), int(scale_max))
    else:
        new_min = original_min + (original_max - original_min) * scale_min
        new_max = original_min + (original_max - original_min) * scale_max

        return (new_min, new_max)


def generate_parameters(parameter_bounds, min_max_labels):
    params = {}
    # Handle all other parameters
    for key, (low, high) in parameter_bounds.items():
        if isinstance(low, bool):
            params[key] = low
        elif isinstance(low, int) and isinstance(high, int):
            if key in min_max_labels:
                ll = random_scale_boundary((low, high), low, high)
                params[f'min_{key}'], params[f'max_{key}'] = ll[0], ll[1]
            else:
                params[key] = random.randint(low, high)
        else:
            if key in min_max_labels:
                ll = random_scale_boundary((low, high), low, high)
                params[f'min_{key}'], params[f'max_{key}'] = ll[0], ll[1]
            else:
                params[key] = random.uniform(low, high)
    return params


# In[8]:


# parameter_bounds = {
#     "model_type": (0, 3),
#     "nb_attributes": (3, 7),
#     "prop_protected_attr": (0.1, 0.4),
#     "nb_groups": (100, 600),
#     "max_group_size": (40, 60),
#     "hiddenlayers_depth": (1, 6),
#     "categorical_outcome": (True, True),  # assuming always True
#     "nb_categories_outcome": (2, 6),
#     "global_iteration_limit": (1000, 1100),
#     "local_iteration_limit": (100, 200),
#     "similarity": (0.0, 1.0),
#     "alea_uncertainty": (0.0, 1.0),
#     "epis_uncertainty": (0.0, 1.0),
#     "magnitude": (0.0, 1.0),
#     "frequency": (0.0, 1.0),
#     "number_of_classes": (2, 6),
# }
#
# k = 1000
#
# for _ in range(k):
#     min_max_labels = ["similarity", "alea_uncertainty", "epis_uncertainty",
#                       "magnitude", "frequency", "number_of_classes"]
#     params = generate_parameters(parameter_bounds, min_max_labels)
#     print(params)
#     couple_tpr, couple_fpr, model_scores, couple_df, protected_attr = run_algo(**params)
#
#     couple_df['couple_tpr'] = couple_tpr
#     couple_df['couple_fpr'] = couple_fpr
#     couple_df['model_scores'] = str(model_scores[0])
#     couple_df['params'] = str(params)
#     couple_df['attr'] = str(protected_attr)
#     print(couple_tpr, couple_fpr)
#     # save_results(0, 0, {}, {})
#
#     couple_df.to_sql('results4', con=engine, if_exists='append', index=False)

import optuna


def objective(trial):
    parameter_bounds = {
        "model_type": (0, 3),
        "nb_attributes": (3, 7),
        "prop_protected_attr": (0.1, 0.4),
        "nb_groups": (100, 600),
        "max_group_size": (40, 60),
        "hiddenlayers_depth": (1, 6),
        "categorical_outcome": True,  # assuming always True
        "nb_categories_outcome": (2, 6),
        "global_iteration_limit": (1000, 1100),
        "local_iteration_limit": (100, 200),
        "similarity": (0.0, 1.0),
        "alea_uncertainty": (0.0, 1.0),
        "epis_uncertainty": (0.0, 1.0),
        "magnitude": (0.0, 1.0),
        "frequency": (0.0, 1.0),
        "number_of_classes": (2, 6),
    }

    min_max_labels = ["similarity", "alea_uncertainty", "epis_uncertainty", "magnitude", "frequency",
                      "number_of_classes"]
    params = {}
    for key, bounds in parameter_bounds.items():
        if isinstance(bounds, bool):
            params[key] = bounds
        elif key in min_max_labels:
            low, high = bounds
            if isinstance(bounds, tuple) and isinstance(bounds[0], int):
                kk = trial.suggest_int(key, low, high)
                kkmax = kk + 1
            else:
                kk = trial.suggest_float(key, low, high)
                kkmax = kk + 0.01
            params[f'min_{key}'], params[f'max_{key}'] = kk, kkmax
        elif isinstance(bounds, tuple) and isinstance(bounds[0], int):
            params[key] = trial.suggest_int(key, bounds[0], bounds[1])
        else:
            params[key] = trial.suggest_float(key, bounds[0], bounds[1])

    # Assuming run_algo is the function you use to evaluate your model
    couple_tpr, couple_fpr, model_scores, couple_df, protected_attr = run_algo(**params)

    # Log additional information if needed
    trial.set_user_attr("couple_fpr", couple_fpr)
    trial.set_user_attr("model_scores", model_scores)
    trial.set_user_attr("protected_attr", protected_attr)

    couple_df.to_sql('results_optuna', con=engine, if_exists='append', index=False)

    return couple_tpr


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # You can adjust the number of trials

print("Best trial:")
trial = study.best_trial
print(" Value: ", trial.value)
print(" Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
