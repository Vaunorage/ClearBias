import pandas as pd

from aequitas_algo.algo import run_aequitas

from data_generator.main2 import generate_data
from data_generator.utils import scale_dataframe, visualize_df

df, protected_attr = generate_data(min_number_of_classes=2, max_number_of_classes=6, nb_attributes=6,
                                   prop_protected_attr=0.2, nb_elems=300, hiddenlayers_depth=3, min_similarity=0.0,
                                   max_similarity=1.0, min_alea_uncertainty=0.0, max_alea_uncertainty=1.0,
                                   min_epis_uncertainty=0.0, max_epis_uncertainty=1.0,
                                   min_magnitude=0.0, max_magnitude=1.0, min_frequency=0.0, max_frequency=1.0,
                                   categorical_outcome=True, nb_categories_outcome=4)

# visualize_df(df, ['granularity', 'alea_uncertainty', 'epis_uncertainty', 'magnitude', 'diff_outcome'],
#              'diff_outcome', 'figure4.png')

# %%
dff = df[[e for e in protected_attr] + ['outcome']]
# scaled_df, min_values, max_values = scale_dataframe(dff)

# %%
results_df = run_aequitas(dff, col_to_be_predicted="outcome",
                          sensitive_param_name_list=[k for k, e in protected_attr.items() if e],
                          perturbation_unit=1, model_type="DecisionTree", threshold=0,
                          global_iteration_limit=1000, local_iteration_limit=100)


# %%

def transform_subgroup(x):
    res = list(map(lambda e: '|'.join(list(map(str, e))), x[list(protected_attr)].values.tolist()))
    res = ["*".join(res), "*".join(res[::-1])]
    return pd.Series(res, index=x.index)


results_df['ind_discr_key'] = results_df.apply(
    lambda x: '|'.join(list(map(str, x[list(protected_attr)].values.tolist()))), axis=1)
df['ind_discr_key'] = df.apply(lambda x: '|'.join(list(map(str, x[list(protected_attr)].values.tolist()))), axis=1)

results_df['couple_discr_key'] = results_df.groupby(['subgroup_num']).apply(transform_subgroup).reset_index(level=0,
                                                                                                            drop=True)
df['couple_discr_key'] = df.groupby(['subgroup_num', 'subgroup_id']).apply(transform_subgroup).reset_index(level=0,
                                                                                                           drop=True).reset_index(
    level=0, drop=True)

# %%
results_df_couple = results_df[['couple_discr_key', 'diff_outcome']].drop_duplicates().reset_index(drop=True)
df2_couple = df[['couple_discr_key', 'diff_outcome', 'intersectionality', 'similarity', 'alea_uncertainty',
          'epis_uncertainty', 'magnitude', 'frequency']].drop_duplicates().reset_index(drop=True)

#%%
merged_df_leftjoin_df2_couple = df2_couple.merge(results_df_couple, on='couple_discr_key', how='inner',
                               suffixes=('_injected', '_found'))

#%%
results_df_ind = results_df[['ind_discr_key', 'diff_outcome']].drop_duplicates().reset_index(drop=True)
df2_ind = df[['ind_discr_key', 'diff_outcome', 'intersectionality', 'similarity', 'alea_uncertainty',
          'epis_uncertainty', 'magnitude', 'frequency']].drop_duplicates().reset_index(drop=True)

merged_df_leftjoin_df2_ind= df2_ind.merge(results_df_ind, on='ind_discr_key', how='inner',
                               suffixes=('_injected', '_found'))

# %%
# a quel point le modele est bon lorsquil y a de lallucination
#pour les cas hallucinés, ils discriminent contre qui ? est ce que ceux qui trouvent sont proches
# faire une analyse pour voir parmis les groupes de discrimination combien il en trouve ?

# contre combien de personnes il trouve la discrimination :
# .shape[0] / merged_df[~merged_df['diff_outcome_injected'].isna()].shape[0]
# precision = sum([1 for e in org_found if e in org1_found]) / len(org1_found)

# il faut sassurer de trouver deux individus pour dire quil ya de la discrimination
# il faut regarder si la magnitude est differente

# %%
print('sss')
# original_df = scale_dataframe(scaled_df, reverse=True, min_values=min_values, max_values=max_values)
