import pandas as pd

from data_generator.main import generate_data
from methods.aequitas_algo.algo import run_aequitas


def run_algo(model_type, min_number_of_classes=2, max_number_of_classes=6, nb_attributes=6,
             prop_protected_attr=0.3, nb_elems=500, hiddenlayers_depth=3, min_similarity=0.0,
             max_similarity=1.0, min_alea_uncertainty=0.0, max_alea_uncertainty=1.0,
             min_epis_uncertainty=0.0, max_epis_uncertainty=1.0,
             min_magnitude=0.0, max_magnitude=1.0, min_frequency=0.0, max_frequency=1.0,
             categorical_outcome=True, nb_categories_outcome=4,
             global_iteration_limit=1000, local_iteration_limit=100):
    modelst = ["DecisionTree", 'DecisionTreeClassifier', "MLPC", 'MLPClassifier', "SVM", 'SVC', "RandomForest",
               'RandomForestClassifier']

    model_type = modelst[model_type]
    df, protected_attr = generate_data(min_number_of_classes=min_number_of_classes, max_number_of_classes=max_number_of_classes, nb_attributes=nb_attributes,
                                       prop_protected_attr=prop_protected_attr, nb_elems=nb_elems, hiddenlayers_depth=hiddenlayers_depth, min_similarity=min_similarity,
                                       max_similarity=max_similarity, min_alea_uncertainty=min_alea_uncertainty, max_alea_uncertainty=max_alea_uncertainty,
                                       min_epis_uncertainty=min_epis_uncertainty, max_epis_uncertainty=max_epis_uncertainty,
                                       min_magnitude=min_magnitude, max_magnitude=max_magnitude, min_frequency=min_frequency, max_frequency=max_frequency,
                                       categorical_outcome=categorical_outcome, nb_categories_outcome=nb_categories_outcome)

    dff = df[[e for e in protected_attr] + ['outcome']]
    results_df, model_scores = run_aequitas(dff, col_to_be_predicted="outcome",
                              sensitive_param_name_list=[k for k, e in protected_attr.items() if e],
                              perturbation_unit=1, model_type=model_type, threshold=0,
                              global_iteration_limit=global_iteration_limit, local_iteration_limit=local_iteration_limit)

    def generate_merged_dataframe(results_df, df, protected_attr):
        def transform_subgroup(x, protected_attr):
            res = list(map(lambda e: '|'.join(list(map(str, e))), x[list(protected_attr)].values.tolist()))
            res = ["*".join(res), "*".join(res[::-1])]
            return pd.Series(res, index=x.index)

        # Creating individual discrimination keys
        results_df['ind_discr_key'] = results_df.apply(
            lambda x: '|'.join(list(map(str, x[list(protected_attr)].values.tolist()))), axis=1)
        df['ind_discr_key'] = df.apply(lambda x: '|'.join(list(map(str, x[list(protected_attr)].values.tolist()))),
                                       axis=1)

        # Creating coupled discrimination keys
        results_df['couple_discr_key'] = results_df.groupby(['subgroup_num']).apply(
            lambda x: transform_subgroup(x, protected_attr)).reset_index(level=0, drop=True)
        df['couple_discr_key'] = df.groupby(['subgroup_num', 'subgroup_id']).apply(
            lambda x: transform_subgroup(x, protected_attr)).reset_index(level=0, drop=True).reset_index(level=0,
                                                                                                         drop=True)

        # Define columns for merging and results
        common_columns = ['diff_outcome', 'intersectionality', 'similarity', 'alea_uncertainty', 'epis_uncertainty',
                          'magnitude', 'frequency']
        results_columns = ['ind_discr_key', 'couple_discr_key', 'diff_outcome']

        # Merging for individual discrimination keys
        results_df_ind = results_df[results_columns].drop_duplicates().reset_index(drop=True)
        df2_ind = df[results_columns[:-1] + common_columns].drop_duplicates().reset_index(drop=True)
        merged_df_leftjoin_df2_ind = df2_ind.merge(results_df_ind, on='ind_discr_key', how='inner',
                                                   suffixes=('_injected', '_found'))
        merged_df_leftjoin_df2_ind['source'] = 'true_positive'

        # Merging for coupled discrimination keys
        results_df_couple = results_df[results_columns].drop_duplicates().reset_index(drop=True)
        df2_couple = df[results_columns[:-1] + common_columns].drop_duplicates().reset_index(drop=True)
        merged_df_leftjoin_df2_couple = df2_couple.merge(results_df_couple, on='couple_discr_key', how='inner',
                                                         suffixes=('_injected', '_found'))
        merged_df_leftjoin_df2_couple['source'] = 'true_positive'

        # Unique entries for individual keys
        df_unique_to_results_ind = pd.merge(results_df_ind[['ind_discr_key', 'diff_outcome']],
                                            df2_ind[['ind_discr_key']],
                                            on='ind_discr_key', how='left', indicator=True)
        df_unique_to_results_ind = df_unique_to_results_ind[df_unique_to_results_ind['_merge'] == 'left_only'].drop(
            columns=['_merge'])
        df_unique_to_results_ind = df_unique_to_results_ind.assign(**{col: None for col in common_columns[1:]})
        df_unique_to_results_ind['diff_outcome_found'] = df_unique_to_results_ind['diff_outcome']
        df_unique_to_results_ind['source'] = 'false_positive'

        # Unique entries for coupled keys
        df_unique_to_results_couple = pd.merge(results_df_couple[['couple_discr_key', 'diff_outcome']],
                                               df2_couple[['couple_discr_key']], on='couple_discr_key', how='left',
                                               indicator=True)
        df_unique_to_results_couple = df_unique_to_results_couple[
            df_unique_to_results_couple['_merge'] == 'left_only'].drop(columns=['_merge'])
        df_unique_to_results_couple = df_unique_to_results_couple.assign(**{col: None for col in common_columns[1:]})
        df_unique_to_results_couple['diff_outcome_found'] = df_unique_to_results_couple['diff_outcome']
        df_unique_to_results_couple['source'] = 'false_positive'

        indv_df = pd.concat([merged_df_leftjoin_df2_ind, df_unique_to_results_ind], ignore_index=True)
        couple_df = pd.concat([merged_df_leftjoin_df2_couple, df_unique_to_results_couple], ignore_index=True)

        return df, results_df, couple_df, indv_df

    df, result_df, couple_df, indv_df = generate_merged_dataframe(results_df, df, protected_attr)

    def calculate_metrics(df, results_df, couple_df):
        unique_keys_in_df = df[
            'couple_discr_key'].nunique()  # Adjust 'ind_discr_key' to the correct column name if necessary
        unique_keys_in_results_df = results_df[
            'couple_discr_key'].nunique()  # Adjust 'ind_discr_key' to the correct column name if necessary

        # Calculate TPR and FPR using the unique keys count
        tpr = couple_df[couple_df['source'] == 'true_positive'].shape[0] / unique_keys_in_df
        fpr = couple_df[couple_df['source'] == 'false_positive'].shape[0] / unique_keys_in_results_df

        return tpr, fpr

    tpr, fpr = calculate_metrics(df, results_df, couple_df)

    return tpr, fpr
