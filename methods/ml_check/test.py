import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dataclasses import dataclass

from data_generator.main import generate_data, GeneratedData
from paths import HERE
from methods.ml_check.ml_check import Assume, Assert, RunChecker, PropertyChecker
from methods.ml_check.util import local_delete, local_load, update_dataframe_types, file_exists

import statistics as st
import math
import warnings

warnings.filterwarnings("ignore")


def delete_all():
    files = ['assumeStmnt', 'assertStmnt', 'Cand-Set', 'CexSet', 'CandidateSet',
             'CandidateSetInst', 'CandidateSetBranch', 'TestDataSMT',
             'TestDataSMTMain', 'DecSmt', 'ToggleBranchSmt',
             'ToggleFeatureSmt', 'TreeOutput', 'SampleFile', 'FinalOutput',
             'ConditionFile', 'MUT', 'DNNSmt', 'TestData', 'TestDataSet', 'CandTestDataSet']
    for ff in files:
        local_delete(ff)


def func_calculate_sem(samples):
    standard_dev = st.pstdev(samples)
    return standard_dev / math.sqrt(len(samples))


def run_analysis(generated_data: GeneratedData, iteration_no: int = 1):
    delete_all()

    # Save training dataframe to CSV
    train_data_loc = HERE.joinpath('methods/ml_check/files/input_data.csv').as_posix()
    generated_data.training_dataframe.to_csv(train_data_loc, header=True, index=False)

    # Initialize data
    input_data = generated_data.training_dataframe.copy()
    categorical_columns = generated_data.categorical_columns
    protected_attributes = generated_data.protected_attributes
    outcome_column = generated_data.outcome_column

    # Separate numerical and categorical columns
    numerical_columns = [col for col in input_data.columns if col not in categorical_columns + [outcome_column]]

    # Preprocessing
    if categorical_columns:
        input_data[categorical_columns] = input_data[categorical_columns].fillna('missing').astype(str)
        label_encoders = {col: LabelEncoder() for col in categorical_columns}
        input_data[categorical_columns] = input_data[categorical_columns].apply(
            lambda col: label_encoders[col.name].fit_transform(col))

    if numerical_columns:
        input_data[numerical_columns] = input_data[numerical_columns].apply(pd.to_numeric, errors='coerce')
        input_data[numerical_columns] = SimpleImputer(strategy='median').fit_transform(input_data[numerical_columns])
        input_data[numerical_columns] = StandardScaler().fit_transform(input_data[numerical_columns])

    # Update data types and rename outcome column
    input_data = update_dataframe_types(input_data, categorical_cols=categorical_columns)
    input_data.rename(columns={outcome_column: generated_data.outcome_column}, inplace=True)

    # Prepare training data
    X = input_data.drop(generated_data.outcome_column, axis=1)
    y = input_data[generated_data.outcome_column]

    # Train model
    model = RandomForestClassifier()
    model.fit(X, y)

    f = open(HERE.joinpath('methods/ml_check/files/fairnessResults.txt').as_posix(), 'w')

    f.write("Result of MLCheck is----- \n")
    cex_count = 0
    cex_count_list = []
    f.write('------MLC_DT results-----\n')

    for no in range(iteration_no):
        PropertyChecker(output_class_name=generated_data.outcome_column, categorical_columns=categorical_columns,
                        no_of_params=2, max_samples=6, mul_cex=True, train_data_available=True, train_ratio=30,
                        no_of_train=1000, model=model, train_data_loc=train_data_loc,
                        white_box_model=['Decision tree'], no_of_class=2)

        for i, col in enumerate(X.columns.tolist()):
            if col in protected_attributes:
                Assume('x[i] != y[i]', col)
            else:
                Assume('x[i] = y[i]', col)
        Assert('model.predict(x) == model.predict(y)')

        obj_faircheck = RunChecker()
        obj_faircheck.run_prop_check()

    if file_exists('DiscriminatoryCases'):
        discrimination_cases = local_load('DiscriminatoryCases')
        print("Discrimination cases found:")
        print(discrimination_cases)

    else:
        discrimination_cases = pd.DataFrame()
        print("No discrimination cases found.")

    dfCexSet = local_load('CexSet')
    cex_count = cex_count + round(dfCexSet.shape[0] / 2)
    cex_count_list.append(round(dfCexSet.shape[0] / 2))
    mean_cex_count = cex_count / iteration_no
    cex_count_sem = func_calculate_sem(cex_count_list)

    f.write('Mean value is: ' + str(mean_cex_count) + '\n')
    f.write('Standard Error of the Mean is: +- ' + str(cex_count_sem) + '\n \n ')

    return discrimination_cases, mean_cex_count, cex_count_sem


if __name__ == '__main__':
    # Example usage
    ge = generate_data(min_number_of_classes=2, max_number_of_classes=6, nb_attributes=6,
                       prop_protected_attr=0.3, nb_groups=500, max_group_size=50, hiddenlayers_depth=3,
                       min_similarity=0.0, max_similarity=1.0, min_alea_uncertainty=0.0, max_alea_uncertainty=1.0,
                       min_epis_uncertainty=0.0, max_epis_uncertainty=1.0, min_magnitude=0.0, max_magnitude=1.0,
                       min_frequency=0.0, max_frequency=1.0, categorical_outcome=True, nb_categories_outcome=4)

    discrimination_cases, mean_cex_count, cex_count_sem = run_analysis(ge, iteration_no=1)

    print(f"Mean CEX count: {mean_cex_count}")
    print(f"Standard Error of the Mean: {cex_count_sem}")
