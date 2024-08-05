import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dataclasses import dataclass

from paths import HERE
from methods.ml_check.ml_check import Assume, Assert, RunChecker, PropertyChecker
from methods.ml_check.util import local_delete, local_load, update_dataframe_types

import statistics as st
import math
import warnings

warnings.filterwarnings("ignore")


@dataclass
class GeneratedData:
    dataframe: pd.DataFrame
    categorical_columns: list
    protected_attributes: dict
    collisions: int
    nb_groups: int
    max_group_size: int
    hiddenlayers_depth: int
    outcome_column: str


def delete_all():
    files = ['assumeStmnt', 'assertStmnt', 'Cand-Set', 'CandidateSet',
             'CandidateSetInst', 'CandidateSetBranch', 'TestDataSMT',
             'TestDataSMTMain', 'DecSmt', 'ToggleBranchSmt',
             'ToggleFeatureSmt', 'TreeOutput', 'SampleFile', 'FinalOutput',
             'ConditionFile', 'MUTWeight', 'MUT', 'DNNSmt',
             'TestData', 'TestDataSet', 'CandTestDataSet']
    for ff in files:
        local_delete(ff)


def func_calculate_sem(samples):
    standard_dev = st.pstdev(samples)
    return standard_dev / math.sqrt(len(samples))


def run_analysis(generated_data: GeneratedData, iteration_no: int = 1):
    delete_all()

    input_data = generated_data.dataframe
    categorical_columns = generated_data.categorical_columns
    protected_attributes = list(generated_data.protected_attributes.keys())
    outcome_column = generated_data.outcome_column

    numerical_columns = [col for col in input_data.columns if col not in categorical_columns + [outcome_column]]

    # Preprocessing
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    numerical_imputer = SimpleImputer(strategy='median')

    input_data[categorical_columns] = categorical_imputer.fit_transform(input_data[categorical_columns])
    input_data[numerical_columns] = numerical_imputer.fit_transform(input_data[numerical_columns])

    label_encoders = {}
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        input_data[column] = label_encoders[column].fit_transform(input_data[column])

    scaler = StandardScaler()
    input_data[numerical_columns] = scaler.fit_transform(input_data[numerical_columns])

    input_data = update_dataframe_types(input_data, categorical_cols=categorical_columns)

    input_data.rename(columns={outcome_column: 'Class'}, inplace=True)

    X = input_data.drop('Class', axis=1)
    y = input_data['Class']

    model = RandomForestClassifier()
    model.fit(X, y)

    f = open(HERE.joinpath('methods/ml_check/files/fairnessResults.txt').as_posix(), 'w')

    f.write("Result of MLCheck is----- \n")
    cex_count = 0
    cex_count_list = []
    f.write('------MLC_DT results-----\n')

    for no in range(iteration_no):
        PropertyChecker(output_class_name='Class', categorical_columns=categorical_columns, no_of_params=2,
                        max_samples=6, mul_cex=True, train_data_available=True, train_ratio=30, no_of_train=1000,
                        model=model, train_data_loc=HERE.joinpath('methods/ml_check/files/input_data.csv').as_posix(),
                        white_box_model=['Decision tree'], no_of_class=2)

        for i, col in enumerate(X.columns.tolist()):
            if col in protected_attributes:
                Assume('x[i] != y[i]', col)
            else:
                Assume('x[i] = y[i]', col)
        Assert('model.predict(x) == model.predict(y)')

        obj_faircheck = RunChecker()
        obj_faircheck.run_prop_check()

    discrimination_cases = local_load('DiscriminatoryCases')
    if not discrimination_cases.empty:
        print("Discrimination cases found:")
        print(discrimination_cases)
    else:
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
    generated_data = GeneratedData(
        dataframe=pd.read_csv('path_to_your_generated_data.csv'),
        categorical_columns=['column1', 'column2', 'column3'],
        protected_attributes={'sensitive_attribute': [0, 1]},
        collisions=5,
        nb_groups=3,
        max_group_size=10,
        hiddenlayers_depth=2,
        outcome_column='target'
    )

    discrimination_cases, mean_cex_count, cex_count_sem = run_analysis(generated_data, iteration_no=1)

    print(f"Mean CEX count: {mean_cex_count}")
    print(f"Standard Error of the Mean: {cex_count_sem}")
