import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from paths import HERE
from methods.ml_check.ml_check import Assume, Assert, propCheck, runChecker
from methods.ml_check.util import local_delete, local_load, update_dataframe_types

import statistics as st
import math

import warnings

warnings.filterwarnings("ignore")

iteration_no = 1


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


if __name__ == '__main__':

    delete_all()

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]

    adult_data = pd.read_csv(url, names=columns, sep=r'\s*,\s*', engine='python', na_values="?")

    adult_data_path = HERE.joinpath('methods/ml_check/files/adult_data.csv')

    categorical_columns = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                           'native_country', 'income']
    numerical_columns = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

    protected_attributes = ['sex']

    categorical_imputer = SimpleImputer(strategy='most_frequent')
    numerical_imputer = SimpleImputer(strategy='median')

    adult_data[categorical_columns] = categorical_imputer.fit_transform(adult_data[categorical_columns])
    adult_data[numerical_columns] = numerical_imputer.fit_transform(adult_data[numerical_columns])

    label_encoders = {}
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        adult_data[column] = label_encoders[column].fit_transform(adult_data[column])

    scaler = StandardScaler()
    adult_data[numerical_columns] = scaler.fit_transform(adult_data[numerical_columns])

    input_data = pd.DataFrame(adult_data, columns=categorical_columns + numerical_columns)
    input_data = update_dataframe_types(input_data, categorical_cols=categorical_columns)

    input_data.rename(columns={'income': 'Class'}, inplace=True)

    input_data.to_csv(adult_data_path, index=False)

    X = input_data.drop('Class', axis=1)
    y = input_data['Class']

    model = RandomForestClassifier()
    model.fit(X, y)

    f = open(HERE.joinpath('methods/ml_check/files/fairnessResults.txt').as_posix(), 'w')

    f.write("Result of MLCheck is----- \n")
    cex_count = 0
    cex_count_list = []
    cex_count_sem = 0
    f.write('------MLC_DT results-----\n')
    for no in range(0, iteration_no):
        propCheck(output_class_name='Class', categorical_columns=categorical_columns, no_of_params=2, max_samples=50,
                  mul_cex=True, train_data_available=True, train_ratio=30, no_of_train=1000, model=model,
                  train_data_loc=adult_data_path.as_posix(), white_box_model=['Decision tree'], no_of_class=2)

        for i, col in enumerate(X.columns.tolist()):
            if col in protected_attributes:
                Assume('x[i] != y[i]', col)
            else:
                Assume('x[i] = y[i]', col)
        Assert('model.predict(x) == model.predict(y)')

        obj_faircheck = runChecker()
        obj_faircheck.runPropCheck()

    dfCexSet = local_load('CexSet')
    cex_count = cex_count + round(dfCexSet.shape[0] / 2)
    cex_count_list.append(round(dfCexSet.shape[0] / 2))
    mean_cex_count = cex_count / iteration_no
    cex_count_sem = func_calculate_sem(cex_count_list)

    f.write('Mean value is: ' + str(mean_cex_count) + '\n')
    f.write('Standard Error of the Mean is: +- ' + str(cex_count_sem) + '\n \n ')
