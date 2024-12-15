import time
from typing import TypedDict, Union, Tuple, Dict, Any, List
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from data_generator.main import DiscriminationData
from path import HERE
from methods.ml_check.ml_check import Assume, Assert, RunChecker, PropertyChecker
from methods.ml_check.util import local_delete, local_load, update_dataframe_types, file_exists

import statistics as st
import math
import warnings

warnings.filterwarnings("ignore")


class MLCheckResultRow(TypedDict, total=False):
    z3_pred: Any
    ind_key: str
    couple_key: str
    outcome: Any
    diff_outcome: Any


MLCheckResultDF = DataFrame


def delete_all() -> None:
    files: List[str] = ['assumeStmnt', 'assertStmnt', 'Cand-Set', 'CexSet', 'CandidateSet',
                        'CandidateSetInst', 'CandidateSetBranch', 'formatted_z3_output',
                        'TestDataSMTMain', 'DecSmt', 'ToggleBranchSmt',
                        'ToggleFeatureSmt', 'TreeOutput', 'SampleFile', 'z3_raw_output',
                        'ConditionFile', 'MUT', 'DNNSmt', 'TestData', 'TestDataSet', 'CandTestDataSet']
    for ff in files:
        local_delete(ff)


def func_calculate_sem(samples: List[float]) -> float:
    standard_dev = st.pstdev(samples)
    return standard_dev / math.sqrt(len(samples))


def run_mlcheck(generated_data: DiscriminationData,
                iteration_no: int = 1,
                max_samples: int = 6,
                mul_cex: bool = True,
                train_data_available: bool = True,
                train_ratio: int = 30,
                no_of_train: int = 1000) -> Tuple[MLCheckResultDF, dict]:
    delete_all()
    start_time = time.time()

    # Save training dataframe to CSV
    train_data_loc: str = HERE.joinpath('methods/ml_check/files/input_data.csv').as_posix()
    generated_data.training_dataframe.to_csv(train_data_loc, header=True, index=False)

    # Initialize data
    input_data: DataFrame = generated_data.training_dataframe.copy()
    categorical_columns: List[str] = generated_data.categorical_columns
    protected_attributes: List[str] = generated_data.protected_attributes
    outcome_column: str = generated_data.outcome_column
    no_of_class_in_outcome: int = generated_data.dataframe[outcome_column].nunique()

    # Separate numerical and categorical columns
    numerical_columns: List[str] = [col for col in input_data.columns if
                                    col not in categorical_columns + [outcome_column]]

    # Preprocessing
    if categorical_columns:
        input_data[categorical_columns] = input_data[categorical_columns].fillna('missing').astype(str)
        label_encoders: Dict[str, LabelEncoder] = {col: LabelEncoder() for col in categorical_columns}
        input_data[categorical_columns] = input_data[categorical_columns].apply(
            lambda col: label_encoders[col.name].fit_transform(col))

    if numerical_columns:
        input_data[numerical_columns] = input_data[numerical_columns].apply(pd.to_numeric, errors='coerce')
        input_data[numerical_columns] = SimpleImputer(strategy='median').fit_transform(input_data[numerical_columns])
        input_data[numerical_columns] = StandardScaler().fit_transform(input_data[numerical_columns])

    # Update data types and rename outcome column
    input_data = update_dataframe_types(input_data, categorical_cols=categorical_columns)
    input_data.rename(columns={outcome_column: generated_data.outcome_column}, inplace=True)

    # Prepare and train model
    X: DataFrame = input_data.drop(generated_data.outcome_column, axis=1)
    y: pd.Series = input_data[generated_data.outcome_column]
    model: RandomForestClassifier = RandomForestClassifier()
    model.fit(X, y)

    # Run MLCheck iterations
    for _ in range(iteration_no):
        PropertyChecker(
            output_class_name=generated_data.outcome_column,
            all_attributes=list(generated_data.attributes),
            categorical_columns=categorical_columns,
            no_of_class=no_of_class_in_outcome,
            model=model,
            train_data_loc=train_data_loc,
            white_box_model=['Decision tree'],
            no_of_params=2,
            max_samples=max_samples,
            mul_cex=mul_cex,
            train_data_available=train_data_available,
            train_ratio=train_ratio,
            no_of_train=no_of_train
        )

        for i, col in enumerate(X.columns.tolist()):
            if col in protected_attributes:
                Assume('x[i] != y[i]', col)
            else:
                Assume('x[i] = y[i]', col)
        Assert('model.predict(x) == model.predict(y)')

        obj_faircheck: RunChecker = RunChecker()
        obj_faircheck.run_prop_check()

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate metrics
    metrics = {}

    if file_exists('DiscriminatoryCases'):
        discrimination_cases: MLCheckResultDF = local_load('DiscriminatoryCases')

        # Count total test cases (Z3 predictions)
        total_samples = discrimination_cases['couple_key'].nunique()

        # Count cases where whitebox prediction matches Z3 prediction
        discrimination_cases['pred_match'] = discrimination_cases['whitebox_pred'] == discrimination_cases['z3_pred']
        matching_predictions = (discrimination_cases
                                .groupby('couple_key')['pred_match']
                                .all()  # Both predictions in the pair must match
                                .sum())

        # Calculate metrics
        metrics['TSN'] = total_samples
        metrics['DSN'] = matching_predictions
        metrics['DSS'] = round(total_time / matching_predictions, 3) if matching_predictions > 0 else 0
        metrics['SUR'] = round((matching_predictions / total_samples) * 100, 2) if total_samples > 0 else 0

        # Add columns for analysis
        discrimination_cases['TSN'] = metrics['TSN']
        discrimination_cases['DSN'] = metrics['DSN']
        discrimination_cases['DSS'] = metrics['DSS']
        discrimination_cases['SUR'] = metrics['SUR']

        print("\nMetrics Summary:")
        print(f"Total Sample Number (TSN): {metrics['TSN']} test pairs")
        print(f"Discriminatory Sample Number (DSN): {metrics['DSN']} matching prediction pairs")
        print(f"Discriminatory Sample Search Time (DSS): {metrics['DSS']} seconds")
        print(f"Success Rate (SUR): {metrics['SUR']}% prediction match rate")
    else:
        discrimination_cases = pd.DataFrame()
        metrics.update({
            'TSN': 0,
            'DSN': 0,
            'DSS': 0,
            'SUR': 0
        })
        print("No discrimination cases found.")

    return discrimination_cases, metrics
