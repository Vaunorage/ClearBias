import json
import pickle
import re
import numpy as np
import pandas as pd
import os

from paths import HERE

files_folder = HERE.joinpath(f"methods/ml_check/files")

from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(outcome_column):
    df = local_load('OracleData')
    X = df.drop(columns=outcome_column).values
    y = df[outcome_column].values

    model = DecisionTreeClassifier(
        criterion="entropy",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None
    )

    model.fit(X, y)
    local_save(model, 'decTreeApprox', force_rewrite=True)

    return model


def get_file_path(var_name, return_hypothetical=False, default_ext=None):
    for ext in ['.csv', '.json', '.pkl', '.txt']:
        var_path = files_folder.joinpath(f"{var_name}{ext}")
        if var_path.exists():
            return var_path

    # Return the hypothetical path if no file exists and return_hypothetical is True
    if return_hypothetical:
        return files_folder.joinpath(f"{var_name}{default_ext}")  # Placeholder for extension
    return None


def file_exists(var_name):
    for ext in ['.csv', '.json', '.pkl', '.txt']:
        var_path = files_folder.joinpath(f"{var_name}{ext}")
        if var_path.exists():
            return True
    return False


def local_delete(var_name):
    deleted_files = []
    for ext in ['.csv', '.json', '.pkl', '.txt']:
        var_path = files_folder.joinpath(f"{var_name}{ext}")
        if var_path.exists():
            var_path.unlink()  # Delete the file
            deleted_files.append(var_path)

    # if not deleted_files:
    #     print(f"No files found to delete with base name '{var_name}'")

    return deleted_files


def local_save(var, var_name, force_rewrite=False):
    var_path = files_folder.joinpath(var_name)
    var_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    if isinstance(var, pd.DataFrame):
        var_path = var_path.with_suffix('.csv')
        if var_path.exists() and not force_rewrite:
            var.to_csv(var_path, mode='a', header=False, index=False)
        else:
            var.to_csv(var_path, index=False)
    elif isinstance(var, dict):
        var_path = var_path.with_suffix('.json')
        with open(var_path, 'w' if force_rewrite else 'a') as file:
            json.dump(var, file)
    elif isinstance(var, str):
        var_path = var_path.with_suffix('.txt')
        with open(var_path, 'w' if force_rewrite else 'a') as file:
            file.write(var + '\n')
    else:
        var_path = var_path.with_suffix('.pkl')
        if var_path.exists() and not force_rewrite:
            with open(var_path, 'rb') as file:
                existing_data = pickle.load(file)
            combined_data = existing_data + var if isinstance(existing_data, list) else [existing_data, var]
            with open(var_path, 'wb') as file:
                pickle.dump(combined_data, file)
        else:
            with open(var_path, 'wb') as file:
                pickle.dump(var, file)


def local_load(var_name):
    var_path_csv = files_folder.joinpath(f"{var_name}.csv")
    var_path_json = files_folder.joinpath(f"{var_name}.json")
    var_path_pkl = files_folder.joinpath(f"{var_name}.pkl")
    var_path_txt = files_folder.joinpath(f"{var_name}.txt")

    if var_path_json.exists():
        with open(var_path_json, 'r') as file:
            data = json.load(file)
    elif var_path_csv.exists():
        data = pd.read_csv(var_path_csv)
    elif var_path_pkl.exists():
        with open(var_path_pkl, 'rb') as file:
            data = pickle.load(file)
    elif var_path_txt.exists():
        with open(var_path_txt, 'r') as file:
            data = file.read()
    else:
        raise FileNotFoundError(f"No saved data found with base name '{var_name}'")

    return data


def run_z3_solver(input_file, output_file):
    z3_path = HERE.joinpath('methods/ml_check/z3_windows/bin/z3.exe')
    os.system(
        f"{z3_path} {files_folder.joinpath(input_file).as_posix()}.txt > {files_folder.joinpath(output_file).as_posix()}.txt")


def update_dataframe_types(df: pd.DataFrame, categorical_cols=None) -> pd.DataFrame:
    def infer_and_convert(series, col_name):
        if categorical_cols and col_name in categorical_cols:
            return series.astype('category')
        try:
            return pd.to_numeric(series, errors='coerce')
        except ValueError:
            pass
        try:
            return pd.to_datetime(series, errors='coerce')
        except ValueError:
            pass
        if series.dropna().map(lambda x: isinstance(x, bool)).all():
            return series.astype(bool)
        return series.astype(str)

    return df.apply(lambda series: infer_and_convert(series, series.name))


def convert_z3_output_to_df(file_content, df, paramDict):
    # Regex pattern to match the define-fun declarations
    pattern = r'\(define-fun\s+(\w+?)(\d)\s+\(\)\s+(\w+)\s*\n\s*(\d+)\)'

    # Find all matches
    matches = re.findall(pattern, file_content)

    # Initialize an empty DataFrame to store results
    no_of_params = int(paramDict['no_of_params'])
    dfAgain = pd.DataFrame(np.zeros((no_of_params, df.shape[1])), columns=df.columns.values)

    # Process each match
    for feature_name, param_no, data_type, value in matches:

        param_no = int(param_no)
        value = int(value)  # assuming the value is always an integer here

        # Assign the value to the correct cell in the DataFrame
        if feature_name in df.columns:
            dfAgain.loc[param_no, feature_name] = value

    return dfAgain


def conv_z3_out_to_data(df):
    paramDict = local_load('param_dict')
    file_content = local_load('z3_raw_output')

    if ('unknown' in file_content[0]):
        raise Exception('Encoding problem')
    if ('model is not available' in file_content[1]):
        return False

    # Create DataFrame from the data dictionary
    dfAgain = convert_z3_output_to_df(file_content, df, paramDict)

    local_save(dfAgain, 'formatted_z3_output', force_rewrite=True)
    return True
