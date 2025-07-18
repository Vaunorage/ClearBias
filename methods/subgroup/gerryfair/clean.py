import argparse
import numpy as np
import pandas as pd

def setup():
    parser = argparse.ArgumentParser(description='Fairness Data Cleaning')
    parser.add_argument('-n', '--name', type=str,
                        help='name of the to store the new datasets (Required)')
    parser.add_argument('-d', '--dataset', type=str,
                        help='name of the original dataset file (Required)')
    parser.add_argument('-a', '--attributes', type=str,
                        help='name of the file representing which attributes are protected (unprotected = 0, protected = 1, label = 2) (Required)')
    parser.add_argument('-c', '--centered', default=False, action='store_true', required=False,
                        help='Include this flag to determine whether data should be centered')
    args = parser.parse_args()
    return [args.name, args.dataset, args.attributes, args.centered]


'''
Clean a dataset, given the filename for the dataset and the filename for the attributes.

Dataset: Filename for dataset. The dataset should be formatted such that categorical variables use one-hot encoding
and the label should be 0/1

Attributes: Filename for the attributes of the dataset. The file should have each column name in a list, and under this
list should have 0 for an unprotected attribute, 1 for a protected attribute, and 2 for the attribute of the label.

'''
def clean_dataset(dataset, attributes, centered):
    df = pd.read_csv(dataset)
    sens_df = pd.read_csv(attributes)

    ## Get and remove label Y
    y_col = [str(c) for c in sens_df.columns if sens_df[c][0] == 2]
    print('label feature: {}'.format(y_col))
    if(len(y_col) > 1):
        raise ValueError('More than 1 label column used')
    if (len(y_col) < 1):
        raise ValueError('No label column used')

    y = df[y_col[0]]

    ## Do not use labels in rest of data
    X = df.loc[:, df.columns != y_col[0]]
    X = X.loc[:, X.columns != 'Unnamed: 0']
    ## Create X_prime, by getting protected attributes
    sens_cols = [str(c) for c in sens_df.columns if sens_df[c][0] == 1]
    print('sensitive features: {}'.format(sens_cols))
    sens_dict = {c: 1 if c in sens_cols else 0 for c in df.columns}
    X, sens_dict = one_hot_code(X, sens_dict)
    sens_names = [key for key in sens_dict.keys() if sens_dict[key] == 1]
    print('there are {} sensitive features including derivative features'.format(len(sens_names)))

    X_prime = X[sens_names]

    #X = X.reset_index(drop=True)
    #X_prime = X_prime.reset_index(drop=True)

    if(centered):
        X = center(X)
        X_prime = center(X_prime)

    return X, X_prime, y



def center(X):
    for col in X.columns:
        X.loc[:, col] = X.loc[:, col]-np.mean(X.loc[:, col])
    return X

def one_hot_code(df1, sens_dict):
    cols = df1.columns
    for c in cols:
        if isinstance(df1[c][0], str):
            column = df1[c]
            df1 = df1.drop(columns=c)
            unique_values = list(set(column))
            n = len(unique_values)
            if n > 2:
                for i in range(n):
                    col_name = '{}.{}'.format(c, i)
                    col_i = [1 if el == unique_values[i] else 0 for el in column]
                    df1[col_name] = col_i
                    sens_dict[col_name] = sens_dict[c]
                del sens_dict[c]
            else:
                col_name = c
                col = [1 if el == unique_values[0] else 0 for el in column]
                df1[col_name] = col
    return df1, sens_dict


# Helper for main method
'''
Given name of dataset, load in the three datasets associated from the clean.py file
'''
def get_data(dataset):
    X = pd.read_csv('dataset/' + dataset + '_features.csv')
    X_prime = pd.read_csv('dataset/' + dataset + '_protectedfeatures.csv')
    y = pd.read_csv('dataset/' + dataset + '_labels.csv', names=['index', 'label'])
    y = y['label']
    return X, X_prime, y





