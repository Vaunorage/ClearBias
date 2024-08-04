import math

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class DatasetConfig:
    """
    Base configuration class for datasets.
    """

    def __init__(self, params, sensitive_param, feature_name, class_name, categorical_features, sens_name):
        self.params = params
        self.sensitive_param = sensitive_param  # List of strings
        self.feature_name = feature_name
        self.class_name = class_name
        self.categorical_features = categorical_features
        self.sens_name = sens_name

    def get_dataframe(self, preprocess=False):
        raise NotImplementedError("This method should be implemented by subclasses")

    def preprocess_data(self, df):
        # Separate features and target
        X = df[self.feature_name]
        y = df[self.class_name]

        # Handle missing values
        imputer = SimpleImputer(strategy='most_frequent')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Encode categorical variables
        le = LabelEncoder()
        for col in self.categorical_features:
            X[col] = le.fit_transform(X[col].astype(str))  # Use column names directly

        # Encode target variable
        y = le.fit_transform(y)

        # Scale numerical features
        numerical_features = [col for col in X.columns if col not in self.categorical_features]
        scaler = StandardScaler()
        X[numerical_features] = scaler.fit_transform(X[numerical_features])

        self.input_bounds = []
        for col in self.feature_name:
            min_val = math.floor(X[col].min())
            max_val = math.ceil(X[col].max())
            self.input_bounds.append([min_val, max_val])

        return pd.concat([X, pd.Series(y, name=self.class_name[0])], axis=1)

    @property
    def sensitive_indices(self):
        return {col: self.feature_name.index(col) for col in self.sensitive_param}


class census(DatasetConfig):
    def __init__(self):
        super().__init__(
            params=12,
            sensitive_param=["sex", "age", "capital-gain"],
            feature_name=["age", "workclass", "education", "marital-status", "occupation", "relationship",
                          "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"],
            class_name=["income"],
            categorical_features=["workclass", "education", "marital-status", "occupation", "relationship",
                                  "race", "sex", "native-country"],
            sens_name={"sex": 'sex', "age": "age"}
        )

    def get_dataframe(self, preprocess=False):
        adult = fetch_ucirepo(id=2)
        X = adult.data.features
        y = adult.data.targets
        df = pd.concat([X, y], axis=1)
        if preprocess:
            df = self.preprocess_data(df)
        X = df[self.feature_name]
        y = df[self.class_name]
        return X, y


class credit(DatasetConfig):
    """
    Configuration of dataset German Credit.
    """

    def __init__(self):
        super().__init__(
            params=20,
            sensitive_param=["sex", "age"],
            feature_name=["checking_status", "duration", "credit_history", "purpose", "credit_amount", "savings_status",
                          "employment", "installment_commitment", "sex", "other_parties", "residence",
                          "property_magnitude", "age", "other_payment_plans", "housing", "existing_credits", "job",
                          "num_dependents", "own_telephone", "foreign_worker"],
            class_name=["class"],
            categorical_features=["checking_status", "credit_history", "purpose", "savings_status", "employment",
                                  "sex", "other_parties", "property_magnitude", "other_payment_plans", "housing",
                                  "job", "own_telephone", "foreign_worker"],
            sens_name={"sex": 'sex', "age": "age"}
        )

    def get_dataframe(self, preprocess=False):
        german_credit = fetch_ucirepo(id=144)
        X = german_credit.data.features
        y = german_credit.data.targets
        df = pd.concat([X, y], axis=1)
        if preprocess:
            df = self.preprocess_data(df)
        X = df[self.feature_name]
        y = df[self.class_name]
        return X, y


class bank(DatasetConfig):
    """
    Configuration of dataset Bank Marketing.
    """

    def __init__(self):
        super().__init__(
            params=16,
            sensitive_param=["age"],
            feature_name=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact",
                          "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"],
            class_name=["y"],
            categorical_features=["job", "marital", "education", "default", "housing", "loan", "contact", "month",
                                  "poutcome"],
            sens_name={"age": 'age'}
        )

    def get_dataframe(self, preprocess=False):
        bank_marketing = fetch_ucirepo(id=222)
        X = bank_marketing.data.features
        y = bank_marketing.data.targets
        df = pd.concat([X, y], axis=1)
        if preprocess:
            df = self.preprocess_data(df)
        X = df[self.feature_name]
        y = df[self.class_name]
        return X, y

# Example usage:
# census_dataset = census()
# df_census, _ = census_dataset.get_dataframe()
# census_bounds = census_dataset.calculate_input_bounds(df_census)
# print("Census Input Bounds:", census_bounds)
# print("Census Sensitive Indices:", census_dataset.sensitive_indices)
#
# credit_dataset = credit()
# df_credit, _ = credit_dataset.get_dataframe()
# credit_bounds = credit_dataset.calculate_input_bounds(df_credit)
# print("Credit Input Bounds:", credit_bounds)
# print("Credit Sensitive Indices:", credit_dataset.sensitive_indices)
#
# bank_dataset = bank()
# df_bank, _ = bank_dataset.get_dataframe()
# bank_bounds = bank_dataset.calculate_input_bounds(df_bank)
# print("Bank Input Bounds:", bank_bounds)
# print("Bank Sensitive Indices:", bank_dataset.sensitive_indices)
