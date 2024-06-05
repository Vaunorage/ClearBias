import pandas as pd
from aif360.detectors import bias_scan
from aif360.detectors.mdss.ScoringFunctions import Bernoulli

from data_generator.main import generate_data

df, protected_attr = generate_data(min_number_of_classes=2, max_number_of_classes=6, nb_attributes=6,
                                   prop_protected_attr=0.3, nb_groups=500, max_group_size=50, hiddenlayers_depth=3,
                                   min_similarity=0.0,
                                   max_similarity=1.0, min_alea_uncertainty=0.0, max_alea_uncertainty=1.0,
                                   min_epis_uncertainty=0.0, max_epis_uncertainty=1.0,
                                   min_magnitude=0.0, max_magnitude=1.0, min_frequency=0.0, max_frequency=1.0,
                                   categorical_outcome=True, nb_categories_outcome=4)


# Define the target variable
target = 'outcome'
features = ['feature1', 'feature2', 'feature3']

# Train a model on the training data
from sklearn.linear_model import LogisticRegression

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Define the observations (ground truth)
observations = pd.Series(y_test, index=y_test.index)

# Define the expectations (predictions)
expectations = pd.Series(y_pred, index=y_pred.index)

# Perform the bias scan
scoring = Bernoulli()
result = bias_scan(data, observations, expectations, scoring=scoring, mode='binary', num_iters=10, penalty=1e-17)

# Print the result
print(result)
