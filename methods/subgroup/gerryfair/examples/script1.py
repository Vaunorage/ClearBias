from methods.subgroup.gerryfair.auditor import Auditor
from methods.subgroup.gerryfair.clean import clean_dataset
from methods.subgroup.gerryfair.model import Model
from path import HERE

dataset = HERE.joinpath("methods/subgroup/gerryfair/dataset/communities.csv")
attributes = HERE.joinpath("methods/subgroup/gerryfair/dataset/communities_protected.csv")
centered = True
X, X_prime, y = clean_dataset(dataset, attributes, centered)
C = 10
printflag = True
gamma = .01
fair_model = Model(C=C, printflag=printflag, gamma=gamma, fairness_def='FN')
max_iters = 50
fair_model.set_options(max_iters=max_iters)

# Train Set
train_size = 1000

X_train = X.iloc[:train_size]
X_prime_train = X_prime.iloc[:train_size]
y_train = y.iloc[:train_size]

# Test Set
X_test = X.iloc[train_size:].reset_index(drop=True)
X_prime_test = X_prime.iloc[train_size:].reset_index(drop=True)
y_test = y.iloc[train_size:].reset_index(drop=True)

#%% Train the model
[errors, fp_difference] = fair_model.train(X_train, X_prime_train, y_train)

#%% Generate predictions
predictions = fair_model.predict(X_train)

#%% Audit predictions
auditor = Auditor(X_prime_train, y_train, 'FN')
[group, fairness_violation] = auditor.audit(predictions)
print(fairness_violation)