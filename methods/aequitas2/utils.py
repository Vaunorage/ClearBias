from random import seed, shuffle
from scipy.optimize import minimize  # for loss func minimization
from multiprocessing import Pool, Process, Queue
from collections import defaultdict
from copy import deepcopy
from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Optional, Dict

SEED = 1122334455
seed(SEED)  # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)


def get_adult_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Fetch and preprocess the UCI Adult dataset.

    Returns:
        Tuple containing:
        - X: Feature matrix
        - Y: Target labels (-1, 1)
        - sensitive: Dictionary with sensitive attributes
    """
    # Fetch Adult dataset
    adult = fetch_ucirepo(id=2)
    df = adult.data.features
    target = adult.data.targets

    # Define columns to use - expanded to match original 13 features
    columns_to_keep = [
        'age', 'workclass', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
    ]

    # Select relevant columns
    df = df[columns_to_keep]

    # Handle missing values
    df = df.replace('?', np.nan)
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical variables
    le_dict = {}
    for col in df.select_dtypes(include=['object']):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Convert to numpy arrays
    X = df.values
    Y = np.where(target.values.ravel() <= '<=50K', -1, 1)

    # Create sensitive attributes dictionary
    sensitive = {
        'sex': df['sex'].values,  # Already encoded
        'race': df['race'].values  # Already encoded
    }

    return X, Y, sensitive


def generate_initial_input(X: np.ndarray, sensitive_param_idx: int, random_seed: int = 42) -> np.ndarray:
    np.random.seed(random_seed)

    # Generate initial input based on median values for continuous features
    # and mode for categorical features
    initial_input = []

    for i in range(X.shape[1]):
        # Get unique values and their counts
        unique_vals, counts = np.unique(X[:, i], return_counts=True)

        if len(unique_vals) > 10:  # Assume continuous feature
            # Use a value near the median with some random variation
            median_val = np.median(X[:, i])
            std_val = np.std(X[:, i])
            initial_val = int(np.clip(
                median_val + np.random.normal(0, std_val * 0.1),
                np.min(X[:, i]),
                np.max(X[:, i])
            ))
        else:  # Categorical feature
            # Use the mode (most common value)
            initial_val = int(unique_vals[np.argmax(counts)])

        initial_input.append(initial_val)

    # Set sensitive parameter to 0 as per original code
    initial_input[sensitive_param_idx - 1] = 0  # -1 for 0-based indexing

    return initial_input


def prepare_data_for_fairness_testing(
        X: np.ndarray,
        Y: np.ndarray,
        sensitive: Dict[str, np.ndarray],
        sensitive_feature_name: str = 'sex'
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], dict]:

    # Ensure all features are float type
    X = X.astype(float)
    Y = Y.astype(float)

    # Ensure sensitive attributes are float type
    for key in sensitive:
        sensitive[key] = sensitive[key].astype(float)

    # Verify the sensitive feature exists
    if sensitive_feature_name not in sensitive:
        raise ValueError(f"Sensitive feature {sensitive_feature_name} not found in sensitive attributes")

    # Generate configurations
    configs = {}

    # Number of features
    configs['params'] = X.shape[1]

    # Find feature index for sensitive parameter (1-based indexing)
    feature_names = [
        'age', 'workclass', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain',
        'capital-loss', 'hours-per-week', 'native-country'
    ]
    sensitive_idx = feature_names.index(sensitive_feature_name) + 1  # Add 1 for 1-based indexing
    configs['sensitive_param'] = sensitive_idx

    # Generate input bounds
    input_bounds = []
    for col in range(X.shape[1]):
        min_val = int(np.floor(np.min(X[:, col])))
        max_val = int(np.ceil(np.max(X[:, col])))
        input_bounds.append((min_val, max_val))
    configs['input_bounds'] = input_bounds

    # Set fairness-related configurations
    configs['name'] = sensitive_feature_name

    # Calculate baseline covariance between sensitive attribute and outcomes
    sensitive_values = sensitive[sensitive_feature_name]
    baseline_cov = np.cov(sensitive_values, Y)[0, 1]

    # Set target covariance threshold as a fraction of baseline
    # Using 0 for strict fairness, but could be adjusted based on requirements
    configs['cov'] = 0

    # Set sensitive attributes list and threshold dictionary
    configs['sensitive_attrs'] = [sensitive_feature_name]
    configs['sensitive_attrs_to_cov_thresh'] = {sensitive_feature_name: configs['cov']}

    # Add additional fairness metrics for reference
    configs['baseline_metrics'] = {
        'baseline_covariance': baseline_cov,
        'sensitive_attr_mean': np.mean(sensitive_values),
        'sensitive_attr_std': np.std(sensitive_values),
        'positive_label_ratio': np.mean(Y == 1),
        'negative_label_ratio': np.mean(Y == -1)
    }

    # Generate initial input
    configs['initial_input'] = generate_initial_input(X, configs['sensitive_param'])

    # Set optimizer configurations
    configs['minimizer'] = {"method": "L-BFGS-B"}
    configs['stepsize'] = 1.0
    configs['global_iteration_limit'] = 6000
    configs['local_iteration_limit'] = 1000

    return X, Y, sensitive, configs


def check_accuracy(model, x_train, y_train, x_test, y_test, y_train_predicted, y_test_predicted):
    """
    returns the train/test accuracy of the model
    we either pass the model (w)
    else we pass y_predicted
    """
    if model is not None and y_test_predicted is not None:
        print("Either the model (w) or the predicted labels should be None")
        raise Exception("Either the model (w) or the predicted labels should be None")

    if model is not None:
        y_test_predicted = np.sign(np.dot(x_test, model))
        y_train_predicted = np.sign(np.dot(x_train, model))

    def get_accuracy(y, Y_predicted):
        correct_answers = (Y_predicted == y).astype(int)  # will have 1 when the prediction and the actual label match
        accuracy = float(sum(correct_answers)) / float(len(correct_answers))
        return accuracy, sum(correct_answers)

    train_score, correct_answers_train = get_accuracy(y_train, y_train_predicted)
    test_score, correct_answers_test = get_accuracy(y_test, y_test_predicted)

    return train_score, test_score, correct_answers_train, correct_answers_test


def test_sensitive_attr_constraint_cov(model, x_arr, y_arr_dist_boundary, x_control, thresh, verbose):
    """
    The covariance is computed b/w the sensitive attr val and the distance from the boundary
    If the model is None, we assume that the y_arr_dist_boundary contains the distace from the decision boundary
    If the model is not None, we just compute a dot product or model and x_arr
    for the case of SVM, we pass the distace from bounday becase the intercept in internalized for the class
    and we have compute the distance using the project function

    this function will return -1 if the constraint specified by thresh parameter is not satifsified
    otherwise it will reutrn +1
    if the return value is >=0, then the constraint is satisfied
    """

    assert (x_arr.shape[0] == x_control.shape[0])
    if len(x_control.shape) > 1:  # make sure we just have one column in the array
        assert (x_control.shape[1] == 1)

    arr = []
    if model is None:
        arr = y_arr_dist_boundary  # simply the output labels
    else:
        arr = np.dot(model, x_arr.T)  # the product with the weight vector -- the sign of this is the output label

    arr = np.array(arr, dtype=np.float64)

    cov = np.dot(x_control - np.mean(x_control), arr) / float(len(x_control))

    ans = thresh - abs(
        cov)  # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied
    # ans = thresh - cov # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied
    if verbose is True:
        print
        "Covariance is", cov
        print
        "Diff is:", ans
        print
    return ans


def print_covariance_sensitive_attrs(model, x_arr, y_arr_dist_boundary, x_control, sensitive_attrs):
    """
    reutrns the covariance between sensitive features and distance from decision boundary
    """

    arr = []
    if model is None:
        arr = y_arr_dist_boundary  # simplt the output labels
    else:
        arr = np.dot(model, x_arr.T)  # the product with the weight vector -- the sign of this is the output label

    sensitive_attrs_to_cov_original = {}
    for attr in sensitive_attrs:

        attr_arr = x_control[attr]

        bin_attr = check_binary(attr_arr)  # check if the attribute is binary (0/1), or has more than 2 vals
        if bin_attr == False:  # if its a non-binary sensitive feature, then perform one-hot-encoding
            attr_arr_transformed, index_dict = get_one_hot_encoding(attr_arr)

        thresh = 0

        if bin_attr:
            cov = thresh - test_sensitive_attr_constraint_cov(None, x_arr, arr, np.array(attr_arr), thresh, False)
            sensitive_attrs_to_cov_original[attr] = cov
        else:  # sensitive feature has more than 2 categorical values

            cov_arr = []
            sensitive_attrs_to_cov_original[attr] = {}
            for attr_val, ind in index_dict.items():
                t = attr_arr_transformed[:, ind]
                cov = thresh - test_sensitive_attr_constraint_cov(None, x_arr, arr, t, thresh, False)
                sensitive_attrs_to_cov_original[attr][attr_val] = cov
                cov_arr.append(abs(cov))

            cov = max(cov_arr)

    return sensitive_attrs_to_cov_original


def get_correlations(model, x_test, y_predicted, x_control_test, sensitive_attrs):
    """
    returns the fraction in positive class for sensitive feature values
    """

    if model is not None:
        y_predicted = np.sign(np.dot(x_test, model))

    y_predicted = np.array(y_predicted)

    out_dict = {}
    for attr in sensitive_attrs:

        attr_val = []
        for v in x_control_test[attr]: attr_val.append(v)
        assert (len(attr_val) == len(y_predicted))

        total_per_val = defaultdict(int)
        attr_to_class_labels_dict = defaultdict(lambda: defaultdict(int))

        for i in range(0, len(y_predicted)):
            val = attr_val[i]
            label = y_predicted[i]

            # val = attr_val_int_mapping_dict_reversed[val] # change values from intgers to actual names
            total_per_val[val] += 1
            attr_to_class_labels_dict[val][label] += 1

        class_labels = set(y_predicted.tolist())

        local_dict_1 = {}
        for k1, v1 in attr_to_class_labels_dict.items():
            total_this_val = total_per_val[k1]

            local_dict_2 = {}
            for k2 in class_labels:  # the order should be the same for printing
                v2 = v1[k2]

                f = float(v2) * 100.0 / float(total_this_val)

                local_dict_2[k2] = f
            local_dict_1[k1] = local_dict_2
        out_dict[attr] = local_dict_1

    return out_dict


def get_constraint_list_cov(x_train, y_train, x_control_train, sensitive_attrs, sensitive_attrs_to_cov_thresh):
    """
    get the list of constraints to be fed to the minimizer
    """

    constraints = []

    for attr in sensitive_attrs:

        attr_arr = x_control_train[attr]
        attr_arr_transformed, index_dict = get_one_hot_encoding(attr_arr)

        if index_dict is None:  # binary attribute
            thresh = sensitive_attrs_to_cov_thresh[attr]
            c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov,
                  'args': (x_train, y_train, attr_arr_transformed, thresh, False)})
            constraints.append(c)
        else:  # otherwise, its a categorical attribute, so we need to set the cov thresh for each value separately

            for attr_val, ind in index_dict.items():
                attr_name = attr_val
                print
                attr, attr_name, sensitive_attrs_to_cov_thresh[attr]
                thresh = sensitive_attrs_to_cov_thresh[attr][attr_name]

                t = attr_arr_transformed[:, ind]
                c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov,
                      'args': (x_train, y_train, t, thresh, False)})
                constraints.append(c)

    return constraints


def train_model(x, y, x_control, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint,
                sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma=None):
    """
    Function that trains the model subject to various fairness constraints.
    If no constraints are given, then simply trains an unaltered classifier.
    """

    print(x[0], y[0], x_control, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint,
          sensitive_attrs, sensitive_attrs_to_cov_thresh)

    assert ((
                    apply_accuracy_constraint == 1 and apply_fairness_constraints == 1) == False)  # both constraints cannot be applied at the same time

    max_iter = 100000  # maximum number of iterations for the minimization algorithm

    if apply_fairness_constraints == 0:
        constraints = []
    else:
        constraints = get_constraint_list_cov(x, y, x_control, sensitive_attrs, sensitive_attrs_to_cov_thresh)

    if apply_accuracy_constraint == 0:  # its not the reverse problem, just train w with cross cov constraints
        f_args = (x, y)
        w = minimize(fun=loss_function,
                     x0=np.random.rand(x.shape[1], ),
                     args=f_args,
                     method='SLSQP',
                     options={"maxiter": max_iter},
                     constraints=constraints
                     )
    else:
        # train on just the loss function
        w = minimize(fun=loss_function,
                     x0=np.random.rand(x.shape[1], ),
                     args=(x, y),
                     method='SLSQP',
                     options={"maxiter": max_iter},
                     constraints=[]
                     )

        old_w = deepcopy(w.x)

        def constraint_gamma_all(w, x, y, initial_loss_arr):
            gamma_arr = np.ones_like(y) * gamma  # set gamma for everyone
            new_loss = loss_function(w, x, y)
            old_loss = sum(initial_loss_arr)
            return ((1.0 + gamma) * old_loss) - new_loss

        def constraint_protected_people(w, x,
                                        y):  # dont confuse the protected here with the sensitive feature protected/non-protected values
            return np.dot(w, x.T)  # if this is positive, the constraint is satisfied

        def constraint_unprotected_people(w, ind, old_loss, x, y):
            new_loss = loss_function(w, np.array([x]), np.array(y))
            return ((1.0 + gamma) * old_loss) - new_loss

        constraints = []
        predicted_labels = np.sign(np.dot(w.x, x.T))
        unconstrained_loss_arr = loss_function(w.x, x, y, return_arr=True)

        if sep_constraint == True:  # separate gemma for different people
            for i in range(0, len(predicted_labels)):
                if predicted_labels[i] == 1.0 and x_control[sensitive_attrs[0]][i] == 1.0:
                    c = ({'type': 'ineq', 'fun': constraint_protected_people, 'args': (x[i], y[i])})
                    constraints.append(c)
                else:
                    c = ({'type': 'ineq', 'fun': constraint_unprotected_people,
                          'args': (i, unconstrained_loss_arr[i], x[i], y[i])})
                    constraints.append(c)
        else:  # same gamma for everyone
            c = ({'type': 'ineq', 'fun': constraint_gamma_all, 'args': (x, y, unconstrained_loss_arr)})
            constraints.append(c)

        def cross_cov_abs_optm_func(weight_vec, x_in, x_control_in_arr):
            cross_cov = (x_control_in_arr - np.mean(x_control_in_arr)) * np.dot(weight_vec, x_in.T)
            return float(abs(sum(cross_cov))) / float(x_in.shape[0])

        w = minimize(fun=cross_cov_abs_optm_func,
                     x0=old_w,
                     args=(x, x_control[sensitive_attrs[0]]),
                     method='SLSQP',
                     options={"maxiter": 100000},
                     constraints=constraints
                     )

    try:
        assert (w.success == True)
    except:
        print("Optimization problem did not converge.. Check the solution returned by the optimizer.")
        print("Returned solution is:")
        print(w)

    return w.x


def split_into_train_test(x_all, y_all, x_control_all, train_fold_size):
    split_point = int(round(float(x_all.shape[0]) * train_fold_size))
    x_all_train = x_all[:split_point]
    x_all_test = x_all[split_point:]
    y_all_train = y_all[:split_point]
    y_all_test = y_all[split_point:]
    x_control_all_train = {}
    x_control_all_test = {}
    for k in x_control_all.keys():
        x_control_all_train[k] = x_control_all[k][:split_point]
        x_control_all_test[k] = x_control_all[k][split_point:]

    return x_all_train, y_all_train, x_control_all_train, x_all_test, y_all_test, x_control_all_test


def get_avg_correlation_dict(correlation_dict_arr):
    # make the structure for the correlation dict
    correlation_dict_avg = {}
    # print correlation_dict_arr
    for k, v in correlation_dict_arr[0].items():
        correlation_dict_avg[k] = {}
        for feature_val, feature_dict in v.items():
            correlation_dict_avg[k][feature_val] = {}
            for class_label, frac_class in feature_dict.items():
                correlation_dict_avg[k][feature_val][class_label] = []

    # populate the correlation dict
    for correlation_dict in correlation_dict_arr:
        for k, v in correlation_dict.items():
            for feature_val, feature_dict in v.items():
                for class_label, frac_class in feature_dict.items():
                    correlation_dict_avg[k][feature_val][class_label].append(frac_class)

    # now take the averages
    for k, v in correlation_dict_avg.items():
        for feature_val, feature_dict in v.items():
            for class_label, frac_class_arr in feature_dict.items():
                correlation_dict_avg[k][feature_val][class_label] = np.mean(frac_class_arr)

    return correlation_dict_avg


def compute_cross_validation_error(x_all, y_all, x_control_all, num_folds, loss_function, apply_fairness_constraints,
                                   apply_accuracy_constraint, sep_constraint, sensitive_attrs,
                                   sensitive_attrs_to_cov_thresh_arr, gamma=None):
    """
    Computes the cross validation error for the classifier subject to various fairness constraints
    """

    train_folds = []
    test_folds = []
    n_samples = len(y_all)
    train_fold_size = 0.7  # the rest of 0.3 is for testing

    # split the data into folds for cross-validation
    for i in range(0, num_folds):
        perm = list(range(0, n_samples))  # shuffle the data before creating each fold
        shuffle(perm)
        x_all_perm = x_all[perm]
        y_all_perm = y_all[perm]
        x_control_all_perm = {}
        for k in x_control_all.keys():
            x_control_all_perm[k] = np.array(x_control_all[k])[perm]

        x_all_train, y_all_train, x_control_all_train, x_all_test, y_all_test, x_control_all_test = split_into_train_test(
            x_all_perm, y_all_perm, x_control_all_perm, train_fold_size)

        train_folds.append([x_all_train, y_all_train, x_control_all_train])
        test_folds.append([x_all_test, y_all_test, x_control_all_test])

    def train_test_single_fold(train_data, test_data, fold_num, output_folds, sensitive_attrs_to_cov_thresh):
        x_train, y_train, x_control_train = train_data
        x_test, y_test, x_control_test = test_data

        w = train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints,
                        apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh,
                        gamma)
        train_score, test_score, correct_answers_train, correct_answers_test = check_accuracy(w, x_train, y_train,
                                                                                              x_test, y_test, None,
                                                                                              None)

        distances_boundary_test = (np.dot(x_test, w)).tolist()
        all_class_labels_assigned_test = np.sign(distances_boundary_test)
        correlation_dict_test = get_correlations(None, None, all_class_labels_assigned_test, x_control_test,
                                                 sensitive_attrs)
        cov_dict_test = print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test,
                                                         sensitive_attrs)

        distances_boundary_train = (np.dot(x_train, w)).tolist()
        all_class_labels_assigned_train = np.sign(distances_boundary_train)
        correlation_dict_train = get_correlations(None, None, all_class_labels_assigned_train, x_control_train,
                                                  sensitive_attrs)
        cov_dict_train = print_covariance_sensitive_attrs(None, x_train, distances_boundary_train, x_control_train,
                                                          sensitive_attrs)

        output_folds.put(
            [fold_num, test_score, train_score, correlation_dict_test, correlation_dict_train, cov_dict_test,
             cov_dict_train])
        return

    output_folds = Queue()
    processes = [Process(target=train_test_single_fold,
                         args=(train_folds[x], test_folds[x], x, output_folds, sensitive_attrs_to_cov_thresh_arr[x]))
                 for x in range(num_folds)]

    # Run processes
    for p in processes:
        p.start()

    # Get the results
    results = [output_folds.get() for p in processes]
    for p in processes:
        p.join()

    test_acc_arr = []
    train_acc_arr = []
    correlation_dict_test_arr = []
    correlation_dict_train_arr = []
    cov_dict_test_arr = []
    cov_dict_train_arr = []

    results = sorted(results, key=lambda x: x[0])  # sort w.r.t fold num
    for res in results:
        fold_num, test_score, train_score, correlation_dict_test, correlation_dict_train, cov_dict_test, cov_dict_train = res

        test_acc_arr.append(test_score)
        train_acc_arr.append(train_score)
        correlation_dict_test_arr.append(correlation_dict_test)
        correlation_dict_train_arr.append(correlation_dict_train)
        cov_dict_test_arr.append(cov_dict_test)
        cov_dict_train_arr.append(cov_dict_train)

    return test_acc_arr, train_acc_arr, correlation_dict_test_arr, correlation_dict_train_arr, cov_dict_test_arr, cov_dict_train_arr


def print_classifier_fairness_stats(acc_arr, correlation_dict_arr, cov_dict_arr, s_attr_name):
    correlation_dict = get_avg_correlation_dict(correlation_dict_arr)
    non_prot_pos = correlation_dict[s_attr_name][1][1]
    prot_pos = correlation_dict[s_attr_name][0][1]
    p_rule = (prot_pos / non_prot_pos) * 100.0

    print("Accuracy: %0.2f" % (np.mean(acc_arr)))
    print("Protected/non-protected in +ve class: %0.0f%% / %0.0f%%" % (prot_pos, non_prot_pos))
    print("P-rule achieved: %0.0f%%" % (p_rule))
    print("Covariance between sensitive feature and decision from distance boundary : %0.3f" % (
        np.mean([v[s_attr_name] for v in cov_dict_arr])))
    print()
    return p_rule


def compute_p_rule(x_control, class_labels):
    """ Compute the p-rule based on Doctrine of disparate impact """

    non_prot_all = sum(x_control == 1.0)  # non-protected group
    prot_all = sum(x_control == 0.0)  # protected group
    non_prot_pos = sum(class_labels[x_control == 1.0] == 1.0)  # non_protected in positive class
    prot_pos = sum(class_labels[x_control == 0.0] == 1.0)  # protected in positive class
    frac_non_prot_pos = float(non_prot_pos) / float(non_prot_all)
    frac_prot_pos = float(prot_pos) / float(prot_all)
    p_rule = (frac_prot_pos / frac_non_prot_pos) * 100.0

    print()
    print("Total data points: %d" % (len(x_control)))
    print("# non-protected examples: %d" % (non_prot_all))
    print("# protected examples: %d" % (prot_all))
    print("Non-protected in positive class: %d (%0.0f%%)" % (non_prot_pos, non_prot_pos * 100.0 / non_prot_all))
    print("Protected in positive class: %d (%0.0f%%)" % (prot_pos, prot_pos * 100.0 / prot_all))
    print("P-rule is: %0.0f%%" % (p_rule))
    return p_rule


def add_intercept(x):
    """ Add intercept to the data before linear classification """
    m, n = x.shape
    intercept = np.ones(m).reshape(m, 1)  # the constant b
    return np.concatenate((intercept, x), axis=1)


def check_binary(arr):
    "give an array of values, see if the values are only 0 and 1"
    s = sorted(set(arr))
    if s[0] == 0 and s[1] == 1:
        return True
    else:
        return False


def get_one_hot_encoding(in_arr):
    """
    input: 1-D arr with int vals -- if not int vals, will raise an error
    output: m (ndarray): one-hot encoded matrix
            d (dict): also returns a dictionary original_val -> column in encoded matrix
    """

    for k in in_arr:
        if not isinstance(k, (np.float64, int, np.int64)):
            print(str(type(k)))
            print("************* ERROR: Input arr does not have integer types")
            return None

    in_arr = np.array(in_arr, dtype=int)
    assert (len(in_arr.shape) == 1)  # no column, means it was a 1-D arr
    attr_vals_uniq_sorted = sorted(list(set(in_arr)))
    num_uniq_vals = len(attr_vals_uniq_sorted)
    if (num_uniq_vals == 2) and (attr_vals_uniq_sorted[0] == 0 and attr_vals_uniq_sorted[1] == 1):
        return in_arr, None

    index_dict = {}  # value to the column number
    for i in range(0, len(attr_vals_uniq_sorted)):
        val = attr_vals_uniq_sorted[i]
        index_dict[val] = i

    out_arr = []
    for i in range(0, len(in_arr)):
        tup = np.zeros(num_uniq_vals)
