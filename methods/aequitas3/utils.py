import numpy as np
from random import seed, shuffle
import loss_funcs as lf  # our implementation of loss funcs
from scipy.optimize import minimize  # for loss func minimization
from multiprocessing import Pool, Process, Queue
from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt  # for plotting stuff
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib  # Updated joblib import

from path import HERE

SEED = 1122334455
seed(SEED)  # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)


def train_model(x, y, x_control, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint,
                sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma=None):
    print(x[0], y[0], x_control, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint,
          sensitive_attrs, sensitive_attrs_to_cov_thresh)

    """

    Function that trains the model subject to various fairness constraints.
    If no constraints are given, then simply trains an unaltered classifier.
    Example usage in: "synthetic_data_demo/decision_boundary_demo.py"

    ----

    Inputs:

    X: (n) x (d+1) numpy array -- n = number of examples, d = number of features, one feature is the intercept
    y: 1-d numpy array (n entries)
    x_control: dictionary of the type {"s": [...]}, key "s" is the sensitive feature name, and the value is a 1-d list with n elements holding the sensitive feature values
    loss_function: the loss function that we want to optimize -- for now we have implementation of logistic loss, but other functions like hinge loss can also be added
    apply_fairness_constraints: optimize accuracy subject to fairness constraint (0/1 values)
    apply_accuracy_constraint: optimize fairness subject to accuracy constraint (0/1 values)
    sep_constraint: apply the fine grained accuracy constraint
        for details, see Section 3.3 of arxiv.org/abs/1507.05259v3
        For examples on how to apply these constraints, see "synthetic_data_demo/decision_boundary_demo.py"
    Note: both apply_fairness_constraints and apply_accuracy_constraint cannot be 1 at the same time
    sensitive_attrs: ["s1", "s2", ...], list of sensitive features for which to apply fairness constraint, all of these sensitive features should have a corresponding array in x_control
    sensitive_attrs_to_cov_thresh: the covariance threshold that the classifier should achieve (this is only needed when apply_fairness_constraints=1, not needed for the other two constraints)
    gamma: controls the loss in accuracy we are willing to incur when using apply_accuracy_constraint and sep_constraint

    ----

    Outputs:

    w: the learned weight vector for the classifier

    """

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
                                        y):  # dont confuse the protected here with the sensitive feature protected/non-protected values -- protected here means that these points should not be misclassified to negative class
            return np.dot(w, x.T)  # if this is positive, the constraint is satisfied

        def constraint_unprotected_people(w, ind, old_loss, x, y):

            new_loss = loss_function(w, np.array([x]), np.array(y))
            return ((1.0 + gamma) * old_loss) - new_loss

        constraints = []
        predicted_labels = np.sign(np.dot(w.x, x.T))
        unconstrained_loss_arr = loss_function(w.x, x, y, return_arr=True)

        if sep_constraint == True:  # separate gemma for different people
            for i in range(0, len(predicted_labels)):
                if predicted_labels[i] == 1.0 and x_control[sensitive_attrs[0]][
                    i] == 1.0:  # for now we are assuming just one sensitive attr for reverse constraint, later, extend the code to take into account multiple sensitive attrs
                    c = ({'type': 'ineq', 'fun': constraint_protected_people, 'args': (x[i], y[
                        i])})  # this constraint makes sure that these people stay in the positive class even in the modified classifier
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


def compute_cross_validation_error(x_all, y_all, x_control_all, num_folds, loss_function, apply_fairness_constraints,
                                   apply_accuracy_constraint, sep_constraint, sensitive_attrs,
                                   sensitive_attrs_to_cov_thresh_arr, gamma=None):
    """
    Computes the cross validation error for the classifier subject to various fairness constraints
    This function is just a wrapper of "train_model(...)", all inputs (except for num_folds) are the same. See the specifications of train_model(...) for more info.

    Returns lists of train/test accuracy (with each list holding values for all folds), the fractions of various sensitive groups in positive class (for train and test sets), and covariance between sensitive feature and distance from decision boundary (again, for both train and test folds).
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

    # Get the reuslts
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
        if str(type(k)) != "<type 'numpy.float64'>" and type(k) != int and type(k) != np.int64:
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
        val = in_arr[i]
        ind = index_dict[val]
        tup[ind] = 1  # set that value of tuple to 1
        out_arr.append(tup)

    return np.array(out_arr), index_dict


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
        print("Covariance is", cov)
        print("Diff is:", ans)
        print()
    return ans


def print_covariance_sensitive_attrs(model, x_arr, y_arr_dist_boundary, x_control, sensitive_attrs):
    """
    reutrns the covariance between sensitive features and distance from decision boundary
    """

    arr = []
    if model is not None:

        if isinstance(y_arr_dist_boundary, dict) is False:
            y_arr_dist_boundary = {}
            for s_attr in sensitive_attrs:
                y_arr_dist_boundary[s_attr] = np.dot(model, x_arr.T)

        for s in sensitive_attrs:
            s_attr_to_fp_fn_original = {}

            if isinstance(x_control[s], float) or isinstance(x_control[s], int):
                s_attr_arr = x_control[s].reshape(-1, 1)
            else:
                s_attr_arr = x_control[s]

            s_attr_arr = np.array(s_attr_arr, dtype=float)

            cov = np.dot(s_attr_arr - np.mean(s_attr_arr), y_arr_dist_boundary[s]) / float(len(y_arr_dist_boundary[s]))

            sensitive_attrs_to_cov_original = {}
            sensitive_attrs_to_cov_original[s] = cov

    else:
        if isinstance(y_arr_dist_boundary, dict) is False:
            y_arr_dist_boundary = {}
            for s_attr in sensitive_attrs:
                y_arr_dist_boundary[s_attr] = y_arr_dist_boundary

        for s in sensitive_attrs:
            s_attr_to_fp_fn_original = {}

            if isinstance(x_control[s], float) or isinstance(x_control[s], int):
                s_attr_arr = x_control[s].reshape(-1, 1)
            else:
                s_attr_arr = x_control[s]

            s_attr_arr = np.array(s_attr_arr, dtype=float)

            cov = np.dot(s_attr_arr - np.mean(s_attr_arr), y_arr_dist_boundary[s]) / float(len(y_arr_dist_boundary[s]))

            sensitive_attrs_to_cov_original = {}
            sensitive_attrs_to_cov_original[s] = cov

    return sensitive_attrs_to_cov_original


def get_correlations(model, x_test, y_predicted, x_control_test, sensitive_attrs):
    """
    returns the fraction in positive class for sensitive feature values
    """

    if model is not None:
        y_predicted = np.sign(np.dot(x_test, model))

    y_predicted = np.array(y_predicted)

    out_dict = {}
    for s_attr in sensitive_attrs:

        s_attr_vals = x_control_test[s_attr]
        uniq_vals = sorted(list(set(s_attr_vals)))

        if len(uniq_vals) <= 1:
            print(
                "!!Looks like the sensitive feature has only one value, skipping the correlation computation for this feature...")
            print("Feature name:", s_attr)
            continue

        total_per_val = defaultdict(int)
        positive_per_val = defaultdict(int)

        for i in range(len(y_predicted)):
            val = s_attr_vals[i]
            label = y_predicted[i]

            if label == 1:
                positive_per_val[val] += 1
            total_per_val[val] += 1

        positive_per_val_ratio = {}
        for k in total_per_val.keys():
            positive_per_val_ratio[k] = float(positive_per_val[k]) / total_per_val[k]

        out_dict[s_attr] = positive_per_val_ratio

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
                print(attr, attr_name, sensitive_attrs_to_cov_thresh[attr])
                thresh = sensitive_attrs_to_cov_thresh[attr][attr_name]

                t = attr_arr_transformed[:, ind]
                c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov,
                      'args': (x_train, y_train, t, thresh, False)})
                constraints.append(c)

    return constraints


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


def plot_cov_thresh_vs_acc_pos_ratio(x_all, y_all, x_control_all, num_folds, loss_function, apply_fairness_constraints,
                                     apply_accuracy_constraint, sep_constraint, sensitive_attrs):
    # very the covariance threshold using a range of decreasing multiplicative factors and see the tradeoffs between accuracy and fairness
    it = 0.05
    cov_range = np.arange(1.0, 0.0 - it, -it).tolist()
    if apply_accuracy_constraint == True:
        if sep_constraint == False:
            it = 0.1
            cov_range = np.arange(0.0, 1.0 + it, it).tolist()
        if sep_constraint == True:
            cov_range = [0, 1, 5, 10, 20, 50, 100, 500, 1000]

    positive_class_label = 1  # positive class is +1
    train_acc = []
    test_acc = []
    positive_per_category = defaultdict(list)  # for each category (male / female), the frac of positive

    # first get the original values of covariance in the unconstrained classifier -- these original values are not needed for reverse constraint    
    test_acc_arr, train_acc_arr, correlation_dict_test_arr, correlation_dict_train_arr, cov_dict_test_arr, cov_dict_train_arr = compute_cross_validation_error(
        x_all, y_all, x_control_all, num_folds, loss_function, 0, apply_accuracy_constraint, sep_constraint,
        sensitive_attrs, [{} for i in range(0, num_folds)], 0)

    for c in cov_range:
        print("LOG: testing for multiplicative factor: %0.2f" % c)
        sensitive_attrs_to_cov_original_arr_multiplied = []
        for sensitive_attrs_to_cov_original in cov_dict_train_arr:
            sensitive_attrs_to_cov_thresh = deepcopy(sensitive_attrs_to_cov_original)
            for k in sensitive_attrs_to_cov_thresh.keys():
                v = sensitive_attrs_to_cov_thresh[k]
                if type(v) == type({}):
                    for k1 in v.keys():
                        v[k1] = v[k1] * c
                else:
                    sensitive_attrs_to_cov_thresh[k] = v * c
            sensitive_attrs_to_cov_original_arr_multiplied.append(sensitive_attrs_to_cov_thresh)

        test_acc_arr, train_acc_arr, correlation_dict_test_arr, correlation_dict_train_arr, cov_dict_test_arr, cov_dict_train_arr = compute_cross_validation_error(
            x_all, y_all, x_control_all, num_folds, loss_function, apply_fairness_constraints,
            apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_original_arr_multiplied,
            c)
        test_acc.append(np.mean(test_acc_arr))

        correlation_dict_train = get_avg_correlation_dict(correlation_dict_train_arr)
        correlation_dict_test = get_avg_correlation_dict(correlation_dict_test_arr)

        # just plot the correlations for the first sensitive attr, the plotting can be extended for the other values, but as a proof of concept, we will jsut show for one
        s = sensitive_attrs[0]

        for k, v in correlation_dict_test[s].items():
            if v.get(positive_class_label) is None:
                positive_per_category[k].append(0.0)
            else:
                positive_per_category[k].append(v[positive_class_label])

    positive_per_category = dict(positive_per_category)

    p_rule_arr = (np.array(positive_per_category[0]) / np.array(positive_per_category[1])) * 100.0

    ax = plt.subplot(2, 1, 1)
    plt.plot(cov_range, positive_per_category[0], "-o", color="green", label="Protected")
    plt.plot(cov_range, positive_per_category[1], "-o", color="blue", label="Non-protected")
    ax.set_xlim([min(cov_range), max(cov_range)])
    plt.xlabel('Multiplicative loss factor')
    plt.ylabel('Perc. in positive class')
    if apply_accuracy_constraint == False:
        plt.gca().invert_xaxis()
        plt.xlabel('Multiplicative covariance factor (c)')
    ax.legend()

    ax = plt.subplot(2, 1, 2)
    plt.scatter(p_rule_arr, test_acc, color="red")
    ax.set_xlim([min(p_rule_arr), max(max(p_rule_arr), 100)])
    plt.xlabel('P% rule')
    plt.ylabel('Accuracy')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.show()


def get_line_coordinates(w, x1, x2):
    y1 = (-w[0] - (w[1] * x1)) / w[2]
    y2 = (-w[0] - (w[1] * x2)) / w[2]
    return y1, y2


def evaluate_sklearn_model(classifier_name, sensitive_param):
    """Evaluate the sklearn model performance"""
    print("Evaluating Sklearn Model...")

    # Load the model
    model = joblib.load(HERE.joinpath(f"methods/aequitas3/{classifier_name}"))

    # Load test data
    X_test = []
    Y_test = []
    with open("cleaned_train", "r") as f:
        next(f)  # Skip header
        for line in f:
            line = line.strip().split(',')
            features = list(map(int, line[:-1]))
            X_test.append(features)
            Y_test.append(1 if int(line[-1]) == 1 else -1)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    # Make predictions
    Y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, pos_label=1)
    recall = recall_score(Y_test, Y_pred, pos_label=1)
    f1 = f1_score(Y_test, Y_pred, pos_label=1)
    conf_matrix = confusion_matrix(Y_test, Y_pred)

    print("\nSklearn Model Performance:")
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1 Score: " + str(f1))
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Evaluate fairness metrics
    sensitive_feature = X_test[:, sensitive_param - 1]  # Get the sensitive attribute (sex)
    predictions_by_group = {
        0: Y_pred[sensitive_feature == 0],
        1: Y_pred[sensitive_feature == 1]
    }
    actual_by_group = {
        0: Y_test[sensitive_feature == 0],
        1: Y_test[sensitive_feature == 1]
    }

    print("\nFairness Metrics:")
    for group in [0, 1]:
        group_acc = accuracy_score(actual_by_group[group], predictions_by_group[group])
        group_size = len(predictions_by_group[group])
        print("Group " + str(group))
        print("  Size: " + str(group_size))
        print("  Accuracy: " + str(group_acc))


def evaluate_direct_model(sensitive_param, ut):
    """Evaluate the direct model performance"""
    print("\nEvaluating Direct Model...")

    # Load and prepare data
    X = []
    Y = []
    sens = []
    with open("cleaned_train", "r") as f:
        next(f)  # Skip header
        for line in f:
            line = line.strip().split(',')
            features = list(map(int, line[:-1]))
            sens.append(features[sensitive_param - 1])
            X.append(features)
            Y.append(1 if int(line[-1]) == 1 else -1)

    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    sensitive = {'sex': np.array(sens, dtype=float)}

    # Train model with the same parameters as in the original script
    loss_function = lf._logistic_loss
    sep_constraint = 0
    sensitive_attrs = ['sex']
    sensitive_attrs_to_cov_thresh = {'sex': 0}
    gamma = None

    model = ut.train_model(X, Y, sensitive, loss_function, 1, 0, sep_constraint,
                           sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)

    # Make predictions
    Y_pred = np.sign(np.dot(X, model))

    # Calculate metrics
    accuracy = accuracy_score(Y, Y_pred)
    precision = precision_score(Y, Y_pred, pos_label=1)
    recall = recall_score(Y, Y_pred, pos_label=1)
    f1 = f1_score(Y, Y_pred, pos_label=1)
    conf_matrix = confusion_matrix(Y, Y_pred)

    print("\nDirect Model Performance:")
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1 Score: " + str(f1))
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Evaluate fairness metrics
    sensitive_feature = X[:, sensitive_param - 1]  # Get the sensitive attribute (sex)
    predictions_by_group = {
        0: Y_pred[sensitive_feature == 0],
        1: Y_pred[sensitive_feature == 1]
    }
    actual_by_group = {
        0: Y[sensitive_feature == 0],
        1: Y[sensitive_feature == 1]
    }

    print("\nFairness Metrics:")
    for group in [0, 1]:
        group_acc = accuracy_score(actual_by_group[group], predictions_by_group[group])
        group_size = len(predictions_by_group[group])
        print("Group " + str(group))
        print("  Size: " + str(group_size))
        print("  Accuracy: " + str(group_acc))
