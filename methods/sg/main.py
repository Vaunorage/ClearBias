import logging
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from methods.utils import train_sklearn_model, check_for_error_condition, make_final_metrics_and_dataframe
from queue import PriorityQueue, Queue

from z3 import *
import copy

from lime import lime_tabular
import time
import random

from data_generator.main import get_real_data, DiscriminationData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Suppress Intel Extension messages
class SklearnexFilter(logging.Filter):
    def filter(self, record):
        return 'sklearnex' not in record.name and 'sklearn.utils.validation._assert_all_finite' not in record.msg


logging.getLogger().addFilter(SklearnexFilter())
logging.getLogger('sklearnex').setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def cluster(data, cluster_num):
    """
    Perform K-means clustering on the data
    :param data: input data to cluster
    :param cluster_num: number of clusters to form
    :return: fitted clustering model
    """
    kmeans = KMeans(n_clusters=cluster_num, random_state=42)
    kmeans.fit(data)
    return kmeans


def global_discovery(iteration, config, random_state=42):
    np.random.seed(random_state)  # Only need to set numpy's seed

    # Extract the bounds
    input_bounds = np.array(config.input_bounds)

    # Get dimensions
    dimensions = len(input_bounds)

    # Calculate the ranges
    lower_bounds = input_bounds[:, 0]
    upper_bounds = input_bounds[:, 1]

    samples = np.random.randint(
        low=lower_bounds,
        high=upper_bounds + 1,
        size=(iteration, dimensions)
    )

    return samples


def seed_test_input(dataset, cluster_num=None, random_seed=42, iter=0):
    """
    Select the seed inputs for fairness testing
    :param dataset: the name of dataset
    :param clusters: the results of K-means clustering
    :param limit: the size of seed inputs wanted
    :return: a sequence of seed inputs
    """
    # build the clustering model
    np.random.seed(random_seed + iter)
    if cluster_num is None:
        cluster_num = max(min(cluster_num, dataset.shape[0]), 10)

    clf = KMeans(
        n_clusters=cluster_num,
        random_state=random_seed + iter,  # Use the provided seed directly
        # n_init=20,
        init='k-means++',
    )
    clf.fit(dataset)

    clusters = [np.where(clf.labels_ == i)[0] for i in range(cluster_num)]
    random.seed(random_seed + iter)
    clusters = sorted(clusters, key=len)  # len(clusters[0][0])==32561
    return clusters


def extract_lime_decision_constraints(ge, model, input, random_state=42):
    """
    Get the path from Local Interpretable Model-agnostic Explanation Tree
    :param X: the whole inputs
    :param model: the model's symbolic output
    :param input: instance to interpret
    :param conf: the configuration of dataset
    :return: the path for the decision of given instance
    """

    # Convert DataFrame to numpy array for LIME
    X_train = ge.xdf.to_numpy()

    # Get feature names from DataFrame
    feature_names = ge.xdf.columns.tolist()

    # Define class names (binary classification: 0 and 1)
    class_names = ge.ydf.unique()

    # Get categorical features (if any are specified in the DataFrame)
    categorical_features = [i for i, dtype in enumerate(ge.xdf.dtypes) if dtype == 'category' or dtype == 'object']

    # Initialize LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        categorical_features=categorical_features,
        discretize_continuous=True,
        random_state=random_state  # Add random state parameter
    )
    o_data, g_data = explainer._LimeTabularExplainer__data_inverse(input, num_samples=5000)
    # print(g_data)
    g_labels = model.predict(g_data)

    # build the interpretable tree
    tree = DecisionTreeClassifier(random_state=random_state)  # min_samples_split=0.05, min_samples_leaf =0.01
    tree.fit(g_data, g_labels)

    # get the path for decision
    path_index = tree.decision_path(np.array([input])).indices
    path = []
    for i in range(len(path_index)):
        node = path_index[i]
        i = i + 1
        f = tree.tree_.feature[node]
        if f != -2:
            left_count = tree.tree_.n_node_samples[tree.tree_.children_left[node]]
            right_count = tree.tree_.n_node_samples[tree.tree_.children_right[node]]
            left_confidence = 1.0 * left_count / (left_count + right_count)
            right_confidence = 1.0 - left_confidence
            if tree.tree_.children_left[node] == path_index[i]:
                path.append([f, "<=", tree.tree_.threshold[node], left_confidence])
            else:
                path.append([f, ">", tree.tree_.threshold[node], right_confidence])
    return path


def remove_sensitive_attributes(input_vector, sensitive_indices):
    """
    Remove multiple sensitive attributes from the input vector
    :param input_vector: original input vector
    :param sensitive_indices: dictionary of sensitive attribute names and their indices
    :return: input vector without sensitive attributes
    """
    result = []
    for i in range(len(input_vector)):
        if i not in sensitive_indices.values():
            result.append(input_vector[i])
    return result


def global_solve(path_constraint, arguments, t, conf):
    """
    Solve the constraint for global generation
    :param path_constraint: the constraint of path
    :param arguments: the name of features in path_constraint
    :param t: test case
    :param conf: the configuration of dataset
    :return: new instance through global generation
    """
    s = Solver()
    for c in path_constraint:
        s.add(arguments[c[0]] >= conf.input_bounds[c[0]][0])
        s.add(arguments[c[0]] <= conf.input_bounds[c[0]][1])
        if c[1] == "<=":
            s.add(arguments[c[0]] <= c[2])
        else:
            s.add(arguments[c[0]] > c[2])

    if s.check() == sat:
        m = s.model()
    else:
        return None

    tnew = copy.deepcopy(t)
    for i in range(len(arguments)):
        if m[arguments[i]] == None:
            continue
        else:
            tnew[i] = int(m[arguments[i]].as_long())
    return tnew.astype('int').tolist()


def local_solve(path_constraint, arguments, t, index, conf):
    """
    Solve the constraint for local generation
    :param path_constraint: the constraint of path
    :param arguments: the name of features in path_constraint
    :param t: test case
    :param index: the index of constraint for local generation
    :param conf: the configuration of dataset
    :return: new instance through global generation
    """
    c = path_constraint[index]
    s = z3.Solver()
    s.add(arguments[c[0]] >= conf.input_bounds[c[0]][0])
    s.add(arguments[c[0]] <= conf.input_bounds[c[0]][1])
    for i in range(len(path_constraint)):
        if path_constraint[i][0] == c[0]:
            if path_constraint[i][1] == "<=":
                s.add(arguments[path_constraint[i][0]] <= path_constraint[i][2])
            else:
                s.add(arguments[path_constraint[i][0]] > path_constraint[i][2])

    if s.check() == sat:
        m = s.model()
    else:
        return None

    tnew = copy.deepcopy(t)
    tnew[c[0]] = int(m[arguments[c[0]]].as_long())
    return tnew.astype('int').tolist()


def average_confidence(path_constraint):
    """
    The average confidence (probability) of path
    :param path_constraint: the constraint of path
    :return: the average confidence
    """
    r = np.mean(np.array(path_constraint)[:, 3].astype(float))
    return r


def gen_arguments(ge):
    """
    Generate the argument for all the features
    :param ge: DiscriminationData object containing the dataset
    :return: a sequence of arguments
    """
    arguments = []
    feature_names = ge.xdf.columns.tolist()
    for feature_name in feature_names:
        arguments.append(z3.Int(str(feature_name)))
    return arguments


def run_sg(data: DiscriminationData, model_type='lr', cluster_num=None, max_tsn=100, random_state=42,
           max_runtime_seconds=3900, one_attr_at_a_time=True, db_path=None, analysis_id=None):
    # store the result of fairness testing
    global_disc_inputs = set()
    global_disc_inputs_list = []
    local_disc_inputs = set()
    local_disc_inputs_list = []
    tot_inputs = set()
    all_discriminations = set()
    all_tot_inputs = []
    early_termination = False

    def should_terminate() -> bool:
        nonlocal early_termination
        current_runtime = time.time() - start_time
        max_runtime_seconds_exceeded = current_runtime > max_runtime_seconds
        tsn_threshold_reached = max_tsn is not None and len(tot_inputs) >= max_tsn

        if max_runtime_seconds_exceeded or tsn_threshold_reached:
            early_termination = True
            return True
        return False

    dsn_by_attr_value = {e: {'TSN': 0, 'DSN': 0} for e in data.protected_attributes}
    dsn_by_attr_value['total'] = 0

    start_time = time.time()

    np.random.seed(random_state)
    random.seed(random_state)

    if not cluster_num:
        cluster_num = len(data.ydf.unique())

    start = time.time()
    f_results = []

    all_inputs = seed_test_input(data.xdf, cluster_num)

    ge_targets_queue = Queue()
    [ge_targets_queue.put(inp) for inp in all_inputs]

    while not should_terminate() and ge_targets_queue.qsize() != 0:

        model, X_train, X_test, y_train, y_test, feature_names = train_sklearn_model(
            data=data.training_dataframe,
            model_type=model_type,
            sensitive_attrs=data.protected_attributes,
            target_col=data.outcome_column,
            random_state=random_state,
        )

        # the rank for priority queue, rank1 is for seed inputs, rank2 for local, rank3 for global
        rank1 = 5
        rank2 = 1
        rank3 = 10
        T1 = 0.3

        # select the seed input for fairness testing
        inputs = ge_targets_queue.get()

        # Get all input data at once and convert to numpy for faster processing
        input_data = copy.deepcopy(data.xdf.iloc[inputs])
        input_data['key'] = input_data[data.non_protected_attributes].apply(lambda x: ''.join(x.astype(str)), axis=1)
        input_data = input_data.drop_duplicates(subset=['key'])
        input_data.drop(columns=['key'], inplace=True)

        # Initialize queue with optimized data insertion
        targets_queue = PriorityQueue()
        [targets_queue.put((rank1, row.tolist())) for row in input_data[::-1].to_numpy()]

        visited_path = []
        l_count = 0
        g_count = 0
        count = 300

        # Generate arguments for Z3 solver
        arguments = gen_arguments(data)

        def add_inputs(input_key):
            if (input_key not in global_disc_inputs) and (input_key not in local_disc_inputs):
                if org_input_rank > 2:
                    global_disc_inputs.add(input_key)
                    global_disc_inputs_list.append(input_key)
                else:
                    local_disc_inputs.add(input_key)
                    local_disc_inputs_list.append(input_key)

        while not should_terminate() and targets_queue.qsize() != 0:

            org_input = targets_queue.get()
            org_input_rank = org_input[0]
            org_input = np.array(org_input[1])
            found, found_df, max_discr, org_df, tested_inp = check_for_error_condition(logger=logger,
                                                                                       dsn_by_attr_value=dsn_by_attr_value,
                                                                                       discrimination_data=data,
                                                                                       model=model,
                                                                                       instance=org_input,
                                                                                       tot_inputs=tot_inputs,
                                                                                       all_discriminations=all_discriminations,
                                                                                       one_attr_at_a_time=one_attr_at_a_time,
                                                                                       db_path=db_path,
                                                                                       analysis_id=analysis_id)

            decision_rules = extract_lime_decision_constraints(data, model, org_input, random_state)

            # Create a version of the input without any sensitive parameters
            input_without_sensitive = remove_sensitive_attributes(org_input.tolist(), data.sensitive_indices_dict)
            input_key = tuple(input_without_sensitive)

            # Track unique inputs and check for discrimination
            if found:
                add_inputs(input_key)

                # Update dsn_by_attr_value for each discriminatory case
                for el in found_df.iterrows():
                    f_results.append((org_df, el[1].to_frame().T))
                    add_inputs(tuple(remove_sensitive_attributes(el[1].tolist(), data.sensitive_indices_dict)))

                    # Record the protected attribute values in the original input that led to discrimination
                    for attr_name in data.protected_attributes:
                        attr_idx = data.sensitive_indices_dict[attr_name]
                        attr_value = int(org_input[attr_idx])

                # local search
                for decision_rule_index in range(len(decision_rules)):
                    if should_terminate():
                        break

                    path_constraint = copy.deepcopy(decision_rules)
                    c = path_constraint[decision_rule_index]
                    if c[0] in data.sensitive_indices_dict.values():
                        continue

                    if c[1] == "<=":
                        c[1] = ">"
                        c[3] = 1.0 - c[3]
                    else:
                        c[1] = "<="
                        c[3] = 1.0 - c[3]

                    if path_constraint not in visited_path:
                        visited_path.append(path_constraint)
                        input = local_solve(path_constraint, arguments, org_input, decision_rule_index, data)
                        l_count += 1
                        if input != None:
                            r = average_confidence(path_constraint)
                            targets_queue.put((rank2 + r, input))

            # global search
            prefix_pred = []
            for c in decision_rules:
                if should_terminate():
                    break

                if c[0] in data.sensitive_indices_dict.values():
                    continue
                if c[3] < T1:
                    break

                n_c = copy.deepcopy(c)

                if n_c[1] == "<=":
                    n_c[1] = ">"
                    n_c[3] = 1.0 - c[3]
                else:
                    n_c[1] = "<="
                    n_c[3] = 1.0 - c[3]
                path_constraint = prefix_pred + [n_c]

                current_runtime = time.time() - start
                if current_runtime >= count:
                    print("Percentage discriminatory inputs - " + str(
                        float(len(global_disc_inputs_list) + len(local_disc_inputs_list))
                        / float(len(tot_inputs)) * 100))
                    print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))
                    print("Total Inputs are " + str(len(tot_inputs)))
                    print('use time:' + str(current_runtime))

                    count += 300

                # filter out the path_constraint already solved before
                if path_constraint not in visited_path:
                    visited_path.append(path_constraint)
                    input = global_solve(path_constraint, arguments, org_input, data)
                    g_count += 1
                    if input != None:
                        r = average_confidence(path_constraint)
                        targets_queue.put((rank3 - r, input))

                prefix_pred = prefix_pred + [c]

    res_df, metrics = make_final_metrics_and_dataframe(data, tot_inputs, all_discriminations, dsn_by_attr_value,
                                                       start_time, logger=logger)

    return res_df, metrics


if __name__ == '__main__':
    ge, ge_schema = get_real_data('adult', use_cache=True)

    res_df, metrics = run_sg(ge, cluster_num=50, max_tsn=20000, max_runtime_seconds=1000)
    print(f"Results DataFrame shape: {res_df.shape}")
    print(f"Metrics: {metrics}")
