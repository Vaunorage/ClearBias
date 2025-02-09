import logging
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from queue import PriorityQueue

from win32comext.shell.demos.servers.folder_view import onSetting1
from z3 import *
import os
import copy

from lime import lime_tabular
import time
import random

from data_generator.main import get_real_data, DiscriminationData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_model(model_type, X, y):
    """
    Train a model based on the specified type
    :param model_type: String specifying model type ('lr', 'rf', 'svm', 'mlp')
    :param X: Feature matrix
    :param y: Target values
    :return: Trained model
    """
    model_type = model_type.lower()
    if model_type == 'lr':
        model = LogisticRegression(random_state=42)
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        model = SVC(kernel='rbf', probability=True, random_state=42)
    elif model_type == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from: 'lr', 'rf', 'svm', 'mlp'")

    # Split data and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Print basic model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    logger.info(f"Model training complete:")
    logger.info(f"Training accuracy: {train_score:.4f}")
    logger.info(f"Testing accuracy: {test_score:.4f}")

    return model


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


def global_discovery(iteration, config):
    input_bounds = config.input_bounds
    params = config.params

    samples = []
    for j in range(iteration):
        x = np.zeros(params)
        for i in range(params):
            # random.seed(time.time())
            x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])
        x = np.array(x)
        samples.append(x)
    # print x
    samples = np.array(samples)
    return samples


def seed_test_input(dataset, cluster_num, limit):
    """
    Select the seed inputs for fairness testing
    :param dataset: the name of dataset
    :param clusters: the results of K-means clustering
    :param limit: the size of seed inputs wanted
    :return: a sequence of seed inputs
    """
    # build the clustering model
    clf = cluster(dataset, cluster_num)
    clusters = [np.where(clf.labels_ == i)[0] for i in range(cluster_num)]
    clusters = sorted(clusters, key=len)  # len(clusters[0][0])==32561
    return clusters[0]


def extract_lime_decision_constraints(ge, model, input):
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
        discretize_continuous=True
    )
    o_data, g_data = explainer._LimeTabularExplainer__data_inverse(input, num_samples=5000)
    # print(g_data)
    g_labels = model.predict(g_data)

    # build the interpretable tree
    tree = DecisionTreeClassifier(random_state=2019)  # min_samples_split=0.05, min_samples_leaf =0.01
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


def check_for_discrimination_case(ge, model, t, sensitive_indices):
    """
    Check whether the test case is an individual discriminatory instance
    :param ge: DiscriminationData object
    :param model: the model's symbolic output
    :param t: test case
    :param sensitive_indices: dictionary of sensitive attribute names and their indices
    :return: whether it is an individual discriminatory instance
    """
    import itertools

    # Get original prediction
    org_df = pd.DataFrame([t], columns=ge.attr_columns)
    label = model.predict(org_df)
    org_df['outcome'] = label

    # Get all possible values for each sensitive attribute
    sensitive_values = {}
    for sens_name, sens_idx in sensitive_indices.items():
        sensitive_values[sens_name] = np.unique(ge.xdf.iloc[:, sens_idx]).tolist()

    # Generate all possible combinations of sensitive attribute values
    sensitive_names = list(sensitive_indices.keys())
    value_combinations = list(itertools.product(*[sensitive_values[name] for name in sensitive_names]))

    # Create new test cases with all combinations
    new_targets = []
    for values in value_combinations:
        # Skip if combination is identical to original
        if all(t[sensitive_indices[name]] == value for name, value in zip(sensitive_names, values)):
            continue

        tnew = pd.DataFrame([t], columns=ge.attr_columns)
        for name, value in zip(sensitive_names, values):
            tnew[name] = value
        new_targets.append(tnew)

    if not new_targets:  # If no new combinations were generated
        return False

    new_targets = pd.concat(new_targets)
    new_targets['outcome'] = model.predict(new_targets)

    # Check if any combination leads to a different prediction
    discriminations = new_targets[new_targets['outcome'] != label[0]]

    return discriminations.shape[0] > 0, org_df, discriminations


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


def symbolic_generation(ge: DiscriminationData, model_type, cluster_num, limit):
    """
    The implementation of symbolic generation
    :param ge: the name of dataset
    :param sensitive_param: the index of sensitive feature
    :param model_type: model type
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :param limit: the maximum number of test case
    """

    start = time.time()

    model = train_model(model_type, ge.xdf, ge.ydf)
    # the rank for priority queue, rank1 is for seed inputs, rank2 for local, rank3 for global
    rank1 = 5
    rank2 = 1
    rank3 = 10
    T1 = 0.3

    # store the result of fairness testing
    global_disc_inputs = set()
    global_disc_inputs_list = []
    local_disc_inputs = set()
    local_disc_inputs_list = []
    tot_inputs = set()

    # select the seed input for fairness testing
    inputs = seed_test_input(ge.xdf, cluster_num, limit)
    print(len(inputs))
    print(inputs)

    # Get all input data at once and convert to numpy for faster processing
    input_data = copy.deepcopy(ge.xdf.iloc[inputs])
    input_data['key'] = input_data[ge.non_protected_attributes].apply(lambda x: ''.join(x.astype(str)), axis=1)
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
    arguments = gen_arguments(ge)

    results = []
    use_time = 0  # Initialize use_time

    while len(tot_inputs) < limit and targets_queue.qsize() != 0:
        use_time = time.time() - start  # Update use_time at the start of each iteration
        if use_time >= 3900:  # Check time limit at the start of each iteration
            break

        org_input = targets_queue.get()
        org_input_rank = org_input[0]
        org_input = np.array(org_input[1])

        found, org_df, found_df = check_for_discrimination_case(ge, model, org_input, ge.sensitive_indices)
        decision_rules = extract_lime_decision_constraints(ge, model, org_input)

        # Create a version of the input without any sensitive parameters
        input_without_sensitive = remove_sensitive_attributes(org_input.tolist(), ge.sensitive_indices)
        input_key = tuple(input_without_sensitive)

        # Track unique inputs and check for discrimination
        tot_inputs.add(input_key)
        if found:
            print(input_key, found, len(tot_inputs), len(results))
            results.append((org_df, found_df))
            if (input_key not in global_disc_inputs) and (input_key not in local_disc_inputs):
                if org_input_rank > 2:
                    global_disc_inputs.add(input_key)
                    global_disc_inputs_list.append(input_key)
                else:
                    local_disc_inputs.add(input_key)
                    local_disc_inputs_list.append(input_key)

                if len(tot_inputs) == limit:
                    break

            # local search
            for decision_rule_index in range(len(decision_rules)):
                path_constraint = copy.deepcopy(decision_rules)
                c = path_constraint[decision_rule_index]
                if c[0] in ge.sensitive_indices.values():
                    continue

                if c[1] == "<=":
                    c[1] = ">"
                    c[3] = 1.0 - c[3]
                else:
                    c[1] = "<="
                    c[3] = 1.0 - c[3]

                end = time.time()
                use_time = end - start
                if use_time >= count:
                    print("Percentage discriminatory inputs - " + str(
                        float(len(global_disc_inputs_list) + len(local_disc_inputs_list))
                        / float(len(tot_inputs)) * 100))
                    print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))
                    print("Total Inputs are " + str(len(tot_inputs)))
                    print('use time:' + str(end - start))
                    count += 300

                if use_time >= 3900:  # Check time limit after each local search
                    break
                if path_constraint not in visited_path:
                    visited_path.append(path_constraint)
                    input = local_solve(path_constraint, arguments, org_input, decision_rule_index, ge)
                    l_count += 1
                    if input != None:
                        r = average_confidence(path_constraint)
                        targets_queue.put((rank2 + r, input))

        # global search
        prefix_pred = []
        for c in decision_rules:
            if c[0] in ge.sensitive_indices.values():
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

            end = time.time()
            use_time = end - start
            if use_time >= count:
                print("Percentage discriminatory inputs - " + str(
                    float(len(global_disc_inputs_list) + len(local_disc_inputs_list))
                    / float(len(tot_inputs)) * 100))
                print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))
                print("Total Inputs are " + str(len(tot_inputs)))
                print('use time:' + str(end - start))
                count += 300

            # filter out the path_constraint already solved before
            if path_constraint not in visited_path:
                visited_path.append(path_constraint)
                input = global_solve(path_constraint, arguments, org_input, ge)
                g_count += 1
                if input != None:
                    r = average_confidence(path_constraint)
                    targets_queue.put((rank3 - r, input))

            prefix_pred = prefix_pred + [c]

            if use_time >= 3900:  # Check time limit after each global search iteration
                break
    res_df = []
    case_id = 0
    for org, counter_org in results:
        for _, counter_examples in counter_org.iterrows():
            indv1 = org.copy()
            indv2 = pd.DataFrame([counter_examples])

            indv_key1 = "|".join(str(x) for x in indv1[ge.attr_columns].iloc[0])
            indv_key2 = "|".join(str(x) for x in indv2[ge.attr_columns].iloc[0])

            # Add the additional columns
            indv1['indv_key'] = indv_key1
            indv2['indv_key'] = indv_key2

            # Create couple_key as before

            couple_key = f"{indv_key1}-{indv_key2}"
            diff_outcome = abs(indv1['outcome'] - indv2['outcome'])

            df_res = pd.concat([indv1, indv2])
            df_res['couple_key'] = couple_key
            df_res['diff_outcome'] = diff_outcome
            df_res['case_id'] = case_id
            res_df.append(df_res)
            case_id += 1

    if len(res_df) != 0:
        res_df = pd.concat(res_df)
    else:
        res_df = pd.DataFrame([])

    # Calculate metrics similar to Aequitas
    end_time = time.time()
    execution_time = end_time - start

    return res_df, execution_time, tot_inputs, results


def run_sg(ge: DiscriminationData, model_type='lr', cluster_num=None, limit=100, iter=2):
    if not cluster_num:
        cluster_num = len(ge.ydf.unique())

    start = time.time()
    results = []
    for i in range(iter):
        res_df, execution_time, tot_inputs, f_results = symbolic_generation(ge=ge,
                                                                 model_type=model_type,
                                                                 cluster_num=cluster_num,
                                                                 limit=limit)
        end = time.time()

        print('total time:' + str(end - start))
        results.append((res_df, execution_time, tot_inputs, f_results))

    results_df = pd.concat([res[0] for res in results])
    results_df.drop_duplicates(subset='couple_key', keep='first', inplace=True)

    tsn = sum([len(res[2]) for res in results])
    dsn = sum([len(res[3]) for res in results])

    metrics = {
        "TSN": tsn,
        "DSN": dsn,
        "DSS": round(sum([res[1] for res in results])/len(results), 2),
        "SUR": round(dsn/tsn, 2)
    }

    return results_df, metrics


if __name__ == '__main__':
    ge, ge_schema = get_real_data('adult')

    res = run_sg(ge)
    print(res)
