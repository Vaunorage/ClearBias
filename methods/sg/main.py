import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import itertools
from sklearn.tree import DecisionTreeClassifier
from methods.utils import train_sklearn_model
from queue import PriorityQueue, Queue

from z3 import *
import copy

from lime import lime_tabular
import time
import random

from data_generator.main import get_real_data, DiscriminationData

# Configure logging
logger = logging.getLogger('SG')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# Suppress Intel Extension messages
class SklearnexFilter(logging.Filter):
    def filter(self, record):
        return 'sklearnex' not in record.name and 'sklearn.utils.validation._assert_all_finite' not in record.msg


logging.getLogger().addFilter(SklearnexFilter())
logging.getLogger('sklearnex').setLevel(logging.WARNING)


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


def check_for_discrimination_case(ge, model, t, sensitive_indices, dsn_by_attr_value, one_attr_at_a_time=False):
    """
    Check whether the test case is an individual discriminatory instance

    Args:
        ge: DiscriminationData object
        model: the model's symbolic output
        t: test case
        sensitive_indices: dictionary of sensitive attribute names and their indices
        one_attr_at_a_time: if True, vary only one sensitive attribute at a time

    Returns:
        whether it is an individual discriminatory instance, original dataframe,
        discriminations dataframe, and tested inputs
    """

    # Get original prediction
    org_df = pd.DataFrame([t], columns=ge.attr_columns)
    label = model.predict(org_df)
    org_df['outcome'] = label

    new_targets = []

    if one_attr_at_a_time:
        # Vary one attribute at a time
        for sens_name, sens_idx in sensitive_indices.items():
            # Get all possible values for this attribute
            values = np.unique(ge.xdf.iloc[:, sens_idx]).tolist()

            # Current value for this attribute
            current_value = t[sens_idx]
            dsn_by_attr_value[sens_name]['TSN'] += 1

            # Create new test cases with different values for this attribute only
            for value in values:
                if current_value == value:
                    continue

                tnew = pd.DataFrame([t], columns=ge.attr_columns)
                tnew[sens_name] = value
                new_targets.append(tnew)
    else:
        # Get all possible values for each sensitive attribute
        sensitive_values = {}
        for sens_name, sens_idx in sensitive_indices.items():
            sensitive_values[sens_name] = np.unique(ge.xdf.iloc[:, sens_idx]).tolist()
            dsn_by_attr_value[sens_name]['TSN'] += 1

        # Generate all possible combinations of sensitive attribute values
        sensitive_names = list(sensitive_indices.keys())
        value_combinations = list(itertools.product(*[sensitive_values[name] for name in sensitive_names]))

        # Create new test cases with all combinations
        for values in value_combinations:
            # Skip if combination is identical to original
            if all(t[sensitive_indices[name]] == value for name, value in zip(sensitive_names, values)):
                continue

            tnew = pd.DataFrame([t], columns=ge.attr_columns)
            for name, value in zip(sensitive_names, values):
                tnew[name] = value
            new_targets.append(tnew)

    if not new_targets:  # If no new combinations were generated
        return False, org_df, pd.DataFrame(), []

    new_targets = pd.concat(new_targets)
    new_targets['outcome'] = model.predict(new_targets)

    # Check if any combination leads to a different prediction
    discriminations = new_targets[new_targets['outcome'] != label[0]]

    if discriminations.shape[0] > 0:
        for el in discriminations.to_numpy():
            for attr in ge.protected_attributes:
                n_el = pd.DataFrame([el], columns=org_df.columns)
                if n_el[attr].iloc[0] != org_df[attr].iloc[0]:
                    dsn_by_attr_value[attr]['DSN'] += 1
                    dsn_by_attr_value['total'] += 1

    tested_inp = new_targets[ge.attr_columns].to_numpy().tolist()

    return discriminations.shape[0] > 0, org_df, discriminations, tested_inp


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


def run_sg(ge: DiscriminationData, model_type='lr', cluster_num=None, max_tsn=100, random_state=42,
           time_limit=3900, one_attr_at_a_time=True):
    # store the result of fairness testing
    global_disc_inputs = set()
    global_disc_inputs_list = []
    local_disc_inputs = set()
    local_disc_inputs_list = []
    tot_inputs = []

    dsn_by_attr_value = {e: {'TSN': 0, 'DSN': 0} for e in ge.protected_attributes}
    dsn_by_attr_value['total'] = 0

    np.random.seed(random_state)
    random.seed(random_state)

    if not cluster_num:
        cluster_num = len(ge.ydf.unique())

    start = time.time()
    f_results = []

    all_inputs = seed_test_input(ge.xdf, cluster_num)

    ge_targets_queue = Queue()
    [ge_targets_queue.put(inp) for inp in all_inputs]

    while len(tot_inputs) < max_tsn and (time.time() - start) <= time_limit and ge_targets_queue.qsize() != 0:

        model, X_train, X_test, y_train, y_test, feature_names = train_sklearn_model(
            data=ge.training_dataframe,
            model_type=model_type,
            sensitive_attrs=ge.protected_attributes,
            target_col=ge.outcome_column,
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

        def add_inputs(input_key):
            if (input_key not in global_disc_inputs) and (input_key not in local_disc_inputs):
                if org_input_rank > 2:
                    global_disc_inputs.add(input_key)
                    global_disc_inputs_list.append(input_key)
                else:
                    local_disc_inputs.add(input_key)
                    local_disc_inputs_list.append(input_key)

        while len(tot_inputs) < max_tsn and (time.time() - start) <= time_limit and targets_queue.qsize() != 0:
            dsn = len(f_results)
            sur = dsn / len(tot_inputs) if len(tot_inputs) > 0 else 0
            logger.info(f"TSN : {len(tot_inputs)} DSN: {dsn} DSR : {sur}")

            org_input = targets_queue.get()
            org_input_rank = org_input[0]
            org_input = np.array(org_input[1])

            found, org_df, found_df, tested_inp = check_for_discrimination_case(ge, model, org_input,
                                                                                ge.sensitive_indices_dict,
                                                                                dsn_by_attr_value,
                                                                                one_attr_at_a_time=one_attr_at_a_time)
            tot_inputs.extend(tested_inp)

            decision_rules = extract_lime_decision_constraints(ge, model, org_input, random_state)

            # Create a version of the input without any sensitive parameters
            input_without_sensitive = remove_sensitive_attributes(org_input.tolist(), ge.sensitive_indices_dict)
            input_key = tuple(input_without_sensitive)

            # Track unique inputs and check for discrimination
            if found:
                add_inputs(input_key)

                # Update dsn_by_attr_value for each discriminatory case
                for el in found_df.iterrows():
                    f_results.append((org_df, el[1].to_frame().T))
                    add_inputs(tuple(remove_sensitive_attributes(el[1].tolist(), ge.sensitive_indices_dict)))

                    # Record the protected attribute values in the original input that led to discrimination
                    for attr_name in ge.protected_attributes:
                        attr_idx = ge.sensitive_indices_dict[attr_name]
                        attr_value = int(org_input[attr_idx])

                # local search
                for decision_rule_index in range(len(decision_rules)):
                    path_constraint = copy.deepcopy(decision_rules)
                    c = path_constraint[decision_rule_index]
                    if c[0] in ge.sensitive_indices_dict.values():
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

                    if use_time >= time_limit:  # Check time limit after each local search
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
                if c[0] in ge.sensitive_indices_dict.values():
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

    end = time.time()
    res_df = []
    case_id = 0
    for org, counter_example in f_results:
        indv1 = org.copy()
        indv2 = counter_example.copy()

        indv_key1 = "|".join(str(x) for x in indv1[ge.attr_columns].iloc[0])
        indv_key2 = "|".join(str(x) for x in indv2[ge.attr_columns].iloc[0])

        # Add the additional columns
        indv1['indv_key'] = indv_key1
        indv2['indv_key'] = indv_key2

        couple_key = f"{indv_key1}-{indv_key2}"
        diff_outcome = abs(indv1['outcome'] - indv2['outcome'])

        df_res = pd.concat([indv1, indv2])
        df_res['couple_key'] = couple_key
        df_res['diff_outcome'] = diff_outcome
        df_res['case_id'] = case_id
        res_df.append(df_res)
        case_id += 1

    if len(res_df) != 0:
        results_df = pd.concat(res_df)
    else:
        results_df = pd.DataFrame([])

    # Calculate metrics similar to Aequitas
    execution_time = end - start

    tsn = len(tot_inputs)
    dsn = len(f_results)

    for k, v in dsn_by_attr_value.items():
        if k != 'total':
            dsn_by_attr_value[k]['SUR'] = dsn_by_attr_value[k]['DSN'] / dsn_by_attr_value[k]['TSN']

    metrics = {
        "TSN": tsn,
        "DSN": dsn,
        "SUR": round(dsn / tsn, 2) if tsn > 0 else 0,
        "DSS": round(execution_time / dsn, 2) if dsn > 0 else 0,
        "dsn_by_attr_value": dsn_by_attr_value  # Add the dsn_by_attr_value tracking to the metrics
    }

    return results_df, metrics


if __name__ == '__main__':
    ge, ge_schema = get_real_data('adult')

    res_df, metrics = run_sg(ge, max_tsn=10000, cluster_num=50, time_limit=1000)
    print(f"Results DataFrame shape: {res_df.shape}")
    print(f"Metrics: {metrics}")

    # Print dsn_by_attr_value in a readable format
    print("\nDiscriminatory samples by attribute value:")
    for attr, value_counts in metrics['dsn_by_attr_value'].items():
        if value_counts:
            total = sum(value_counts.values())
            print(f"  {attr} ({total} total):")
            for value, count in sorted(value_counts.items()):
                print(f"    Value {value}: {count} ({(count / total) * 100:.1f}%)")
