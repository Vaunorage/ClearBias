import sys
import argparse
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from queue import PriorityQueue
from z3 import *
import os
import copy
import signal
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import joblib
import random
import lime
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import logging

from data_generator.main import DiscriminationData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

args = None


class Config:
    def __init__(self, input_bounds, params, feature_name=None):
        self.input_bounds = input_bounds
        self.params = params
        self.feature_name = feature_name if feature_name is not None else [f'x{i}' for i in range(params)]


class QueueItem:
    def __init__(self, priority, data):
        self.priority = priority
        self.data = data.tolist() if isinstance(data, np.ndarray) else data

    def __lt__(self, other):
        return self.priority < other.priority

    def __eq__(self, other):
        return self.priority == other.priority


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


def seed_test_input(X, cluster_num, limit):
    """
    Select the seed inputs for fairness testing
    :param X: input data matrix
    :param cluster_num: number of clusters to form
    :param limit: the size of seed inputs wanted
    :return: a sequence of seed inputs
    """
    # build the clustering model
    clf = cluster(X, cluster_num)
    clusters = [np.where(clf.labels_ == i)[0] for i in range(cluster_num)]

    # Select samples from each cluster
    selected_indices = []
    max_per_cluster = limit // cluster_num + 1

    for cluster_indices in clusters:
        # Take up to max_per_cluster samples from each cluster
        n_samples = min(len(cluster_indices), max_per_cluster)
        if n_samples > 0:
            selected = np.random.choice(cluster_indices, size=n_samples, replace=False)
            selected_indices.extend(selected)

        if len(selected_indices) >= limit:
            break

    # Trim to exact limit if we went over
    selected_indices = selected_indices[:limit]

    return X[selected_indices]


def getPath(X, preds, input, conf):
    """
    Get the path from Local Interpretable Model-agnostic Explanation Tree
    :param X: the whole inputs
    :param preds: the model's symbolic output
    :param input: instance to interpret
    :param conf: the configuration of dataset
    :return: the path for the decision of given instance
    """

    # use the original implementation of LIME
    explainer = LimeTabularExplainer(X, feature_names=conf.feature_name, class_names=['0', '1'],
                                     categorical_features=[],
                                     discretize_continuous=True)
    o_data, g_data = explainer._LimeTabularExplainer__data_inverse(input, num_samples=5000)
    # print(g_data)
    g_labels = preds(g_data)

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


def check_for_error_condition(conf, preds, t, sens):
    """
    Check whether the test case is an individual discriminatory instance
    :param conf: the configuration of dataset
    :param preds: the model's symbolic output
    :param t: test case
    :param sens: the index of sensitive feature
    :return: tuple (is_discriminatory, original_prediction, counter_example, counter_prediction)
            or (False, None, None, None) if no discrimination found
    """
    label = preds(np.array([t]))
    for val in range(conf.input_bounds[sens - 1][0], conf.input_bounds[sens - 1][1] + 1):
        if val != t[sens - 1]:
            tnew = copy.deepcopy(t)
            tnew[sens - 1] = val
            label_new = preds(np.array([tnew]))
            if label_new != label:
                return True, label, tnew, label_new
    return False, None, None, None


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


def gen_arguments(conf):
    """
    Generate the argument for all the features
    :param conf: the configuration of dataset
    :return: a sequence of arguments
    """
    arguments = []
    for i in range(conf.params):
        arguments.append(z3.Int(conf.feature_name[i]))
        # arguments.append(conf.feature_name[i])
    return arguments


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


class SymbolicGenerationResult:
    def __init__(self, outcome: int, diff_outcome: float, type_str: str,
                 sensitive_attribute: str, total_inputs: int, discriminatory_inputs: int,
                 percentage_discriminatory_inputs: float, case_id: int,
                 indv_key: str, couple_key: str):
        self.outcome = outcome
        self.diff_outcome = diff_outcome
        self.type = type_str
        self.Sensitive_Attribute = sensitive_attribute
        self.Total_Inputs = total_inputs
        self.Discriminatory_Inputs = discriminatory_inputs
        self.Percentage_Discriminatory_Inputs = percentage_discriminatory_inputs
        self.case_id = case_id
        self.indv_key = indv_key
        self.couple_key = couple_key



def symbolic_generation(ge: DiscriminationData, model_type, cluster_num, limit, iter):
    """
    The implementation of symbolic generation
    :param ge: pandas DataFrame containing the data
    :param model_type: String specifying the type of model to train ('lr', 'rf', 'svm', 'mlp')
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :param limit: the maximum number of test case
    :param iter: iteration number
    :return: List of SymbolicGenerationResult objects
    """
    start_time = time.time()
    count = 300  # Initialize counter for periodic reporting

    logger.info(f"Starting symbolic generation with model_type={model_type}, cluster_num={cluster_num}, limit={limit}")

    dataset = ge.dataframe
    # Get data from DataFrame
    logger.info("Preparing data...")
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values
    logger.info(f"Data shape: X={X.shape}, Y={Y.shape}")

    # Set up input bounds based on the data
    logger.info("Calculating input bounds...")
    input_bounds = []
    for col_idx in range(X.shape[1]):
        min_val = int(X[:, col_idx].min())
        max_val = int(X[:, col_idx].max())
        input_bounds.append([min_val, max_val])
    logger.info(f"Input bounds: {input_bounds}")

    # Get feature names
    feature_names = list(dataset.iloc[:, :-1].columns) if hasattr(dataset, 'columns') else [f'x{i}' for i in
                                                                                            range(X.shape[1])]
    logger.info(f"Feature names: {feature_names}")

    # Create config object
    config = Config(input_bounds=input_bounds,
                    params=X.shape[1],
                    feature_name=feature_names)
    logger.info("Configuration object created")

    # Protected indices setup
    protected_indices = list(range(X.shape[1]))
    sensitive_param = protected_indices[0] + 1
    logger.info(f"Using sensitive parameter index: {sensitive_param}")

    # Train the model
    logger.info(f"Training {model_type} model...")
    model = train_model(model_type, X, Y)

    # Get seed inputs through clustering
    logger.info("Generating seed inputs through clustering...")
    inputs = seed_test_input(X, cluster_num, limit)
    logger.info(f"Generated {len(inputs)} seed inputs")

    # Initialize queue and tracking variables
    logger.info("Initializing search process...")
    q = PriorityQueue()
    rank1 = 1
    tot_inputs = set()
    results = []  # List to store SymbolicGenerationResult objects
    case_id = 0  # Counter for unique case IDs

    # Add inputs to queue
    for inp in inputs[::-1]:
        q.put(QueueItem(rank1, inp))
    logger.info(f"Added {len(inputs)} inputs to queue")

    visited_path = []
    l_count = 0
    g_count = 0

    logger.info("Starting main search loop...")
    while len(tot_inputs) < limit and q.qsize() != 0:
        current_time = time.time()
        use_time = current_time - start_time

        if use_time >= count:
            logger.info(f"Progress update at {use_time:.2f} seconds:")
            logger.info(f"- Discriminatory inputs found: {len(results)}")
            logger.info(f"- Total inputs processed: {len(tot_inputs)}")
            logger.info(f"- Local searches: {l_count}")
            logger.info(f"- Global searches: {g_count}")
            logger.info(f"- Queue size: {q.qsize()}")
            count += 300

        if use_time >= 3900:
            logger.warning("Time limit reached (65 minutes)")
            break

        # Get next input from queue
        t = q.get()
        t_rank = t.priority
        t = np.array(t.data)

        # Check for discrimination
        logger.debug(f"Checking input: {t}")
        is_disc, orig_pred, counter_ex, counter_pred = check_for_error_condition(config, model.predict, t,
                                                                                 sensitive_param)
        if is_disc:
            # First, get the schema to know which attributes are protected
            schema = ge.schema()

            # Create formatted feature names based on protected attributes
            formatted_feature_names = []
            for i, name in enumerate(feature_names):
                # Check if this attribute is protected by looking it up in the schema's protected_attr list
                is_protected = schema.protected_attr[i]
                formatted_name = f"Attr{i}_{'T' if is_protected else 'X'}"
                formatted_feature_names.append(formatted_name)

            # Create DataFrames with the properly formatted column names
            indv1 = pd.DataFrame([t], columns=formatted_feature_names)
            indv2 = pd.DataFrame([counter_ex], columns=formatted_feature_names)

            indv_key1 = "|".join(str(x) for x in t)
            indv_key2 = "|".join(str(x) for x in counter_ex)

            # Add the additional columns
            indv1['outcome'] = orig_pred
            indv1['indv_key'] = indv_key1
            indv2['outcome'] = counter_pred
            indv2['indv_key'] = indv_key2

            # Create couple_key as before

            couple_key = f"{indv_key1}-{indv_key2}"
            diff_outcome = abs(orig_pred - counter_pred)

            df_res = pd.concat([indv1, indv2])
            df_res['couple_key'] = couple_key
            df_res['diff_outcome'] = diff_outcome
            df_res['case_id'] = case_id
            results.append(df_res)
            case_id += 1

            # Log in a format similar to Aequitas
            logger.info(
                f"Found discrimination: {df_res['couple_key']} (outcome={df_res['outcome']}, diff={df_res['diff_outcome']})")

            tot_inputs.add(tuple(t.tolist()))

        # Get explanation path
        p = getPath(X, model.predict, t, config)
        path_str = str([(config.feature_name[c[0]], c[1], c[2], c[3]) for c in p])
        logger.debug(f"Path for input: {path_str}")

        if str(p) not in visited_path:
            visited_path.append(str(p))

            # Local search
            logger.debug("Performing local search...")
            for i in range(len(p)):
                path_constraint = copy.deepcopy(p)
                c = path_constraint[i]

                if c[1] == "<=":
                    c[1] = ">"
                    c[3] = 1.0 - c[3]
                else:
                    c[1] = "<="
                    c[3] = 1.0 - c[3]

                input = local_solve(path_constraint, gen_arguments(config), t, i, config)
                if input is not None:
                    l_count += 1
                    r = average_confidence(path_constraint)
                    q.put(QueueItem(1 + r, input))
                    logger.debug(f"Added new input from local search: {input}")

            # Global search
            logger.debug("Performing global search...")
            prefix_pred = []
            for i in range(len(p)):
                c = p[i]
                path_constraint = prefix_pred

                if c[1] == "<=":
                    n_c = copy.deepcopy(c)
                    n_c[1] = ">"
                    n_c[3] = 1.0 - c[3]
                else:
                    n_c = copy.deepcopy(c)
                    n_c[1] = "<="
                    n_c[3] = 1.0 - c[3]

                path_constraint = prefix_pred + [n_c]
                input = global_solve(path_constraint, gen_arguments(config), t, config)

                if input is not None:
                    g_count += 1
                    r = average_confidence(path_constraint)
                    q.put(QueueItem(10 - r, input))
                    logger.debug(f"Added new input from global search: {input}")

                prefix_pred = prefix_pred + [c]

    logger.info("Search completed!")
    logger.info(f"Final statistics:")
    logger.info(f"- Total time: {time.time() - start_time:.2f} seconds")
    logger.info(f"- Total discriminatory inputs found: {len(results)}")
    logger.info(f"- Total inputs processed: {len(tot_inputs)}")
    logger.info(f"- Local searches performed: {l_count}")
    logger.info(f"- Global searches performed: {g_count}")

    # create the folder for storing the fairness testing result
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    if not os.path.exists('./results/' + "generated" + '/'):
        os.makedirs('./results/' + "generated" + '/')
    if not os.path.exists('./results/' + "generated" + '/' + str(sensitive_param)):
        os.makedirs('./results/' + "generated" + '/' + str(sensitive_param))

    # storing the fairness testing result
    np.save('./results/' + "generated" + '/' + str(sensitive_param) + '/global_samples_symbolic{}.npy'.format(iter),
            np.array([r['couple_key'] for r in results]))
    np.save('./results/' + "generated" + '/' + str(sensitive_param) + '/local_samples_symbolic{}.npy'.format(iter),
            np.array([r['couple_key'] for r in results]))

    if not results:
        return pd.DataFrame()  # Return empty DataFrame if no results

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Calculate global statistics
    total_inputs = len(tot_inputs)
    discriminatory_inputs = len(results)
    percentage = (discriminatory_inputs / total_inputs * 100) if total_inputs > 0 else 0

    # Update global statistics in each row
    df["Total Inputs"] = total_inputs
    df["Discriminatory Inputs"] = discriminatory_inputs
    df["Percentage Discriminatory Inputs"] = percentage
    df["TSN"] = total_inputs
    df["DSN"] = discriminatory_inputs
    df["SUR"] = percentage

    # Reorder columns to match desired format
    desired_cols = [f"Attr{i}_{'T' if str(i) in df.columns[0] else 'X'}" for i in range(len(feature_names))] + [
        "outcome", "diff_outcome", "type", "Sensitive Attribute", "Total Inputs",
        "Discriminatory Inputs", "Percentage Discriminatory Inputs", "TSN", "DSN",
        "DSS", "SUR", "case_id", "indv_key", "couple_key"
    ]

    # Reorder columns and return
    return df[desired_cols]
