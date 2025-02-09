import sys

from data_generator.main import get_real_data, DiscriminationData
from path import HERE

sys.path.append("C:/Users/gen06917/PycharmProjects/ClearBias")
import numpy as np
import tensorflow as tf
import os
import itertools

# Disable eager execution for TF 1.x compatibility
tf.compat.v1.disable_eager_execution()

import sys, os
import copy
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import basinhopping

from adf_model.tutorial_models import dnn
from adf_utils.utils_tf import model_prediction, model_argmax
from adf_utils.config import census, credit, bank
from adf_tutorial.utils import cluster, gradient_graph

# step size of perturbation
perturbation_size = 1


def load_census_data():
    """Load Census Income dataset"""
    # Load the data
    data = pd.read_csv(HERE.joinpath('methods/adf/datasets/census'), header=None)

    # Encode categorical variables
    le = LabelEncoder()
    for i in range(len(data.columns)):
        if data[i].dtype == 'object':
            data[i] = le.fit_transform(data[i].astype(str))

    # Split features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    return X, y, (None, X.shape[1]), 2


def load_credit_data():
    """Load German Credit dataset"""
    # Load the data
    data = pd.read_csv('datasets/credit_sample', header=None)

    # Encode categorical variables
    le = LabelEncoder()
    for i in range(len(data.columns)):
        if data[i].dtype == 'object':
            data[i] = le.fit_transform(data[i].astype(str))

    # Split features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values - 1  # Convert to 0-based indexing

    return X, y, (X.shape[1],), 2


def load_bank_data():
    """Load Bank Marketing dataset"""
    # Load the data
    data = pd.read_csv('datasets/bank', header=None)

    # Encode categorical variables
    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col].astype(str))

    # Split features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    return X, y, (X.shape[1],), 2


def check_for_error_condition(ge: DiscriminationData, sess, x, preds, t, sens_indices):
    """
    Check whether the test case is an individual discriminatory instance
    :param ge: the DiscriminationData object
    :param sess: TF session
    :param x: input placeholder
    :param preds: the model's symbolic output
    :param t: test case
    :param sens_indices: list of indices of sensitive features
    :return: tuple (is_discriminatory, (original_features, original_outcome, counter_features, counter_outcome))
    """
    t = t.astype('int')
    # Reshape t to (1, -1) for model input
    t_reshaped = t.reshape(1, -1)
    label = model_argmax(sess, x, preds, t_reshaped)

    # For each sensitive feature
    for sens_idx in sens_indices:
        # Get the possible values for this sensitive feature
        unique_vals = np.unique(ge.xdf.iloc[:, sens_idx])

        # check for all the possible values of sensitive feature
        for val in unique_vals:
            if val != t[sens_idx]:
                # Create a copy of the test case
                t_changed = t.copy()
                t_changed[sens_idx] = val

                # Reshape for model input
                t_changed_reshaped = t_changed.reshape(1, -1)
                label_changed = model_argmax(sess, x, preds, t_changed_reshaped)

                # If we find any case where changing the sensitive feature changes the prediction,
                # this is a discriminatory instance
                if label_changed != label:
                    return True, (tuple(t), label, tuple(t_changed), label_changed)

    return False, None


def seed_test_input(dataset, cluster_num):
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
    clusters = sorted(clusters, key=lambda x: x.shape[0])  # len(clusters[0][0])==32561
    return clusters


def clip(input, ge: DiscriminationData):
    """
    Clip the generating instance with each feature to make sure it is valid
    :param input: generating instance
    :param ge: the DiscriminationData object
    :return: a valid generating instance
    """
    for i in range(len(input)):
        input[i] = max(input[i], ge.input_bounds[i][0])
        input[i] = min(input[i], ge.input_bounds[i][1])
    return input


class Local_Perturbation(object):
    """
    The implementation of local perturbation
    """

    def __init__(self, sess, grad, x, n_values, sensitive_indices, input_shape, ge: DiscriminationData):
        """
        Initial function of local perturbation
        :param sess: TF session
        :param grad: the gradient graph
        :param x: input placeholder
        :param n_values: dictionary mapping sensitive feature indices to their new values
        :param sensitive_indices: list of indices of sensitive features
        :param input_shape: the shape of dataset
        :param ge: the DiscriminationData object
        """
        self.sess = sess
        self.grad = grad
        self.x = x
        self.n_values = n_values
        self.input_shape = input_shape
        self.sensitive_indices = sensitive_indices
        self.ge = ge

    def __call__(self, x):
        """
        Local perturbation
        :param x: input instance for local perturbation
        :return: new potential individual discriminatory instance
        """
        # perturbation
        s = np.random.choice([1.0, -1.0]) * perturbation_size

        n_x = x.copy()
        for sens_idx, n_value in self.n_values.items():
            n_x[sens_idx - 1] = n_value

        # compute the gradients of an individual discriminatory instance pairs
        ind_grad = self.sess.run(self.grad, feed_dict={self.x: np.array([x])})
        n_ind_grad = self.sess.run(self.grad, feed_dict={self.x: np.array([n_x])})

        # Convert gradients to float arrays to avoid boolean operations
        ind_grad = np.array(ind_grad, dtype=np.float32)
        n_ind_grad = np.array(n_ind_grad, dtype=np.float32)
        zero_array = np.zeros(self.input_shape, dtype=np.float32)

        if np.array_equal(ind_grad[0], zero_array) and np.array_equal(n_ind_grad[0], zero_array):
            # Create probability distribution avoiding all sensitive features
            probs = 1.0 / (self.input_shape - len(self.sensitive_indices)) * np.ones(self.input_shape)
            for sens_idx in self.sensitive_indices:
                probs[sens_idx - 1] = 0
        else:
            # normalize the reciprocal of gradients (prefer the low impactful feature)
            grad_sum = 1.0 / (np.abs(ind_grad[0]) + np.abs(n_ind_grad[0]))
            for sens_idx in self.sensitive_indices:
                grad_sum[sens_idx - 1] = 0
            probs = grad_sum / np.sum(grad_sum)
        probs = probs / probs.sum()

        # randomly choose the feature for local perturbation
        available_indices = [i for i in range(self.input_shape) if i + 1 not in self.sensitive_indices]
        if available_indices:
            index = np.random.choice(available_indices, p=probs[available_indices])
            local_cal_grad = np.zeros(self.input_shape)
            local_cal_grad[index] = 1.0

            x = clip(x + s * local_cal_grad, self.ge).astype("int")

        return x


def dnn_fair_testing(ge: DiscriminationData, cluster_num, max_global, max_local, max_iter):
    """
    The implementation of ADF
    :param ge: DiscriminationData object containing dataset and metadata
    :param cluster_num: the number of clusters to form
    :param max_global: the maximum number of samples for global search
    :param max_local: the maximum number of samples for local search
    :param max_iter: the maximum iteration of global perturbation
    """
    # Get sensitive feature indices from ge
    sensitive_indices = ge.sensitive_indices
    if not isinstance(sensitive_indices, list):
        sensitive_indices = [sensitive_indices]

    dataset = "census"
    # prepare the testing data and model
    # X, Y, input_shape, nb_classes = data[dataset]()
    X, Y, input_shape, nb_classes = ge.xdf.to_numpy(), ge.ydf.to_numpy(), (None, ge.xdf.shape[1]), \
        ge.ydf.unique().shape[0]
    print("Input shape:", input_shape)  # Debug print
    tf.compat.v1.random.set_random_seed(1234)

    # Create a TensorFlow session with v1 compatibility
    sess = tf.compat.v1.Session()
    x = tf.compat.v1.placeholder(tf.float32, shape=input_shape)
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)
    preds = model(x)

    # Initialize all variables
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    saver = tf.compat.v1.train.Saver()

    # Try to load the model first
    model_path = HERE.joinpath("methods/adf/models/census/trained_model.model")
    if os.path.exists(model_path.with_suffix(".model.index")):
        print("Loading existing model from:", model_path)
        saver.restore(sess, str(model_path))
    else:
        print("No existing model found. Training a new model...")
        # Convert labels to one-hot encoding
        nb_classes = len(np.unique(Y))
        Y_one_hot = tf.keras.utils.to_categorical(Y, nb_classes)

        # Normalize input data
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # Avoid division by zero
        X_normalized = (X - X_mean) / X_std

        # Define training parameters
        batch_size = 32
        epochs = 10
        learning_rate = 0.01  # Increased learning rate

        # Define optimizer and loss
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        # Define training ops with proper variable scope
        with tf.compat.v1.variable_scope('training'):
            logits = model(x)  # Get logits from model
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
            train_op = optimizer.minimize(loss)

            # Add accuracy metric
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Training loop
        n_batches = int(np.ceil(len(X_normalized) / batch_size))

        # Initialize variables after model creation
        sess.run(tf.compat.v1.global_variables_initializer())

        # Training history
        history = {'loss': [], 'accuracy': []}

        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(len(X_normalized))
            X_shuffled = X_normalized[indices]
            Y_shuffled = Y_one_hot[indices]

            epoch_loss = 0
            epoch_acc = 0

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_normalized))

                batch_x = X_shuffled[start_idx:end_idx]
                batch_y = Y_shuffled[start_idx:end_idx]

                # Run training step and get metrics
                _, batch_loss, batch_acc = sess.run(
                    [train_op, loss, accuracy],
                    feed_dict={x: batch_x, y: batch_y}
                )

                epoch_loss += batch_loss
                epoch_acc += batch_acc

            # Calculate average epoch metrics
            epoch_loss /= n_batches
            epoch_acc /= n_batches

            # Store history
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

            # Early stopping if loss isn't changing
            if epoch > 0 and abs(history['loss'][-1] - history['loss'][-2]) < 1e-4:
                print("Loss has stopped decreasing. Early stopping...")
                break

        # Final evaluation on full dataset
        final_acc = sess.run(accuracy, feed_dict={x: X_normalized, y: Y_one_hot})
        print(f"\nFinal Test Accuracy: {final_acc:.4f}")

        # Save the trained model
        save_path = HERE.joinpath("methods/adf/models/census/")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_save_path = save_path.joinpath("trained_model.model")
        save_path = saver.save(sess, str(model_save_path))
        print("Model saved in path:", save_path)

    # construct the gradient graph
    grad_0 = gradient_graph(x, preds)

    # build the clustering model
    # clf = cluster(X, cluster_num)
    # clusters = [np.where(clf.labels_ == i) for i in range(cluster_num)]

    # store the result of fairness testing
    tot_inputs = set()
    global_disc_inputs = set()
    global_disc_inputs_list = []
    local_disc_inputs = set()
    local_disc_inputs_list = []
    value_list = []
    suc_idx = []
    discriminatory_pairs = []  # List to store (original_features, original_outcome, counter_features, counter_outcome)
    unique_disc_pairs = set()  # Set to track unique discriminatory pairs

    def evaluate_local(inp):
        """
        Evaluate whether the test input after local perturbation is an individual discriminatory instance
        :param inp: test input
        :return: float value indicating whether it is an individual discriminatory instance
        """
        # Ensure inp is a 1D array before passing to check_for_error_condition
        inp = inp.reshape(-1)
        result, disc_tuple = check_for_error_condition(ge, sess, x, preds, inp, ge.sensitive_indices.values())

        # Create temp without sensitive features for tracking
        temp = []
        for i in range(len(inp)):
            if i not in ge.sensitive_indices.values():
                temp.append(inp[i])
        temp = tuple(temp)

        tot_inputs.add(temp)
        if result and (temp not in global_disc_inputs) and (temp not in local_disc_inputs):
            local_disc_inputs.add(temp)
            local_disc_inputs_list.append(temp)
            if disc_tuple is not None:
                # Only add if this pair hasn't been seen before
                if disc_tuple not in unique_disc_pairs:
                    unique_disc_pairs.add(disc_tuple)
                    discriminatory_pairs.append(disc_tuple)

        return float(not result)

    # select the seed input for fairness testing
    clusters = seed_test_input(X, min(max_global, len(X)))

    for iter_num, cluster in enumerate(clusters):
        if iter_num > max_iter:
            break

        for index in cluster:
            sample = X[index:index + 1]

            probs = model_prediction(sess, x, preds, sample)[0]
            label = np.argmax(probs)
            prob = probs[label]
            max_diff = 0
            n_values = {}

            # Get all possible values for each sensitive attribute
            sensitive_values = {}
            for sens_name, sens_idx in ge.sensitive_indices.items():
                sensitive_values[sens_name] = np.unique(ge.xdf.iloc[:, sens_idx]).tolist()

            # Generate all possible combinations of sensitive attribute values
            sensitive_names = list(ge.sensitive_indices.keys())
            value_combinations = list(itertools.product(*[sensitive_values[name] for name in sensitive_names]))

            for values in value_combinations:
                # Skip if combination is identical to original
                if all(sample[0][ge.sensitive_indices[name]] == value for name, value in zip(sensitive_names, values)):
                    continue

                tnew = pd.DataFrame(sample, columns=ge.attr_columns)
                for name, value in zip(sensitive_names, values):
                    tnew[name] = value
                n_sample = tnew.to_numpy()
                n_probs = model_prediction(sess, x, preds, n_sample)[0]
                n_label = np.argmax(n_probs)
                n_prob = n_probs[n_label]
                print(sample, label, n_sample, n_label)
                if label != n_label:
                    for i, (name, value) in enumerate(zip(sensitive_names, values)):
                        n_values[ge.sensitive_indices[name]] = value
                    break
                else:
                    prob_diff = abs(prob - n_prob)
                    if prob_diff > max_diff:
                        max_diff = prob_diff
                        for i, (name, value) in enumerate(zip(sensitive_names, values)):
                            n_values[ge.sensitive_indices[name]] = value

            sample_key = copy.deepcopy(sample[0].astype('int').tolist())
            sample_key = [sample_key[i] for i in range(len(sample_key)) if i not in ge.sensitive_indices.values()]

            # if get an individual discriminatory instance
            if label != n_label and (tuple(sample_key) not in global_disc_inputs) and (
                    tuple(sample_key) not in local_disc_inputs):
                global_disc_inputs_list.append(sample_key)
                global_disc_inputs.add(tuple(sample_key))
                suc_idx.append(index)

                # start local perturbation
                minimizer = {"method": "L-BFGS-B"}
                local_perturbation = Local_Perturbation(sess, grad_0, x, n_values, ge.sensitive_indices.values(),
                                                        input_shape[1],
                                                        ge)
                basinhopping(evaluate_local, sample.flatten(), stepsize=1.0, take_step=local_perturbation,
                             minimizer_kwargs=minimizer,
                             niter=max_local)

                print(len(local_disc_inputs_list),
                      "Percentage discriminatory inputs of local search- " + str(
                          float(len(local_disc_inputs)) / float(len(tot_inputs)) * 100))
                break

        # global perturbation
        s_grad = sess.run(tf.sign(grad_0), feed_dict={x: sample})
        n_grad = sess.run(tf.sign(grad_0), feed_dict={x: n_sample})

        # find the feature with same impact
        if np.zeros(len(ge.attr_columns)).tolist() == s_grad[0].tolist():
            g_diff = n_grad[0]
        elif np.zeros(len(ge.attr_columns)).tolist() == n_grad[0].tolist():
            g_diff = s_grad[0]
        else:
            g_diff = np.array(s_grad[0] == n_grad[0], dtype=float)

        # Zero out all sensitive features
        for sens_idx in sensitive_indices[0].values():
            g_diff[sens_idx - 1] = 0

        if np.zeros(input_shape[1]).tolist() == g_diff.tolist():
            # Get all non-sensitive feature indices
            available_indices = [i for i in range(len(g_diff)) if i + 1 not in sensitive_indices]
            if available_indices:
                index = np.random.choice(available_indices)
                g_diff[index] = 1.0

        cal_grad = s_grad * g_diff
        sample[0] = clip(sample[0] + perturbation_size * cal_grad[0], ge).astype("int")

    # print the overview information of result
    print("Total Inputs are " + str(len(tot_inputs)))
    print("Total discriminatory inputs of global search- " + str(len(global_disc_inputs)))
    print("Total discriminatory inputs of local search- " + str(len(local_disc_inputs)))
    print("Total discriminatory pairs found- " + str(len(discriminatory_pairs)))

    return discriminatory_pairs


def main():
    ge, ge_schema = get_real_data('adult')

    dnn_fair_testing(ge=ge,
                     cluster_num=4,
                     max_global=1000,
                     max_local=1000,
                     max_iter=1000)


if __name__ == '__main__':
    main()
