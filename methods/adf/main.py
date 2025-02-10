import copy
import itertools
import time
import logging
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import tensorflow as tf
import pandas as pd
from scipy.optimize import basinhopping
import os

from data_generator.main import get_real_data, DiscriminationData
from path import HERE
from adf_model.tutorial_models import dnn
from methods.adf.utils_tf import model_prediction, model_argmax, cluster, gradient_graph

# Disable eager execution for TF 1.x compatibility
tf.compat.v1.disable_eager_execution()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('ADF')

# Configuration constants
PERTURBATION_SIZE = 1.0


def check_for_error_condition(
        ge: DiscriminationData,
        sess: tf.compat.v1.Session,
        x: tf.Tensor,
        preds: tf.Tensor,
        t: np.ndarray,
        sens_indices: List[int]
) -> Tuple[bool, Optional[Tuple[tuple, int, tuple, int]]]:
    """Check whether the test case is an individual discriminatory instance.
    
    Args:
        ge (DiscriminationData): The discrimination data object
        sess (tf.Session): TensorFlow session
        x (tf.Tensor): Input placeholder
        preds (tf.Tensor): Model's symbolic output
        t (np.ndarray): Test case
        sens_indices (List[int]): List of indices of sensitive features
        
    Returns:
        tuple: Contains:
            - is_discriminatory (bool): Whether instance is discriminatory
            - discrimination_info (tuple): Details about discrimination if found, else None
    """
    t = t.astype('int')
    t_reshaped = t.reshape(1, -1)
    label = model_argmax(sess, x, preds, t_reshaped)

    for sens_idx in sens_indices:
        unique_vals = np.unique(ge.xdf.iloc[:, sens_idx])

        for val in unique_vals:
            if val != t[sens_idx]:
                t_changed = t.copy()
                t_changed[sens_idx] = val
                t_changed_reshaped = t_changed.reshape(1, -1)
                label_changed = model_argmax(sess, x, preds, t_changed_reshaped)

                if label_changed != label:
                    return True, (tuple(t), label, tuple(t_changed), label_changed)

    return False, None


def seed_test_input(dataset: str, cluster_num: int) -> List[np.ndarray]:
    """Select the seed inputs for fairness testing.
    
    Args:
        dataset (str): The name of dataset
        cluster_num (int): The number of clusters to form
        
    Returns:
        List[np.ndarray]: A sequence of seed inputs
    """
    clf = cluster(dataset, cluster_num)
    clusters = [np.where(clf.labels_ == i)[0] for i in range(cluster_num)]
    clusters = sorted(clusters, key=lambda x: x.shape[0])  # len(clusters[0][0])==32561
    return clusters


def clip(input: np.ndarray, ge: DiscriminationData) -> np.ndarray:
    """Clip the generating instance with each feature to make sure it is valid.
    
    Args:
        input (np.ndarray): Generating instance
        ge (DiscriminationData): The discrimination data object
        
    Returns:
        np.ndarray: A valid generating instance
    """
    for i in range(len(input)):
        input[i] = max(input[i], ge.input_bounds[i][0])
        input[i] = min(input[i], ge.input_bounds[i][1])
    return input


class LocalPerturbation:
    """Implementation of local perturbation strategy."""

    def __init__(
            self,
            sess: tf.compat.v1.Session,
            grad: tf.Tensor,
            x: tf.Tensor,
            n_values: Dict[int, Any],
            sensitive_indices: List[int],
            input_shape: int,
            ge: DiscriminationData
    ):
        """Initialize local perturbation.
        
        Args:
            sess (tf.Session): TensorFlow session
            grad (tf.Tensor): Gradient graph
            x (tf.Tensor): Input placeholder
            n_values (Dict[int, Any]): Dictionary mapping sensitive feature indices to new values
            sensitive_indices (List[int]): List of indices of sensitive features
            input_shape (int): Shape of input data
            ge (DiscriminationData): Discrimination data object
        """
        self.sess = sess
        self.grad = grad
        self.x = x
        self.n_values = n_values
        self.input_shape = input_shape
        self.sensitive_indices = sensitive_indices
        self.ge = ge

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Perform local perturbation on input instance.
        
        Args:
            x (np.ndarray): Input instance for local perturbation
            
        Returns:
            np.ndarray: New potential individual discriminatory instance
        """
        s = np.random.choice([1.0, -1.0]) * PERTURBATION_SIZE

        n_x = x.copy()
        for sens_idx, n_value in self.n_values.items():
            n_x[sens_idx - 1] = n_value

        ind_grad = self.sess.run(self.grad, feed_dict={self.x: np.array([x])})
        n_ind_grad = self.sess.run(self.grad, feed_dict={self.x: np.array([n_x])})

        ind_grad = np.array(ind_grad, dtype=np.float32)
        n_ind_grad = np.array(n_ind_grad, dtype=np.float32)
        zero_array = np.zeros(self.input_shape, dtype=np.float32)

        if np.array_equal(ind_grad[0], zero_array) and np.array_equal(n_ind_grad[0], zero_array):
            probs = 1.0 / (self.input_shape - len(self.sensitive_indices)) * np.ones(self.input_shape)
            for sens_idx in self.sensitive_indices:
                probs[sens_idx - 1] = 0
        else:
            grad_sum = 1.0 / (np.abs(ind_grad[0]) + np.abs(n_ind_grad[0]))
            for sens_idx in self.sensitive_indices:
                grad_sum[sens_idx - 1] = 0
            probs = grad_sum / np.sum(grad_sum)
        probs = probs / probs.sum()

        available_indices = [i for i in range(self.input_shape) if i + 1 not in self.sensitive_indices]
        if available_indices:
            index = np.random.choice(available_indices, p=probs[available_indices])
            local_cal_grad = np.zeros(self.input_shape)
            local_cal_grad[index] = 1.0

            x = clip(x + s * local_cal_grad, self.ge).astype("int")

        return x


def dnn_fair_testing(
        ge: DiscriminationData,
        max_tsn: int,
        max_global: int,
        max_local: int,
        max_iter: int
) -> [pd.DataFrame, Dict[str, float]]:
    """The implementation of ADF.

    Args:
        ge (DiscriminationData): DiscriminationData object containing dataset and metadata
        max_tsn (int): The number of clusters to form
        max_global (int): The maximum number of samples for global search
        max_local (int): The maximum number of samples for local search
        max_iter (int): The maximum iteration of global perturbation

    Returns:
        Tuple containing:
            - List[Tuple[tuple, int, tuple, int]]: List of discriminatory pairs
            - Dict[str, float]: Dictionary containing the following metrics:
                - TSN: Total Sample Number
                - DSN: Discriminatory Sample Number
                - SUR: Success Rate (DSN/TSN)
                - DSS: Discriminatory Sample Search time (Time/DSN)
    """
    X, Y, input_shape, nb_classes = ge.xdf.to_numpy(), ge.ydf.to_numpy(), (None, ge.xdf.shape[1]), \
        ge.ydf.unique().shape[0]
    logger.info(f"Input shape: {input_shape}")
    tf.compat.v1.random.set_random_seed(1234)

    sess = tf.compat.v1.Session()
    x = tf.compat.v1.placeholder(tf.float32, shape=input_shape)
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)
    preds = model(x)

    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    # saver = tf.compat.v1.train.Saver()

    # model_path = HERE.joinpath("methods/adf/models/census/trained_model.model")
    # if os.path.exists(model_path.with_suffix(".model.index")):
    #     logger.info(f"Loading existing model from: {model_path}")
    #     saver.restore(sess, str(model_path))
    # else:
    logger.info("No existing model found. Training a new model...")
    Y_one_hot = tf.keras.utils.to_categorical(Y, nb_classes)

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero
    X_normalized = (X - X_mean) / X_std

    batch_size = 32
    epochs = 10
    learning_rate = 0.01  # Increased learning rate

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    with tf.compat.v1.variable_scope('training'):
        logits = model(x)  # Get logits from model
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
        train_op = optimizer.minimize(loss)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    n_batches = int(np.ceil(len(X_normalized) / batch_size))

    sess.run(tf.compat.v1.global_variables_initializer())

    history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
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

            _, batch_loss, batch_acc = sess.run(
                [train_op, loss, accuracy],
                feed_dict={x: batch_x, y: batch_y}
            )

            epoch_loss += batch_loss
            epoch_acc += batch_acc

        epoch_loss /= n_batches
        epoch_acc /= n_batches

        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        if epoch > 0 and abs(history['loss'][-1] - history['loss'][-2]) < 1e-4:
            logger.info("Loss has stopped decreasing. Early stopping...")
            break

    final_acc = sess.run(accuracy, feed_dict={x: X_normalized, y: Y_one_hot})
    logger.info(f"Final Test Accuracy: {final_acc:.4f}")

    # save_path = HERE.joinpath("methods/adf/models/census/")
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # model_save_path = save_path.joinpath("trained_model.model")
    # save_path = saver.save(sess, str(model_save_path))
    # logger.info(f"Model saved in path: {save_path}")

    grad_0 = gradient_graph(x, preds)

    tot_inputs = set()
    global_disc_inputs = set()
    global_disc_inputs_list = []
    local_disc_inputs = set()
    local_disc_inputs_list = []
    value_list = []
    suc_idx = []
    discriminatory_pairs = []
    unique_disc_pairs = set()

    start_time = time.time()

    def evaluate_local(inp):
        result, disc_tuple = check_for_error_condition(ge, sess, x, preds, inp, ge.sensitive_indices.values())

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
                if disc_tuple not in unique_disc_pairs:
                    unique_disc_pairs.add(disc_tuple)
                    discriminatory_pairs.append(disc_tuple)

        current_dsn = len(global_disc_inputs) + len(local_disc_inputs)
        logger.info(
            f"[Real-time Metrics] TSN: {len(tot_inputs)} DSN: {current_dsn}")
        return float(not result)

    clusters = seed_test_input(X, min(max_global, len(X)))

    for iter_num, cluster in enumerate(clusters):
        if iter_num > max_iter or len(tot_inputs) > max_tsn:
            break

        for index in cluster:
            if len(tot_inputs) > max_tsn:
                break

            sample = X[index:index + 1]

            probs = model_prediction(sess, x, preds, sample)[0]
            label = np.argmax(probs)
            prob = probs[label]
            max_diff = 0
            n_values = {}

            sensitive_values = {}
            for sens_name, sens_idx in ge.sensitive_indices.items():
                sensitive_values[sens_name] = np.unique(ge.xdf.iloc[:, sens_idx]).tolist()

            value_combinations = list(itertools.product(*[sensitive_values[name] for name in sensitive_values.keys()]))

            for values in value_combinations:
                if all(sample[0][ge.sensitive_indices[name]] == value for name, value in
                       zip(sensitive_values.keys(), values)):
                    continue

                tnew = pd.DataFrame(sample, columns=ge.attr_columns)
                for name, value in zip(sensitive_values.keys(), values):
                    tnew[name] = value
                n_sample = tnew.to_numpy()
                n_probs = model_prediction(sess, x, preds, n_sample)[0]
                n_label = np.argmax(n_probs)
                n_prob = n_probs[n_label]
                logger.debug(
                    f"Sample comparison - Original: {sample}, Label: {label}, New: {n_sample}, New Label: {n_label}")
                if label != n_label:
                    for i, (name, value) in enumerate(zip(sensitive_values.keys(), values)):
                        n_values[ge.sensitive_indices[name]] = value
                    break
                else:
                    prob_diff = abs(prob - n_prob)
                    if prob_diff > max_diff:
                        max_diff = prob_diff
                        for i, (name, value) in enumerate(zip(sensitive_values.keys(), values)):
                            n_values[ge.sensitive_indices[name]] = value

            sample_key = copy.deepcopy(sample[0].astype('int').tolist())
            sample_key = [sample_key[i] for i in range(len(sample_key)) if i not in ge.sensitive_indices.values()]

            if label != n_label and (tuple(sample_key) not in global_disc_inputs) and (
                    tuple(sample_key) not in local_disc_inputs):
                global_disc_inputs_list.append(sample_key)
                global_disc_inputs.add(tuple(sample_key))
                suc_idx.append(index)
                current_dsn = len(global_disc_inputs) + len(local_disc_inputs)
                logger.info(
                    f"[Real-time Metrics] DSN: {current_dsn} (Global: {len(global_disc_inputs)}, Local: {len(local_disc_inputs)})")
                minimizer = {"method": "L-BFGS-B"}
                local_perturbation = LocalPerturbation(sess, grad_0, x, n_values, ge.sensitive_indices.values(),
                                                       input_shape[1], ge)
                basinhopping(evaluate_local, sample.flatten(), stepsize=1.0, take_step=local_perturbation,
                             minimizer_kwargs=minimizer, niter=max_local)

                logger.info(f"Local search discriminatory inputs: {len(local_disc_inputs_list)}")
                logger.info(
                    f"Percentage discriminatory inputs of local search: {float(len(local_disc_inputs)) / float(len(tot_inputs)) * 100:.2f}%")
                break

        s_grad = sess.run(tf.sign(grad_0), feed_dict={x: sample})
        n_grad = sess.run(tf.sign(grad_0), feed_dict={x: n_sample})

        g_diff = np.array(s_grad[0] == n_grad[0], dtype=float)

        for sens_idx in ge.sensitive_indices.values():
            g_diff[sens_idx - 1] = 0

        if np.zeros(input_shape[1]).tolist() == g_diff.tolist():
            available_indices = [i for i in range(len(g_diff)) if i + 1 not in ge.sensitive_indices.values()]
            if available_indices:
                index = np.random.choice(available_indices)
                g_diff[index] = 1.0

        cal_grad = s_grad * g_diff
        sample[0] = clip(sample[0] + PERTURBATION_SIZE * cal_grad[0], ge).astype("int")

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate metrics
    tsn = len(tot_inputs)  # Total Sample Number
    dsn = len(global_disc_inputs) + len(local_disc_inputs)  # Discriminatory Sample Number
    sur = dsn / tsn if tsn > 0 else 0  # Success Rate
    dss = total_time / dsn if dsn > 0 else float('inf')  # Discriminatory Sample Search time

    metrics = {
        'TSN': tsn,
        'DSN': dsn,
        'SUR': sur,
        'DSS': dss
    }

    logger.info(f"Total Inputs: {len(tot_inputs)}")
    logger.info(f"Global Search Discriminatory Inputs: {len(global_disc_inputs)}")
    logger.info(f"Local Search Discriminatory Inputs: {len(local_disc_inputs)}")
    logger.info(f"Success Rate (SUR): {metrics['SUR']:.4f}")
    logger.info(f"Average Search Time per Discriminatory Sample (DSS): {metrics['DSS']:.4f} seconds")
    logger.info(f"Total Discriminatory Pairs Found: {len(discriminatory_pairs)}")

    res_df = []
    case_id = 0
    for org, org_outcome, counter_org, counter_org_outcome in discriminatory_pairs:
        indv1 = pd.DataFrame([list(org)], columns=ge.attr_columns)
        indv2 = pd.DataFrame([list(counter_org)], columns=ge.attr_columns)

        indv_key1 = "|".join(str(x) for x in indv1[ge.attr_columns].iloc[0])
        indv_key2 = "|".join(str(x) for x in indv2[ge.attr_columns].iloc[0])

        # Add the additional columns
        indv1['indv_key'] = indv_key1
        indv1['outcome'] = int(org_outcome)
        indv2['indv_key'] = indv_key2
        indv2['outcome'] = int(counter_org_outcome)

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

    res_df['TSN'] = tsn
    res_df['DSN'] = dsn
    res_df['SUR'] = sur
    res_df['DSS'] = dss

    return res_df, metrics


def main():
    ge, ge_schema = get_real_data('adult')

    discriminatory_df, metrics = dnn_fair_testing(ge=ge, max_tsn=500, max_global=1000,
                                                  max_local=100, max_iter=1000)

    print(discriminatory_df)


if __name__ == '__main__':
    main()
