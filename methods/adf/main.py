import copy
import os
from collections import deque

from sklearn.neural_network import MLPClassifier, MLPRegressor
from tqdm import tqdm

os.environ['PYTHONHASHSEED'] = '0'
import itertools
import time
import logging
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping
from methods.utils import train_sklearn_model
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearnex import patch_sklearn
from data_generator.main import DiscriminationData
from sklearn.neural_network._base import DERIVATIVES
from sklearn.utils.extmath import safe_sparse_dot

patch_sklearn()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('ADF')

# Set global random seed for numpy
np.random.seed(42)

# Force sklearn to use a single thread
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Configuration constants
MIN_PERTURBATION_SIZE = 0.5
MAX_PERTURBATION_SIZE = 2.0
MIN_EPSILON = 1e-6
MAX_EPSILON = 1e-4
MIN_DISTANCE_THRESHOLD = 0.05
MAX_DISTANCE_THRESHOLD = 0.2
PERTURBATION_SIZE = 1.0
MAX_SIMILAR_FEATURES = 0.9

# Previous solutions for diversity checking
previous_solutions = set()
feature_value_counts = {}


def seed_test_input(X: np.ndarray, cluster_num: int, random_seed: int = 42) -> List[np.ndarray]:
    """Select seed inputs for fairness testing.

    Args:
        X: Input data
        cluster_num: Number of clusters
        random_seed: Random seed for reproducibility

    Returns:
        List of seed inputs
    """
    # Set the random seed
    np.random.seed(random_seed)

    cluster_num = max(min(cluster_num, X.shape[0]), 10)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=cluster_num,
        random_state=random_seed,  # Use the provided seed directly
        n_init=20,
        init='k-means++',
    )
    kmeans.fit(X_scaled)

    clusters = [np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)]

    # Use RandomState instance for shuffling to ensure reproducibility
    clusters = sorted(clusters, key=lambda x: len(x), reverse=True)

    logger.info(f"Created {len(clusters)} clusters with sizes: {[len(c) for c in clusters]}")

    return clusters


def clip(input_data: np.ndarray, data_obj) -> np.ndarray:
    """Clip values to valid ranges.

    Args:
        input_data: Input to clip
        data_obj: Data object with bounds

    Returns:
        Clipped input
    """
    for i in range(len(input_data)):
        input_data[i] = max(input_data[i], data_obj.input_bounds[i][0])
        input_data[i] = min(input_data[i], data_obj.input_bounds[i][1])
    return input_data


def get_gradient(mlp, X):
    # Forward pass
    n_samples, n_features = X.shape
    activations = [X] + [None] * (mlp.n_layers_ - 1)
    activations = mlp._forward_pass(activations)

    # Backward pass
    deltas = [None] * (mlp.n_layers_ - 1)
    last = mlp.n_layers_ - 2

    # Output layer gradient
    if hasattr(mlp, "n_outputs_"):  # Classifier
        if mlp.n_outputs_ == 1:  # Binary classification
            deltas[last] = np.ones((n_samples, 1))
        else:  # Multiclass classification
            deltas[last] = np.ones((n_samples, mlp.n_outputs_))
    else:  # Regressor
        deltas[last] = np.ones((n_samples, mlp.hidden_layer_sizes[-1]))

    # Backpropagate through hidden layers
    inplace_derivative = DERIVATIVES[mlp.activation]
    for i in range(mlp.n_layers_ - 2, 0, -1):
        deltas[i - 1] = safe_sparse_dot(deltas[i], mlp.coefs_[i].T)
        inplace_derivative(activations[i], deltas[i - 1])

    # Compute the input gradient
    first_layer = 0
    input_gradient = safe_sparse_dot(deltas[first_layer], mlp.coefs_[first_layer].T)

    return input_gradient


def global_perturbation(model, x, protected_indices, input_bounds, step_size=0.1):
    """
    Perturb the input x to generate a new discriminatory input.

    Args:
        model: The trained model.
        x: The current input instance.
        protected_indices: Indices of the protected attributes.
        input_bounds: Bounds for each feature in the input.
        step_size: The step size for perturbation.

    Returns:
        The perturbed input x.
    """
    # Step 1: Generate perturbed instances by varying protected attributes
    X_perturbed = []
    for i in protected_indices:
        for value in range(int(input_bounds[i][0]), int(input_bounds[i][1]) + 1):
            if value != x[i]:
                x_perturbed = x.copy()
                x_perturbed[i] = value
                X_perturbed.append(x_perturbed)

    # Step 2: Select the instance x' that maximizes the absolute difference in predictions
    max_diff = -1
    x_prime = None
    for x_p in X_perturbed:
        pred_x = model.predict_proba([x])[0]
        pred_x_p = model.predict_proba([x_p])[0]
        diff = np.abs(pred_x_p - pred_x).max()
        if diff > max_diff:
            max_diff = diff
            x_prime = x_p

    if x_prime is None:
        return x

    # Step 3: Compute gradients of the model's loss function with respect to x and x'
    if isinstance(model, (MLPClassifier, MLPRegressor)):
        # Use get_gradient for MLPClassifier and MLPRegressor
        grad_x = get_gradient(model, x.reshape(1, -1)).flatten()
        grad_x_prime = get_gradient(model, x_prime.reshape(1, -1)).flatten()
    else:
        # Fallback to finite difference method for other models
        eps = 1e-6
        grad_x = np.zeros_like(x)
        grad_x_prime = np.zeros_like(x)

        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps

            pred_plus = model.predict_proba([x_plus])[0]
            pred_minus = model.predict_proba([x_minus])[0]
            grad_x[i] = np.max(pred_plus - pred_minus) / (2 * eps)

            x_prime_plus = x_prime.copy()
            x_prime_plus[i] += eps
            x_prime_minus = x_prime.copy()
            x_prime_minus[i] -= eps

            pred_plus = model.predict_proba([x_prime_plus])[0]
            pred_minus = model.predict_proba([x_prime_minus])[0]
            grad_x_prime[i] = np.max(pred_plus - pred_minus) / (2 * eps)

    # Step 4: Determine the direction of perturbation based on the sign of the gradients
    dir_perturb = np.zeros_like(x)
    for i in range(len(x)):
        if i not in protected_indices:  # Only perturb non-protected attributes
            if np.sign(grad_x[i]) == np.sign(grad_x_prime[i]):
                dir_perturb[i] = np.sign(grad_x[i])

    # Step 5: Update x by adding the perturbation direction scaled by the step size
    x_new = x + dir_perturb * step_size

    # Step 6: Clip the updated x to ensure it stays within valid bounds
    x_new = np.clip(x_new, [b[0] for b in input_bounds], [b[1] for b in input_bounds])

    return x_new


class LocalPerturbation:
    def __init__(self, model, protected_indices, input_bounds, step_size=0.1):
        self.model = model
        self.protected_indices = protected_indices
        self.input_bounds = input_bounds
        self.step_size = step_size

    def __call__(self, x: np.ndarray):
        """
        Perturb the input x to generate a new discriminatory input in the local search phase.

        Args:
            model: The trained model.
            x: The current input instance.
            protected_indices: Indices of the protected attributes.
            input_bounds: Bounds for each feature in the input.
            step_size: The step size for perturbation.

        Returns:
            The perturbed input x.
        """
        # Step 1: Compute gradients of the model's loss function with respect to x
        if isinstance(self.model, (MLPClassifier, MLPRegressor)):
            # Use get_gradient for MLPClassifier and MLPRegressor
            grad_x = get_gradient(self.model, x.reshape(1, -1)).flatten()
        else:
            # Fallback to finite difference method for other models
            eps = 1e-6
            grad_x = np.zeros_like(x)

            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps

                pred_plus = self.model.predict_proba([x_plus])[0]
                pred_minus = self.model.predict_proba([x_minus])[0]
                grad_x[i] = np.max(pred_plus - pred_minus) / (2 * eps)

        # Step 2: Normalize the gradients to get perturbation probabilities
        saliency = np.abs(grad_x)
        saliency[self.protected_indices] = 0  # Do not perturb protected attributes
        saliency_sum = np.sum(saliency)
        if saliency_sum == 0:
            return x  # No valid attributes to perturb

        probabilities = saliency / saliency_sum

        # Step 3: Select an attribute to perturb based on the probabilities
        selected_attr = np.random.choice(len(x), p=probabilities)

        # Step 4: Randomly choose the direction of perturbation
        direction = np.random.choice([-1, 1])

        # Step 5: Update the selected attribute by adding the perturbation direction scaled by the step size
        x_new = x.copy()
        x_new[selected_attr] += direction * self.step_size

        # Step 6: Clip the updated x to ensure it stays within valid bounds
        x_new = np.clip(x_new, [b[0] for b in self.input_bounds], [b[1] for b in self.input_bounds])

        return x_new


def adf_fairness_testing(
        discrimination_data: DiscriminationData,
        max_global: int = 2000,
        max_local: int = 2000,
        cluster_num: int = 10,
        random_seed: int = 42,
        max_runtime_seconds: int = 3600,  # Default 1 hour time limit
        max_tsn: int = None,  # New parameter for TSN threshold
        step_size: float = 0.4,
        one_attr_at_a_time: bool = False
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Implementation of ADF fairness testing.

    Args:
        discrimination_data: Data object containing dataset and metadata
        max_global: Maximum samples for global search
        max_local: Maximum samples for local search
        max_iter: Maximum iterations for global perturbation
        cluster_num: Number of clusters
        random_seed: Random seed for reproducibility
        max_runtime_seconds: Maximum runtime in seconds before early termination
        max_tsn: Maximum number of test samples (TSN) to generate before termination

    Returns:
        Results DataFrame and metrics dictionary
    """

    # Set both numpy and random seeds
    np.random.seed(random_seed)

    logger = logging.getLogger("ADF")

    start_time = time.time()

    dsn_by_attr_value = {e: {'TSN': 0, 'DSN': 0} for e in discrimination_data.protected_attributes}
    dsn_by_attr_value['total'] = 0

    data = discrimination_data.training_dataframe.copy()

    logger.info(f"Dataset shape: {data.shape}")
    logger.info(f"Protected attributes: {discrimination_data.protected_attributes}")
    logger.info(f"Time limit: {max_runtime_seconds} seconds")
    if max_tsn:
        logger.info(f"Target TSN: {max_tsn}")

    model, X_train, X_test, y_train, y_test, feature_names = train_sklearn_model(
        data=data,
        model_type='mlp',
        target_col=discrimination_data.outcome_column,
        sensitive_attrs=list(discrimination_data.protected_attributes),
        random_state=random_seed
    )

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    logger.info(f"Model training score: {train_score:.4f}")
    logger.info(f"Model test score: {test_score:.4f}")

    X = discrimination_data.xdf.to_numpy()

    logger.info(f"Testing data shape: {X.shape}")

    global_disc_inputs = set()
    local_disc_inputs = set()
    tot_inputs = set()
    all_discriminations = set()

    def should_terminate() -> bool:
        current_runtime = time.time() - start_time
        time_limit_exceeded = current_runtime > max_runtime_seconds
        tsn_threshold_reached = len(tot_inputs) >= max_tsn if max_tsn is not None else False

        return time_limit_exceeded or tsn_threshold_reached

    def check_for_error_condition(model, instance, protected_indices, input_bounds, tot_inputs, all_discriminations,
                                  one_attr_at_a_time=False):
        # Ensure instance is integer and within bounds
        for i, (low, high) in enumerate(input_bounds):
            instance[i] = max(int(low), min(int(high), instance[i]))

        # Convert to DataFrame for prediction
        instance = pd.DataFrame([instance], columns=discrimination_data.attr_columns)

        # Get original prediction
        label = model.predict(instance)[0]

        new_df = []

        if one_attr_at_a_time:
            # Vary one attribute at a time
            for i, attr_idx in enumerate(protected_indices):
                attr_name = discrimination_data.protected_attributes[i]
                current_value = instance[attr_name].values[0]

                # Get all possible values for this attribute
                values = range(int(input_bounds[attr_idx][0]), int(input_bounds[attr_idx][1]) + 1)

                # Create variants with different values for this attribute only
                for value in values:
                    if int(current_value) == value:
                        continue

                    new_instance = instance.copy()
                    new_instance[attr_name] = value
                    new_df.append(new_instance)
                    dsn_by_attr_value[attr_name]['TSN'] += 1
        else:
            # Generate all possible combinations of protected attribute values
            protected_values = []
            for idx in protected_indices:
                values = range(int(input_bounds[idx][0]), int(input_bounds[idx][1]) + 1)
                protected_values.append(list(values))

            # Create variants with all combinations of protected attributes
            for values in itertools.product(*protected_values):
                if tuple(instance[discrimination_data.protected_attributes].values[0]) != values:
                    new_instance = instance.copy()
                    for i, attr in enumerate(discrimination_data.protected_attributes):
                        new_instance[attr] = values[i]
                        dsn_by_attr_value[attr]['TSN'] += 1
                    new_df.append(new_instance)

        if not new_df:  # If no combinations were found
            return False

        new_df = pd.concat(new_df)
        new_predictions = model.predict(new_df)
        new_df['outcome'] = new_predictions

        # Add to total inputs
        # for row in new_df.to_numpy():
        #     tot_inputs.add(tuple(row.astype(int)))
        inst_key = tuple(instance.values[0].astype(int))
        if inst_key not in tot_inputs:
            tot_inputs.add(inst_key)

        # Find discriminatory instances (different outcome)
        discrimination_df = new_df[new_df['outcome'] != label]

        tsn = len(tot_inputs)
        dsn = len(all_discriminations)
        sur = dsn / tsn if tsn > 0 else 0
        logger.info(f"Current Metrics - TSN: {tsn}, DSN: {dsn}, SUR: {sur:.4f}")

        # Record discriminatory pairs and update attribute value counts
        for _, row in discrimination_df.iterrows():
            # Create the discrimination pair tuple
            disc_pair = (tuple(instance.values[0].astype(int)), int(label),
                         tuple(row[discrimination_data.attr_columns].astype(int)), int(row['outcome']))

            # Only count if this is a new discrimination
            if disc_pair not in all_discriminations:
                all_discriminations.add(disc_pair)

                n_inp = pd.DataFrame(np.expand_dims(disc_pair[0], 0), columns=discrimination_data.attr_columns)
                n_counter = pd.DataFrame(np.expand_dims(disc_pair[2], 0), columns=discrimination_data.attr_columns)

                # Update counts for each protected attribute value in both original and variant
                for i, attr in enumerate(discrimination_data.protected_attributes):
                    if n_inp[attr].iloc[0] != n_counter[attr].iloc[0]:
                        dsn_by_attr_value[attr]['DSN'] += 1
                        dsn_by_attr_value['total'] += 1

        return discrimination_df.shape[0] > 0, discrimination_df

    # GLOBAL SEARCH
    clusters = seed_test_input(X, cluster_num, random_seed=random_seed)

    def round_robin_clusters(clusters):
        queues = [deque(cluster) for cluster in clusters]
        iter_cycle = itertools.cycle(queues)  # Create a cycle over non-empty queues

        while queues:
            queue = next(iter_cycle)  # Get the next queue
            if queue:
                yield queue.popleft()  # Pop from front
                if not queue:
                    queues.remove(queue)  # Remove empty queues
                    iter_cycle = itertools.cycle(queues)

    for inst_num, instance_id in enumerate(round_robin_clusters(clusters)):
        instance = X[instance_id]
        if inst_num > max_global or should_terminate():
            break

        for glob_iter in range(max_global):
            result, result_df = check_for_error_condition(model, instance,
                                                          discrimination_data.sensitive_indices,
                                                          discrimination_data.input_bounds, tot_inputs,
                                                          all_discriminations,
                                                          one_attr_at_a_time=one_attr_at_a_time)

            if result:
                global_disc_inputs.add(tuple(instance.astype(float).tolist()))
                break
            else:
                last_instance = copy.deepcopy(instance)
                instance = global_perturbation(model, instance, discrimination_data.sensitive_indices,
                                               discrimination_data.input_bounds, step_size=step_size)
                if (instance == last_instance).all():
                    # the pertubations will lead to no where better to skip this instance
                    break

    def evaluate_local(inp):
        is_discriminatory, _ = check_for_error_condition(model, inp, discrimination_data.sensitive_indices,
                                                         discrimination_data.input_bounds, tot_inputs,
                                                         all_discriminations, one_attr_at_a_time=one_attr_at_a_time)

        if is_discriminatory:
            local_disc_inputs.add(tuple(inp.astype(float).tolist()))
        return float(not is_discriminatory)

    # LOCAL SEARCH
    for glob_num, global_inp in enumerate(global_disc_inputs):
        if glob_num > max_local:
            break

        local_perturbation = LocalPerturbation(model, discrimination_data.sensitive_indices,
                                               discrimination_data.input_bounds, step_size=step_size)

        minimizer = {"method": "L-BFGS-B"}

        try:
            basinhopping(evaluate_local, np.array(global_inp),
                         stepsize=step_size, take_step=local_perturbation,
                         minimizer_kwargs=minimizer, niter=max_local,
                         accept_test=lambda: not should_terminate())
        except:
            break

    # Calculate final results
    end_time = time.time()
    total_time = end_time - start_time

    # Log final results
    tsn = len(tot_inputs)  # Total Sample Number
    dsn = len(all_discriminations)  # Discriminatory Sample Number
    sur = dsn / tsn if tsn > 0 else 0  # Success Rate
    dss = total_time / dsn if dsn > 0 else float('inf')  # Discriminatory Sample Search time

    for k, v in dsn_by_attr_value.items():
        if k != 'total':
            dsn_by_attr_value[k]['SUR'] = dsn_by_attr_value[k]['DSN'] / dsn_by_attr_value[k]['TSN']
            dsn_by_attr_value[k]['DSS'] = dss

    # Log dsn_by_attr_value counts
    logger.info("\nDiscrimination counts by protected attribute values:")
    metrics = {
        'TSN': tsn,
        'DSN': dsn,
        'SUR': sur,
        'DSS': dss,
        'total_time': total_time,
        'time_limit_reached': max_runtime_seconds is not None and total_time >= max_runtime_seconds,
        'max_tsn_reached': max_tsn is not None and tsn >= max_tsn,
        'dsn_by_attr_value': dsn_by_attr_value
    }

    logger.info("\nFinal Results:")
    logger.info(f"Total inputs tested: {tsn}")
    logger.info(f"Global discriminatory inputs: {len(global_disc_inputs)}")
    logger.info(f"Local discriminatory inputs: {len(local_disc_inputs)}")
    logger.info(f"Total discriminatory pairs: {dsn}")
    logger.info(f"Success rate (SUR): {sur:.4f}")
    logger.info(f"Avg. search time per discriminatory sample (DSS): {dss:.4f} seconds")
    logger.info(f"Discrimination by attribute value: {dsn_by_attr_value}")
    logger.info(f"Total time: {total_time:.2f} seconds")

    # Generate result dataframe
    res_df = []
    case_id = 0
    for org, org_res, counter_org, counter_org_res in all_discriminations:
        indv1 = pd.DataFrame([list(org)], columns=discrimination_data.attr_columns)
        indv2 = pd.DataFrame([list(counter_org)], columns=discrimination_data.attr_columns)

        indv_key1 = "|".join(str(x) for x in indv1[discrimination_data.attr_columns].iloc[0])
        indv_key2 = "|".join(str(x) for x in indv2[discrimination_data.attr_columns].iloc[0])

        # Add the additional columns
        indv1['indv_key'] = indv_key1
        indv1['outcome'] = org_res
        indv2['indv_key'] = indv_key2
        indv2['outcome'] = counter_org_res

        # Create couple_key
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

    # Add metrics to result dataframe
    res_df['TSN'] = tsn
    res_df['DSN'] = dsn
    res_df['SUR'] = sur
    res_df['DSS'] = dss

    return res_df, metrics


if __name__ == "__main__":
    from data_generator.main import get_real_data, generate_from_real_data, DiscriminationData

    data_obj, schema = get_real_data('adult', use_cache=True)

    results_df, metrics = adf_fairness_testing(
        data_obj,
        max_global=20000,
        max_local=100,
        cluster_num=50,
        max_runtime_seconds=3600,  # 1 hour time limit
        max_tsn=20000,  # Set target TSN threshold
        step_size=0.05,
        one_attr_at_a_time=True
    )

    print(f"\nTesting Metrics: {metrics}")
