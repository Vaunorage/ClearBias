import os
import random
from collections import deque

from sklearn.neural_network import MLPClassifier, MLPRegressor

import itertools
import time
import logging
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping
from methods.utils import train_sklearn_model, check_for_error_condition, make_final_metrics_and_dataframe
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data_generator.main import DiscriminationData
from sklearn.neural_network._base import DERIVATIVES
from sklearn.utils.extmath import safe_sparse_dot

# patch_sklearn()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('ADF')

# Set global random seed for numpy

# Force sklearn to use a single thread
os.environ['PYTHONHASHSEED'] = '0'
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


def run_adf(data: DiscriminationData, max_global: int = 2000, max_local: int = 2000, cluster_num: int = 10,
            random_seed: int = 42, max_runtime_seconds: int = 3600, max_tsn: int = None,
            step_size: float = 0.4, one_attr_at_a_time: bool = False, db_path=None, analysis_id=None, use_cache=True) -> \
        Tuple[pd.DataFrame, Dict[str, float]]:
    """Implementation of ADF fairness testing.

    Args:
        data: Data object containing dataset and metadata
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
    random.seed(random_seed)

    logger = logging.getLogger("ADF")

    start_time = time.time()

    dsn_by_attr_value = {e: {'TSN': 0, 'DSN': 0} for e in data.protected_attributes}
    dsn_by_attr_value['total'] = 0

    logger.info(f"Dataset shape: {data.dataframe.shape}")
    logger.info(f"Protected attributes: {data.protected_attributes}")
    logger.info(f"Time limit: {max_runtime_seconds} seconds")
    if max_tsn:
        logger.info(f"Target TSN: {max_tsn}")

    model, X_train, X_test, y_train, y_test, feature_names, metrics = train_sklearn_model(
        data=data.training_dataframe.copy(),
        model_type='mlp',
        target_col=data.outcome_column,
        sensitive_attrs=list(data.protected_attributes),
        random_state=random_seed,
        use_cache=use_cache
    )

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    logger.info(f"Model training score: {train_score:.4f}")
    logger.info(f"Model test score: {test_score:.4f}")

    X = data.xdf.to_numpy()

    logger.info(f"Testing data shape: {X.shape}")

    global_disc_inputs = set()
    local_disc_inputs = set()

    tot_inputs = set()
    all_tot_inputs = []
    all_discriminations = set()

    # Early termination flag - will be used to break out of the local search
    early_termination = False

    def should_terminate() -> bool:
        nonlocal early_termination
        current_runtime = time.time() - start_time
        time_limit_exceeded = current_runtime > max_runtime_seconds
        tsn_threshold_reached = max_tsn is not None and len(tot_inputs) >= max_tsn

        if time_limit_exceeded or tsn_threshold_reached:
            early_termination = True
            return True
        return False

    clusters = seed_test_input(X, cluster_num, random_seed=random_seed)

    def round_robin_clusters(clusters):
        queues = [deque(cluster) for cluster in clusters if len(cluster)!=0]
        if not queues:
            return
        while queues:
            i = 0
            while i < len(queues):
                if queues[i]:
                    yield queues[i].popleft()
                    if not queues[i]:
                        queues.pop(i)
                    else:
                        i += 1
                else:
                    i += 1

    # Global search phase
    for inst_num, instance_id in enumerate(round_robin_clusters(clusters)):
        if inst_num >= max_global or should_terminate():
            break

        instance = X[instance_id]

        for it_global in range(10):
            result, result_df, max_discr, org_df, tested_inp = check_for_error_condition(
                logger=logger,
                model=model,
                instance=instance,
                dsn_by_attr_value=dsn_by_attr_value,
                discrimination_data=data,
                tot_inputs=tot_inputs,
                all_discriminations=all_discriminations,
                one_attr_at_a_time=one_attr_at_a_time,
                db_path=db_path,
                analysis_id=analysis_id
            )

            if should_terminate():
                break

            if result:
                global_disc_inputs.add(tuple(instance.astype(float).tolist()))
                break
            else:
                instance = global_perturbation(
                    model,
                    instance,
                    data.sensitive_indices,
                    data.input_bounds,
                    step_size=step_size
                )

        if should_terminate():
            break

    class EarlyTerminationCallback:
        def __init__(self):
            self.should_stop = False

        def __call__(self, x, f, accept):
            if should_terminate():
                self.should_stop = True
                return True  # Signal to stop
            return False  # Continue normally

    def evaluate_local(inp):
        if early_termination:
            return 0.0  # Return immediately if early termination is flagged

        is_discriminatory, found_df, max_discr, org_df, tested_inp = check_for_error_condition(
            logger=logger,
            model=model,
            instance=inp,
            dsn_by_attr_value=dsn_by_attr_value,
            discrimination_data=data,
            tot_inputs=tot_inputs,
            all_discriminations=all_discriminations,
            one_attr_at_a_time=one_attr_at_a_time,
            db_path=db_path,
            analysis_id=analysis_id
        )

        if is_discriminatory:
            local_disc_inputs.add(tuple(inp.astype(float).tolist()))

        return float(not is_discriminatory)

    # LOCAL SEARCH phase
    for glob_num, global_inp in enumerate(global_disc_inputs):
        if glob_num >= max_local or should_terminate():
            break

        local_perturbation = LocalPerturbation(
            model,
            data.sensitive_indices,
            data.input_bounds,
            step_size=step_size
        )

        minimizer = {"method": "L-BFGS-B"}
        callback = EarlyTerminationCallback()

        basinhopping(
            evaluate_local,
            np.array(global_inp),
            stepsize=step_size,
            take_step=local_perturbation,
            minimizer_kwargs=minimizer,
            niter=max_local,
            callback=callback,  # Add callback to check termination after each iteration
            accept_test=lambda f_new, x_new, f_old, x_old: not should_terminate(),
            seed=random_seed
        )

        # Break out early if the callback indicates we should stop
        if callback.should_stop or early_termination:
            logger.info("Early termination triggered during local search")
            break

    res_df, metrics = make_final_metrics_and_dataframe(data, tot_inputs, all_discriminations,
                                                       dsn_by_attr_value, start_time, logger=logger)
    return res_df, metrics


if __name__ == "__main__":
    from data_generator.main import get_real_data, DiscriminationData

    data_obj, schema = get_real_data('adult', use_cache=True)

    results_df, metrics = run_adf(data_obj, max_global=100, max_local=100, cluster_num=50,
                                  max_runtime_seconds=60, max_tsn=20000, step_size=0.1)

    print(f"\nTesting Metrics: {metrics}")
