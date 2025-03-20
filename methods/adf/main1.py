import os

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


def is_diverse_enough(new_solution: np.ndarray) -> bool:
    """Check if a new solution is diverse enough compared to previous ones.
    
    Args:
        new_solution: New candidate solution
        
    Returns:
        bool: True if solution is diverse enough
    """
    new_solution_tuple = tuple(new_solution)
    if new_solution_tuple in previous_solutions:
        return False

    current_threshold = min(
        MAX_DISTANCE_THRESHOLD,
        MIN_DISTANCE_THRESHOLD + (len(previous_solutions) * 0.001)
    )

    if len(previous_solutions) > 0:
        similar_solutions = 0
        for prev_sol in previous_solutions:
            feature_similarities = np.abs(np.array(new_solution) - np.array(prev_sol)) < current_threshold
            similarity_ratio = np.mean(feature_similarities)
            if similarity_ratio > MAX_SIMILAR_FEATURES:
                similar_solutions += 1
                if similar_solutions > 3:
                    return False

            if np.linalg.norm(np.array(new_solution) - np.array(prev_sol)) < current_threshold * 0.5:
                return False

    for i, value in enumerate(new_solution):
        if i not in feature_value_counts:
            feature_value_counts[i] = {}
        rounded_value = round(value, 1)
        feature_value_counts[i][rounded_value] = feature_value_counts[i].get(rounded_value, 0) + 1

        total_count = sum(feature_value_counts[i].values())
        if feature_value_counts[i][rounded_value] / total_count > 0.5:
            return False

    return True


def model_prediction(model, x):
    """Get model predictions.

    Args:
        model: Trained model
        x: Input data (DataFrame or numpy array)

    Returns:
        Model predictions probabilities
    """
    if isinstance(x, pd.DataFrame):
        probs = model.predict_proba(x)
    else:
        x_df = pd.DataFrame(x, columns=model.feature_names_in_)
        probs = model.predict_proba(x_df)
    return probs


def model_argmax(model, x):
    """Get model class predictions.

    Args:
        model: Trained model
        x: Input data (DataFrame or numpy array)

    Returns:
        Predicted classes
    """
    if isinstance(x, pd.DataFrame):
        preds = model.predict(x)
    else:
        x_df = pd.DataFrame(x, columns=model.feature_names_in_)
        preds = model.predict(x_df)
    return preds


def check_for_error_condition(
        data_obj,
        model,
        x: np.ndarray,
        dsn_by_attr_value,
        one_attr_at_a_time: bool = False
) -> Tuple[bool, set, list]:
    """Check if test case shows discrimination.

    Args:
        data_obj: Data object containing metadata
        model: Trained model
        x: Test instance
        one_attr_at_a_time: If True, vary only one protected attribute at a time

    Returns:
        Tuple[bool, set, list]: (Has discrimination, discriminatory pairs, all tested cases)
    """
    inp_df = pd.DataFrame([x], columns=data_obj.feature_names)
    original_pred = model_argmax(model, inp_df)
    inp_df['outcome'] = original_pred

    protected_values = {}
    for attr in data_obj.protected_attributes:
        protected_values[attr] = sorted(data_obj.training_dataframe[attr].unique())

    test_cases = []

    if one_attr_at_a_time:
        # Vary one attribute at a time
        for attr in data_obj.protected_attributes:
            # Get all possible values for this attribute
            values = protected_values[attr]

            # For each possible value that is different from the current one
            for value in values:
                if inp_df[attr].iloc[0] == value:
                    continue

                # Create a new test case changing only this attribute
                new_case = inp_df.copy()
                new_case[attr] = value
                test_cases.append(new_case)
                dsn_by_attr_value[attr]['TSN'] += 1

    else:
        # Original approach: vary all combinations of protected attributes
        attr_names = list(protected_values.keys())
        attr_values = list(protected_values.values())
        combinations = list(itertools.product(*attr_values))

        for combination in combinations:
            if tuple(map(int, inp_df[attr_names].iloc[0].tolist())) == combination:
                continue

            new_case = inp_df.copy()
            for attr, value in zip(attr_names, combination):
                new_case[attr] = value
                dsn_by_attr_value[attr]['TSN'] += 1

            test_cases.append(new_case)

    if not test_cases:
        return False, set(), set()

    test_cases_df = pd.concat(test_cases)
    test_cases_df['outcome'] = model_argmax(model, test_cases_df[data_obj.feature_names])

    discriminations = test_cases_df[abs(test_cases_df['outcome'] - original_pred) > 0]
    if discriminations.empty:
        return False, set(), set()

    disc_pairs = set()
    for el in discriminations.to_numpy():
        disc_pairs.add((tuple(inp_df.iloc[0]), tuple(el)))
        for attr in data_obj.protected_attributes:
            n_el = pd.DataFrame(np.expand_dims(el, 0), columns=inp_df.columns)
            if n_el[attr].iloc[0] != inp_df[attr].iloc[0]:
                dsn_by_attr_value[attr]['DSN'] += 1
                dsn_by_attr_value['total'] += 1

    all_tested = set([tuple(e) for e in test_cases_df.values])

    return True, disc_pairs, all_tested


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


class LocalPerturbation:
    """Local perturbation strategy."""

    def __init__(
            self,
            model,
            n_values: Dict[int, Any],
            sensitive_indices: List[int],
            input_shape: int,
            data_obj,
            random_state,
    ):
        self.model = model
        self.n_values = n_values
        self.input_shape = input_shape
        self.sensitive_indices = sensitive_indices
        self.data_obj = data_obj
        self.rng = np.random.RandomState(random_state)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Perform local perturbation.

        Args:
            x: Input instance

        Returns:
            Perturbed instance
        """
        s = self.rng.uniform(MIN_PERTURBATION_SIZE, MAX_PERTURBATION_SIZE) * self.rng.choice([1.0, -1.0])

        n_x = x.copy()
        for sens_idx, n_value in self.n_values.items():
            n_x[sens_idx - 1] = n_value

        eps = self.rng.uniform(MIN_EPSILON, MAX_EPSILON)
        ind_grad = np.zeros(self.input_shape)
        n_ind_grad = np.zeros(self.input_shape)

        for i in range(self.input_shape):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps

            pred_plus = model_prediction(self.model, x_plus.reshape(1, -1))[0]
            pred_minus = model_prediction(self.model, x_minus.reshape(1, -1))[0]
            ind_grad[i] = np.max(pred_plus - pred_minus) / (2 * eps)

            nx_plus = n_x.copy()
            nx_plus[i] += eps
            nx_minus = n_x.copy()
            nx_minus[i] -= eps

            pred_plus = model_prediction(self.model, nx_plus.reshape(1, -1))[0]
            pred_minus = model_prediction(self.model, nx_minus.reshape(1, -1))[0]
            n_ind_grad[i] = np.max(pred_plus - pred_minus) / (2 * eps)

        if np.allclose(ind_grad, 0) and np.allclose(n_ind_grad, 0):
            n_sensitive = len(self.sensitive_indices)
            if self.input_shape > n_sensitive:
                probs = 1.0 / (self.input_shape - n_sensitive) * np.ones(self.input_shape)
                for sens_idx in self.sensitive_indices:
                    probs[sens_idx - 1] = 0
            else:
                return x
        else:
            grad_sum = np.abs(ind_grad) + np.abs(n_ind_grad)
            grad_sum = np.where(grad_sum > 0, 1.0 / grad_sum, 0)
            for sens_idx in self.sensitive_indices:
                grad_sum[sens_idx - 1] = 0
            total = np.sum(grad_sum)
            if total > 0:
                probs = grad_sum / total
            else:
                return x

        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs = probs / probs_sum
        else:
            return x

        available_indices = [i for i in range(self.input_shape) if i + 1 not in self.sensitive_indices]
        if available_indices and not np.any(np.isnan(probs[available_indices])):
            index = self.rng.choice(available_indices, p=probs[available_indices])

            local_cal_grad = np.zeros(self.input_shape)
            local_cal_grad[index] = 1.0

            x_new = x + s * local_cal_grad
            x_new = clip(x_new, self.data_obj)

            return x_new

        return x


class BasinhoppingCallback:
    def __init__(self, max_niter, progress_bar=None):
        self.niter = 0
        self.max_niter = max_niter
        self.progress_bar = progress_bar

    def __call__(self, x, f, accept):
        # Update tqdm progress bar if provided
        if self.progress_bar is not None:
            self.progress_bar.update(1)
            self.progress_bar.set_description(f"f={f:.4f} accepted={accept}")
        return False


def adf_fairness_testing(
        data_obj: DiscriminationData,
        max_global: int = 2000,
        max_local: int = 2000,
        max_iter: int = 100,
        cluster_num: int = 10,
        random_seed: int = 42,
        max_runtime_seconds: int = 3600,  # Default 1 hour time limit
        max_tsn: int = None,  # New parameter for TSN threshold
        one_attr_at_a_time: bool = False
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Implementation of ADF fairness testing.

    Args:
        data_obj: Data object containing dataset and metadata
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
    rng = np.random.RandomState(random_seed)

    logger = logging.getLogger("ADF")

    start_time = time.time()

    global_disc_inputs = []
    local_disc_inputs = []

    dsn_by_attr_value = {e: {'TSN': 0, 'DSN': 0} for e in data_obj.protected_attributes}
    dsn_by_attr_value['total'] = 0

    data = data_obj.training_dataframe.copy()

    logger.info(f"Dataset shape: {data.shape}")
    logger.info(f"Protected attributes: {data_obj.protected_attributes}")
    logger.info(f"Time limit: {max_runtime_seconds} seconds")
    if max_tsn:
        logger.info(f"Target TSN: {max_tsn}")

    model, X_train, X_test, y_train, y_test, feature_names = train_sklearn_model(
        data=data,
        model_type='rf',
        target_col=data_obj.outcome_column,
        sensitive_attrs=list(data_obj.protected_attributes),
        random_state=random_seed
    )

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    logger.info(f"Model training score: {train_score:.4f}")
    logger.info(f"Model test score: {test_score:.4f}")

    X = data_obj.xdf.to_numpy()
    y = data_obj.ydf.to_numpy()

    logger.info(f"Testing data shape: {X.shape}")
    input_shape = X.shape[1]

    tot_inputs = set()
    discriminatory_pairs = set()

    log_count = 0

    def log_metrics():
        nonlocal log_count
        if log_count % 100 == 0:
            dsr = 100 * len(discriminatory_pairs) / len(tot_inputs) if len(
                tot_inputs) > 0 else 0
            logger.info(f"Current Metrics - TSN: {len(tot_inputs)}, DSN: {len(discriminatory_pairs)}, DSR: {dsr:.4f}")
        log_count += 1

    # Function to check if we've exceeded the time limit
    def time_limit_exceeded() -> bool:
        current_runtime = time.time() - start_time
        return current_runtime > max_runtime_seconds

    # Function to check if we've reached the TSN threshold
    def tsn_threshold_reached() -> bool:
        if max_tsn is None:
            return False
        return len(tot_inputs) >= max_tsn

    # Function to check if any termination condition is met
    def should_terminate() -> bool:
        return time_limit_exceeded() or tsn_threshold_reached()

    def evaluate_local(inp):
        """Evaluate local perturbation results."""
        # Check termination criteria first
        if should_terminate():
            # Return a value that will terminate the optimization
            return 0.0

        discr_key = []
        for i in range(len(inp)):
            if i not in data_obj.sensitive_indices.values():
                discr_key.append(inp[i])
        discr_key = tuple(discr_key)

        result = False
        if (discr_key not in tot_inputs):
            result, discr_res, all_tested = check_for_error_condition(data_obj, model, inp,
                                                                      dsn_by_attr_value,
                                                                      one_attr_at_a_time=one_attr_at_a_time)

            tot_inputs.add(discr_key)
            if result:
                previous_solutions.add(tuple(inp))

                for el in all_tested:
                    tot_inputs.add(el)

                for el in discr_res:
                    discriminatory_pairs.add(el)

        log_metrics()

        return float(not result)

    clusters = seed_test_input(X, cluster_num, random_seed=random_seed)
    all_cluster_indices = [idx for cluster in clusters for idx in cluster]

    # Make sure we can restart the search by tracking explored indices
    explored_indices = set()

    # To support multiple passes, we'll use an outer loop that continues until termination
    search_iteration = 0
    while not should_terminate():
        logger.info(f"Starting search iteration {search_iteration + 1}")

        # Reset iteration counter for each search pass
        iter_num = 0

        # If we've explored all clusters and need more samples
        if len(explored_indices) >= len(all_cluster_indices):
            logger.info("All indices explored. Resetting exploration to continue search.")
            explored_indices = set()

            # Reshuffle clusters for more diversity in the next pass
            for cluster in clusters:
                rng.shuffle(cluster)

        for _, cluster in enumerate(clusters):
            # Check if termination criteria have been met
            if should_terminate():
                logger.info("Termination criteria met during cluster exploration.")
                break

            if iter_num > max_iter:
                logger.info(f"Reached max_iter ({max_iter}). Moving to next search iteration.")
                break

            # Modified this condition to avoid early termination when we want to reach max_tsn
            # Only use this condition if max_tsn is not specified
            if max_tsn is None and len(discriminatory_pairs) < 100 and len(tot_inputs) > max_global:
                continue

            for idx, index in enumerate(cluster):
                # Skip already explored indices unless we've reset for a new pass
                if index in explored_indices:
                    continue

                explored_indices.add(index)

                # Check termination criteria periodically
                if should_terminate():
                    break

                # Modified this condition to avoid early termination when we want to reach max_tsn
                # Only use this condition if max_tsn is not specified
                if max_tsn is None and len(discriminatory_pairs) < 100 and len(tot_inputs) > max_global:
                    continue

                if iter_num > max_iter:
                    break

                sample = X[index:index + 1]

                probs = model_prediction(model, sample)[0]
                label = np.argmax(probs)
                prob = probs[label]
                max_diff = 0
                n_values = {}

                sensitive_values = {}
                for sens_name, sens_idx in data_obj.sensitive_indices.items():
                    sensitive_values[sens_name] = np.unique(data_obj.xdf.iloc[:, sens_idx]).tolist()

                value_combinations = list(
                    itertools.product(*[sensitive_values[name] for name in sensitive_values.keys()]))

                for values in value_combinations:
                    # Check termination criteria inside the innermost loop for responsiveness
                    if should_terminate():
                        break

                    if all(sample[0][sens_idx] == value
                           for name, value in zip(sensitive_values.keys(), values)):
                        continue

                    tnew = pd.DataFrame(sample, columns=data_obj.feature_names)
                    for name, value in zip(sensitive_values.keys(), values):
                        tnew[name] = value
                    n_sample = tnew.to_numpy()

                    discr_key = []
                    for i in range(len(n_sample[0])):
                        if i not in data_obj.sensitive_indices.values():
                            discr_key.append(n_sample[0][i])
                    discr_key = tuple(discr_key)

                    tot_inputs.add(discr_key)

                    n_probs = model_prediction(model, n_sample)[0]
                    n_label = np.argmax(n_probs)
                    n_prob = n_probs[n_label]

                    if label != n_label:
                        for name, value in zip(sensitive_values.keys(), values):
                            n_values[data_obj.sensitive_indices[name]] = value

                        global_disc_inputs.append(discr_key)

                        log_metrics()

                        # Create a tqdm progress bar
                        with tqdm(total=max_local, desc="Basinhopping progress") as progress_bar:
                            # Modify basinhopping to respect termination criteria
                            minimizer = {
                                "method": "L-BFGS-B",
                                "options": {
                                    "maxiter": 10,
                                    "ftol": 1e-6,
                                    "gtol": 1e-5,
                                    "maxls": 20  # Limit line search steps for determinism
                                }
                            }

                            local_perturbation = LocalPerturbation(
                                model, n_values, data_obj.sensitive_indices.values(),
                                X.shape[1], data_obj, random_seed
                            )

                            # Custom accept_test function to check termination criteria
                            def termination_check(f_new, x_new, f_old, x_old):
                                return not should_terminate()

                            # Create callback for tqdm updates
                            callback = BasinhoppingCallback(max_local, progress_bar)

                            basinhopping(
                                evaluate_local,
                                sample.flatten(),
                                stepsize=rng.uniform(MIN_PERTURBATION_SIZE, MAX_PERTURBATION_SIZE),
                                take_step=local_perturbation,
                                minimizer_kwargs=minimizer,
                                niter=max_local,
                                T=1.0,
                                interval=40,
                                niter_success=12,
                                seed=random_seed,  # Add seed for basinhopping
                                accept_test=termination_check,  # Add termination check
                                callback=callback  # Add the callback for tqdm updates
                            )

                            # Check if termination criteria were met during basinhopping
                            if should_terminate():
                                break
                    else:
                        prob_diff = abs(prob - n_prob)
                        if prob_diff > max_diff:
                            max_diff = prob_diff
                            for name, value in zip(sensitive_values.keys(), values):
                                n_values[data_obj.sensitive_indices[name]] = value

                if should_terminate():
                    break

                iter_num += 1
                log_metrics()

            # Check termination after processing a cluster
            if should_terminate():
                break

        # Increment search iteration counter
        search_iteration += 1

        # Log progress after each search iteration
        current_tsn = len(tot_inputs)
        current_dsn = len(discriminatory_pairs)
        current_dsr = current_dsn / current_tsn if current_tsn > 0 else 0
        logger.info(f"Completed search iteration {search_iteration}")
        logger.info(f"Current TSN: {current_tsn}, DSN: {current_dsn}, DSR: {current_dsr:.4f}")

        # If we reached max_tsn, break out of the loop
        if should_terminate():
            logger.info("Termination criteria met after search iteration.")
            break

    # Record the termination reasons
    time_limit_reached = time_limit_exceeded()
    tsn_threshold_reached = tsn_threshold_reached()

    end_time = time.time()
    total_time = end_time - start_time

    tsn = len(tot_inputs)
    dsn = len(discriminatory_pairs)
    sur = dsn / tsn if tsn > 0 else 0
    dss = total_time / dsn if dsn > 0 else float('inf')

    for k, v in dsn_by_attr_value.items():
        if k != 'total':
            dsn_by_attr_value[k]['SUR'] = dsn_by_attr_value[k]['DSN'] / dsn_by_attr_value[k]['TSN']

    metrics = {
        'TSN': tsn,
        'DSN': dsn,
        'SUR': sur,
        'DSS': total_time / dsn if dsn > 0 else float('inf'),
        'total_time': total_time,
        'time_limit_reached': time_limit_reached,
        'tsn_threshold_reached': tsn_threshold_reached,
        'search_iterations': search_iteration,
        'dsn_by_attr_value': dsn_by_attr_value
    }

    logger.info("Final Results:")
    logger.info(f"Total Time: {total_time:.2f}s")
    logger.info(f"Search Iterations: {search_iteration}")

    termination_reasons = []
    if time_limit_reached:
        termination_reasons.append(f"Time limit of {max_runtime_seconds}s reached")
    if tsn_threshold_reached:
        termination_reasons.append(f"TSN threshold of {max_tsn} reached")

    if termination_reasons:
        logger.info(f"Termination reason(s): {', '.join(termination_reasons)}")

    logger.info(f"Final Metrics - TSN: {tsn}, DSN: {dsn}, DSR: {metrics['SUR']:.4f}")
    logger.info(f"Success Rate: {metrics['SUR']:.4f}")
    logger.info(f"Search Time per Discriminatory Sample: {metrics['DSS']:.2f}s")

    all_rows = []
    feature_cols = data_obj.feature_names
    all_cols = feature_cols + ['outcome', 'indv_key', 'couple_key', 'diff_outcome', 'case_id']

    # Pre-allocate lists for faster append operations
    for case_id, (org, counter_org) in enumerate(discriminatory_pairs):
        # Extract features and outcome
        org_features = list(org[:-1])
        counter_features = list(counter_org[:-1])
        org_outcome = org[-1]
        counter_outcome = counter_org[-1]

        # Create keys directly from features without creating DataFrames
        indv_key1 = "|".join(str(x) for x in org_features)
        indv_key2 = "|".join(str(x) for x in counter_features)
        couple_key = f"{indv_key1}-{indv_key2}"
        diff_outcome = abs(org_outcome - counter_outcome)

        # Add both rows directly to the list
        all_rows.append(org_features + [org_outcome, indv_key1, couple_key, diff_outcome, case_id])
        all_rows.append(counter_features + [counter_outcome, indv_key2, couple_key, diff_outcome, case_id])

    # Create DataFrame in one operation
    if all_rows:
        res_df = pd.DataFrame(all_rows, columns=all_cols)

        # Add metrics columns all at once
        res_df['TSN'] = tsn
        res_df['DSN'] = dsn
        res_df['SUR'] = sur
        res_df['DSS'] = dss
        res_df['time_limit_reached'] = time_limit_reached
        res_df['tsn_threshold_reached'] = tsn_threshold_reached
        res_df['search_iterations'] = search_iteration

        # Remove duplicates only once at the end
        res_df.drop_duplicates(subset=['indv_key'], inplace=True)
    else:
        res_df = pd.DataFrame(columns=all_cols + ['TSN', 'DSN', 'SUR', 'DSS', 'time_limit_reached',
                                                  'tsn_threshold_reached', 'search_iterations'])

    return res_df, metrics


if __name__ == "__main__":
    from data_generator.main import get_real_data, generate_from_real_data, DiscriminationData

    data_obj, schema = get_real_data('adult')

    results_df, metrics = adf_fairness_testing(
        data_obj,
        max_global=5000,
        max_local=2000,
        max_iter=100,
        cluster_num=50,
        max_runtime_seconds=3600,  # 1 hour time limit
        max_tsn=20000,  # Set target TSN threshold
        one_attr_at_a_time=True
    )

    print(f"\nTesting Metrics: {metrics}")
