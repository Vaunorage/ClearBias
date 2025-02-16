import copy
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('ADF')

# Configuration constants
PERTURBATION_SIZE = 1.0


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
        # Convert numpy array to DataFrame using the stored feature names
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
        # Convert numpy array to DataFrame using the stored feature names
        x_df = pd.DataFrame(x, columns=model.feature_names_in_)
        preds = model.predict(x_df)
    return preds


def check_for_error_condition(
        data_obj,
        model,
        x: np.ndarray
) -> Tuple[bool, set, list]:
    """Check if test case shows discrimination.

    Args:
        data_obj: Data object containing metadata
        model: Trained model
        x: Test instance

    Returns:
        Tuple[bool, set]: (Has discrimination, discriminatory pairs)
    """
    inp_df = pd.DataFrame([x], columns=data_obj.feature_names)
    original_pred = model_argmax(model, inp_df)
    inp_df['outcome'] = original_pred

    # Get all unique values for each protected attribute
    protected_values = {}
    for attr in data_obj.protected_attributes:
        protected_values[attr] = sorted(data_obj.dataframe[attr].unique())

    # Generate all possible combinations
    attr_names = list(protected_values.keys())
    attr_values = list(protected_values.values())
    combinations = list(itertools.product(*attr_values))

    test_cases = []
    for combination in combinations:
        if all(inp_df[attr].iloc[0] == value
               for attr, value in zip(attr_names, combination)):
            continue

        new_case = inp_df.copy()
        for attr, value in zip(attr_names, combination):
            new_case[attr] = value
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

    all_tested = set([tuple(e) for e in test_cases_df.values])

    return True, disc_pairs, all_tested


def seed_test_input(X: np.ndarray, cluster_num: int) -> List[np.ndarray]:
    """Select seed inputs for fairness testing.

    Args:
        X: Input data
        cluster_num: Number of clusters

    Returns:
        List of seed inputs
    """

    # Standardize features for better clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform clustering
    kmeans = KMeans(
        n_clusters=min(cluster_num, X.shape[0]),
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)

    # Get indices for each cluster
    clusters = [np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)]

    # Sort clusters by size to prioritize larger clusters
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
            data_obj
    ):
        self.model = model
        self.n_values = n_values
        self.input_shape = input_shape
        self.sensitive_indices = sensitive_indices
        self.data_obj = data_obj

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Perform local perturbation.

        Args:
            x: Input instance

        Returns:
            Perturbed instance
        """
        s = np.random.choice([1.0, -1.0]) * PERTURBATION_SIZE

        n_x = x.copy()
        for sens_idx, n_value in self.n_values.items():
            n_x[sens_idx - 1] = n_value

        # Get gradient approximations
        eps = 1e-6
        ind_grad = np.zeros(self.input_shape)
        n_ind_grad = np.zeros(self.input_shape)

        for i in range(self.input_shape):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps

            # Get probabilities for x
            pred_plus = model_prediction(self.model, x_plus.reshape(1, -1))[0]
            pred_minus = model_prediction(self.model, x_minus.reshape(1, -1))[0]
            # Use the maximum probability difference as gradient
            ind_grad[i] = np.max(pred_plus - pred_minus) / (2 * eps)

            # Get probabilities for n_x
            nx_plus = n_x.copy()
            nx_plus[i] += eps
            nx_minus = n_x.copy()
            nx_minus[i] -= eps

            pred_plus = model_prediction(self.model, nx_plus.reshape(1, -1))[0]
            pred_minus = model_prediction(self.model, nx_minus.reshape(1, -1))[0]
            # Use the maximum probability difference as gradient
            n_ind_grad[i] = np.max(pred_plus - pred_minus) / (2 * eps)

        if np.allclose(ind_grad, 0) and np.allclose(n_ind_grad, 0):
            # Handle case where all gradients are zero
            n_sensitive = len(self.sensitive_indices)
            if self.input_shape > n_sensitive:  # Check to avoid division by zero
                probs = 1.0 / (self.input_shape - n_sensitive) * np.ones(self.input_shape)
                for sens_idx in self.sensitive_indices:
                    probs[sens_idx - 1] = 0
            else:
                return x  # Return unchanged if no valid features to perturb
        else:
            # Handle case with non-zero gradients
            grad_sum = np.abs(ind_grad) + np.abs(n_ind_grad)
            grad_sum = np.where(grad_sum > 0, 1.0 / grad_sum, 0)  # Avoid division by zero
            for sens_idx in self.sensitive_indices:
                grad_sum[sens_idx - 1] = 0
            total = np.sum(grad_sum)
            if total > 0:  # Check if we have any valid probabilities
                probs = grad_sum / total
            else:
                return x  # Return unchanged if no valid probabilities

        # Normalize probabilities if they exist
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs = probs / probs_sum
        else:
            return x  # Return unchanged if no valid probabilities

        available_indices = [i for i in range(self.input_shape) if i + 1 not in self.sensitive_indices]
        if available_indices and not np.any(np.isnan(probs[available_indices])):
            index = np.random.choice(available_indices, p=probs[available_indices])
            local_cal_grad = np.zeros(self.input_shape)
            local_cal_grad[index] = 1.0

            x = clip(x + s * local_cal_grad, self.data_obj)

        return x


def adf_fairness_testing(
        data_obj,
        max_global: int = 100,
        max_local: int = 1000,
        max_iter: int = 10,
        cluster_num: int = 4
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Implementation of ADF fairness testing.

    Args:
        data_obj: Data object containing dataset and metadata
        max_global: Maximum samples for global search
        max_local: Maximum samples for local search
        max_iter: Maximum iterations for global perturbation
        cluster_num: Number of clusters

    Returns:
        Results DataFrame and metrics dictionary
    """
    logger = logging.getLogger("ADF")

    start_time = time.time()

    # Initialize metrics
    global_disc_inputs = []
    local_disc_inputs = []

    # Get data and normalize
    data = data_obj.dataframe.copy()

    logger.info(f"Dataset shape: {data.shape}")
    logger.info(f"Protected attributes: {data_obj.protected_attributes}")

    # Train model
    model, X_train, X_test, y_train, y_test, feature_names = train_sklearn_model(
        data=data,
        model_type='rf',
        target_col=data_obj.outcome_column,
        sensitive_attrs=list(data_obj.protected_attributes)
    )

    # Evaluate model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    logger.info(f"Model training score: {train_score:.4f}")
    logger.info(f"Model test score: {test_score:.4f}")

    # Get full dataset in numpy format for testing
    X = data_obj.xdf.to_numpy()
    y = data_obj.ydf.to_numpy()

    logger.info(f"Testing data shape: {X.shape}")
    input_shape = X.shape[1]

    # Initialize tracking sets
    tot_inputs = set()
    discriminatory_pairs = set()

    def log_metrics():
        dsr = 100 * len(discriminatory_pairs) / len(tot_inputs) if len(tot_inputs) > 0 else 0  # Discriminatory Sample Ratio
        logger.info(f"Current Metrics - TSN: {len(tot_inputs)}, DSN: {len(discriminatory_pairs)}, DSR: {dsr:.4f}")

    def evaluate_local(inp):
        """Evaluate local perturbation results."""
        discr_key = []
        for i in range(len(inp)):
            if i not in data_obj.sensitive_indices.values():
                discr_key.append(inp[i])
        discr_key = tuple(discr_key)

        result = False
        if (discr_key not in tot_inputs):
            result, discr_res, all_tested = check_for_error_condition(data_obj, model, inp)

            tot_inputs.add(discr_key)

            if result:
                for el in all_tested:
                    tot_inputs.add(el)

                for el in discr_res:
                    discriminatory_pairs.add(el)

        log_metrics()

        return float(not result)

    # Get seed inputs
    clusters = seed_test_input(X, min(max_global, len(X)))

    # Main testing loop
    for iter_num, cluster in enumerate(clusters):
        if iter_num > max_iter or len(tot_inputs) > max_global:
            break

        for index in cluster:
            if len(tot_inputs) > max_global:
                break

            sample = X[index:index + 1]

            probs = model_prediction(model, sample)[0]
            label = np.argmax(probs)
            prob = probs[label]
            max_diff = 0
            n_values = {}

            # Get all possible protected attribute combinations
            sensitive_values = {}
            for sens_name, sens_idx in data_obj.sensitive_indices.items():
                sensitive_values[sens_name] = np.unique(data_obj.xdf.iloc[:, sens_idx]).tolist()

            value_combinations = list(itertools.product(*[sensitive_values[name] for name in sensitive_values.keys()]))

            for values in value_combinations:
                if all(sample[0][sens_idx] == value
                       for name, value in zip(sensitive_values.keys(), values)):
                    continue

                tnew = pd.DataFrame(sample, columns=data_obj.feature_names)
                for name, value in zip(sensitive_values.keys(), values):
                    tnew[name] = value
                n_sample = tnew.to_numpy()

                n_probs = model_prediction(model, n_sample)[0]
                n_label = np.argmax(n_probs)
                n_prob = n_probs[n_label]

                if label != n_label:
                    for name, value in zip(sensitive_values.keys(), values):
                        n_values[data_obj.sensitive_indices[name]] = value
                    break
                else:
                    prob_diff = abs(prob - n_prob)
                    if prob_diff > max_diff:
                        max_diff = prob_diff
                        for name, value in zip(sensitive_values.keys(), values):
                            n_values[data_obj.sensitive_indices[name]] = value

            sample_key = copy.deepcopy(sample[0].astype('int').tolist())
            sample_key = [sample_key[i] for i in range(len(sample_key))
                          if i not in data_obj.sensitive_indices.values()]

            if label != n_label and (tuple(sample_key) not in tot_inputs):
                global_disc_inputs.append(sample_key)
                tot_inputs.add(tuple(sample_key))

                log_metrics()

                # Local search
                minimizer = {"method": "L-BFGS-B"}
                local_perturbation = LocalPerturbation(
                    model, n_values, data_obj.sensitive_indices.values(),
                    X.shape[1], data_obj
                )

                basinhopping(
                    evaluate_local,
                    sample.flatten(),
                    stepsize=1.0,
                    take_step=local_perturbation,
                    minimizer_kwargs=minimizer,
                    niter=max_local
                )

                logger.info(f"Local search discriminatory inputs: {len(local_disc_inputs)}")
                logger.info(
                    f"Percentage discriminatory inputs of local search: "
                    f"{float(len(local_disc_inputs)) / float(len(tot_inputs)) * 100:.2f}%"
                )
                break

            log_metrics()

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate metrics
    tsn = len(tot_inputs)
    dsn = len(global_disc_inputs) + len(local_disc_inputs)  # Discriminatory Sample Number
    sur = dsn / tsn if tsn > 0 else 0  # Success Rate
    dss = total_time / dsn if dsn > 0 else float('inf')  # Discriminatory Sample Search time

    metrics = {
        'tsn': tsn,
        'dsn': dsn,
        'dsr': dsn / tsn if tsn > 0 else 0,
        'sur': dsn / tsn if tsn > 0 else 0,  # Success Rate
        'dss': total_time / dsn if dsn > 0 else float('inf'),  # Discriminatory Sample Search time
        'total_time': total_time
    }

    # Final log
    logger.info("Final Results:")
    logger.info(f"Total Time: {total_time:.2f}s")
    logger.info(f"Final Metrics - TSN: {tsn}, DSN: {dsn}, DSR: {metrics['dsr']:.4f}")
    logger.info(f"Success Rate: {metrics['sur']:.4f}")
    logger.info(f"Search Time per Discriminatory Sample: {metrics['dss']:.2f}s")

    # Create results dataframe
    res_df = []
    case_id = 0
    for org, counter_org in discriminatory_pairs:
        indv1 = pd.DataFrame([list(org)], columns=data_obj.feature_names + ['outcome'])
        indv2 = pd.DataFrame([list(counter_org)], columns=data_obj.feature_names + ['outcome'])

        indv_key1 = "|".join(str(x) for x in indv1[data_obj.feature_names].iloc[0])
        indv_key2 = "|".join(str(x) for x in indv2[data_obj.feature_names].iloc[0])

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
        res_df = pd.concat(res_df)
    else:
        res_df = pd.DataFrame([])

    # Add metrics to results
    res_df['TSN'] = tsn
    res_df['DSN'] = dsn
    res_df['SUR'] = sur
    res_df['DSS'] = dss

    # Log final results
    logger.info(f"Total Inputs: {len(tot_inputs)}")
    logger.info(f"Global Search Discriminatory Inputs: {len(global_disc_inputs)}")
    logger.info(f"Local Search Discriminatory Inputs: {len(local_disc_inputs)}")
    logger.info(f"Success Rate (SUR): {metrics['SUR']:.4f}")
    logger.info(f"Average Search Time per Discriminatory Sample (DSS): {metrics['DSS']:.4f} seconds")
    logger.info(f"Total Discriminatory Pairs Found: {len(discriminatory_pairs)}")

    return res_df, metrics


if __name__ == "__main__":
    # Example usage
    from data_generator.main import get_real_data, generate_from_real_data

    # Generate synthetic data
    data_obj, schema = get_real_data('adult')

    # Run fairness testing
    results_df, metrics = adf_fairness_testing(
        data_obj,
        max_global=100,
        max_local=1000,
        max_iter=10,
        cluster_num=4
    )

    print("\nTesting Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
