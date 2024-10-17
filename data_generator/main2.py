import copy
import math
import random
import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh
from tqdm import tqdm
from dataclasses import dataclass, field
from pandas import DataFrame
from typing import Literal, TypeVar, Any, List, Dict, Tuple, Union
from scipy.stats import norm, multivariate_normal, beta, stats, gaussian_kde
import itertools
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import bernoulli
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
# Ignore specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class GaussianCopulaCategorical:
    def __init__(self, marginals, correlation_matrix, excluded_combinations=None):
        self.marginals = [np.array(m) for m in marginals]
        self.correlation_matrix = np.array(correlation_matrix)
        self.dim = len(marginals)
        self.excluded_combinations = set(map(tuple, excluded_combinations or []))
        self.cum_probabilities = [np.cumsum(m[:-1]) for m in self.marginals]

    def is_excluded(self, sample):
        return tuple(sample) in self.excluded_combinations

    def generate_samples(self, n_samples):
        samples = []
        while len(samples) < n_samples:
            gaussian_samples = multivariate_normal.rvs(mean=np.zeros(self.dim),
                                                       cov=self.correlation_matrix,
                                                       size=1).flatten()
            uniform_samples = norm.cdf(gaussian_samples)
            categorical_sample = np.zeros(self.dim, dtype=int)
            for i in range(self.dim):
                categorical_sample[i] = np.searchsorted(self.cum_probabilities[i], uniform_samples[i])
            if not self.is_excluded(categorical_sample):
                samples.append(categorical_sample)
        return np.array(samples)


def coefficient_of_variation(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    if mean == 0 or np.isnan(mean) or np.isnan(std_dev):
        return 0  # or another appropriate value
    cv = (std_dev / mean) * 100
    return cv


def generate_data_schema(min_number_of_classes, max_number_of_classes, nb_attributes, prop_protected_attr):
    attr_categories = []
    attr_names = []

    if nb_attributes < 2:
        raise ValueError("nb_attributes must be at least 2 to ensure both protected and unprotected attributes.")

    # Ensure at least one protected and one unprotected attribute
    protected_attr = [True, False]

    # Randomly assign additional attributes based on prop_protected_attr
    for _ in range(nb_attributes - 2):
        protected_attr.append(random.random() < prop_protected_attr)

    # Shuffle to randomize the order
    random.shuffle(protected_attr)

    i_t = 1
    i_x = 1

    for i in range(nb_attributes):
        num_classes = random.randint(min_number_of_classes, max_number_of_classes)
        attribute_set = [-1] + list(range(0, num_classes))
        attr_categories.append(attribute_set)

        if protected_attr[i]:
            attr_names.append(f"Attr{i_t}_T")
            i_t += 1
        else:
            attr_names.append(f"Attr{i_x}_X")
            i_x += 1

    return attr_categories, protected_attr, attr_names


def bin_array_values(array, num_bins):
    min_val = np.min(array)
    max_val = np.max(array)
    bins = np.linspace(min_val, max_val, num_bins + 1)
    binned_indices = np.digitize(array, bins) - 1
    return binned_indices


AttrCol = TypeVar('AttrCol', bound=str)


class DiscriminationDataFrame(DataFrame):
    group_key: str
    subgroup_key: str
    indv_key: str
    group_size: int
    min_number_of_classes: int
    max_number_of_classes: int
    nb_attributes: int
    prop_protected_attr: float
    nb_groups: int
    max_group_size: int
    hiddenlayers_depth: int
    granularity: int
    intersectionality: int
    similarity: float
    alea_uncertainty: float
    epis_uncertainty: float
    magnitude: float
    frequency: float
    outcome: float
    collisions: int
    diff_outcome: float
    diff_variation: float

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, key: AttrCol) -> Any:
        return super().__getitem__(key)

    def get_attr_columns(self) -> list[str]:
        return [col for col in self.columns if col.startswith('Attr')]

    def get_attr_column(self, index: int, protected: bool) -> Any:
        suffix = '_T' if protected else '_X'
        col_name = f'Attr{index}{suffix}'
        return self[col_name] if col_name in self.columns else None


@dataclass
class DiscriminationData:
    dataframe: DiscriminationDataFrame
    categorical_columns: List[str]
    attributes: Dict[str, bool]
    collisions: int
    nb_groups: int
    max_group_size: int
    hiddenlayers_depth: int
    outcome_column: Literal['outcome']
    relevance_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    attr_possible_values: Dict[str, List[int]] = field(default_factory=dict)  # Add possible values for each attribute

    @property
    def attr_columns(self) -> List[str]:
        return self.dataframe.get_attr_columns()

    @property
    def protected_attributes(self):
        return [k for k, v in self.attributes.items() if v]

    @property
    def feature_names(self):
        return list(self.attributes)

    @property
    def sensitive_indices(self):
        return {k: i for i, (k, v) in enumerate(self.attributes.items()) if v}

    @property
    def training_dataframe(self):
        return self.dataframe[list(self.attributes) + [self.outcome_column]]

    @property
    def xdf(self):
        return self.dataframe[list(self.attributes)]

    @property
    def ydf(self):
        return self.dataframe[self.outcome_column]

    def __post_init__(self):
        self.input_bounds = []
        for col in list(self.attributes):
            min_val = math.floor(self.xdf[col].min())
            max_val = math.ceil(self.xdf[col].max())
            self.input_bounds.append([min_val, max_val])


def generate_subgroup2_probabilities(subgroup1_sample, subgroup_sets, similarity, sets_attr):
    subgroup2_probabilities = []

    for i, (sample_value, possible_values) in enumerate(zip(subgroup1_sample, subgroup_sets)):
        n = len(possible_values)
        probs = np.zeros(n)

        if sets_attr[i]:  # If the attribute is protected
            # Ensure that subgroup2 has a different value than subgroup1 for protected attributes
            sample_index = possible_values.index(sample_value)
            different_indices = [j for j in range(n) if j != sample_index]

            # If the protected attribute has only one possible value, skip changing it
            if len(possible_values) == 1:
                probs[sample_index] = 1.0  # Only one value available, assign it
            elif not different_indices:
                # If no different values are available, use the same value as subgroup1
                probs[sample_index] = 1.0
            else:
                chosen_index = random.choice(different_indices)
                probs[chosen_index] = 1.0
        else:
            # Apply similarity for non-protected attributes
            sample_index = possible_values.index(sample_value)
            probs[sample_index] = similarity
            remaining_prob = 1 - similarity
            for j in range(n):
                if j != sample_index:
                    probs[j] = remaining_prob / (n - 1)

            # Add noise to the probabilities to simulate realistic variation
            noise = np.random.dirichlet(np.ones(n) * (1 - similarity) * 10)
            probs = (1 - (1 - similarity) / 2) * probs + (1 - similarity) / 2 * noise
            probs /= probs.sum()

        subgroup2_probabilities.append(probs.tolist())

    return subgroup2_probabilities


class OutcomeGenerator:
    def __init__(self, weights: np.ndarray, bias: float, subgroup_bias: float):
        self.weights = weights
        self.bias = bias
        self.subgroup_bias = subgroup_bias

    def generate_outcome(self, sample: List[int], is_subgroup1: bool) -> float:
        x = np.array(sample, dtype=float)
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        weighted_sum = np.dot(x, self.weights) + self.bias
        if is_subgroup1:
            weighted_sum += self.subgroup_bias
        else:
            weighted_sum -= self.subgroup_bias
        sigmoid = 1 / (1 + np.exp(-weighted_sum))
        return norm.cdf(norm.ppf(sigmoid) * 0.5)


class IndividualsGenerator:
    def __init__(self, schema, graph, gen_order, outcome_weights, outcome_bias, subgroup_bias,
                 epis_uncertainty, alea_uncertainty, n_estimators=50):
        self.schema = schema
        self.graph = np.array(graph)
        self.gen_order = [i - 1 for i in gen_order]
        self.n_attributes = len(schema)
        self.outcome_weights = outcome_weights
        self.outcome_bias = outcome_bias
        self.subgroup_bias = subgroup_bias
        self.epis_uncertainty = epis_uncertainty
        self.alea_uncertainty = alea_uncertainty
        self.n_estimators = n_estimators

    def _compute_support_degrees(self, n, p):
        def objective_positive(theta):
            return -min(
                (theta ** p * (1 - theta) ** n) / ((p / (n + p)) ** p * (n / (n + p)) ** n),
                2 * theta - 1
            )

        def objective_negative(theta):
            return -min(
                (theta ** p * (1 - theta) ** n) / ((p / (n + p)) ** p * (n / (n + p)) ** n),
                1 - 2 * theta
            )

        res_pos = minimize_scalar(objective_positive, bounds=(0, 1), method='bounded')
        res_neg = minimize_scalar(objective_negative, bounds=(0, 1), method='bounded')

        return -res_pos.fun, -res_neg.fun

    def generate_sample(self, predetermined: Union[List[int], List[Tuple[int, int]]] = None) -> List[int]:
        sample = [None] * self.n_attributes

        if predetermined:
            if isinstance(predetermined[0], tuple):
                for attr, value in predetermined:
                    sample[attr] = value
            else:
                for attr, value in enumerate(predetermined):
                    if value != -1:
                        sample[attr] = value

        for attr in self.gen_order:
            if sample[attr] is None:
                sample[attr] = self._generate_value(attr, sample)

        return sample

    def _generate_value(self, attr: int, partial_sample: List[int]) -> int:
        probabilities = np.ones(len(self.schema[attr]))

        for other_attr, other_value in enumerate(partial_sample):
            if other_value is not None:
                correlation = self.graph[attr][other_attr]
                for value in range(len(self.schema[attr])):
                    if value == other_value:
                        probabilities[value] *= correlation
                    else:
                        probabilities[value] *= (1 - correlation) / (len(self.schema[attr]) - 1)

        probabilities /= probabilities.sum()
        return np.random.choice(len(self.schema[attr]), p=probabilities)

    def generate_outcome(self, sample: List[int], is_subgroup1: bool) -> float:
        x = np.array(sample, dtype=float)
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        weighted_sum = np.dot(x, self.outcome_weights) + self.outcome_bias
        if is_subgroup1:
            weighted_sum += self.subgroup_bias
        else:
            weighted_sum -= self.subgroup_bias

        # Generate multiple predictions to simulate a random forest
        predictions = []
        for _ in range(self.n_estimators):
            # Add noise to simulate different trees
            noisy_sum = weighted_sum + np.random.normal(0, self.alea_uncertainty)
            pred = 1 / (1 + np.exp(-noisy_sum))
            predictions.append(bernoulli.rvs(pred))

        n = sum(1 - p for p in predictions)
        p = sum(predictions)

        pi_1, pi_0 = self._compute_support_degrees(n, p)

        # Use beta distribution to generate epistemic uncertainty
        a, b = 2, 5  # Shape parameters for beta distribution
        epistemic = beta.rvs(a, b) * min(pi_1, pi_0)

        aleatoric = 1 - max(pi_1, pi_0)

        # Adjust the final prediction based on uncertainties
        final_pred = (p / self.n_estimators) * (1 - epistemic) + 0.5 * epistemic

        return final_pred, epistemic, aleatoric

    def generate_sample_with_outcome(self, predetermined: Union[List[int], List[Tuple[int, int]]] = None,
                                     is_subgroup1: bool = True) -> Tuple[List[int], float, float, float]:
        sample = self.generate_sample(predetermined)
        outcome, epistemic, aleatoric = self.generate_outcome(sample, is_subgroup1)
        return sample, outcome, epistemic, aleatoric

    def generate_dataset_with_outcome(self, n_samples: int,
                                      predetermined_values: Union[List[int], List[Tuple[int, int]]],
                                      is_subgroup1: bool) -> List[Tuple[List[int], float, float, float]]:
        return [self.generate_sample_with_outcome(predetermined_values, is_subgroup1) for _ in range(n_samples)]


class CollisionTracker:
    def __init__(self, nb_attributes):
        self.used_combinations = set()
        self.nb_attributes = nb_attributes

    def is_collision(self, possibility):
        return tuple(possibility) in self.used_combinations

    def add_combination(self, possibility):
        self.used_combinations.add(tuple(possibility))


def calculate_actual_similarity(data):
    def calculate_group_similarity(group_data):
        subgroup_keys = group_data['subgroup_key'].unique()
        if len(subgroup_keys) != 2:
            return np.nan

        subgroup1 = subgroup_keys[0].split('|')
        subgroup2 = subgroup_keys[1].split('|')

        matching_attrs = sum(a == b for a, b in zip(subgroup1, subgroup2) if a != '*' and b != '*')
        total_fixed_attrs = sum(a != '*' and b != '*' for a, b in zip(subgroup1, subgroup2))

        return matching_attrs / total_fixed_attrs if total_fixed_attrs > 0 else 1.0

    return data.dataframe.groupby('group_key', group_keys=False).apply(calculate_group_similarity)


class UncertaintyRandomForest(RandomForestClassifier):
    def __init__(self, n_estimators=100, max_depth=10, **kwargs):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, **kwargs)

    def fit(self, X, y):
        # Remove feature names before fitting
        X_array = X.values if hasattr(X, 'values') else X
        super().fit(X_array, y)
        return self

    def _compute_support_degrees(self, n, p):
        if n + p == 0:
            return 0, 0  # or another appropriate value

        def objective_positive(theta):
            if theta == 0 or n + p == 0:
                return 0
            return -min(
                (theta ** p * (1 - theta) ** n) / ((p / (n + p)) ** p * (n / (n + p)) ** n),
                2 * theta - 1
            )

        def objective_negative(theta):
            if theta == 1 or n + p == 0:
                return 0
            return -min(
                (theta ** p * (1 - theta) ** n) / ((p / (n + p)) ** p * (n / (n + p)) ** n),
                1 - 2 * theta
            )

        res_pos = minimize_scalar(objective_positive, bounds=(0, 1), method='bounded')
        res_neg = minimize_scalar(objective_negative, bounds=(0, 1), method='bounded')

        return -res_pos.fun, -res_neg.fun

    def predict_with_uncertainty(self, X):
        # Remove feature names before prediction
        X_array = X.values if hasattr(X, 'values') else X

        predictions = []
        for tree in self.estimators_:
            leaf_id = tree.apply(X_array)
            predictions.append(tree.tree_.value[leaf_id].reshape(-1, self.n_classes_))

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)

        epistemic = np.zeros(X_array.shape[0])
        aleatoric = np.zeros(X_array.shape[0])

        for i in range(X_array.shape[0]):
            n = np.sum(predictions[:, i, 0])
            p = np.sum(predictions[:, i, 1])
            pi_1, pi_0 = self._compute_support_degrees(n, p)

            epistemic[i] = min(pi_1, pi_0)
            aleatoric[i] = 1 - max(pi_1, pi_0)

        return mean_pred, epistemic, aleatoric


def calculate_actual_uncertainties(data):
    """
    Calculate the actual epistemic and aleatoric uncertainties for each group.
    """
    X = data.dataframe[data.feature_names]
    y = data.dataframe[data.outcome_column]

    urf = UncertaintyRandomForest(n_estimators=50, random_state=42)
    urf.fit(X, y)

    _, epistemic, aleatoric = urf.predict_with_uncertainty(X)

    data.dataframe['calculated_epistemic'] = epistemic
    data.dataframe['calculated_aleatoric'] = aleatoric

    return data.dataframe.groupby('group_key').agg({
        'calculated_epistemic': 'mean',
        'calculated_aleatoric': 'mean'
    })


def calculate_actual_mean_diff_outcome(data):
    def calculate_group_diff(group_data):
        subgroup_keys = group_data['subgroup_key'].unique()
        if len(subgroup_keys) != 2:
            return np.nan

        subgroup1_outcome = group_data[group_data['subgroup_key'] == subgroup_keys[0]][data.outcome_column].mean()
        subgroup2_outcome = group_data[group_data['subgroup_key'] == subgroup_keys[1]][data.outcome_column].mean()

        return abs(subgroup1_outcome - subgroup2_outcome) / max(subgroup1_outcome, subgroup2_outcome)

    return data.dataframe.groupby('group_key', group_keys=False).apply(calculate_group_diff)


def calculate_actual_metrics_and_relevance(data):
    """
    Calculate the actual metrics and relevance for each group.
    """
    actual_similarity = calculate_actual_similarity(data)
    actual_uncertainties = calculate_actual_uncertainties(data)
    actual_mean_diff_outcome = calculate_actual_mean_diff_outcome(data)

    # Merge these metrics into the main dataframe
    data.dataframe['actual_similarity'] = data.dataframe['group_key'].map(actual_similarity)
    data.dataframe['actual_epis_uncertainty'] = data.dataframe['group_key'].map(
        actual_uncertainties['calculated_epistemic'])
    data.dataframe['actual_alea_uncertainty'] = data.dataframe['group_key'].map(
        actual_uncertainties['calculated_aleatoric'])
    data.dataframe['actual_mean_diff_outcome'] = data.dataframe['group_key'].map(actual_mean_diff_outcome)

    # Calculate the new relevance metric
    relevance = calculate_relevance(data)

    # Merge the relevance metrics back into the dataframe, avoiding duplicates
    existing_columns = set(data.dataframe.columns)
    new_columns = [col for col in relevance.columns if col not in existing_columns]
    data.dataframe = data.dataframe.merge(relevance[new_columns], left_on='group_key', right_index=True, how='left')

    # Store the relevance metrics in the DiscriminationData object
    data.relevance_metrics = relevance

    # Clean up any remaining duplicate columns
    data.dataframe = data.dataframe.loc[:, ~data.dataframe.columns.duplicated()]

    return data


def calculate_relevance(data):
    """
    Calculate the relevance metric for each group based on the updated formula.
    """

    def calculate_group_relevance(group_data):
        subgroup_keys = group_data['subgroup_key'].unique()
        if len(subgroup_keys) != 2:
            return pd.Series({
                'relevance': np.nan,
                'calculated_magnitude': np.nan,
                'calculated_group_size': np.nan,
                'calculated_granularity': np.nan,
                'calculated_intersectionality': np.nan,
                'calculated_uncertainty': np.nan,
                'calculated_similarity': np.nan,
                'calculated_subgroup_ratio': np.nan
            })

        S1 = group_data[group_data['subgroup_key'] == subgroup_keys[0]]
        S2 = group_data[group_data['subgroup_key'] == subgroup_keys[1]]

        # Calculate magnitude
        mu1 = S1[data.outcome_column].mean()
        mu2 = S2[data.outcome_column].mean()
        magnitude = abs(mu1 - mu2) / max(mu1, mu2) if max(mu1, mu2) != 0 else 0

        # Calculate other factors
        group_size = len(group_data) / len(data.dataframe)
        granularity = group_data['granularity'].iloc[0] / len(data.attributes)
        intersectionality = group_data['intersectionality'].iloc[0] / len(data.protected_attributes)
        uncertainty = 1 - (
                group_data['actual_epis_uncertainty'].mean() + group_data['actual_alea_uncertainty'].mean()) / 2
        similarity = group_data['actual_similarity'].iloc[0]
        subgroup_ratio = max(len(S1), len(S2)) / min(len(S1), len(S2))

        # Define weights (you may want to adjust these)
        w_f, w_g, w_i, w_u, w_s, w_r = 1, 1, 1, 1, 1, 1
        Z = w_f + w_g + w_i + w_u + w_s + w_r

        # Calculate OtherFactors
        other_factors = (w_f * group_size + w_g * granularity + w_i * intersectionality +
                         w_u * uncertainty + w_s * similarity + w_r * (1 / subgroup_ratio)) / Z

        # Calculate relevance (you may want to adjust alpha)
        alpha = 1
        relevance = magnitude * (1 + alpha * other_factors)

        return pd.Series({
            'relevance': relevance,
            'calculated_magnitude': magnitude,
            'calculated_group_size': group_size,
            'calculated_granularity': granularity,
            'calculated_intersectionality': intersectionality,
            'calculated_uncertainty': uncertainty,
            'calculated_similarity': similarity,
            'calculated_subgroup_ratio': subgroup_ratio
        })

    return data.dataframe.groupby('group_key').apply(calculate_group_relevance)


def calculate_weights(n):
    return [1 / (i + 1) for i in range(n)]


def safe_normalize(p):
    """Safely normalize an array, handling the case where sum is zero."""
    sum_p = np.sum(p)
    if sum_p == 0:
        # If all probabilities are zero, return a uniform distribution
        return np.ones_like(p) / len(p)
    return np.array(p) / sum_p


def create_group(
        possibility, attr_categories, sets_attr, correlation_matrix, gen_order, W,
        subgroup_bias,
        min_similarity, max_similarity, min_alea_uncertainty, max_alea_uncertainty,
        min_epis_uncertainty, max_epis_uncertainty, min_frequency, max_frequency,
        min_diff_subgroup_size, max_diff_subgroup_size, max_group_size, attr_names
):
    # Separate non-protected and protected attributes and reorder them
    non_protected_columns = [attr for attr, protected in zip(attr_names, sets_attr) if not protected]
    protected_columns = [attr for attr, protected in zip(attr_names, sets_attr) if protected]

    # Reorder attr_names so that non-protected attributes come first
    ordered_attr_names = non_protected_columns + protected_columns

    # Adjust attr_categories and sets_attr in the same order
    ordered_attr_categories = [attr_categories[attr_names.index(attr)] for attr in ordered_attr_names]
    ordered_sets_attr = [sets_attr[attr_names.index(attr)] for attr in ordered_attr_names]

    # Function to create sets based on the new ordering
    def make_sets(possibility):
        ress_set = []
        for ind in range(len(ordered_attr_categories)):
            if ind in possibility:
                ss = copy.deepcopy(ordered_attr_categories[ind])
                ss.remove(-1)
                ress_set.append(ss)
            else:
                ress_set.append([-1])
        return ress_set

    subgroup_sets = make_sets(possibility)

    similarity = random.uniform(min_similarity, max_similarity)
    alea_uncertainty = random.uniform(min_alea_uncertainty, max_alea_uncertainty)
    epis_uncertainty = random.uniform(min_epis_uncertainty, max_epis_uncertainty)
    frequency = random.uniform(min_frequency, max_frequency)

    # Generate samples based on the new ordered attributes
    subgroup1_p_vals = [random.choices(list(range(len(e))), k=len(e)) for e in subgroup_sets]
    subgroup1_p_vals = [safe_normalize(p) for p in subgroup1_p_vals]
    subgroup1_sample = GaussianCopulaCategorical(subgroup1_p_vals, correlation_matrix).generate_samples(1)
    subgroup1_vals = [subgroup_sets[i][e] for i, e in enumerate(subgroup1_sample[0])]

    subgroup2_p_vals = generate_subgroup2_probabilities(subgroup1_vals, subgroup_sets, similarity, ordered_sets_attr)
    subgroup2_sample = GaussianCopulaCategorical(subgroup2_p_vals, correlation_matrix,
                                                 list(subgroup1_sample)).generate_samples(1)
    subgroup2_vals = [subgroup_sets[i][e] for i, e in enumerate(subgroup2_sample[0])]

    total_group_size = math.ceil(max_group_size * frequency)

    diff_percentage = random.uniform(min_diff_subgroup_size, max_diff_subgroup_size)
    diff_size = int(total_group_size * diff_percentage)

    subgroup1_size = max(1, (total_group_size + diff_size) // 2)
    subgroup2_size = max(1, total_group_size - subgroup1_size)

    generator = IndividualsGenerator(
        schema=ordered_attr_categories,
        graph=correlation_matrix,
        gen_order=gen_order,
        outcome_weights=W[-1],
        outcome_bias=0,
        subgroup_bias=subgroup_bias,
        epis_uncertainty=epis_uncertainty,
        alea_uncertainty=alea_uncertainty
    )

    # Generate dataset for subgroup 1 and subgroup 2
    subgroup1_data = generator.generate_dataset_with_outcome(subgroup1_size, subgroup1_vals, is_subgroup1=True)
    subgroup1_individuals = [sample for sample, _, _, _ in subgroup1_data]
    subgroup1_individuals_df = pd.DataFrame(subgroup1_individuals, columns=ordered_attr_names)
    subgroup1_individuals_df['outcome'] = [outcome for _, outcome, _, _ in subgroup1_data]
    subgroup1_individuals_df['epis_uncertainty'] = [epis for _, _, epis, _ in subgroup1_data]
    subgroup1_individuals_df['alea_uncertainty'] = [alea for _, _, _, alea in subgroup1_data]

    subgroup2_data = generator.generate_dataset_with_outcome(subgroup2_size, subgroup2_vals, is_subgroup1=False)
    subgroup2_individuals = [sample for sample, _, _, _ in subgroup2_data]
    subgroup2_individuals_df = pd.DataFrame(subgroup2_individuals, columns=ordered_attr_names)
    subgroup2_individuals_df['outcome'] = [outcome for _, outcome, _, _ in subgroup2_data]
    subgroup2_individuals_df['epis_uncertainty'] = [epis for _, _, epis, _ in subgroup2_data]
    subgroup2_individuals_df['alea_uncertainty'] = [alea for _, _, _, alea in subgroup2_data]

    # Create keys based on the ordered attribute values
    subgroup1_key = '|'.join(list(map(lambda x: '*' if x == -1 else str(x), subgroup1_vals)))
    subgroup2_key = '|'.join(list(map(lambda x: '*' if x == -1 else str(x), subgroup2_vals)))
    group_key = subgroup1_key + '-' + subgroup2_key

    # Assign the new keys to the dataframe
    subgroup1_individuals_df['subgroup_key'] = subgroup1_key
    subgroup2_individuals_df['subgroup_key'] = subgroup2_key

    subgroup1_individuals_df['indv_key'] = subgroup1_individuals_df[ordered_attr_names].apply(
        lambda x: '|'.join(list(x.astype(str))), axis=1)
    subgroup2_individuals_df['indv_key'] = subgroup2_individuals_df[ordered_attr_names].apply(
        lambda x: '|'.join(list(x.astype(str))), axis=1)

    result_df = pd.concat([subgroup1_individuals_df, subgroup2_individuals_df])

    result_df['group_key'] = group_key
    result_df['granularity'] = len(
        [i for i in possibility if i in range(len(ordered_attr_names)) and not ordered_sets_attr[i]])
    result_df['intersectionality'] = len(
        [i for i in possibility if i in range(len(ordered_attr_names)) and ordered_sets_attr[i]])
    result_df['similarity'] = similarity
    result_df['epis_uncertainty'] = epis_uncertainty
    result_df['alea_uncertainty'] = alea_uncertainty
    result_df['frequency'] = frequency
    result_df['group_size'] = len(subgroup1_individuals + subgroup2_individuals)
    result_df['diff_subgroup_size'] = diff_percentage

    return result_df


def generate_valid_correlation_matrix(n):
    A = np.random.rand(n, n)
    A = (A + A.T) / 2
    eigenvalues, eigenvectors = eigh(A)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    A = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    D = np.diag(1 / np.sqrt(np.diag(A)))
    correlation_matrix = D @ A @ D

    return correlation_matrix


def estimate_min_attributes_and_classes(num_groups, max_iterations=1000):
    def calculate_combinations(num_attributes, num_classes):
        return sum(math.comb(num_attributes, i) * (num_classes - 1) ** i for i in range(1, num_attributes + 1))

    min_attributes = 2
    min_classes = 2

    for _ in range(max_iterations):
        combinations = calculate_combinations(min_attributes, min_classes)

        if combinations >= num_groups:
            return min_attributes, min_classes

        if min_attributes <= min_classes:
            min_attributes += 1
        else:
            min_classes += 1

    return min_attributes, min_classes


def generate_data(
        gen_order: List[int] = None,
        correlation_matrix=None,
        W: np.ndarray = None,
        nb_groups=100,
        nb_attributes=20,
        min_number_of_classes=None,
        max_number_of_classes=None,
        prop_protected_attr=0.2,
        max_group_size=100,
        min_similarity=0.0,
        max_similarity=1.0,
        min_alea_uncertainty=0.0,
        max_alea_uncertainty=1.0,
        min_epis_uncertainty=0.0,
        max_epis_uncertainty=1.0,
        min_frequency=0.0,
        max_frequency=1.0,
        min_diff_subgroup_size=0.0,
        max_diff_subgroup_size=0.5,
        categorical_outcome: bool = True,
        nb_categories_outcome: int = 6
):
    outcome_column = 'outcome'

    if correlation_matrix is None:
        correlation_matrix = generate_valid_correlation_matrix(nb_attributes)

    if min_number_of_classes is None or max_number_of_classes is None:
        min_number_of_classes, max_number_of_classes = estimate_min_attributes_and_classes(nb_groups)
        max_number_of_classes = int(min_number_of_classes * 1.5)

    if gen_order is None:
        gen_order = list(range(1, nb_attributes + 1))
        random.shuffle(gen_order)

    if W is None:
        hiddenlayers_depth = 3
        W = np.random.uniform(low=0.0, high=1.0, size=(hiddenlayers_depth, nb_attributes))

    attr_categories, sets_attr, attr_names = generate_data_schema(
        min_number_of_classes, max_number_of_classes, nb_attributes, prop_protected_attr
    )

    protected_indexes = [index for index, value in enumerate(sets_attr) if value]
    unprotected_indexes = [index for index, value in enumerate(sets_attr) if not value]

    collision_tracker = CollisionTracker(nb_attributes)

    results = []
    collisions = 0

    with tqdm(total=nb_groups, desc="Generating data") as pbar:
        while len(results) < nb_groups:
            granularity = random.randint(1, max(1, len(unprotected_indexes)))
            intersectionality = random.randint(1, max(1, len(protected_indexes)))

            subgroup_bias = random.uniform(0.1, 0.5)

            # Randomly sample from unprotected and protected indexes without generating all combinations
            possible_gran = random.sample(unprotected_indexes, granularity)
            possible_intersec = random.sample(protected_indexes, intersectionality)

            possibility = tuple(possible_gran + possible_intersec)  # Combine the gran and intersec lists into a tuple

            if not collision_tracker.is_collision(possibility):
                collision_tracker.add_combination(possibility)
                group = create_group(
                    possibility, attr_categories, sets_attr, correlation_matrix, gen_order, W,
                    subgroup_bias,
                    min_similarity, max_similarity, min_alea_uncertainty, max_alea_uncertainty,
                    min_epis_uncertainty, max_epis_uncertainty, min_frequency, max_frequency,
                    min_diff_subgroup_size, max_diff_subgroup_size, max_group_size, attr_names
                )
                results.append(group)
                pbar.update(1)
            else:
                collisions += 1

            if collisions > nb_groups * 2:
                print(f"\nWarning: Unable to generate {nb_groups} groups. Generated {len(results)} groups.")
                pbar.total = len(results)
                pbar.refresh()
                break

    results = pd.concat(results, ignore_index=True)
    results['collisions'] = collisions

    for column in attr_names + [outcome_column]:
        results[column] = pd.to_numeric(results[column], errors='ignore')

    if categorical_outcome:
        results[outcome_column] = bin_array_values(results[outcome_column], nb_categories_outcome)
    else:
        results[outcome_column] = (results[outcome_column] - results[outcome_column].min()) / (
                results[outcome_column].max() - results[outcome_column].min())

    results = results.sort_values(['group_key', 'indv_key'])

    results[f'diff_outcome'] = results.groupby(['group_key', 'indv_key'])[outcome_column].diff().abs().bfill()
    results['diff_variation'] = coefficient_of_variation(results['diff_outcome'])

    protected_attr = {k: e for k, e in zip(attr_names, sets_attr)}

    attr_possible_values = {attr_name: values for attr_name, values in zip(attr_names, attr_categories)}

    results = DiscriminationDataFrame(results)

    data = DiscriminationData(
        dataframe=results,
        categorical_columns=list(attr_names) + [outcome_column],
        attributes=protected_attr,
        collisions=collisions,
        nb_groups=results['group_key'].nunique(),
        max_group_size=max_group_size,
        hiddenlayers_depth=W.shape[0],
        outcome_column=outcome_column,
        attr_possible_values=attr_possible_values  # Include the possible values
    )

    data = calculate_actual_metrics_and_relevance(data)

    return data

# # %%
# nb_attributes = 20
# correlation_matrix = generate_valid_correlation_matrix(nb_attributes)
#
# data = generate_data(
#     nb_attributes=nb_attributes,
#     correlation_matrix=correlation_matrix,
#     min_number_of_classes=2,
#     max_number_of_classes=9,
#     prop_protected_attr=0.4,
#     nb_groups=500,
#     max_group_size=100,
#     categorical_outcome=True,
#     nb_categories_outcome=4)
#
# print(f"Generated {len(data.dataframe)} samples in {data.nb_groups} groups")
# print(f"Collisions: {data.collisions}")
#
#
# # %%
#
# def unique_individuals_ratio(data: pd.DataFrame, individual_col: str, attr_possible_values: Dict[str, List[int]]):
#     unique_individuals_count = data[individual_col].nunique()
#
#     # Calculate the total possible unique individuals by taking the product of possible values for each attribute
#     possible_unique_individuals = np.prod([len(values) for values in attr_possible_values.values()])
#
#     if possible_unique_individuals == 0:
#         return 0, 0  # To handle division by zero if no data or attributes
#
#     # Calculate the number of duplicates
#     total_individuals = data.shape[0]
#     duplicates_count = total_individuals - unique_individuals_count
#
#     # Calculate the ratio
#     ratio = unique_individuals_count / possible_unique_individuals
#
#     return ratio, duplicates_count, total_individuals
#
#
# def individuals_in_multiple_groups(data: pd.DataFrame, individual_col: str, group_col: str) -> int:
#     group_counts = data.groupby(individual_col)[group_col].nunique()
#
#     # Create the histogram
#     plt.figure(figsize=(10, 6))
#     counts, bins, patches = plt.hist(group_counts, bins=range(1, group_counts.max() + 2), edgecolor='black',
#                                      align='left')
#
#     # Add text annotations on top of each bar
#     for count, patch in zip(counts, patches):
#         plt.text(patch.get_x() + patch.get_width() / 2, count, f'{int(count)}', ha='center', va='bottom')
#
#     plt.title('Histogram of Individuals Belonging to Multiple Groups with Counts')
#     plt.xlabel('Number of Groups')
#     plt.ylabel('Number of Individuals')
#     plt.xticks(range(1, group_counts.max() + 1))
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.show()
#
#
# # Example usage:
# individual_col = 'indv_key'
# group_col = 'group_key'
#
# unique_ratio, duplicates_count, total = unique_individuals_ratio(data.dataframe, 'indv_key', data.attr_possible_values)
# individuals_in_multiple_groups_count = individuals_in_multiple_groups(data.dataframe, individual_col, group_col)
#
# print(f"Unique Individuals Ratio: {unique_ratio}, duplicate : {duplicates_count}, total: {total}")
# print(f"Individuals in Multiple Groups: {individuals_in_multiple_groups_count}")
#
#
# # %%
# def create_parallel_coordinates_plot(data):
#     group_properties = data.groupby('group_key').agg({
#         'granularity': 'mean',
#         'intersectionality': 'mean',
#         'diff_subgroup_size': 'mean',
#         'actual_similarity': 'mean',
#         'actual_alea_uncertainty': 'mean',
#         'actual_epis_uncertainty': 'mean',
#         'actual_mean_diff_outcome': 'mean',
#         'relevance': 'mean'
#     }).reset_index().copy()
#
#     group_properties.rename(columns={'actual_similarity': 'similarity',
#                                      'actual_alea_uncertainty': 'alea_uncertainty',
#                                      'actual_epis_uncertainty': 'epis_uncertainty',
#                                      'actual_mean_diff_outcome': 'diff_outcome'}, inplace=True)
#
#     for column in group_properties.columns:
#         if column != 'group_key':
#             group_properties[column] = pd.to_numeric(group_properties[column], errors='coerce')
#
#     # Remove any rows with NaN values
#     group_properties = group_properties.dropna()
#
#     # Normalize the data to a 0-1 range for each property
#     columns_to_plot = ['granularity', 'intersectionality', 'diff_subgroup_size', 'similarity',
#                        'alea_uncertainty', 'epis_uncertainty', 'diff_outcome', 'relevance']
#     normalized_data = group_properties[columns_to_plot].copy()
#     for column in columns_to_plot:
#         min_val = normalized_data[column].min()
#         max_val = normalized_data[column].max()
#         if min_val != max_val:
#             normalized_data[column] = (normalized_data[column] - min_val) / (max_val - min_val)
#         else:
#             normalized_data[column] = 0.5  # Set to middle value if all values are the same
#
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(12, 6))
#
#     # Create x-coordinates for each property
#     x = list(range(len(columns_to_plot)))
#
#     # Create colormap
#     norm = Normalize(vmin=group_properties['relevance'].min(), vmax=group_properties['relevance'].max())
#     cmap = plt.get_cmap('viridis')
#
#     # Plot each group
#     for i, row in normalized_data.iterrows():
#         y = row[columns_to_plot].values
#         color = cmap(norm(group_properties.loc[i, 'relevance']))
#         ax.plot(x, y, c=color, alpha=0.5)
#
#     # Customize the plot
#     ax.set_xticks(x)
#     ax.set_xticklabels(columns_to_plot, rotation=45, ha='right')
#     ax.set_ylim(0, 1)
#     ax.set_title('Parallel Coordinates Plot of Discrimination Metrics')
#     ax.set_xlabel('Metrics')
#     ax.set_ylabel('Normalized Values')
#
#     # Add gridlines
#     ax.grid(True, axis='x', linestyle='--', alpha=0.7)
#
#     # Add colorbar
#     sm = ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax, label='Relevance')
#
#     plt.tight_layout()
#     plt.show()
#
#
# create_parallel_coordinates_plot(data.dataframe)
#
#
# # %%
#
# def plot_and_print_metric_distributions(data, num_bins=10):
#     metrics = [
#         'granularity', 'intersectionality', 'diff_subgroup_size', 'actual_similarity',
#         'actual_alea_uncertainty', 'actual_epis_uncertainty', 'actual_mean_diff_outcome', 'relevance'
#     ]
#
#     group_properties = data.groupby('group_key').agg({metric: 'mean' for metric in metrics}).reset_index()
#
#     fig, axes = plt.subplots(4, 2, figsize=(15, 20), tight_layout=True)
#     axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration
#
#     for i, (metric, ax) in enumerate(zip(metrics, axes)):
#         # Remove NaN values
#         clean_data = group_properties[metric].dropna()
#
#         if clean_data.empty:
#             print(f"\nWarning: All values for {metric} are NaN. Skipping this metric.")
#             ax.text(0.5, 0.5, f"No valid data for {metric}", ha='center', va='center')
#             continue
#
#         try:
#             # Determine the number of bins
#             unique_values = clean_data.nunique()
#             actual_bins = min(num_bins, unique_values)
#
#             # Create histogram
#             if actual_bins == unique_values:
#                 bins = np.sort(clean_data.unique())
#             else:
#                 bins = actual_bins
#
#             n, bins, patches = ax.hist(clean_data, bins=bins, edgecolor='black')
#
#             # Add labels and title
#             ax.set_xlabel(metric)
#             ax.set_ylabel('Frequency')
#             ax.set_title(f'Distribution of {metric}')
#
#             # Add percentage labels on top of each bar
#             total_count = len(clean_data)
#             for j, rect in enumerate(patches):
#                 height = rect.get_height()
#                 percentage = height / total_count * 100
#                 ax.text(rect.get_x() + rect.get_width() / 2., height,
#                         f'{percentage:.1f}%',
#                         ha='center', va='bottom', rotation=90, fontsize=8)
#
#             # Adjust y-axis to make room for percentage labels
#             ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
#
#             # Print bin information
#             print(f"\nDistribution of {metric}:")
#             print(f"Total data points: {total_count}")
#             print(f"Number of bins: {actual_bins}")
#             print("\nBin ranges, counts, and percentages:")
#             for k in range(len(n)):
#                 bin_start = bins[k]
#                 bin_end = bins[k + 1] if k < len(bins) - 1 else bin_start
#                 count = n[k]
#                 percentage = (count / total_count) * 100
#                 print(f"Bin {k + 1}: {bin_start:.2f} to {bin_end:.2f}")
#                 print(f"  Count: {count}")
#                 print(f"  Percentage: {percentage:.1f}%")
#
#         except Exception as e:
#             print(f"\nError processing {metric}: {str(e)}")
#             ax.text(0.5, 0.5, f"Error processing {metric}", ha='center', va='center')
#
#     plt.show()
#
#
# # Usage
# plot_and_print_metric_distributions(data.dataframe)
#
#
# # %%
#
# def test_models_on_generated_data(data):
#     # Extract features and target
#     X = data.dataframe[data.feature_names]
#     y = data.dataframe[data.outcome_column]
#
#     # Encode categorical variables
#     label_encoders = {}
#     for column in X.columns:
#         if X[column].dtype == 'object':
#             le = LabelEncoder()
#             X[column] = le.fit_transform(X[column])
#             label_encoders[column] = le
#
#     # Encode target variable if it's categorical
#     if y.dtype == 'object':
#         le = LabelEncoder()
#         y = le.fit_transform(y)
#
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Initialize models
#     rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
#
#     # Train and evaluate Random Forest
#     rf_model.fit(X_train, y_train)
#     rf_predictions = rf_model.predict(X_test)
#     rf_accuracy = accuracy_score(y_test, rf_predictions)
#     rf_f1 = f1_score(y_test, rf_predictions, average='weighted')
#     rf_recall = recall_score(y_test, rf_predictions, average='weighted')
#
#     # Feature importance
#     rf_feature_importance = rf_model.feature_importances_
#
#     # Sort feature importances
#     rf_sorted_idx = np.argsort(rf_feature_importance)
#     rf_top_features = X.columns[rf_sorted_idx][-5:][::-1]
#
#     # Print results
#     print("Random Forest Results:")
#     print(f"Accuracy: {rf_accuracy:.4f}")
#     print(f"F1 Score: {rf_f1:.4f}")
#     print(f"Recall: {rf_recall:.4f}")
#     print("Top 5 important features:", ", ".join(rf_top_features))
#
#     return {
#         'rf_accuracy': rf_accuracy,
#         'rf_f1': rf_f1,
#         'rf_recall': rf_recall,
#         'rf_top_features': rf_top_features
#     }
#
#
# results = test_models_on_generated_data(data)
#
#
# # %%
#
# def plot_correlation_matrices(input_correlation_matrix, generated_data, figsize=(30, 10)):
#     attr_columns = [col for col in generated_data.dataframe.columns if col.startswith('Attr')]
#
#     generated_correlation_matrix = generated_data.dataframe[attr_columns].corr(method='spearman')
#
#     assert input_correlation_matrix.shape == generated_correlation_matrix.shape, "Correlation matrices have different shapes"
#
#     if isinstance(input_correlation_matrix, np.ndarray):
#         input_correlation_matrix = pd.DataFrame(input_correlation_matrix, columns=attr_columns, index=attr_columns)
#
#     # Calculate the absolute difference matrix
#     abs_diff_matrix = np.abs(input_correlation_matrix - generated_correlation_matrix)
#
#     # Create a custom colormap for the absolute difference (blue to white to red)
#     colors = ['#053061', '#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
#     n_bins = 256  # Increase for smoother gradient
#     custom_cmap = LinearSegmentedColormap.from_list('custom_blue_red', colors, N=n_bins)
#
#     # Create a figure with three subplots side by side
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
#
#     # Function to plot heatmap with adjusted parameters
#     def plot_heatmap(data, ax, title, cmap='coolwarm', vmin=-1, vmax=1):
#         sns.heatmap(data, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, center=0,
#                     annot=True, fmt='.2f', square=True, cbar=False,
#                     annot_kws={'size': 11}, linewidths=0.5)
#         ax.set_title(title, fontsize=16, pad=20)
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
#         ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
#
#     # Plot input correlation matrix
#     plot_heatmap(input_correlation_matrix, ax1, 'Input Correlation Matrix')
#
#     # Plot generated correlation matrix
#     plot_heatmap(generated_correlation_matrix, ax2, 'Generated Data Correlation Matrix')
#
#     # Plot absolute difference matrix with custom colormap
#     plot_heatmap(abs_diff_matrix, ax3, 'Absolute Difference Matrix', cmap=custom_cmap, vmin=0, vmax=1)
#
#     # Add a color bar for the absolute difference matrix
#     cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
#     sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(0, 1))
#     sm.set_array([])
#     cbar = fig.colorbar(sm, cax=cbar_ax)
#     cbar.ax.tick_params(labelsize=10)
#
#     # Adjust layout and display the plot
#     plt.tight_layout(rect=[0, 0, 0.92, 1])
#     plt.show()
#
#     # Calculate and print summary statistics
#     mean_diff = np.mean(abs_diff_matrix)
#     max_diff = np.max(abs_diff_matrix)
#     print(f"Mean absolute difference between matrices: {mean_diff:.4f}")
#     print(f"Maximum absolute difference between matrices: {max_diff:.4f}")
#
#
# plot_correlation_matrices(correlation_matrix, data)
#
# # %%
#
# input_correlation_matrices, generated_data_list = [], []
# nb_attributes = 20
# for da in range(3):
#     correlation_matrix = generate_valid_correlation_matrix(nb_attributes)
#
#     data = generate_data(
#         nb_attributes=nb_attributes,
#         correlation_matrix=correlation_matrix,
#         min_number_of_classes=8,
#         max_number_of_classes=10,
#         prop_protected_attr=0.4,
#         nb_groups=200,
#         max_group_size=100,
#         categorical_outcome=True,
#         nb_categories_outcome=4)
#
#     input_correlation_matrices.append(correlation_matrix)
#     generated_data_list.append(data)
#
# # %%
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.colors import LinearSegmentedColormap
#
#
# def plot_aggregate_correlation_matrices(input_correlation_matrices, generated_data_list, figsize=(30, 10)):
#     # Get all unique column names across all datasets
#     all_columns = set()
#     for data in generated_data_list:
#         all_columns.update(data.feature_names)
#     all_columns = sorted(list(all_columns))
#
#     # Initialize a dictionary to store difference matrices
#     diff_matrices = {col1: {col2: [] for col2 in all_columns} for col1 in all_columns}
#
#     # Calculate the difference matrices and store them
#     for input_corr, generated_data in zip(input_correlation_matrices, generated_data_list):
#         feature_names = generated_data.feature_names
#         generated_corr = generated_data.dataframe[feature_names].corr(method='spearman')
#
#         if isinstance(input_corr, np.ndarray):
#             input_corr = pd.DataFrame(input_corr, columns=feature_names, index=feature_names)
#
#         for col1 in feature_names:
#             for col2 in feature_names:
#                 diff = abs(input_corr.loc[col1, col2] - generated_corr.loc[col1, col2])
#                 diff_matrices[col1][col2].append(diff)
#
#     # Calculate aggregate statistics
#     aggregate_diff_matrix = pd.DataFrame({col1: {col2: np.mean(values) if values else np.nan
#                                                  for col2, values in col_dict.items()}
#                                           for col1, col_dict in diff_matrices.items()})
#
#     variance_matrix = pd.DataFrame({col1: {col2: np.var(values) if len(values) > 1 else np.nan
#                                            for col2, values in col_dict.items()}
#                                     for col1, col_dict in diff_matrices.items()})
#
#     # Calculate summary statistics
#     mean_diff = np.nanmean(aggregate_diff_matrix.values)
#     max_diff = np.nanmax(aggregate_diff_matrix.values)
#     median_diff = np.nanmedian(aggregate_diff_matrix.values)
#     mean_variance = np.nanmean(variance_matrix.values)
#     max_variance = np.nanmax(variance_matrix.values)
#
#     # Create a custom colormap for the absolute difference (blue to white to red)
#     colors = ['#053061', '#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
#     n_bins = 256
#     custom_cmap = LinearSegmentedColormap.from_list('custom_blue_red', colors, N=n_bins)
#
#     # Create a figure with two subplots side by side
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
#
#     # Plot aggregate difference matrix
#     sns.heatmap(aggregate_diff_matrix, ax=ax1, cmap=custom_cmap, vmin=0, vmax=1, center=0,
#                 annot=True, fmt='.2f', square=True, cbar=True,
#                 annot_kws={'size': 8}, linewidths=0.5)
#     ax1.set_title('Aggregate Absolute Difference Matrix', fontsize=16, pad=20)
#     ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='right', fontsize=8)
#     ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=8)
#
#     # Plot variance matrix
#     sns.heatmap(variance_matrix, ax=ax2, cmap='viridis', vmin=0, center=0,
#                 annot=True, fmt='.2f', square=True, cbar=True,
#                 annot_kws={'size': 8}, linewidths=0.5)
#     ax2.set_title('Variance of Absolute Differences', fontsize=16, pad=20)
#     ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, ha='right', fontsize=8)
#     ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8)
#
#     # Adjust layout and display the plot
#     plt.tight_layout()
#     plt.show()
#
#     # Print summary statistics
#     print(f"Mean absolute difference across all matrices: {mean_diff:.4f}")
#     print(f"Maximum absolute difference across all matrices: {max_diff:.4f}")
#     print(f"Median absolute difference across all matrices: {median_diff:.4f}")
#     print(f"Mean variance of differences: {mean_variance:.4f}")
#     print(f"Maximum variance of differences: {max_variance:.4f}")
#
#
# # Example usage:
# plot_aggregate_correlation_matrices(input_correlation_matrices, generated_data_list, figsize=(30, 10))
#
#
# # %%
#
# def plot_metric_distributions(data_list, num_bins=10):
#     metrics = [
#         'granularity', 'intersectionality', 'diff_subgroup_size', 'actual_similarity',
#         'actual_alea_uncertainty', 'actual_epis_uncertainty', 'actual_mean_diff_outcome', 'relevance'
#     ]
#
#     fig, axes = plt.subplots(4, 2, figsize=(15, 20), tight_layout=True)
#     axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration
#
#     for i, (metric, ax) in enumerate(zip(metrics, axes)):
#         all_data = []
#         for data in data_list:
#             group_properties = data.dataframe.groupby('group_key').agg({metric: 'mean'}).reset_index()
#             all_data.extend(group_properties[metric].dropna())
#
#         if not all_data:
#             print(f"\nWarning: All values for {metric} are NaN. Skipping this metric.")
#             ax.text(0.5, 0.5, f"No valid data for {metric}", ha='center', va='center')
#             continue
#
#         try:
#             # Create histogram
#             counts, bin_edges = np.histogram(all_data, bins=num_bins)
#             bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
#
#             # Calculate error (assuming Poisson distribution for bin counts)
#             errors = np.sqrt(counts)
#
#             # Plot histogram without error bars
#             ax.bar(bin_centers, counts, width=np.diff(bin_edges), alpha=0.7)
#
#             # Add error bars separately
#             ax.errorbar(bin_centers, counts, yerr=errors, fmt='none', ecolor='black', capsize=3)
#
#             # Add labels and title
#             ax.set_xlabel(metric)
#             ax.set_ylabel('Frequency')
#             ax.set_title(f'Distribution of {metric}')
#
#             # Print statistics
#             print(f"\nDistribution of {metric}:")
#             print(f"Total data points: {len(all_data)}")
#             print(f"Number of bins: {num_bins}")
#             print("\nBin ranges, counts, and errors:")
#             for j in range(len(counts)):
#                 print(f"Bin {j + 1}: {bin_edges[j]:.2f} to {bin_edges[j + 1]:.2f}")
#                 print(f"  Count: {counts[j]}")
#                 print(f"  Error: ±{errors[j]:.2f}")
#
#         except Exception as e:
#             print(f"\nError processing {metric}: {str(e)}")
#             ax.text(0.5, 0.5, f"Error processing {metric}", ha='center', va='center')
#
#     plt.show()
#
#
# plot_metric_distributions(generated_data_list)
#
# # %%
#
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, f1_score, recall_score
#
#
# def test_models_on_multiple_datasets(datasets):
#     all_results = []
#
#     for data in datasets:
#         # Extract features and target
#         X = data.dataframe[data.feature_names]
#         y = data.dataframe[data.outcome_column]
#
#         # Encode categorical variables
#         label_encoders = {}
#         for column in X.columns:
#             if X[column].dtype == 'object':
#                 le = LabelEncoder()
#                 X[column] = le.fit_transform(X[column])
#                 label_encoders[column] = le
#
#         # Encode target variable if it's categorical
#         if y.dtype == 'object':
#             le = LabelEncoder()
#             y = le.fit_transform(y)
#
#         # Split the data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#         # Initialize models
#         rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
#
#         # Train and evaluate Random Forest
#         rf_model.fit(X_train, y_train)
#         rf_predictions = rf_model.predict(X_test)
#         rf_accuracy = accuracy_score(y_test, rf_predictions)
#         rf_f1 = f1_score(y_test, rf_predictions, average='weighted')
#         rf_recall = recall_score(y_test, rf_predictions, average='weighted')
#
#         # Feature importance
#         rf_feature_importance = rf_model.feature_importances_
#
#         # Sort feature importances
#         rf_sorted_idx = np.argsort(rf_feature_importance)
#         rf_top_features = X.columns[rf_sorted_idx][-5:][::-1]
#
#         # Store results
#         all_results.append({
#             'rf_accuracy': rf_accuracy,
#             'rf_f1': rf_f1,
#             'rf_recall': rf_recall,
#             'rf_top_features': rf_top_features
#         })
#
#     # Calculate mean and variance for each metric
#     metrics = ['rf_accuracy', 'rf_f1', 'rf_recall']
#     summary = {}
#
#     for metric in metrics:
#         values = [result[metric] for result in all_results]
#         summary[f'{metric}_mean'] = np.mean(values)
#         summary[f'{metric}_variance'] = np.var(values)
#
#     # Summarize top features
#     all_top_features = [result['rf_top_features'] for result in all_results]
#     feature_counts = {}
#     for top_features in all_top_features:
#         for feature in top_features:
#             if feature in feature_counts:
#                 feature_counts[feature] += 1
#             else:
#                 feature_counts[feature] = 1
#
#     most_common_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5]
#     summary['most_common_top_features'] = [feature for feature, count in most_common_features]
#
#     # Print results
#     print("Summary of Results:")
#     for metric in metrics:
#         print(f"{metric} - Mean: {summary[f'{metric}_mean']:.4f}, Variance: {summary[f'{metric}_variance']:.4f}")
#     print("Most common top features:", ", ".join(summary['most_common_top_features']))
#
#     return summary, all_results
#
#
# # Example usage:
# results_summary, individual_results = test_models_on_multiple_datasets(generated_data_list)
