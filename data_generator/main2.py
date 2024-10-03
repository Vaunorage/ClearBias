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
from scipy.stats import norm, multivariate_normal, beta
import itertools
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import minimize_scalar
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings

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
        raise ValueError("nb_attributes must be at least 2 to ensure both 0 and 1 can be included.")

    protected_attr = [False] * (nb_attributes - 2) + [False, True]

    for i in range(nb_attributes - 2):
        protected_attr[i] = True if random.random() < prop_protected_attr else False

    random.shuffle(protected_attr)

    for i in range(nb_attributes):
        base_name = 'Attr' + str(i + 1)
        num_classes = random.randint(min_number_of_classes, max_number_of_classes)
        attribute_set = [-1] + list(range(0, num_classes))
        attr_categories.append(attribute_set)

        if protected_attr[i]:
            attr_names.append(base_name + '_T')
        else:
            attr_names.append(base_name + '_X')

    if not any(protected_attr):
        random_index = random.randint(0, nb_attributes - 1)
        protected_attr[random_index] = True
        attr_names[random_index] = attr_names[random_index][:-2] + '_T'

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
    couple_key: str
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


def generate_subgroup2_probabilities(subgroup1_sample, subgroup_sets, similarity):
    subgroup2_probabilities = []

    for sample_value, possible_values in zip(subgroup1_sample, subgroup_sets):
        n = len(possible_values)
        probs = np.zeros(n)
        sample_index = possible_values.index(sample_value)
        probs[sample_index] = similarity
        remaining_prob = 1 - similarity
        for i in range(n):
            if i != sample_index:
                probs[i] = remaining_prob / (n - 1)
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


def calculate_relevance(data):
    """
    Calculate the relevance metric for each group.
    """
    actual_similarity = calculate_actual_similarity(data)
    actual_uncertainties = calculate_actual_uncertainties(data)
    actual_mean_diff_outcome = calculate_actual_mean_diff_outcome(data)

    relevance = pd.DataFrame({
        'actual_similarity': actual_similarity,
        'actual_epis_uncertainty': actual_uncertainties['calculated_epistemic'],
        'actual_alea_uncertainty': actual_uncertainties['calculated_aleatoric'],
        'actual_mean_diff_outcome_subgroups': actual_mean_diff_outcome
    })

    relevance['relevance'] = relevance.sum(axis=1)

    return relevance


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
        min_similarity, max_similarity, min_alea_uncertainty, max_alea_uncertainty,
        min_epis_uncertainty, max_epis_uncertainty, min_frequency, max_frequency,
        min_diff_subgroup_size, max_diff_subgroup_size, max_group_size, attr_names
):
    def make_sets(possiblity):
        ress_set = []
        for ind in range(len(attr_categories)):
            if ind in possiblity:
                ss = copy.deepcopy(attr_categories[ind])
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

    subgroup1_p_vals = [random.choices(list(range(len(e))), k=len(e)) for e in subgroup_sets]
    subgroup1_p_vals = [safe_normalize(p) for p in subgroup1_p_vals]
    subgroup1_sample = GaussianCopulaCategorical(subgroup1_p_vals, correlation_matrix).generate_samples(1)
    subgroup1_vals = [subgroup_sets[i][e] for i, e in enumerate(subgroup1_sample[0])]

    subgroup2_p_vals = generate_subgroup2_probabilities(subgroup1_vals, subgroup_sets, similarity)
    subgroup2_sample = GaussianCopulaCategorical(subgroup2_p_vals, correlation_matrix,
                                                 list(subgroup1_sample)).generate_samples(1)
    subgroup2_vals = [subgroup_sets[i][e] for i, e in enumerate(subgroup2_sample[0])]

    total_group_size = math.ceil(max_group_size * frequency)

    diff_percentage = random.uniform(min_diff_subgroup_size, max_diff_subgroup_size)
    diff_size = int(total_group_size * diff_percentage)

    subgroup1_size = max(1, (total_group_size + diff_size) // 2)
    subgroup2_size = max(1, total_group_size - subgroup1_size)

    subgroup_bias = random.uniform(0.1, 0.5)
    generator = IndividualsGenerator(
        schema=attr_categories,
        graph=correlation_matrix,
        gen_order=gen_order,
        outcome_weights=W[-1],
        outcome_bias=0,
        subgroup_bias=subgroup_bias,
        epis_uncertainty=epis_uncertainty,
        alea_uncertainty=alea_uncertainty
    )

    subgroup1_data = generator.generate_dataset_with_outcome(subgroup1_size, subgroup1_vals, is_subgroup1=True)
    subgroup1_individuals = [sample for sample, _, _, _ in subgroup1_data]
    subgroup1_individuals_df = pd.DataFrame(subgroup1_individuals, columns=attr_names)
    subgroup1_individuals_df['outcome'] = [outcome for _, outcome, _, _ in subgroup1_data]
    subgroup1_individuals_df['epis_uncertainty'] = [epis for _, _, epis, _ in subgroup1_data]
    subgroup1_individuals_df['alea_uncertainty'] = [alea for _, _, _, alea in subgroup1_data]

    subgroup2_data = generator.generate_dataset_with_outcome(subgroup2_size, subgroup2_vals, is_subgroup1=False)
    subgroup2_individuals = [sample for sample, _, _, _ in subgroup2_data]
    subgroup2_individuals_df = pd.DataFrame(subgroup2_individuals, columns=attr_names)
    subgroup2_individuals_df['outcome'] = [outcome for _, outcome, _, _ in subgroup2_data]
    subgroup2_individuals_df['epis_uncertainty'] = [epis for _, _, epis, _ in subgroup2_data]
    subgroup2_individuals_df['alea_uncertainty'] = [alea for _, _, _, alea in subgroup2_data]

    subgroup1_key = '|'.join(list(map(lambda x: '*' if x == -1 else str(x), subgroup1_vals)))
    subgroup2_key = '|'.join(list(map(lambda x: '*' if x == -1 else str(x), subgroup2_vals)))
    group_key = subgroup1_key + '-' + subgroup2_key

    subgroup1_individuals_df['subgroup_key'] = subgroup1_key
    subgroup2_individuals_df['subgroup_key'] = subgroup2_key

    subgroup1_individuals_df['indv_key'] = subgroup1_individuals_df[attr_names].apply(
        lambda x: '|'.join(list(x.astype(str))), axis=1)
    subgroup2_individuals_df['indv_key'] = subgroup2_individuals_df[attr_names].apply(
        lambda x: '|'.join(list(x.astype(str))), axis=1)

    subgroup1_individuals_df['couple_key'] = subgroup1_individuals_df.index
    subgroup2_individuals_df['couple_key'] = subgroup2_individuals_df.index

    result_df = pd.concat([subgroup1_individuals_df, subgroup2_individuals_df])

    result_df['group_key'] = group_key
    result_df['granularity'] = len([i for i in possibility if i in range(len(attr_names)) and not sets_attr[i]])
    result_df['intersectionality'] = len([i for i in possibility if i in range(len(attr_names)) and sets_attr[i]])
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
        W: np.ndarray = None,
        nb_groups=100,
        nb_attributes=None,
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

    nb_protected = sum(sets_attr)
    nb_unprotected = len(sets_attr) - nb_protected

    protected_indexes = [index for index, value in enumerate(sets_attr) if value]
    unprotected_indexes = [index for index, value in enumerate(sets_attr) if not value]

    collision_tracker = CollisionTracker(nb_attributes)

    granularity_weights = calculate_weights(nb_unprotected)
    intersectionality_weights = calculate_weights(nb_protected)

    results = []
    collisions = 0

    with tqdm(total=nb_groups, desc="Generating data") as pbar:
        while len(results) < nb_groups:
            granularity = random.choices(range(1, len(unprotected_indexes) + 1), weights=granularity_weights)[0]
            intersectionality = random.choices(range(1, len(protected_indexes) + 1), weights=intersectionality_weights)[
                0]

            possible_gran = list(itertools.combinations(unprotected_indexes, granularity))
            possible_intersec = list(itertools.combinations(protected_indexes, intersectionality))
            possibilities = list(itertools.product(possible_gran, possible_intersec))
            possibilities = list(map(lambda x: tuple(itertools.chain.from_iterable(x)), possibilities))

            random.shuffle(possibilities)

            group_generated = False
            for possibility in possibilities:
                if not collision_tracker.is_collision(possibility):
                    collision_tracker.add_combination(possibility)
                    group = create_group(
                        possibility, attr_categories, sets_attr, correlation_matrix, gen_order, W,
                        min_similarity, max_similarity, min_alea_uncertainty, max_alea_uncertainty,
                        min_epis_uncertainty, max_epis_uncertainty, min_frequency, max_frequency,
                        min_diff_subgroup_size, max_diff_subgroup_size, max_group_size, attr_names
                    )
                    results.append(group)
                    pbar.update(1)
                    group_generated = True
                    break

            if not group_generated:
                collisions += 1
                if collisions > nb_groups * 2:
                    print(f"Warning: Unable to generate {nb_groups} groups. Generated {len(results)} groups.")
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

    results = results.sort_values(['group_key', 'couple_key'])

    results[f'diff_outcome'] = results.groupby(['group_key', 'couple_key'])[outcome_column].diff().abs().bfill()
    results['diff_variation'] = coefficient_of_variation(results['diff_outcome'])

    protected_attr = {k: e for k, e in zip(attr_names, sets_attr)}

    results = DiscriminationDataFrame(results)

    data = DiscriminationData(
        dataframe=results,
        categorical_columns=list(attr_names) + [outcome_column],
        attributes=protected_attr,
        collisions=collisions,
        nb_groups=nb_groups,
        max_group_size=max_group_size,
        hiddenlayers_depth=W.shape[0],
        outcome_column=outcome_column
    )

    # Calculate and add relevance metrics
    relevance_metrics = calculate_relevance(data)
    data.relevance_metrics = relevance_metrics

    # Merge relevance metrics with the main dataframe
    data.dataframe = data.dataframe.merge(relevance_metrics, left_on='group_key', right_index=True, how='left')

    return data


# %%

data = generate_data(
    nb_attributes=10,
    min_number_of_classes=8,
    max_number_of_classes=10,
    prop_protected_attr=0.4,
    nb_groups=1000,
    max_group_size=50,
    categorical_outcome=True,
    nb_categories_outcome=4
)

print(f"Generated {len(data.dataframe)} samples in {data.nb_groups} groups")
print(f"Collisions: {data.collisions}")


# %%%

def plot_and_print_metric_distributions_with_uncertainty(data_list, num_bins=10):
    metrics = [
        'granularity', 'intersectionality', 'diff_subgroup_size', 'actual_similarity',
        'actual_alea_uncertainty', 'actual_epis_uncertainty', 'actual_mean_diff_outcome_subgroups', 'relevance'
    ]

    # Combine all experiment results
    combined_data = pd.concat([exp_data for exp_data in data_list if not exp_data.empty])

    fig, axes = plt.subplots(4, 2, figsize=(15, 20), tight_layout=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        # Remove NaN values
        clean_data = combined_data[metric].dropna()

        if clean_data.empty:
            print(f"\nWarning: All values for {metric} are NaN. Skipping this metric.")
            ax.text(0.5, 0.5, f"No valid data for {metric}", ha='center', va='center')
            continue

        try:
            # Create histogram
            n, bins, patches = ax.hist(clean_data, bins=num_bins, edgecolor='black')

            # Calculate bin centers
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            # Calculate mean and standard error for each bin across experiments
            bin_means = []
            bin_errors = []
            for j in range(len(n)):
                bin_data = [exp_data[(exp_data[metric] >= bins[j]) & (exp_data[metric] < bins[j + 1])][metric]
                            for exp_data in data_list]
                bin_counts = [len(bd) for bd in bin_data]
                bin_means.append(np.mean(bin_counts))
                bin_errors.append(stats.sem(bin_counts) if len(bin_counts) > 0 else 0)

            # Plot error bars
            ax.errorbar(bin_centers, bin_means, yerr=bin_errors, fmt='none', ecolor='red', capsize=3)

            # Add labels and title
            ax.set_xlabel(metric)
            ax.set_ylabel('Average Frequency')
            ax.set_title(f'Distribution of {metric}')

            # Add percentage labels on top of each bar
            total_count = sum(bin_means)
            for j, (mean, rect) in enumerate(zip(bin_means, patches)):
                percentage = mean / total_count * 100
                ax.text(rect.get_x() + rect.get_width() / 2., mean,
                        f'{percentage:.1f}%',
                        ha='center', va='bottom', rotation=90, fontsize=8)

            # Adjust y-axis to make room for percentage labels
            ax.set_ylim(top=ax.get_ylim()[1] * 1.2)

            # Print bin information
            print(f"\nDistribution of {metric}:")
            print(f"Total experiments: {len(data_list)}")
            print("\nBin ranges, average counts, percentages, and standard errors:")
            for k in range(len(bin_means)):
                bin_start = bins[k]
                bin_end = bins[k + 1]
                avg_count = bin_means[k]
                percentage = (avg_count / total_count) * 100
                std_error = bin_errors[k]
                print(f"Bin {k + 1}: {bin_start:.2f} to {bin_end:.2f}")
                print(f"  Average Count: {avg_count:.2f}")
                print(f"  Percentage: {percentage:.1f}%")
                print(f"  Standard Error: {std_error:.4f}")

        except Exception as e:
            print(f"\nError processing {metric}: {str(e)}")
            ax.text(0.5, 0.5, f"Error processing {metric}", ha='center', va='center')

    plt.show()


# %%

def run_experiments_and_visualize(
        num_experiments: int = 10,
        nb_attributes: int = 10,
        min_number_of_classes: int = 8,
        max_number_of_classes: int = 10,
        prop_protected_attr: float = 0.4,
        nb_groups: int = 1000,
        max_group_size: int = 50,
        categorical_outcome: bool = True,
        nb_categories_outcome: int = 4
) -> None:
    experiment_results: List[pd.DataFrame] = []

    for i in range(num_experiments):
        print(f"Running experiment {i + 1}/{num_experiments}")
        data = generate_data(
            nb_attributes=nb_attributes,
            min_number_of_classes=min_number_of_classes,
            max_number_of_classes=max_number_of_classes,
            prop_protected_attr=prop_protected_attr,
            nb_groups=nb_groups,
            max_group_size=max_group_size,
            categorical_outcome=categorical_outcome,
            nb_categories_outcome=nb_categories_outcome
        )
        experiment_results.append(data.dataframe)

    print("All experiments completed. Plotting results...")
    plot_and_print_metric_distributions_with_uncertainty(experiment_results)


# %%

run_experiments_and_visualize()


# %%
def create_parallel_coordinates_plot(data):
    # Group the data by group_key and calculate mean values for each property
    group_properties = data.groupby('group_key').agg({
        'granularity': 'mean',
        'intersectionality': 'mean',
        'diff_subgroup_size': 'mean',
        'actual_similarity': 'mean',
        'actual_alea_uncertainty': 'mean',
        'actual_epis_uncertainty': 'mean',
        'actual_mean_diff_outcome_subgroups': 'mean',
        'relevance': 'mean'
    }).reset_index().copy()

    group_properties.rename(columns={'actual_similarity': 'similarity',
                                     'actual_alea_uncertainty': 'alea_uncertainty',
                                     'actual_epis_uncertainty': 'epis_uncertainty',
                                     'actual_mean_diff_outcome_subgroups': 'diff_outcome'}, inplace=True)

    for column in group_properties.columns:
        if column != 'group_key':
            group_properties[column] = pd.to_numeric(group_properties[column], errors='coerce')

    # Remove any rows with NaN values
    group_properties = group_properties.dropna()

    # Normalize the data to a 0-1 range for each property
    columns_to_plot = ['granularity', 'intersectionality', 'diff_subgroup_size', 'similarity',
                       'alea_uncertainty', 'epis_uncertainty', 'diff_outcome', 'relevance']
    normalized_data = group_properties[columns_to_plot].copy()
    for column in columns_to_plot:
        min_val = normalized_data[column].min()
        max_val = normalized_data[column].max()
        if min_val != max_val:
            normalized_data[column] = (normalized_data[column] - min_val) / (max_val - min_val)
        else:
            normalized_data[column] = 0.5  # Set to middle value if all values are the same

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create x-coordinates for each property
    x = list(range(len(columns_to_plot)))

    # Create colormap
    norm = Normalize(vmin=group_properties['relevance'].min(), vmax=group_properties['relevance'].max())
    cmap = plt.get_cmap('viridis')

    # Plot each group
    for i, row in normalized_data.iterrows():
        y = row[columns_to_plot].values
        color = cmap(norm(group_properties.loc[i, 'relevance']))
        ax.plot(x, y, c=color, alpha=0.5)

    # Customize the plot
    ax.set_xticks(x)
    ax.set_xticklabels(columns_to_plot, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.set_title('Parallel Coordinates Plot of Discrimination Metrics')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Normalized Values')

    # Add gridlines
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Relevance')

    plt.tight_layout()
    plt.show()


create_parallel_coordinates_plot(data.dataframe)


# %%
def plot_and_print_metric_distributions(data, num_bins=10):
    metrics = [
        'granularity', 'intersectionality', 'diff_subgroup_size', 'actual_similarity',
        'actual_alea_uncertainty', 'actual_epis_uncertainty', 'actual_mean_diff_outcome_subgroups', 'relevance'
    ]

    group_properties = data.groupby('group_key').agg({metric: 'mean' for metric in metrics}).reset_index()

    fig, axes = plt.subplots(4, 2, figsize=(15, 20), tight_layout=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        # Remove NaN values
        clean_data = group_properties[metric].dropna()

        if clean_data.empty:
            print(f"\nWarning: All values for {metric} are NaN. Skipping this metric.")
            ax.text(0.5, 0.5, f"No valid data for {metric}", ha='center', va='center')
            continue

        try:
            # Create histogram
            n, bins, patches = ax.hist(clean_data, bins=num_bins, edgecolor='black')

            # Add labels and title
            ax.set_xlabel(metric)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {metric}')

            # Add percentage labels on top of each bar
            total_count = len(clean_data)
            for j, rect in enumerate(patches):
                height = rect.get_height()
                percentage = height / total_count * 100
                ax.text(rect.get_x() + rect.get_width() / 2., height,
                        f'{percentage:.1f}%',
                        ha='center', va='bottom', rotation=90, fontsize=8)

            # Adjust y-axis to make room for percentage labels
            ax.set_ylim(top=ax.get_ylim()[1] * 1.2)

            # Print bin information
            print(f"\nDistribution of {metric}:")
            print(f"Total data points: {total_count}")
            print("\nBin ranges, counts, and percentages:")
            for k in range(len(n)):
                bin_start = bins[k]
                bin_end = bins[k + 1]
                count = n[k]
                percentage = (count / total_count) * 100
                print(f"Bin {k + 1}: {bin_start:.2f} to {bin_end:.2f}")
                print(f"  Count: {count}")
                print(f"  Percentage: {percentage:.1f}%")

        except Exception as e:
            print(f"\nError processing {metric}: {str(e)}")
            ax.text(0.5, 0.5, f"Error processing {metric}", ha='center', va='center')

    plt.show()


# Usage
plot_and_print_metric_distributions(data.dataframe)


# %%

def create_parallel_coordinates_plot(data):
    group_properties = data.groupby('group_key').agg({
        'granularity': 'mean',
        'intersectionality': 'mean',
        'diff_subgroup_size': 'mean',
        'actual_similarity': 'mean',
        'actual_alea_uncertainty': 'mean',
        'actual_epis_uncertainty': 'mean',
        'actual_mean_diff_outcome_subgroups': 'mean',
        'relevance': 'mean'
    }).reset_index()

    group_properties.rename(columns={
        'actual_similarity': 'similarity',
        'actual_alea_uncertainty': 'alea_uncertainty',
        'actual_epis_uncertainty': 'epis_uncertainty',
        'actual_mean_diff_outcome_subgroups': 'diff_outcome'
    }, inplace=True)

    group_properties = group_properties.dropna()

    columns_to_plot = ['granularity', 'intersectionality', 'diff_subgroup_size', 'similarity',
                       'alea_uncertainty', 'epis_uncertainty', 'diff_outcome', 'relevance']
    normalized_data = group_properties[columns_to_plot].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = list(range(len(columns_to_plot)))

    norm = Normalize(vmin=group_properties['relevance'].min(), vmax=group_properties['relevance'].max())
    cmap = plt.get_cmap('viridis')

    for _, row in normalized_data.iterrows():
        y = row[columns_to_plot].values
        color = cmap(norm(group_properties.loc[_, 'relevance']))
        ax.plot(x, y, c=color, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(columns_to_plot, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.set_title('Parallel Coordinates Plot of Discrimination Metrics')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Normalized Values')
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Relevance')

    plt.tight_layout()
    plt.show()


def plot_metric_distributions(data, num_bins=10):
    metrics = [
        'granularity', 'intersectionality', 'diff_subgroup_size', 'actual_similarity',
        'actual_alea_uncertainty', 'actual_epis_uncertainty', 'actual_mean_diff_outcome_subgroups', 'relevance'
    ]

    group_properties = data.groupby('group_key').agg({metric: 'mean' for metric in metrics}).reset_index()

    fig, axes = plt.subplots(4, 2, figsize=(15, 20), tight_layout=True)
    axes = axes.flatten()

    for metric, ax in zip(metrics, axes):
        n, bins, patches = ax.hist(group_properties[metric], bins=num_bins, edgecolor='black')

        ax.set_xlabel(metric)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {metric}')

        for rect in patches:
            height = rect.get_height()
            percentage = height / len(group_properties) * 100
            ax.text(rect.get_x() + rect.get_width() / 2., height,
                    f'{percentage:.1f}%',
                    ha='center', va='bottom', rotation=90, fontsize=8)

        ax.set_ylim(top=ax.get_ylim()[1] * 1.2)

    plt.show()


def plot_correlation_heatmap(data):
    """
    Plot a correlation heatmap for the metrics in the dataset.
    """
    metrics = [
        'granularity', 'intersectionality', 'diff_subgroup_size', 'actual_similarity',
        'actual_alea_uncertainty', 'actual_epis_uncertainty', 'actual_mean_diff_outcome_subgroups', 'relevance'
    ]

    group_properties = data.groupby('group_key').agg({metric: 'mean' for metric in metrics}).reset_index()
    correlation_matrix = group_properties[metrics].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of Discrimination Metrics')
    plt.tight_layout()
    plt.show()


def plot_pairplot(data):
    """
    Create a pairplot to visualize relationships between metrics.
    """
    metrics = [
        'granularity', 'intersectionality', 'diff_subgroup_size', 'actual_similarity',
        'actual_alea_uncertainty', 'actual_epis_uncertainty', 'actual_mean_diff_outcome_subgroups', 'relevance'
    ]

    group_properties = data.groupby('group_key').agg({metric: 'mean' for metric in metrics}).reset_index()
    sns.pairplot(group_properties[metrics], corner=True, diag_kind='kde')
    plt.suptitle('Pairplot of Discrimination Metrics', y=1.02)
    plt.tight_layout()
    plt.show()


def analyze_group_composition(data):
    """
    Analyze the composition of groups based on protected attributes.
    """
    protected_attrs = [attr for attr, is_protected in data.attributes.items() if is_protected]

    group_composition = data.dataframe.groupby('group_key')[protected_attrs].agg(
        lambda x: x.value_counts().index[0])

    composition_summary = group_composition.apply(pd.Series.value_counts).fillna(0)

    plt.figure(figsize=(12, 6))
    composition_summary.plot(kind='bar', stacked=True)
    plt.title('Group Composition Based on Protected Attributes')
    plt.xlabel('Protected Attributes')
    plt.ylabel('Number of Groups')
    plt.legend(title='Attribute Value', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return composition_summary


def evaluate_fairness(data):
    """
    Evaluate fairness metrics for the generated dataset.
    """
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric

    protected_attribute = data.protected_attributes[0]

    dataset = BinaryLabelDataset(
        df=data.dataframe,
        label_name=data.outcome_column,
        protected_attribute_names=[protected_attribute],
        favorable_label=1,
        unfavorable_label=0
    )

    metric = BinaryLabelDatasetMetric(dataset,
                                      unprivileged_groups=[{protected_attribute: 0}],
                                      privileged_groups=[{protected_attribute: 1}])

    fairness_metrics = {
        "Statistical Parity Difference": metric.statistical_parity_difference(),
        "Disparate Impact": metric.disparate_impact(),
        "Equal Opportunity Difference": metric.equal_opportunity_difference(),
        "Average Odds Difference": metric.average_odds_difference(),
        "Theil Index": metric.theil_index()
    }

    return fairness_metrics


def analyze_data(data):
    print(f"Generated {len(data.dataframe)} samples in {data.nb_groups} groups")
    print(f"Collisions: {data.collisions}")

    create_parallel_coordinates_plot(data.dataframe)
    plot_metric_distributions(data.dataframe)
    plot_correlation_heatmap(data.dataframe)
    plot_pairplot(data.dataframe)

    composition_summary = analyze_group_composition(data)
    print("\nGroup Composition Summary:")
    print(composition_summary)

    fairness_metrics = evaluate_fairness(data)
    print("\nFairness Metrics:")
    for metric, value in fairness_metrics.items():
        print(f"{metric}: {value:.4f}")

    return data


# %%
import seaborn as sns

np.random.seed(42)
random.seed(42)

# Generate and analyze data
generated_data = analyze_data(data)
