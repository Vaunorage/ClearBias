import copy
import datetime
import hashlib
import json
import math
import os
import pickle
import random
from pathlib import Path

import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh
from tqdm import tqdm
from dataclasses import dataclass, field
from pandas import DataFrame
from typing import Literal, TypeVar, Any, List, Dict, Tuple, Optional
from scipy.stats import norm, multivariate_normal, spearmanr
import pandas as pd
import warnings

from data_generator.main_old import DiscriminationData
from path import HERE
from uncertainty_quantification.main import UncertaintyRandomForest, AleatoricUncertainty, EpistemicUncertainty

warnings.filterwarnings("ignore")
# Ignore specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def generate_cache_key(params: dict) -> str:
    # Sort the parameters to ensure consistent ordering
    ordered_params = dict(sorted(params.items()))

    # Convert numpy arrays and other complex types to serializable format
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple, dict, str, int, float, bool, type(None))):
            return obj
        return str(obj)

    serializable_params = {k: make_serializable(v) for k, v in ordered_params.items()}

    # Create a hash of the parameters
    param_str = json.dumps(serializable_params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()


class DataCache:

    def __init__(self):
        self.cache_dir = HERE.joinpath(".cache/discrimination_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a metadata file to track cache contents
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load metadata from file or create if doesn't exist."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def get_cache_path(self, cache_key: str) -> Path:
        """Get the full path for a cache file."""
        return self.cache_dir / f"{cache_key}.pkl"

    def save(self, data: DiscriminationData, params: dict):
        cache_key = generate_cache_key(params)
        cache_path = self.get_cache_path(cache_key)

        # Save the data
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

        # Update metadata
        self.metadata[cache_key] = {
            'params': params,
            'created_at': str(datetime.datetime.now()),
            'file_path': str(cache_path)
        }
        self._save_metadata()

    def load(self, params: dict) -> Optional[DiscriminationData]:
        cache_key = generate_cache_key(params)
        cache_path = self.get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cached data: {e}")
                return None
        return None

    def clear(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        self.metadata = {}
        self._save_metadata()

    def get_cache_info(self) -> dict:
        """Get information about the cache contents."""
        return {
            'cache_dir': str(self.cache_dir),
            'num_cached_items': len(self.metadata),
            'total_size_mb': sum(os.path.getsize(f) for f in self.cache_dir.glob("*.pkl")) / (1024 * 1024),
            'items': self.metadata
        }


class GaussianCopulaCategorical:
    def __init__(self, marginals, correlation_matrix, excluded_combinations=None, corr_matrix_randomness=1.0):
        self.marginals = [np.array(m) for m in marginals]
        self.correlation_matrix = np.array(correlation_matrix)
        self.dim = len(marginals)
        self.excluded_combinations = set(map(tuple, excluded_combinations or []))
        self.corr_matrix_randomness = np.clip(corr_matrix_randomness, 0.0, 1.0)

        # Validate and normalize probability distributions
        self.validate_and_normalize_marginals()

        # Convert marginal probabilities to cumulative probabilities
        self.cum_probabilities = []
        for m in self.marginals:
            cum_prob = np.cumsum(m)
            # Ensure last probability is exactly 1
            cum_prob = cum_prob / cum_prob[-1]
            self.cum_probabilities.append(cum_prob[:-1])

    def validate_and_normalize_marginals(self):
        """Validate and normalize probability distributions, handling edge cases."""
        for i in range(len(self.marginals)):
            m = self.marginals[i]

            # Replace any NaN values with 0
            m = np.nan_to_num(m, nan=0.0)

            # Ensure all probabilities are non-negative
            m = np.maximum(m, 0.0)

            # If sum is 0, create uniform distribution
            if np.sum(m) == 0:
                m = np.ones_like(m) / len(m)
            else:
                # Normalize to sum to 1
                m = m / np.sum(m)

            self.marginals[i] = m

    def is_excluded(self, sample):
        return tuple(sample) in self.excluded_combinations

    def generate_samples(self, n_samples):
        samples = []
        attempts = 0
        max_attempts = n_samples * 10

        while len(samples) < n_samples and attempts < max_attempts:
            try:
                if self.corr_matrix_randomness == 1.0:
                    # Completely random sampling
                    categorical_sample = np.array([
                        np.random.choice(len(m), p=m)
                        for m in self.marginals
                    ])
                elif self.corr_matrix_randomness == 0.0:
                    # Strict correlation-based sampling
                    gaussian_samples = np.random.multivariate_normal(
                        mean=np.zeros(self.dim),
                        cov=self.correlation_matrix,
                        size=1
                    ).flatten()
                    uniform_samples = norm.cdf(gaussian_samples)
                    categorical_sample = np.array([
                        np.searchsorted(cum_prob, u)
                        for cum_prob, u in zip(self.cum_probabilities, uniform_samples)
                    ])
                else:
                    # Blend between random and correlated sampling
                    if np.random.random() < self.corr_matrix_randomness:
                        # Random sampling
                        categorical_sample = np.array([
                            np.random.choice(len(m), p=m)
                            for m in self.marginals
                        ])
                    else:
                        # Correlation-based sampling
                        gaussian_samples = np.random.multivariate_normal(
                            mean=np.zeros(self.dim),
                            cov=self.correlation_matrix,
                            size=1
                        ).flatten()
                        uniform_samples = norm.cdf(gaussian_samples)
                        categorical_sample = np.array([
                            np.searchsorted(cum_prob, u)
                            for cum_prob, u in zip(self.cum_probabilities, uniform_samples)
                        ])

                if not self.is_excluded(categorical_sample):
                    samples.append(categorical_sample)

            except (ValueError, IndexError) as e:
                print(f"Warning: Caught error during sample generation: {str(e)}")
                # Re-normalize probabilities and try again
                self.validate_and_normalize_marginals()

            attempts += 1

        if len(samples) < n_samples:
            raise RuntimeError(f"Could only generate {len(samples)} valid samples out of {n_samples} requested")

        return np.array(samples)


def coefficient_of_variation(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    if mean == 0 or np.isnan(mean) or np.isnan(std_dev):
        return 0  # or another appropriate value
    cv = (std_dev / mean) * 100
    return cv


@dataclass
class DataSchema:
    attr_categories: List[List[str]]
    protected_attr: List[str]
    attr_names: List[str]
    categorical_distribution: Dict[str, List[float]]
    correlation_matrix: np.ndarray
    gen_order: List[str]


def generate_schema_from_dataframe(
        df: pd.DataFrame,
        protected_columns: List[str] = None,
        attr_prefix: str = "Attr",
        ensure_positive_definite: bool = True
) -> Tuple[DataSchema, np.ndarray]:
    # Filter columns that start with the attribute prefix
    attr_columns = list(df.columns)

    if not attr_columns:
        raise ValueError(f"No columns found starting with '{attr_prefix}'")

    # If protected_columns not provided, infer from '_T' suffix
    if protected_columns is None:
        protected_columns = [col for col in attr_columns if col.endswith('_T')]

    # Generate attr_categories
    attr_categories = []
    for col in attr_columns:
        # Get unique values excluding NaN
        unique_vals = sorted(df[col].dropna().unique())
        # Convert to int if possible
        try:
            unique_vals = [int(x) for x in unique_vals]
        except (ValueError, TypeError):
            pass
        # Ensure -1 is first if present
        if -1 in unique_vals:
            unique_vals.remove(-1)
            unique_vals = [-1] + sorted(unique_vals)
        attr_categories.append(unique_vals)

    # Generate protected_attr flags
    protected_attr = [col in protected_columns for col in attr_columns]

    # Calculate correlation matrix using Spearman correlation
    correlation_matrix = np.zeros((len(attr_columns), len(attr_columns)))
    for i, col1 in enumerate(attr_columns):
        for j, col2 in enumerate(attr_columns):
            # Handle the case where we have constant columns
            if df[col1].nunique() == 1 or df[col2].nunique() == 1:
                correlation_matrix[i, j] = 1.0 if i == j else 0.0
            else:
                # Calculate Spearman correlation, ignoring -1 values
                mask = (df[col1] != -1) & (df[col2] != -1)
                if mask.any():
                    corr, _ = spearmanr(df[col1][mask], df[col2][mask], nan_policy='omit')
                    correlation_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                else:
                    correlation_matrix[i, j] = 1.0 if i == j else 0.0

    # Ensure the matrix is symmetric
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2

    if ensure_positive_definite:
        # Make the correlation matrix positive definite if needed
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        if np.any(eigenvalues < 0):
            # Replace negative eigenvalues with small positive values
            eigenvalues[eigenvalues < 0] = 1e-6
            correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

            # Rescale to ensure diagonal elements are 1
            scaling = np.sqrt(np.diag(correlation_matrix))
            correlation_matrix = correlation_matrix / scaling[:, None] / scaling[None, :]

    # Calculate categorical distribution
    categorical_distribution = {}
    for i, col in enumerate(attr_columns):
        # Calculate probabilities for each category, excluding -1
        value_counts = df[col][df[col] != -1].value_counts(normalize=True)
        # Convert to dictionary and ensure all categories are present
        probs = {cat: 0.0 for cat in attr_categories[i] if cat != -1}
        probs.update(value_counts.to_dict())
        categorical_distribution[col] = probs

    # Generate generation order if not provided (random order)
    gen_order = list(range(1, len(attr_columns) + 1))
    random.shuffle(gen_order)

    # Create DataSchema
    schema = DataSchema(
        attr_categories=attr_categories,
        protected_attr=protected_attr,
        attr_names=attr_columns,
        categorical_distribution=categorical_distribution,
        correlation_matrix=correlation_matrix,
        gen_order=gen_order
    )

    return schema, correlation_matrix


def generate_data_schema(min_number_of_classes, max_number_of_classes, nb_attributes,
                         prop_protected_attr) -> DataSchema:
    attr_categories = []
    attr_names = []

    if nb_attributes < 2:
        raise ValueError("nb_attributes must be at least 2 to ensure both protected and unprotected attributes.")

    protected_attr = [True, False]

    for _ in range(nb_attributes - 2):
        protected_attr.append(random.random() < prop_protected_attr)

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

    correlation_matrix = generate_valid_correlation_matrix(nb_attributes)

    categorical_distribution = generate_categorical_distribution(attr_categories, attr_names)

    # Generate generation order if not provided
    gen_order = list(range(1, nb_attributes + 1))
    random.shuffle(gen_order)

    res = DataSchema(attr_categories=attr_categories, protected_attr=protected_attr, attr_names=attr_names,
                     categorical_distribution=categorical_distribution, correlation_matrix=correlation_matrix,
                     gen_order=gen_order)
    return res


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


def sort_two_strings(str1: str, str2: str) -> tuple[str, str]:
    if str1 <= str2:
        return (str1, str2)
    return (str2, str1)


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
        return list(self.attributes)

    @property
    def protected_attributes(self):
        return [k for k, v in self.attributes.items() if v]

    @property
    def non_protected_attributes(self):
        return [k for k, v in self.attributes.items() if not v]

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

    @property
    def schema(self):
        return '|'.join(
            [''.join(list(map(str, e))).replace('-1', '') for e in self.attr_possible_values.values()])

    def __post_init__(self):
        self.input_bounds = []
        for col in list(self.attributes):
            min_val = math.floor(self.xdf[col].min())
            max_val = math.ceil(self.xdf[col].max())
            self.input_bounds.append([min_val, max_val])

    @staticmethod
    def generate_individual_synth_combinations(df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = list(filter(lambda x: 'Attr' in x, df.columns))

        all_combinations = []

        grouped = df.groupby('group_key')

        for group_key, group_data in grouped:
            subgroups = group_data['subgroup_key'].unique()

            if len(subgroups) != 2:
                raise ValueError(f"Group {group_key} does not have exactly 2 subgroups")

            subgroup1_data = df[
                (df['group_key'] == group_key) &
                (df['subgroup_key'] == subgroups[0])
                ]

            subgroup2_data = df[
                (df['group_key'] == group_key) &
                (df['subgroup_key'] == subgroups[1])
                ]

            for idx1, row1 in subgroup1_data.iterrows():
                for idx2, row2 in subgroup2_data.iterrows():
                    # Sort the individual keys
                    sorted_keys = sort_two_strings(row1['indv_key'], row2['indv_key'])

                    combination = {
                        'group_key': group_key,
                        'subgroup1_key': subgroups[0],
                        'subgroup2_key': subgroups[1],
                        'indv_key_1': sorted_keys[0],
                        'indv_key_2': sorted_keys[1]
                    }

                    # Need to match the features with the correct sorted individual
                    if sorted_keys[0] == row1['indv_key']:
                        # Keys are already in original order
                        for feature in feature_cols:
                            combination[f'{feature}_1'] = row1[feature]
                            combination[f'{feature}_2'] = row2[feature]
                    else:
                        # Keys were swapped, so swap the features too
                        for feature in feature_cols:
                            combination[f'{feature}_1'] = row2[feature]
                            combination[f'{feature}_2'] = row1[feature]

                    all_combinations.append(combination)

        result_df = pd.DataFrame(all_combinations)

        result_df['couple_key'] = result_df.apply(lambda x: f"{x['indv_key_1']}-{x['indv_key_2']}", axis=1)

        column_order = ['group_key', 'subgroup1_key', 'subgroup2_key',
                        'indv_key_1', 'indv_key_2', 'couple_key']

        column_order.extend([f'{feature}_1' for feature in feature_cols])
        column_order.extend([f'{feature}_2' for feature in feature_cols])

        res = result_df[column_order]

        return res.drop_duplicates()


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
    def __init__(self, schema: DataSchema, graph, gen_order, outcome_weights, outcome_bias, subgroup_bias,
                 epis_uncertainty, alea_uncertainty, corr_matrix_randomness=1.0, n_estimators=50,
                 categorical_distribution=None, categorical_influence=0.5):
        self.schema = schema
        self.graph = np.array(graph)
        self.gen_order = [i - 1 for i in gen_order]
        self.n_attributes = len(schema.attr_names)
        self.outcome_weights = outcome_weights
        self.outcome_bias = outcome_bias
        self.subgroup_bias = subgroup_bias
        self.epis_uncertainty = epis_uncertainty
        self.alea_uncertainty = alea_uncertainty
        self.n_estimators = n_estimators
        self.corr_matrix_randomness = np.clip(corr_matrix_randomness, 0.0, 1.0)
        self.categorical_distribution = categorical_distribution or {}
        self.categorical_influence = np.clip(categorical_influence, 0.0, 1.0)

    def _get_attribute_probabilities(self, attr_idx, n_values):
        """
        Get the probability distribution for a specific attribute.

        Args:
            attr_idx (int): Index of the attribute
            n_values (int): Number of possible values for the attribute

        Returns:
            np.ndarray: Probability distribution for the attribute
        """
        attr_name = self.schema.attr_names[attr_idx]
        if attr_name in self.categorical_distribution:
            return np.array(self.categorical_distribution[attr_name])
        return np.ones(n_values) / n_values  # Uniform distribution if not specified

    def generate_dataset_with_outcome(self, n_samples, predetermined_values, is_subgroup1):
        samples = np.zeros((n_samples, self.n_attributes), dtype=int)

        # Fill in predetermined values
        if predetermined_values:
            mask = np.array(predetermined_values) != -1
            samples[:, mask] = np.array(predetermined_values)[mask][None, :]

        # Generate remaining values
        for attr in self.gen_order:
            mask = samples[:, attr] == 0
            if not np.any(mask):
                continue

            n_to_generate = np.sum(mask)
            n_values = len(self.schema.attr_categories[attr]) - 1

            # Get categorical probabilities
            base_categorical_probs = self._get_attribute_probabilities(attr, n_values)
            categorical_probs = np.tile(base_categorical_probs, (n_to_generate, 1))

            # Calculate correlation-based probabilities
            other_attrs = np.arange(self.n_attributes) != attr
            other_attrs_indices = np.where(other_attrs)[0]
            other_values = samples[mask][:, other_attrs]
            correlations = self.graph[attr][other_attrs]
            correlations = np.clip(correlations, 0.001, 0.999)  # Avoid extreme values

            # Initialize correlation probabilities
            corr_probs = np.ones((n_to_generate, n_values))
            for value in range(n_values):
                prob_multipliers = np.ones(n_to_generate)
                for i, other_attr_idx in enumerate(other_attrs_indices):
                    corr = correlations[i]
                    attr_values = other_values[:, i]
                    matches = (attr_values == value)
                    divisor = max(n_values - 1, 1e-10)
                    prob_multipliers *= np.where(matches, corr, (1 - corr) / divisor)
                corr_probs[:, value] = prob_multipliers

            # Handle NaN and zero values in correlation probabilities
            corr_probs = np.nan_to_num(corr_probs, nan=1.0 / n_values)
            zero_rows = np.all(corr_probs == 0, axis=1)
            corr_probs[zero_rows] = 1.0 / n_values

            # Normalize correlation probabilities
            row_sums = corr_probs.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            corr_probs /= row_sums

            uniform_probs = np.ones((n_to_generate, n_values)) / n_values

            # Step 1: Blend correlation matrix with uniform distribution
            correlation_blend = (1 - self.corr_matrix_randomness) * corr_probs + \
                                self.corr_matrix_randomness * uniform_probs

            # Step 2: Blend the result with categorical distribution
            final_probs = ((1 - self.categorical_influence) * uniform_probs
                           + self.categorical_influence + categorical_probs)

            # Add small noise to prevent identical outcomes
            # noise = np.random.normal(0, 0.01, final_probs.shape)
            # final_probs = np.abs(final_probs + noise)

            # Ensure valid probabilities
            final_probs = np.maximum(final_probs, 0)
            row_sums = final_probs.sum(axis=1, keepdims=True)
            final_probs /= row_sums

            # Generate values
            samples[mask, attr] = np.array([np.random.choice(n_values, p=p) for p in final_probs])

        # Generate outcomes
        X = (samples - np.mean(samples, axis=0)) / (np.std(samples, axis=0) + 1e-8)

        ensemble_preds = []
        for i in range(self.n_estimators):
            weight_noise = np.random.normal(0, self.epis_uncertainty, size=self.outcome_weights.shape)
            perturbed_weights = self.outcome_weights * (1 + weight_noise)
            base_pred = np.dot(X, perturbed_weights) + self.outcome_bias
            base_pred += self.subgroup_bias if is_subgroup1 else -self.subgroup_bias
            noisy_pred = base_pred + np.random.normal(0, self.alea_uncertainty, size=n_samples)
            probs = 1 / (1 + np.exp(-noisy_pred))
            ensemble_preds.append(probs)

        ensemble_preds = np.array(ensemble_preds).T
        final_predictions = np.mean(ensemble_preds, axis=1)
        epistemic_uncertainty = np.var(ensemble_preds, axis=1)
        aleatoric_uncertainty = np.mean(ensemble_preds * (1 - ensemble_preds), axis=1)

        return [(samples[i].tolist(),
                 final_predictions[i],
                 epistemic_uncertainty[i],
                 aleatoric_uncertainty[i])
                for i in range(n_samples)]


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


def calculate_actual_uncertainties(data):
    X = data.dataframe[data.feature_names].values  # Convert to numpy array
    y = data.dataframe[data.outcome_column].values  # Convert to numpy array

    # Initialize base random forest for comparison
    urf = UncertaintyRandomForest(n_estimators=50, random_state=42)
    urf.fit(X, y)
    mean_pred, base_epistemic, base_aleatoric = urf.predict_with_uncertainty(X)

    # Initialize result columns
    data.dataframe['calculated_epistemic_random_forest'] = base_epistemic
    data.dataframe['calculated_aleatoric_random_forest'] = base_aleatoric

    # Calculate all Aleatoric Uncertainties
    aleatoric_methods = ['entropy', 'probability_margin', 'label_smoothing']
    for method in aleatoric_methods:
        aleatoric_model = AleatoricUncertainty(
            method=method,
            temperature=1.0
        )
        aleatoric_model.fit(X, y)
        probs, aleatoric_uncertainty = aleatoric_model.predict_uncertainty(X)
        data.dataframe[f'calculated_aleatoric_{method}'] = aleatoric_uncertainty

    # Calculate all Epistemic Uncertainties
    epistemic_methods = ['ensemble', 'mc_dropout', 'evidential']
    epistemic_params = {
        'ensemble': {'n_estimators': 5, 'dropout_rate': None, 'n_forward_passes': None},
        'mc_dropout': {'n_estimators': None, 'dropout_rate': 0.5, 'n_forward_passes': 30},
        'evidential': {'n_estimators': None, 'dropout_rate': None, 'n_forward_passes': None}
    }

    for method in epistemic_methods:
        params = epistemic_params[method]
        epistemic_model = EpistemicUncertainty(
            method=method,
            n_estimators=params['n_estimators'],
            dropout_rate=params['dropout_rate'],
            n_forward_passes=params['n_forward_passes']
        )
        epistemic_model.fit(X, y)
        probs, epistemic_uncertainty = epistemic_model.predict_uncertainty(X)
        data.dataframe[f'calculated_epistemic_{method}'] = epistemic_uncertainty

    # Aggregate results by group
    uncertainty_columns = [col for col in data.dataframe.columns
                           if col.startswith('calculated_')]

    res = data.dataframe.groupby('group_key')[uncertainty_columns].agg('mean')

    # Add combined uncertainty metrics for all combinations
    for epistemic_method in epistemic_methods:
        for aleatoric_method in aleatoric_methods:
            combined_col = f'combined_{epistemic_method}_{aleatoric_method}'
            res[combined_col] = (
                    res[f'calculated_epistemic_{epistemic_method}'] +
                    res[f'calculated_aleatoric_{aleatoric_method}']
            )

    # Add summary statistics
    res['calculated_epistemic'] = res[[col for col in res.columns
                                       if col.startswith('calculated_epistemic')]].mean(axis=1)
    res['calculated_aleatoric'] = res[[col for col in res.columns
                                       if col.startswith('calculated_aleatoric')]].mean(axis=1)
    res['calculated_combined'] = res[[col for col in res.columns
                                      if col.startswith('combined')]].mean(axis=1)
    return res


def calculate_actual_mean_diff_outcome(data):
    def calculate_group_diff(group_data):
        subgroup_keys = group_data['subgroup_key'].unique()
        if len(subgroup_keys) != 2:
            return np.nan

        subgroup1_outcome = group_data[group_data['subgroup_key'] == subgroup_keys[0]][data.outcome_column].mean()
        subgroup2_outcome = group_data[group_data['subgroup_key'] == subgroup_keys[1]][data.outcome_column].mean()

        return abs(subgroup1_outcome - subgroup2_outcome) / max(subgroup1_outcome, subgroup2_outcome)

    return data.dataframe.groupby('group_key', group_keys=False).apply(calculate_group_diff)


def calculate_actual_diff_outcome_from_avg(data):
    avg_outcome = data.dataframe[data.outcome_column].mean()

    def calculate_group_diff(group_data):
        unique_subgroups = group_data['subgroup_key'].unique()
        subgroup1_outcome = group_data[group_data['subgroup_key'] == unique_subgroups[0]][data.outcome_column]
        subgroup2_outcome = group_data[group_data['subgroup_key'] == unique_subgroups[1]][data.outcome_column]
        res = (abs(avg_outcome - subgroup1_outcome.mean()) + abs(avg_outcome - subgroup2_outcome.mean())) / 2
        return res

    return data.dataframe.groupby('group_key', group_keys=False).apply(calculate_group_diff)


def calculate_actual_metrics(data):
    """
    Calculate the actual metrics and relevance for each group.
    """
    actual_similarity = calculate_actual_similarity(data)
    actual_uncertainties = calculate_actual_uncertainties(data)
    actual_mean_diff_outcome = calculate_actual_mean_diff_outcome(data)
    actual_diff_outcome_from_avg = calculate_actual_diff_outcome_from_avg(data)

    # Merge these metrics into the main dataframe
    data.dataframe['calculated_similarity'] = data.dataframe['group_key'].map(actual_similarity)
    data.dataframe['calculated_epistemic_group'] = data.dataframe['group_key'].map(
        actual_uncertainties['calculated_epistemic'])
    data.dataframe['calculated_aleatoric_group'] = data.dataframe['group_key'].map(
        actual_uncertainties['calculated_aleatoric'])
    data.dataframe['calculated_magnitude'] = data.dataframe['group_key'].map(actual_mean_diff_outcome)
    data.dataframe['calculated_mean_demographic_disparity'] = data.dataframe['group_key'].map(
        actual_diff_outcome_from_avg)
    data.dataframe['calculated_uncertainty_group'] = data.dataframe.apply(
        lambda x: (x['calculated_epistemic_group'] + x['calculated_aleatoric_group']) / 2, axis=1)

    data.dataframe['calculated_intersectionality'] = data.dataframe['intersectionality_param'].copy() / len(
        data.protected_attributes)
    data.dataframe['calculated_granularity'] = data.dataframe['granularity_param'].copy() / len(
        data.non_protected_attributes)
    data.dataframe['calculated_group_size'] = data.dataframe['group_size'].copy()
    data.dataframe['calculated_subgroup_ratio'] = data.dataframe['diff_subgroup_size'].copy()
    data.dataframe = data.dataframe.loc[:, ~data.dataframe.columns.duplicated()]

    return data


def calculate_weights(n):
    return [1 / (i + 1) for i in range(n)]


def safe_normalize(p):
    """Safely normalize an array, handling the case where sum is zero."""
    sum_p = np.sum(p)
    if sum_p == 0:
        # If all probabilities are zero, return a uniform distribution
        return np.ones_like(p) / len(p)
    elif np.isnan(sum_p):
        # If the sum is NaN, replace NaNs with 0 and normalize
        p = np.nan_to_num(p)
        sum_p = np.sum(p)
        if sum_p == 0:
            return np.ones_like(p) / len(p)
        return p / sum_p
    return p / sum_p


def create_group(granularity, intersectionality,
                 possibility, data_schema, attr_categories, W,
                 subgroup_bias, corr_matrix_randomness,
                 categorical_influence,
                 min_similarity, max_similarity, min_alea_uncertainty, max_alea_uncertainty,
                 min_epis_uncertainty, max_epis_uncertainty, min_frequency, max_frequency,
                 min_diff_subgroup_size, max_diff_subgroup_size, min_group_size, max_group_size):
    attr_categories, sets_attr, correlation_matrix, gen_order, categorical_distribution, attr_names = (
        data_schema.attr_categories, data_schema.protected_attr, data_schema.correlation_matrix, data_schema.gen_order,
        data_schema.categorical_distribution, data_schema.attr_names)

    # Separate non-protected and protected attributes and reorder them
    non_protected_columns = [attr for attr, protected in zip(attr_names, sets_attr) if not protected]
    protected_columns = [attr for attr, protected in zip(attr_names, sets_attr) if protected]

    # Reorder attr_names so that non-protected attributes come first
    attr_names = non_protected_columns + protected_columns

    # Adjust attr_categories and sets_attr in the same order
    attr_categories = [attr_categories[attr_names.index(attr)] for attr in attr_names]
    sets_attr = [sets_attr[attr_names.index(attr)] for attr in attr_names]

    # Function to create sets based on the new ordering
    def make_sets(possibility):
        ress_set = []
        for ind in range(len(attr_categories)):
            if ind in possibility:
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

    # Generate samples based on the new ordered attributes
    subgroup1_p_vals = [random.choices(list(range(len(e))), k=len(e)) for e in subgroup_sets]
    subgroup1_p_vals = [safe_normalize(p) for p in subgroup1_p_vals]
    subgroup1_sample = GaussianCopulaCategorical(
        subgroup1_p_vals, correlation_matrix, corr_matrix_randomness=corr_matrix_randomness).generate_samples(1)
    subgroup1_vals = [subgroup_sets[i][e] for i, e in enumerate(subgroup1_sample[0])]

    subgroup2_p_vals = generate_subgroup2_probabilities(subgroup1_vals, subgroup_sets, similarity, sets_attr)
    subgroup2_sample = GaussianCopulaCategorical(subgroup2_p_vals, correlation_matrix,
                                                 list(subgroup1_sample),
                                                 corr_matrix_randomness=corr_matrix_randomness).generate_samples(1)
    subgroup2_vals = [subgroup_sets[i][e] for i, e in enumerate(subgroup2_sample[0])]

    # Calculate total group size based on frequency while respecting min and max constraints
    total_group_size = max(min_group_size, math.ceil(max_group_size * frequency))
    total_group_size = min(total_group_size, max_group_size)

    diff_percentage = random.uniform(min_diff_subgroup_size, max_diff_subgroup_size)
    diff_size = int(total_group_size * diff_percentage)

    # Ensure each subgroup meets minimum size requirements
    subgroup1_size = max(min_group_size // 2, (total_group_size + diff_size) // 2)
    subgroup2_size = max(min_group_size // 2, total_group_size - subgroup1_size)

    generator = IndividualsGenerator(
        schema=data_schema,
        graph=correlation_matrix,
        gen_order=gen_order,
        outcome_weights=W[-1],
        outcome_bias=0,
        subgroup_bias=subgroup_bias,
        epis_uncertainty=epis_uncertainty,
        alea_uncertainty=alea_uncertainty,
        corr_matrix_randomness=corr_matrix_randomness,
        categorical_distribution=categorical_distribution,
        categorical_influence=categorical_influence
    )

    # Generate dataset for subgroup 1 and subgroup 2
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

    # Create keys based on the ordered attribute values
    subgroup1_key = '|'.join(list(map(lambda x: '*' if x == -1 else str(x), subgroup1_vals)))
    subgroup2_key = '|'.join(list(map(lambda x: '*' if x == -1 else str(x), subgroup2_vals)))
    group_key = subgroup1_key + '-' + subgroup2_key

    # Assign the new keys to the dataframe
    subgroup1_individuals_df['subgroup_key'] = subgroup1_key
    subgroup2_individuals_df['subgroup_key'] = subgroup2_key

    subgroup1_individuals_df['indv_key'] = subgroup1_individuals_df[attr_names].apply(
        lambda x: '|'.join(list(x.astype(str))), axis=1)
    subgroup2_individuals_df['indv_key'] = subgroup2_individuals_df[attr_names].apply(
        lambda x: '|'.join(list(x.astype(str))), axis=1)

    result_df = pd.concat([subgroup1_individuals_df, subgroup2_individuals_df])

    result_df['group_key'] = group_key
    result_df['granularity_param'] = granularity
    result_df['intersectionality_param'] = intersectionality
    result_df['similarity_param'] = similarity
    result_df['epis_uncertainty_param'] = epis_uncertainty
    result_df['alea_uncertainty_param'] = alea_uncertainty
    result_df['frequency_param'] = frequency
    result_df['group_size'] = len(subgroup1_individuals + subgroup2_individuals)
    result_df['diff_subgroup_size'] = diff_percentage

    return result_df


def generate_categorical_distribution(attr_categories, attr_names, bias_strength=0.5):
    """
    Generates a categorical distribution for attributes with controlled randomness.

    Args:
        attr_categories (List[List[int]]): List of possible values for each attribute
        attr_names (List[str]): List of attribute names
        bias_strength (float): Controls how biased the distributions are (0.0 to 1.0)
                             0.0 = nearly uniform, 1.0 = strongly biased to one category

    Returns:
        Dict[str, List[float]]: Dictionary mapping attribute names to probability distributions
    """
    distribution = {}

    for attr_name, categories in zip(attr_names, attr_categories):
        n_categories = len(categories) - 1

        # Generate random probabilities
        if bias_strength > 0:
            # Generate biased probabilities
            probs = np.random.dirichlet([1 - bias_strength] * n_categories)

            # Make one category dominant based on bias_strength
            dominant_idx = np.random.randint(n_categories)
            probs = (1 - bias_strength) * probs + bias_strength * np.eye(n_categories)[dominant_idx]
        else:
            # Generate nearly uniform probabilities
            probs = np.ones(n_categories) / n_categories
            probs += np.random.uniform(-0.1, 0.1, n_categories)

        # Ensure probabilities sum to 1
        probs = probs / np.sum(probs)

        distribution[attr_name] = probs.tolist()

    return distribution


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
        min_group_size=10,
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
        min_granularity=1,
        max_granularity=None,
        min_intersectionality=1,
        max_intersectionality=None,
        categorical_outcome: bool = True,
        nb_categories_outcome: int = 6,
        use_cache: bool = True,
        corr_matrix_randomness=0.5,
        categorical_distribution: Dict[str, List[float]] = None,
        categorical_influence: float = 0.5,
        data_schema: DataSchema = None
) -> DiscriminationData:
    # Validate min_group_size
    if min_group_size >= max_group_size:
        raise ValueError("min_group_size must be less than max_group_size")

    if min_group_size < 2:
        raise ValueError("min_group_size must be at least 2 to ensure both subgroups have at least one member")

    cache = DataCache()

    # Create parameters dictionary for cache key generation
    params = copy.deepcopy(locals())
    params = {k: v for k, v in params.items() if
              ((v is None or isinstance(v, (int, float, str, bool))) and '_debug_' not in k)}

    # Try to load from cache if use_cache is True
    if use_cache:
        cached_data = cache.load(params)
        if cached_data is not None:
            print("Using cached data")
            return cached_data

    outcome_column = 'outcome'

    # Handle data_schema-derived attributes
    if data_schema is not None:
        # Update nb_attributes based on schema
        nb_attributes = len(data_schema.attr_categories)

        # Extract attributes from schema
        attr_categories = data_schema.attr_categories
        sets_attr = data_schema.protected_attr
        attr_names = data_schema.attr_names

        # Update proportion of protected attributes
        prop_protected_attr = sum(sets_attr) / len(sets_attr)

        # Update min/max number of classes based on schema
        if min_number_of_classes is None:
            min_number_of_classes = min(len(cats) for cats in attr_categories)
        if max_number_of_classes is None:
            max_number_of_classes = max(len(cats) for cats in attr_categories)

        if correlation_matrix is None:
            correlation_matrix = data_schema.correlation_matrix

        if categorical_distribution is None:
            categorical_distribution = data_schema.categorical_distribution

        if gen_order is None:
            gen_order = data_schema.gen_order

    else:
        if min_number_of_classes is None or max_number_of_classes is None:
            min_number_of_classes, max_number_of_classes = estimate_min_attributes_and_classes(nb_groups)
            max_number_of_classes = int(min_number_of_classes * 1.5)

        data_schema = generate_data_schema(
            min_number_of_classes, max_number_of_classes, nb_attributes, prop_protected_attr
        )
        attr_categories, sets_attr, attr_names = data_schema.attr_categories, data_schema.protected_attr, data_schema.attr_names

        correlation_matrix = data_schema.correlation_matrix
        categorical_distribution = data_schema.categorical_distribution
        gen_order = data_schema.gen_order

    # Generate weights if not provided
    if W is None:
        hiddenlayers_depth = 3
        W = np.random.uniform(low=0.0, high=1.0, size=(hiddenlayers_depth, nb_attributes))

    # Get protected and unprotected indices
    protected_indexes = [index for index, value in enumerate(sets_attr) if value]
    unprotected_indexes = [index for index, value in enumerate(sets_attr) if not value]

    # Update max_granularity and max_intersectionality based on available attributes
    max_granularity = max(1, len(unprotected_indexes)) if (max_granularity is None) or (
            max_granularity > len(unprotected_indexes)) else max_granularity
    max_intersectionality = max(1, len(protected_indexes)) if (max_intersectionality is None) or (
            max_intersectionality > len(protected_indexes)) else max_intersectionality

    # Validate granularity and intersectionality constraints
    assert 0 < min_granularity <= max_granularity <= len(
        unprotected_indexes), 'min_granularity must be between 0 and max_granularity'
    assert 0 < min_intersectionality <= max_intersectionality <= len(
        protected_indexes), 'min_intersectionality must be between 0 and max_intersectionality'

    collision_tracker = CollisionTracker(nb_attributes)

    results = []
    collisions = 0

    with tqdm(total=nb_groups, desc="Generating data") as pbar:
        while len(results) < nb_groups:
            granularity = random.randint(min_granularity, max_granularity)
            intersectionality = random.randint(min_intersectionality, max_intersectionality)

            subgroup_bias = random.uniform(0.1, 0.5)

            possible_gran = random.sample(unprotected_indexes, granularity)
            possible_intersec = random.sample(protected_indexes, intersectionality)

            possibility = tuple(possible_gran + possible_intersec)

            if not collision_tracker.is_collision(possibility):
                collision_tracker.add_combination(possibility)
                group = create_group(
                    granularity, intersectionality,
                    possibility, data_schema, attr_categories, W,
                    subgroup_bias, corr_matrix_randomness,
                    categorical_influence,
                    min_similarity, max_similarity, min_alea_uncertainty, max_alea_uncertainty,
                    min_epis_uncertainty, max_epis_uncertainty, min_frequency, max_frequency,
                    min_diff_subgroup_size, max_diff_subgroup_size, min_group_size, max_group_size
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

    results_d = DiscriminationDataFrame(results)

    data = DiscriminationData(
        dataframe=results_d,
        categorical_columns=list(attr_names) + [outcome_column],
        attributes=protected_attr,
        collisions=collisions,
        nb_groups=results_d['group_key'].nunique(),
        max_group_size=max_group_size,
        hiddenlayers_depth=W.shape[0],
        outcome_column=outcome_column,
        attr_possible_values=attr_possible_values
    )

    data = calculate_actual_metrics(data)

    if use_cache:
        cache.save(data, params)

    return data
