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
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from dataclasses import dataclass, field
from pandas import DataFrame
from typing import Literal, TypeVar, Any, List, Dict, Tuple, Optional, Union
from scipy.stats import norm, multivariate_normal, spearmanr, gaussian_kde
import pandas as pd
import warnings
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from ucimlrepo import fetch_ucirepo
from scipy.spatial.distance import jensenshannon
import numpy as np
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


def is_numeric_column(series: pd.Series) -> bool:
    """
    Check if a column should be treated as numeric for KDE.
    """
    if not np.issubdtype(series.dtype, np.number):
        return False

    n_unique = len(series.dropna().unique())
    return n_unique >= 5


def is_integer_column(series: pd.Series) -> bool:
    """
    Check if a numeric column contains only integers.
    """
    return np.all(series.dropna() == series.dropna().astype(int))


def get_unique_samples(kde: gaussian_kde, n_samples: int, is_integer: bool = False,
                       max_attempts: int = 1000) -> np.ndarray:
    """
    Get unique samples from KDE with appropriate type handling.
    """
    samples = kde.resample(min(n_samples * 2, max_attempts))[0]

    if is_integer:
        samples = np.round(samples).astype(int)

    unique_samples = np.unique(samples)

    attempts = 1
    while len(unique_samples) < n_samples and attempts < max_attempts:
        new_samples = kde.resample(n_samples)[0]
        if is_integer:
            new_samples = np.round(new_samples).astype(int)
        unique_samples = np.unique(np.concatenate([unique_samples, new_samples]))
        attempts += 1

    if len(unique_samples) > n_samples:
        indices = np.linspace(0, len(unique_samples) - 1, n_samples).astype(int)
        unique_samples = np.sort(unique_samples)[indices]

    return unique_samples


@dataclass
class DataSchema:
    attr_categories: List[List[str]]
    protected_attr: List[str]
    attr_names: List[str]
    categorical_distribution: Dict[str, List[float]]
    correlation_matrix: np.ndarray
    gen_order: List[str]
    category_maps: Dict[str, Dict[int, str]] = None  # Add category maps to store encoding/decoding mappings
    column_mapping: Dict[str, str] = None


def create_kde_encoding(series: pd.Series, n_samples: int = 100) -> Tuple[
    np.ndarray, gaussian_kde, List[Union[int, float]], Dict[Union[int, float], str], np.ndarray]:
    """
    Create KDE from the series and sample fixed points from it.

    Args:
        series: Input pandas Series containing numeric values
        n_samples: Number of points to sample from the KDE

    Returns:
        Tuple containing:
        - encoded values (np.ndarray)
        - KDE object (gaussian_kde)
        - categories list (List[Union[int, float]])
        - category map (Dict[Union[int, float], str])
        - probability distribution (np.ndarray)
    """
    non_nan_mask = pd.notna(series)
    values = series[non_nan_mask].to_numpy()

    if len(values) == 0:
        # Handle empty/all-NaN series
        return (np.full(len(series), -1),
                None,
                [-1, 0],
                {-1: 'nan', 0: '0'},
                np.array([1.0]))  # Default probability for empty series

    # Reshape values for KDE if necessary
    values = values.reshape(-1, 1) if len(values.shape) == 1 else values

    # Create and fit KDE
    kde = gaussian_kde(values.T)
    is_integer = is_integer_column(series)

    # Get sampled points
    sampled_points = get_unique_samples(kde, n_samples, is_integer)

    # Calculate probabilities at sampled points
    sampled_points_reshaped = sampled_points.reshape(1, -1)
    probabilities = kde.evaluate(sampled_points_reshaped)

    # Normalize probabilities to sum to 1
    probabilities = probabilities / np.sum(probabilities)

    # Create categories starting from -1 (missing) then 0 to n-1
    categories = [-1] + list(range(len(sampled_points)))

    # Create mapping dictionary
    if is_integer:
        category_map = {
            -1: 'nan',
            **{i: str(int(point)) for i, point in enumerate(sampled_points)}
        }
    else:
        category_map = {
            -1: 'nan',
            **{i: f"{point:.3f}" for i, point in enumerate(sampled_points)}
        }

    # Encode original values
    encoded = np.full(len(series), -1)  # Default to -1 for missing values
    valid_mask = non_nan_mask
    if valid_mask.any():
        valid_values = series[valid_mask].to_numpy()
        # Find nearest sampled point for each value
        nearest_indices = np.array([
            np.abs(sampled_points - val).argmin()
            for val in valid_values
        ])
        encoded[valid_mask] = nearest_indices

    return encoded, kde, categories, category_map, probabilities


def generate_schema_from_dataframe(
        df: pd.DataFrame,
        protected_columns: List[str] = None,
        attr_prefix: str = None,
        outcome_column: str = 'outcome',
        ensure_positive_definite: bool = True,
        n_samples: int = 100,
        use_attr_naming_pattern: bool = False
) -> Tuple[DataSchema, np.ndarray, Dict[str, str]]:
    """Generate a DataSchema and correlation matrix from a pandas DataFrame using KDE for numeric columns."""
    if outcome_column not in df.columns:
        raise ValueError(f"Outcome column '{outcome_column}' not found in DataFrame")

    new_cols = []
    for k, v in enumerate(df.columns):
        if v == outcome_column:
            attr_col = 'outcome'
        else:
            attr_col = f"Attr{k}_{'T' if v in protected_columns else 'X'}"

        new_cols.append(attr_col)

    new_cols_mapping = {k: v for k, v in zip(new_cols, df.columns)}
    reverse_cols_mapping = {v: k for k, v in zip(new_cols, df.columns)}

    if use_attr_naming_pattern:
        df.columns = new_cols

    # Handle attribute naming pattern
    if use_attr_naming_pattern:
        protected_columns = [col for col in df.columns if col.endswith('_T')]
        attr_columns = [col for col in df.columns if col.endswith('_X') or col.endswith('_T')]
    else:
        if attr_prefix:
            attr_columns = [col for col in df.columns if col.startswith(attr_prefix)]
        else:
            attr_columns = [col for col in df.columns if col != outcome_column]

    if not attr_columns:
        raise ValueError("No attribute columns found")

    attr_categories = []
    encoded_df = pd.DataFrame(index=df.index)
    binning_info = {}
    kde_distributions = {}
    label_encoders = {}
    category_maps = {}  # Store category maps for each column
    categorical_distribution = {}  # Initialize categorical_distribution here

    for col in attr_columns:
        if is_numeric_column(df[col]):
            encoded_vals, kde, categories, category_map, probs = create_kde_encoding(df[col], n_samples)

            encoded_df[col] = encoded_vals

            attr_categories.append(categories)
            category_maps[col] = category_map
            categorical_distribution[col] = probs.tolist()  # Store the KDE-based probabilities

            if kde is not None:
                kde_distributions[col] = kde
                binning_info[col] = {
                    'strategy': 'kde',
                    'n_samples': n_samples,
                    'is_integer': is_integer_column(df[col])
                }
        else:
            le = LabelEncoder()
            non_nan_vals = df[col].dropna().unique()

            if len(non_nan_vals) > 0:
                str_vals = [str(x) for x in non_nan_vals]
                str_vals = list(dict.fromkeys(str_vals))

                le.fit(str_vals)
                encoded = np.full(len(df), -1)

                mask = df[col].notna()
                if mask.any():
                    encoded[mask] = le.transform([str(x) for x in df[col][mask]])

                categories = [-1] + list(range(len(str_vals)))
                category_map = {-1: 'nan', **{i: val for i, val in enumerate(str_vals)}}

                label_encoders[col] = le
                category_maps[col] = category_map
            else:
                encoded = np.full(len(df), -1)
                categories = [-1, 0]
                category_map = {-1: 'nan', 0: 'empty'}
                category_maps[col] = category_map

            encoded_df[col] = encoded
            attr_categories.append(categories)

            # Calculate categorical distribution for non-numeric columns
            valid_counts = encoded_df[col][encoded_df[col] >= 0].value_counts().sort_index()
            total_valid = valid_counts.sum()
            probs = np.zeros(len(categories) - 1)  # -1 to exclude the -1 category
            if total_valid > 0:
                for idx, count in valid_counts.items():
                    probs[int(idx)] = count / total_valid
            categorical_distribution[col] = probs.tolist()

    # Calculate correlation matrix
    correlation_matrix = calculate_correlation_matrix(encoded_df, attr_columns, ensure_positive_definite)

    schema = DataSchema(
        attr_categories=attr_categories,
        protected_attr=[col in protected_columns for col in attr_columns],
        attr_names=attr_columns,
        categorical_distribution=categorical_distribution,
        correlation_matrix=correlation_matrix,
        gen_order=list(range(len(attr_columns))),
        category_maps=category_maps,
        column_mapping=new_cols_mapping
    )

    n_encoded_df = copy.deepcopy(encoded_df)
    n_encoded_df['outcome'] = LabelEncoder().fit_transform(df['outcome'].astype(str))

    return schema, correlation_matrix, new_cols_mapping, n_encoded_df


def decode_dataframe(df: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    """
    Decode a dataframe using the schema's category maps.
    """
    decoded_df = pd.DataFrame(index=df.index)

    for col in schema.attr_names:
        if col in df.columns:
            category_map = schema.category_maps[col]
            decoded_df[col] = df[col].map(
                lambda x: category_map.get(int(x) if isinstance(x, (int, float)) else x, 'unknown'))

    # Copy any columns that weren't in the schema
    for col in df.columns:
        if col not in schema.attr_names:
            decoded_df[col] = df[col]

    return decoded_df


def calculate_correlation_matrix(encoded_df, attr_columns, ensure_positive_definite):
    """Helper function to calculate the correlation matrix"""
    correlation_matrix = np.zeros((len(attr_columns), len(attr_columns)))
    for i, col1 in enumerate(attr_columns):
        for j, col2 in enumerate(attr_columns):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                # Update mask to check for non-negative values (exclude missing values)
                mask = (encoded_df[col1] >= 0) & (encoded_df[col2] >= 0)
                if mask.any():
                    corr, _ = spearmanr(encoded_df[col1][mask], encoded_df[col2][mask])
                    correlation_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                    correlation_matrix[j, i] = correlation_matrix[i, j]
                else:
                    correlation_matrix[i, j] = 0.0
                    correlation_matrix[j, i] = 0.0

    if ensure_positive_definite:
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        if np.any(eigenvalues < 0):
            eigenvalues[eigenvalues < 0] = 1e-6
            correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            scaling = np.sqrt(np.diag(correlation_matrix))
            correlation_matrix = correlation_matrix / scaling[:, None] / scaling[None, :]

    return correlation_matrix


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
        samples = np.full((n_samples, self.n_attributes), -1, dtype=int)

        # Fill in predetermined values
        if predetermined_values:
            mask = np.array(predetermined_values) != -1
            samples[:, mask] = np.array(predetermined_values)[mask][None, :]

        # Generate remaining values
        for attr in self.gen_order:
            mask = samples[:, attr] == -1
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


@dataclass
class GroupDefinition:
    group_size: int
    subgroup_bias: int
    similarity: float
    alea_uncertainty: float
    epis_uncertainty: float
    frequency: float
    avg_diff_outcome: int
    diff_subgroup_size: float
    subgroup1: dict  # {'Attr1_T': 3, 'Attr2_T': 1, 'Attr3_X': 3},
    subgroup2: dict  # {'Attr1_T': 2, 'Attr2_T': 2, 'Attr3_X': 2}


def calculate_actual_similarity(data):
    def calculate_group_similarity(group_data):
        subgroup_keys = group_data['subgroup_key'].unique()
        if len(subgroup_keys) != 2:
            return np.nan

        # Get data for each subgroup
        subgroup1_data = group_data[group_data['subgroup_key'] == subgroup_keys[0]]
        subgroup2_data = group_data[group_data['subgroup_key'] == subgroup_keys[1]]

        # Calculate similarity across all attribute distributions
        similarities = []

        for attr in data.attr_columns:
            # Get distributions of values for this attribute in each subgroup
            vals1 = subgroup1_data[attr].value_counts(normalize=True).sort_index()
            vals2 = subgroup2_data[attr].value_counts(normalize=True).sort_index()

            # Align the distributions
            all_values = sorted(set(vals1.index) | set(vals2.index))
            dist1 = np.array([vals1.get(v, 0) for v in all_values])
            dist2 = np.array([vals2.get(v, 0) for v in all_values])

            # Calculate Jensen-Shannon divergence
            js_distance = jensenshannon(dist1, dist2)

            # Apply non-linear transformation to spread out values
            # This will push values away from 0.5
            if np.isnan(js_distance):
                sim = 1.0
            else:
                # Convert distance to similarity
                raw_sim = 1.0 - js_distance

                # Apply power transformation to increase sensitivity
                # This pushes high similarities higher and low similarities lower
                # Adjust the power (3.0) to control sensitivity
                sim = pow(raw_sim, 3.0) if raw_sim >= 0.5 else 1.0 - pow(1.0 - raw_sim, 3.0)

            similarities.append(sim)

        # Weight more discriminative attributes higher
        # Attributes with similarity far from 0.5 contribute more
        weights = [abs(s - 0.5) + 0.5 for s in similarities]
        weighted_sum = sum(s * w for s, w in zip(similarities, weights))
        weighted_avg = weighted_sum / sum(weights) if sum(weights) > 0 else 0.5

        # Final transformation to spread values further
        spread_factor = 2.0  # Adjust this to control overall spread
        final_sim = 0.5 + spread_factor * (weighted_avg - 0.5)

        # Ensure result stays in [0,1] range
        return max(0.0, min(1.0, final_sim))

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
                try:
                    ss.remove(-1)
                except:
                    pass
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
    subgroup1_individuals_df = pd.DataFrame(subgroup1_individuals, columns=data_schema.attr_names)
    subgroup1_individuals_df['outcome'] = [outcome for _, outcome, _, _ in subgroup1_data]
    subgroup1_individuals_df['epis_uncertainty'] = [epis for _, _, epis, _ in subgroup1_data]
    subgroup1_individuals_df['alea_uncertainty'] = [alea for _, _, _, alea in subgroup1_data]

    subgroup2_data = generator.generate_dataset_with_outcome(subgroup2_size, subgroup2_vals, is_subgroup1=False)
    subgroup2_individuals = [sample for sample, _, _, _ in subgroup2_data]
    subgroup2_individuals_df = pd.DataFrame(subgroup2_individuals, columns=data_schema.attr_names)
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
        data_schema: DataSchema = None,
        predefined_groups=None,
        extra_rows=None
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

    # Generate extra rows if requested
    if extra_rows and extra_rows > 0:
        # Initialize the generator for individual samples using the schema and correlation matrix
        generator = IndividualsGenerator(
            schema=data_schema,
            graph=correlation_matrix,
            gen_order=gen_order,
            outcome_weights=W[-1],
            outcome_bias=0,
            subgroup_bias=0,  # Neutral bias for extra rows
            epis_uncertainty=np.mean([min_epis_uncertainty, max_epis_uncertainty]),
            alea_uncertainty=np.mean([min_alea_uncertainty, max_alea_uncertainty]),
            corr_matrix_randomness=corr_matrix_randomness,
            categorical_distribution=categorical_distribution,
            categorical_influence=categorical_influence
        )

        print(f"\nGenerating {extra_rows} additional rows...")

        # Generate extra samples with a balanced approach (half with is_subgroup1=True, half with False)
        half_extra = extra_rows // 2
        remaining_extra = extra_rows - half_extra

        # Generate the first half with is_subgroup1=True
        extra_data1 = generator.generate_dataset_with_outcome(half_extra, None, is_subgroup1=True)
        extra_samples1 = [sample for sample, _, _, _ in extra_data1]
        extra_df1 = pd.DataFrame(extra_samples1, columns=data_schema.attr_names)
        extra_df1['outcome'] = [outcome for _, outcome, _, _ in extra_data1]
        extra_df1['epis_uncertainty'] = [epis for _, _, epis, _ in extra_data1]
        extra_df1['alea_uncertainty'] = [alea for _, _, _, alea in extra_data1]

        # Generate the second half with is_subgroup1=False
        extra_data2 = generator.generate_dataset_with_outcome(remaining_extra, None, is_subgroup1=False)
        extra_samples2 = [sample for sample, _, _, _ in extra_data2]
        extra_df2 = pd.DataFrame(extra_samples2, columns=data_schema.attr_names)
        extra_df2['outcome'] = [outcome for _, outcome, _, _ in extra_data2]
        extra_df2['epis_uncertainty'] = [epis for _, _, epis, _ in extra_data2]
        extra_df2['alea_uncertainty'] = [alea for _, _, _, alea in extra_data2]

        # Combine both halves
        extra_df = pd.concat([extra_df1, extra_df2], ignore_index=True)

        # Add necessary columns to match the main dataframe structure
        extra_df['group_key'] = 'extra'
        extra_df['subgroup_key'] = extra_df.index.map(lambda i: f'extra_subgroup{i % 2 + 1}')
        extra_df['indv_key'] = extra_df[attr_names].apply(lambda x: '|'.join(list(x.astype(str))), axis=1)

        # Add the parameters columns with average values
        extra_df['granularity_param'] = (min_granularity + max_granularity) / 2
        extra_df['intersectionality_param'] = (min_intersectionality + max_intersectionality) / 2
        extra_df['similarity_param'] = (min_similarity + max_similarity) / 2
        extra_df['epis_uncertainty_param'] = (min_epis_uncertainty + max_epis_uncertainty) / 2
        extra_df['alea_uncertainty_param'] = (min_alea_uncertainty + max_alea_uncertainty) / 2
        extra_df['frequency_param'] = (min_frequency + max_frequency) / 2
        extra_df['group_size'] = extra_rows
        extra_df['diff_subgroup_size'] = (min_diff_subgroup_size + max_diff_subgroup_size) / 2
        extra_df['collisions'] = collisions

        # Append the extra rows to the main dataframe
        results = pd.concat([results, extra_df], ignore_index=True)
        print(f"Added {extra_rows} extra rows to the dataset.")

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


def generate_from_real_data(dataset_name, use_cache=False, *args, **kwargs):
    if dataset_name == 'adult':
        adult = fetch_ucirepo(id=2)
        df1 = adult['data']['original']
        df1.drop(columns=['fnlwgt'], inplace=True)

        schema, correlation_matrix, column_mapping, enc_df = generate_schema_from_dataframe(df1,
                                                                                            protected_columns=['race',
                                                                                                               'sex'],
                                                                                            outcome_column='income',
                                                                                            use_attr_naming_pattern=True)
    elif dataset_name == 'credit':
        df1 = fetch_ucirepo(id=144)
        df1 = df1['data']['original']
        schema, correlation_matrix, new_cols_mapping, enc_df = generate_schema_from_dataframe(df1,
                                                                                              protected_columns=[
                                                                                                  'Attribute8',
                                                                                                  'Attribute12'],
                                                                                              outcome_column='Attribute20',
                                                                                              use_attr_naming_pattern=True,
                                                                                              ensure_positive_definite=True)
    elif dataset_name == 'bank':
        # Bank Marketing dataset
        bank_marketing = fetch_ucirepo(id=222)
        df = pd.concat([bank_marketing.data.features, bank_marketing.data.targets], axis=1)

        if 'protected_columns' not in kwargs:
            kwargs['protected_columns'] = ['age', 'marital', 'education']
        if 'outcome_column' not in kwargs:
            kwargs['outcome_column'] = 'y'

        # Ensure all column names are valid Python identifiers
        df.columns = [col.replace('-', '_') for col in df.columns]

        # Update protected columns if they were renamed
        kwargs['protected_columns'] = [col.replace('-', '_') for col in kwargs['protected_columns']]
        kwargs['outcome_column'] = kwargs['outcome_column'].replace('-', '_')

        schema, correlation_matrix, column_mapping, enc_df = generate_schema_from_dataframe(
            df,
            protected_columns=kwargs['protected_columns'],
            outcome_column=kwargs['outcome_column']
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    # Ensure schema attribute names are unique
    schema.attr_names = list(dict.fromkeys(schema.attr_names))

    # Generate the data using the schema
    data = generate_data(
        correlation_matrix=correlation_matrix,
        data_schema=schema,
        use_cache=use_cache,
        *args,
        **kwargs
    )

    # data = decode_dataframe(data.dataframe, schema)
    return data, schema


def get_real_data(
        dataset_name: str,
        protected_columns: Optional[List[str]] = None,
        outcome_column: Optional[str] = None,
        use_cache=False,
        *args, **kwargs
) -> Tuple[DiscriminationData, DataSchema]:
    """
    Fetch and process real datasets into the same format as generate_data output.

    Args:
        dataset_name (str): Name of the dataset ('adult', 'credit', 'bank', etc.)
        protected_columns (List[str], optional): List of column names to treat as protected attributes
        outcome_column (str, optional): Name of the column to use as outcome
        *args, **kwargs: Additional arguments passed to generate_data

    Returns:
        Tuple[DiscriminationData, DataSchema]: Processed data and schema matching generate_data format
    """
    dataset_configs = {
        'adult': {
            'id': 2,
            'protected_columns': ['race', 'sex'] if protected_columns is None else protected_columns,
            'outcome_column': 'income' if outcome_column is None else outcome_column,
            'drop_columns': ['fnlwgt']
        },
        'credit': {
            'id': 144,
            'protected_columns': ['Attribute8', 'Attribute12'] if protected_columns is None else protected_columns,
            'outcome_column': 'Attribute20' if outcome_column is None else outcome_column,
            'drop_columns': []
        },
        'bank': {
            'id': 222,
            'protected_columns': ['age', 'marital', 'education'] if protected_columns is None else protected_columns,
            'outcome_column': 'y' if outcome_column is None else outcome_column,
            'drop_columns': []
        }
    }

    if dataset_name not in dataset_configs:
        raise ValueError(f"Dataset {dataset_name} not supported. Available datasets: {list(dataset_configs.keys())}")

    config = dataset_configs[dataset_name]

    # Fetch the dataset
    dataset = fetch_ucirepo(id=config['id'])
    df = dataset['data']['original']

    if dataset_name == 'adult':
        df['income'] = df['income'].apply(lambda x: x.replace('.', ''))

    if config['drop_columns']:
        df = df.drop(columns=config['drop_columns'])

    # Generate schema from the dataframe
    schema, correlation_matrix, column_mapping, enc_df = generate_schema_from_dataframe(
        df,
        protected_columns=config['protected_columns'],
        outcome_column=config['outcome_column'],
        use_attr_naming_pattern=True,
        ensure_positive_definite=True
    )

    data = DiscriminationData(
        dataframe=enc_df,
        categorical_columns=list(schema.attr_names) + ['outcome'],
        attributes={k: v for k, v in zip(schema.attr_names, schema.protected_attr)},
        collisions=0,
        nb_groups=0,
        max_group_size=0,
        hiddenlayers_depth=0,
        outcome_column='outcome',
        attr_possible_values={}
    )

    return data, schema


def generate_from_real_data(dataset_name, use_cache=False, extra_rows=None, *args, **kwargs):
    if dataset_name == 'adult':
        adult = fetch_ucirepo(id=2)
        df1 = adult['data']['original']
        df1.drop(columns=['fnlwgt'], inplace=True)

        schema, correlation_matrix, column_mapping, enc_df = generate_schema_from_dataframe(df1,
                                                                                            protected_columns=['race',
                                                                                                               'sex'],
                                                                                            outcome_column='income',
                                                                                            use_attr_naming_pattern=True)
    elif dataset_name == 'credit':
        df1 = fetch_ucirepo(id=144)
        df1 = df1['data']['original']
        schema, correlation_matrix, new_cols_mapping, enc_df = generate_schema_from_dataframe(df1,
                                                                                              protected_columns=[
                                                                                                  'Attribute8',
                                                                                                  'Attribute12'],
                                                                                              outcome_column='Attribute20',
                                                                                              use_attr_naming_pattern=True,
                                                                                              ensure_positive_definite=True)
    elif dataset_name == 'bank':
        # Bank Marketing dataset
        bank_marketing = fetch_ucirepo(id=222)
        df1 = pd.concat([bank_marketing.data.features, bank_marketing.data.targets], axis=1)

        # Ensure all column names are valid Python identifiers
        df1.columns = [col.replace('-', '_') for col in df1.columns]

        schema, correlation_matrix, column_mapping, enc_df = generate_schema_from_dataframe(
            df1,
            protected_columns=['age', 'marital', 'education'],
            outcome_column='y',
            use_attr_naming_pattern=True,
            ensure_positive_definite=True
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    if use_cache:
        cache = DataCache()
        data = cache.load(kwargs)
        if data is not None:
            return data, schema

    data = generate_data(
        gen_order=schema.gen_order,
        correlation_matrix=correlation_matrix,
        data_schema=schema,
        use_cache=use_cache,
        extra_rows=extra_rows,
        *args, **kwargs
    )

    if use_cache:
        cache.save(data, kwargs)

    return data, schema
