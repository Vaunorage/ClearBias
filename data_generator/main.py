import copy
import datetime
import hashlib
import json
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass, field
from scipy.stats import spearmanr
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.sampling import Condition
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo

from data_generator.main_old2 import safe_normalize, GaussianCopulaCategorical, generate_subgroup2_probabilities, \
    bin_array_values, coefficient_of_variation, calculate_actual_metrics


class DataCache:
    def __init__(self):
        self.cache_dir = Path(".cache/discrimination_data")
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

    def save(self, data, params: dict):
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

    def load(self, params: dict) -> Optional:
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


@dataclass
class DataSchema:
    attr_categories: List[List[str]]
    protected_attr: List[bool]
    attr_names: List[str]
    categorical_distribution: Dict[str, List[float]]
    correlation_matrix: np.ndarray
    gen_order: List[str]
    category_maps: Dict[str, Dict[int, str]] = None  # Add category maps for encoding/decoding
    column_mapping: Dict[str, str] = None
    synthesizer: Any = None  # Store the trained SDV synthesizer
    sdv_metadata: Any = None  # Store the SDV metadata

    def to_sdv_metadata(self) -> SingleTableMetadata:
        """Convert this schema to SDV SingleTableMetadata."""
        if self.sdv_metadata is not None:
            return self.sdv_metadata

        metadata = SingleTableMetadata()

        # Add columns to the metadata
        for i, (attr_name, attr_cats, is_protected) in enumerate(zip(
                self.attr_names, self.attr_categories, self.protected_attr)):

            # Determine the SDV column type
            if -1 in attr_cats:  # If -1 is in categories, it handles missing values
                cats_without_missing = [c for c in attr_cats if c != -1]
            else:
                cats_without_missing = attr_cats

            # Check if this is a categorical or numerical column
            if all(isinstance(v, (int, float)) for v in cats_without_missing):
                # This is a numerical column
                metadata.add_column(attr_name, sdtype='numerical')
            else:
                # This is a categorical column
                metadata.add_column(attr_name, sdtype='categorical')

        # Add outcome column
        metadata.add_column('outcome', sdtype='categorical')

        return metadata

    def sample(self, num_rows=100, conditions=None):
        """Sample from the fitted synthesizer if available."""
        if self.synthesizer is None:
            raise ValueError("No synthesizer has been fitted to this schema")

        if conditions:
            return self.synthesizer.sample_from_conditions(conditions=conditions)
        else:
            return self.synthesizer.sample(num_rows=num_rows)

    def decode_dataframe(self, df):
        """Decode a dataframe using the category maps."""
        if not self.category_maps:
            return df

        decoded_df = pd.DataFrame(index=df.index)

        for col in self.attr_names:
            if col in df.columns and col in self.category_maps:
                category_map = self.category_maps[col]
                decoded_df[col] = df[col].map(
                    lambda x: category_map.get(int(x) if isinstance(x, (int, float)) else x, 'unknown'))

        # Copy non-schema columns
        for col in df.columns:
            if col not in decoded_df.columns:
                decoded_df[col] = df[col]

        return decoded_df


@dataclass
class DiscriminationDataFrame(pd.DataFrame):
    """A custom DataFrame subclass for discrimination data."""

    @property
    def _constructor(self):
        return DiscriminationDataFrame


@dataclass
class DiscriminationData:
    dataframe: pd.DataFrame
    categorical_columns: List[str]
    attributes: Dict[str, bool]
    collisions: int
    nb_groups: int
    max_group_size: int
    hiddenlayers_depth: int
    schema: DataSchema
    outcome_column: str = 'outcome'
    relevance_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    attr_possible_values: Dict[str, List[int]] = field(default_factory=dict)

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

    # @property
    # def schema(self):
    #     return '|'.join(
    #         [''.join(list(map(str, e))).replace('-1', '') for e in self.attr_possible_values.values()])

    def __post_init__(self):
        self.input_bounds = []
        for col in list(self.attributes):
            min_val = math.floor(self.xdf[col].min())
            max_val = math.ceil(self.xdf[col].max())
            self.input_bounds.append([min_val, max_val])


def create_sdv_numerical_distributions(data_schema):
    """Create numerical distributions configuration for SDV GaussianCopulaSynthesizer."""
    numerical_distributions = {}

    for attr_name, attr_cats in zip(data_schema.attr_names, data_schema.attr_categories):
        # Check if attribute is numerical
        if all(isinstance(c, (int, float)) for c in attr_cats if c != -1):
            # Use beta distribution for protected attributes and normal for non-protected
            is_protected = data_schema.protected_attr[data_schema.attr_names.index(attr_name)]
            if is_protected:
                numerical_distributions[attr_name] = 'beta'
            else:
                numerical_distributions[attr_name] = 'truncnorm'

    return numerical_distributions


def generate_data_schema(min_number_of_classes, max_number_of_classes, nb_attributes,
                         prop_protected_attr, fit_synthesizer=True, n_samples=1000) -> DataSchema:
    """Generate a data schema for synthetic data."""
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

    # Create SDV metadata
    sdv_metadata = SingleTableMetadata()

    # Add columns to the metadata
    for attr_name, attr_cats, is_protected in zip(attr_names, attr_categories, protected_attr):
        # Determine if this is categorical or numerical
        if all(isinstance(v, (int, float)) for v in attr_cats if v != -1):
            sdv_metadata.add_column(attr_name, sdtype='numerical')
        else:
            sdv_metadata.add_column(attr_name, sdtype='categorical')

    # Add outcome column
    sdv_metadata.add_column('outcome', sdtype='categorical')

    # Initialize schema
    res = DataSchema(attr_categories=attr_categories,
                     protected_attr=protected_attr,
                     attr_names=attr_names,
                     categorical_distribution=categorical_distribution,
                     correlation_matrix=correlation_matrix,
                     gen_order=gen_order,
                     sdv_metadata=sdv_metadata)

    # Create and fit synthesizer if requested
    if fit_synthesizer:
        # Create numerical distribution settings
        numerical_distributions = create_sdv_numerical_distributions(res)

        # Create the synthesizer with appropriate settings
        synthesizer = GaussianCopulaSynthesizer(
            sdv_metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            numerical_distributions=numerical_distributions,
            default_distribution='beta'
        )

        # Generate some initial data to fit the synthesizer
        # This creates a small sample dataset based on the schema
        initial_data = {}
        for k, v in categorical_distribution.items():
            initial_data[k] = random.choices(list(range(len(v))), weights=v, k=n_samples)
        initial_data = pd.DataFrame(initial_data)
        # Generate simple outcome values (0/1)
        initial_data['outcome'] = random.choices([0, 1], k=n_samples)

        # Convert to DataFrame
        initial_df = pd.DataFrame(initial_data)

        # Fit the synthesizer
        print("Fitting initial GaussianCopulaSynthesizer...")
        synthesizer.fit(initial_df)

        # Add the synthesizer to the schema
        res.synthesizer = synthesizer

    return res


def generate_valid_correlation_matrix(n):
    """Generate a valid correlation matrix."""
    A = np.random.rand(n, n)
    A = (A + A.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    A = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    D = np.diag(1 / np.sqrt(np.diag(A)))
    correlation_matrix = D @ A @ D

    return correlation_matrix


def generate_categorical_distribution(attr_categories, attr_names, bias_strength=0.5):
    """
    Generate a categorical distribution for attributes with controlled randomness.

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
        n_categories = len(categories) - 1  # Exclude -1 which represents missing values

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


def generate_data_from_distribution(distribution, attr_categories, attr_names, n_samples=1000,
                                    preset_values=None, excluded_values=None):
    """
    Generate synthetic data using categorical distributions.

    Args:
        distribution (Dict[str, List[float]]): Dictionary mapping attribute names to probability distributions
                                             (output from generate_categorical_distribution function)
        attr_categories (List[List[int]]): List of possible values for each attribute
        attr_names (List[str]): List of attribute names
        n_samples (int): Number of samples to generate
        preset_values (List[List]): List of lists of possible values for each attribute.
                                  Each position corresponds to the attr_names position.
                                  If None for an attribute, random values will be generated.
                                  If a list is provided, values will be randomly selected from that list.
                                  Default is None (no preset values).
        excluded_values (List[List]): List of lists of values to exclude for each attribute.
                                     Each position corresponds to the attr_names position.
                                     If None for an attribute, no values will be excluded.
                                     If a list is provided, those values will be excluded.
                                     Default is None (no excluded values).

    Returns:
        pd.DataFrame: DataFrame with generated categorical data
    """

    # Initialize default values for preset_values and excluded_values if not provided
    if preset_values is None:
        preset_values = [None] * len(attr_names)
    elif len(preset_values) < len(attr_names):
        preset_values.extend([None] * (len(attr_names) - len(preset_values)))

    if excluded_values is None:
        excluded_values = [None] * len(attr_names)
    elif len(excluded_values) < len(attr_names):
        excluded_values.extend([None] * (len(attr_names) - len(excluded_values)))

    # Initialize dataframe to store generated data
    generated_data = defaultdict(list)

    for _ in range(n_samples):
        for i, attr_name in enumerate(attr_names):
            # Get the distribution for the current attribute
            valid_probs = distribution.get(attr_name, [])
            probs = distribution.get(attr_name, [])
            valid_categories = list(filter(lambda x: x != -1, attr_categories[i]))
            categories = list(filter(lambda x: x != -1, attr_categories[i]))

            if preset_values[i] == [-1]:
                generated_data[attr_name].append(-1)
            else:
                # Apply preset values filter if specified
                if preset_values[i] is not None and len(preset_values[i]) > 0 and preset_values[i] != [-1]:
                    # Get indices of categories that are in preset_values
                    valid_indices = [idx for idx, cat in enumerate(categories)
                                     if cat in preset_values[i]]

                    if not valid_indices:
                        raise ValueError(f"None of the preset values for attribute {attr_name} exist in its categories")

                    # Filter categories to only include preset values
                    valid_categories = [categories[idx] for idx in valid_indices]
                    valid_probs = [probs[idx] for idx in valid_indices]

                # Apply excluded values filter if specified
                if excluded_values[i] and excluded_values[i] != [-1]:
                    # Keep indices of categories that are not in excluded_values
                    keep_indices = [idx for idx, cat in enumerate(valid_categories)
                                    if cat not in excluded_values[i]]

                    if not keep_indices:
                        raise ValueError(f"All possible values for attribute {attr_name} are excluded")

                    # Update categories and probabilities
                    valid_categories = [valid_categories[idx] for idx in keep_indices]
                    valid_probs = [valid_probs[idx] for idx in keep_indices]

                # Normalize probabilities
                sum_probs = sum(valid_probs)
                if sum_probs > 0:
                    valid_probs = [p / sum_probs for p in valid_probs]
                else:
                    raise ValueError(f"Sum of probabilities for attribute {attr_name} is zero after filtering")

                # Generate a random value based on the distribution
                value = np.random.choice(valid_categories, p=valid_probs)

                # Add the generated value to the dataframe
                generated_data[attr_name].append(value)

    # Convert the dictionary to a pandas DataFrame
    return pd.DataFrame(generated_data)


def estimate_min_attributes_and_classes(num_groups, max_iterations=1000):
    """Estimate minimum attributes and classes needed for the desired number of groups."""

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


def generate_outcomes_with_uncertainty(
        df: pd.DataFrame,
        feature_columns: List[str],
        weights: Optional[np.ndarray] = None,
        bias: float = 0.0,
        group_column: Optional[str] = None,
        group_bias: float = 0.0,
        epistemic_uncertainty: float = 0.1,
        aleatoric_uncertainty: float = 0.1,
        n_estimators: int = 50,
        random_state: Optional[int] = None
):
    """
    Generate prediction outcomes with epistemic and aleatoric uncertainty
    for a pandas DataFrame.

    Args:
        df: Input DataFrame containing features
        feature_columns: List of column names to use as features
        weights: Optional weight vector for features. If None, random weights are generated
        bias: Bias term to add to the linear combination
        group_column: Optional column name for group membership (for group bias)
        group_bias: Bias to apply based on group membership
        epistemic_uncertainty: Amount of model uncertainty (weight perturbation)
        aleatoric_uncertainty: Amount of irreducible noise in predictions
        n_estimators: Number of models in the ensemble
        random_state: Optional random seed for reproducibility

    Returns:
        DataFrame with original data plus prediction, epistemic_uncertainty,
        and aleatoric_uncertainty columns
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Extract features and standardize
    X = df[feature_columns].values
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    # Generate random weights if not provided
    if weights is None:
        weights = np.random.normal(0, 1, size=X.shape[1])

    # Get group information if specified
    group_effects = np.zeros(len(df))
    if group_column is not None and group_column in df.columns:
        group_values = df[group_column].values
        unique_groups = np.unique(group_values)
        if len(unique_groups) == 2:
            # Binary group case
            group_effects = np.where(group_values == unique_groups[0],
                                     group_bias, -group_bias)
        else:
            # Multi-group case (assign random biases)
            group_map = {g: np.random.normal(0, group_bias)
                         for g in unique_groups}
            group_effects = np.array([group_map[g] for g in group_values])

    # Generate ensemble predictions
    ensemble_preds = []
    for i in range(n_estimators):
        # Add epistemic uncertainty through weight perturbation
        weight_noise = np.random.normal(0, epistemic_uncertainty, size=weights.shape)
        perturbed_weights = weights * (1 + weight_noise)

        # Compute base prediction
        base_pred = np.dot(X, perturbed_weights) + bias + group_effects

        # Add aleatoric uncertainty
        noisy_pred = base_pred + np.random.normal(0, aleatoric_uncertainty, size=len(df))

        # Convert to probability
        probs = 1 / (1 + np.exp(-noisy_pred))
        ensemble_preds.append(probs)

    # Calculate final predictions and uncertainties
    ensemble_preds = np.array(ensemble_preds).T
    final_predictions = np.mean(ensemble_preds, axis=1)
    epistemic_uncertainty = np.var(ensemble_preds, axis=1)
    aleatoric_uncertainty = np.mean(ensemble_preds * (1 - ensemble_preds), axis=1)

    return final_predictions, epistemic_uncertainty, aleatoric_uncertainty


def fill_nan_values(df, data_schema=None):
    """
    Fill NaN values in a dataframe using smart defaults.

    Args:
        df: DataFrame containing NaN values to fill
        data_schema: Optional DataSchema object for schema-aware defaults

    Returns:
        DataFrame with NaN values filled
    """
    # Create a copy to avoid modifying the original
    filled_df = df.copy()

    # Process each column
    for col in filled_df.columns:
        # Skip columns with no NaN values
        if not filled_df[col].isna().any():
            continue

        # For numeric columns
        if pd.api.types.is_numeric_dtype(filled_df[col]):
            # Try to use the median of non-NaN values
            median_val = filled_df[col].median()
            # If median is also NaN, use 0
            filled_df[col].fillna(0 if pd.isna(median_val) else median_val, inplace=True)

        # For categorical columns
        else:
            # Try to use mode first
            if not filled_df[col].mode().empty:
                mode_val = filled_df[col].mode().iloc[0]
                filled_df[col].fillna(mode_val, inplace=True)
            # If no mode or schema is available, use a schema-based default
            elif data_schema is not None and col in data_schema.attr_names:
                col_idx = data_schema.attr_names.index(col)
                valid_values = [v for v in data_schema.attr_categories[col_idx] if v != -1]
                default_val = valid_values[0] if valid_values else "unknown"
                filled_df[col].fillna(default_val, inplace=True)
            # Last resort - use "unknown"
            else:
                filled_df[col].fillna("unknown", inplace=True)

    return filled_df


import contextlib
import io
import sys


@contextlib.contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


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

    subgroup1_vals = generate_data_from_distribution(data_schema.categorical_distribution, attr_categories,
                                                     data_schema.attr_names, n_samples=1,
                                                     preset_values=subgroup_sets).values.tolist()[0]

    subgroup2_vals = generate_data_from_distribution(data_schema.categorical_distribution, attr_categories,
                                                     data_schema.attr_names, n_samples=1,
                                                     preset_values=subgroup_sets,
                                                     excluded_values=list(
                                                         map(lambda x: [x], subgroup1_vals))).values.tolist()[0]

    # Calculate total group size based on frequency while respecting min and max constraints
    total_group_size = max(min_group_size, math.ceil(max_group_size * frequency))
    total_group_size = min(total_group_size, max_group_size)

    diff_percentage = random.uniform(min_diff_subgroup_size, max_diff_subgroup_size)
    diff_size = int(total_group_size * diff_percentage)

    # Ensure each subgroup meets minimum size requirements
    subgroup1_size = max(min_group_size // 2, (total_group_size + diff_size) // 2)
    subgroup2_size = max(min_group_size // 2, total_group_size - subgroup1_size)

    # Generate dataset for subgroup 1 and subgroup 2
    with suppress_stdout():
        subgroup1_individuals_df = data_schema.synthesizer.sample_from_conditions(conditions=[Condition(
            num_rows=subgroup1_size,
            column_values={k: v for k, v in zip(data_schema.attr_names, subgroup1_vals) if v != -1}
        )])
    subgroup1_individuals_df = fill_nan_values(subgroup1_individuals_df, data_schema)

    subgroup1_outcome, subgroup1_epistemic_uncertainty, subgroup1_aleatoric_uncertainty = generate_outcomes_with_uncertainty(
        df=subgroup1_individuals_df,
        feature_columns=data_schema.attr_names,
        weights=W[-1], bias=0.0, group_column=None,
        group_bias=0.0, epistemic_uncertainty=epis_uncertainty,
        aleatoric_uncertainty=alea_uncertainty, n_estimators=50, random_state=None)

    subgroup1_individuals_df['outcome'] = subgroup1_outcome
    subgroup1_individuals_df['epis_uncertainty'] = subgroup1_epistemic_uncertainty
    subgroup1_individuals_df['alea_uncertainty'] = subgroup1_aleatoric_uncertainty

    with suppress_stdout():
        subgroup2_individuals_df = data_schema.synthesizer.sample_from_conditions(conditions=[Condition(
            num_rows=subgroup2_size,
            column_values={k: v for k, v in zip(data_schema.attr_names, subgroup2_vals) if v != -1}
        )])
    subgroup2_individuals_df = fill_nan_values(subgroup2_individuals_df, data_schema)

    subgroup2_outcome, subgroup2_epistemic_uncertainty, subgroup2_aleatoric_uncertainty = generate_outcomes_with_uncertainty(
        df=subgroup2_individuals_df,
        feature_columns=data_schema.attr_names,
        weights=W[-1], bias=subgroup_bias, group_column=None,
        group_bias=0.0, epistemic_uncertainty=epis_uncertainty,
        aleatoric_uncertainty=alea_uncertainty, n_estimators=50, random_state=None)

    subgroup2_individuals_df['outcome'] = subgroup2_outcome
    subgroup2_individuals_df['epis_uncertainty'] = subgroup2_epistemic_uncertainty
    subgroup2_individuals_df['alea_uncertainty'] = subgroup2_aleatoric_uncertainty

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
    result_df['group_size'] = subgroup1_individuals_df.shape[0] + subgroup2_individuals_df.shape[0]
    result_df['diff_subgroup_size'] = diff_percentage

    return result_df


class CollisionTracker:
    def __init__(self, nb_attributes):
        self.used_combinations = set()
        self.nb_attributes = nb_attributes

    def is_collision(self, possibility):
        return tuple(possibility) in self.used_combinations

    def add_combination(self, possibility):
        self.used_combinations.add(tuple(possibility))


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

    data = DiscriminationData(
        dataframe=results,
        categorical_columns=list(attr_names) + [outcome_column],
        attributes=protected_attr,
        collisions=collisions,
        nb_groups=results['group_key'].nunique(),
        max_group_size=max_group_size,
        hiddenlayers_depth=W.shape[0],
        outcome_column=outcome_column,
        attr_possible_values=attr_possible_values,
        schema=data_schema
    )

    data = calculate_actual_metrics(data)

    if use_cache:
        cache.save(data, params)

    return data


def is_numeric_type(series):
    """
    Check if a pandas Series contains numeric data suitable for SDV numerical modeling.
    """
    # Check if dtype is numeric
    if pd.api.types.is_numeric_dtype(series):
        # Check if it has enough unique values to be treated as numerical
        n_unique = series.nunique()
        if n_unique >= 5:  # SDV typically treats columns with 5+ values as numeric
            return True
    return False


def calculate_sdv_correlation_matrix(encoded_df, attr_columns, ensure_positive_definite=True):
    """
    Calculate a correlation matrix suitable for SDV from encoded data.

    Args:
        encoded_df: DataFrame with encoded values
        attr_columns: List of attribute column names
        ensure_positive_definite: Whether to ensure the matrix is positive definite

    Returns:
        Correlation matrix as numpy array
    """
    n_cols = len(attr_columns)
    correlation_matrix = np.zeros((n_cols, n_cols))

    # Fill the diagonal with 1.0
    np.fill_diagonal(correlation_matrix, 1.0)

    # Calculate pairwise correlations
    for i, col1 in enumerate(attr_columns):
        for j in range(i + 1, n_cols):  # Only need to calculate upper triangle
            col2 = attr_columns[j]

            # Filter out missing values (-1)
            mask = (encoded_df[col1] >= 0) & (encoded_df[col2] >= 0)

            if mask.sum() > 1:  # Need at least 2 points for correlation
                try:
                    # Use Spearman correlation as it works better with ordinal data
                    corr, _ = spearmanr(encoded_df[col1][mask], encoded_df[col2][mask])

                    # Handle NaN or infinity
                    if np.isnan(corr) or np.isinf(corr):
                        corr = 0.0

                    # Keep correlations in reasonable bounds
                    corr = np.clip(corr, -0.99, 0.99)

                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr  # Mirror to lower triangle
                except:
                    # If correlation calculation fails, use zero
                    correlation_matrix[i, j] = 0.0
                    correlation_matrix[j, i] = 0.0
            else:
                correlation_matrix[i, j] = 0.0
                correlation_matrix[j, i] = 0.0

    # Ensure the matrix is positive definite if requested
    if ensure_positive_definite:
        # Check eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)

        # If any eigenvalues are negative or very close to zero, adjust them
        if np.any(eigenvalues < 1e-6):
            # Set small eigenvalues to a small positive number
            eigenvalues[eigenvalues < 1e-6] = 1e-6

            # Reconstruct the correlation matrix
            correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

            # Normalize to ensure diagonal is 1
            d = np.sqrt(np.diag(correlation_matrix))
            correlation_matrix = correlation_matrix / np.outer(d, d)

            # Ensure it's symmetric (might have small numerical errors)
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2

            # Final check for diagonal elements
            np.fill_diagonal(correlation_matrix, 1.0)

    return correlation_matrix


def create_sdv_numerical_distributions(schema):
    """Create numerical distributions configuration for SDV GaussianCopulaSynthesizer."""
    numerical_distributions = {}

    for attr_name, attr_cats in zip(schema.attr_names, schema.attr_categories):
        # Check if attribute is numerical
        if all(isinstance(c, (int, float)) for c in attr_cats if c != -1):
            # Use beta distribution for protected attributes and truncnorm for non-protected
            is_protected = schema.protected_attr[schema.attr_names.index(attr_name)]
            if is_protected:
                numerical_distributions[attr_name] = 'beta'
            else:
                numerical_distributions[attr_name] = 'truncnorm'

    return numerical_distributions


def generate_schema_from_dataframe(
        df: pd.DataFrame,
        protected_columns: List[str] = None,
        attr_prefix: str = None,
        outcome_column: str = 'outcome',
        ensure_positive_definite: bool = True,
        n_samples: int = 100,
        use_attr_naming_pattern: bool = False,
        fit_synthesizer: bool = True,
        synthesizer_params: Dict = None
) -> Tuple[DataSchema, np.ndarray, Dict[str, str], pd.DataFrame]:
    """
    Generate a DataSchema with fitted copula synthesizer from a pandas DataFrame.

    Args:
        df: Input DataFrame
        protected_columns: List of column names that are protected attributes
        attr_prefix: Prefix for attribute columns if not using naming pattern
        outcome_column: Name of the outcome column
        ensure_positive_definite: Whether to ensure the correlation matrix is positive definite
        n_samples: Number of samples to use for binning of numerical columns
        use_attr_naming_pattern: Whether to use attribute naming pattern (Attr{n}_T/X)
        fit_synthesizer: Whether to fit a GaussianCopulaSynthesizer
        synthesizer_params: Parameters for the GaussianCopulaSynthesizer

    Returns:
        Tuple of (DataSchema, correlation_matrix, column_mapping, encoded_dataframe, simple_encoded_dataframe)
    """
    if outcome_column not in df.columns:
        raise ValueError(f"Outcome column '{outcome_column}' not found in DataFrame")

    # Create column mapping
    new_cols = []
    for k, v in enumerate(df.columns):
        if v == outcome_column:
            attr_col = 'outcome'
        else:
            attr_col = f"Attr{k + 1}_{'T' if v in protected_columns else 'X'}"
        new_cols.append(attr_col)

    new_cols_mapping = {k: v for k, v in zip(new_cols, df.columns)}

    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # Rename columns if using attribute naming pattern
    if use_attr_naming_pattern:
        df_copy.columns = new_cols
        protected_columns = [col for col in df_copy.columns if col.endswith('_T')]
        attr_columns = [col for col in df_copy.columns if col.endswith('_X') or col.endswith('_T')]
    else:
        if attr_prefix:
            attr_columns = [col for col in df_copy.columns if col.startswith(attr_prefix)]
        else:
            attr_columns = [col for col in df_copy.columns if col != outcome_column]

    if not attr_columns:
        raise ValueError("No attribute columns found")

    # Initialize data structures
    attr_categories = []
    encoded_df = pd.DataFrame(index=df_copy.index)
    simple_encoded_df = pd.DataFrame(index=df_copy.index)
    category_maps = {}  # Store category maps for each column
    categorical_distribution = {}  # Store categorical distributions

    # Store simple encoding maps for reference
    simple_category_maps = {}

    # Process each attribute column
    for col in attr_columns:
        # Check if column is numeric
        is_numeric = is_numeric_type(df_copy[col])

        if is_numeric:
            # Handle numeric column - use SDV-compatible binning approach
            values = df_copy[col].dropna().values

            if len(values) > 0:
                # Create bins based on quantiles for more uniform distribution
                num_bins = min(n_samples, len(np.unique(values)))

                # Use quantiles to create bins for numerical data
                quantiles = np.linspace(0, 1, num_bins + 1)
                bins = np.quantile(values, quantiles)

                # Handle edge case where all values are the same
                if len(np.unique(bins)) == 1:
                    bins = np.array([bins[0] - 1, bins[0], bins[0] + 1])
                    num_bins = 2

                # Encode values
                encoded = np.full(len(df_copy), -1)
                mask = df_copy[col].notna()

                if mask.any():
                    # Get bin indices (0 to num_bins-1)
                    bin_indices = np.digitize(df_copy[col][mask], bins) - 1
                    bin_indices = np.clip(bin_indices, 0, num_bins - 1)  # Ensure within range
                    encoded[mask] = bin_indices

                categories = [-1] + list(range(num_bins))

                # Create category map with bin representations
                category_map = {-1: 'nan'}
                for i in range(num_bins):
                    lower = bins[i]
                    upper = bins[i + 1] if i < num_bins - 1 else bins[i] + (bins[i] - bins[i - 1])
                    category_map[i] = f"[{lower:.4g}, {upper:.4g}]"

                # Calculate distribution
                counts = np.bincount(encoded[encoded >= 0].astype(int), minlength=num_bins)
                probs = counts / np.sum(counts) if np.sum(counts) > 0 else np.ones(num_bins) / num_bins

                # Simple encoding for numerical column (no binning)
                # For simple encoding, we just keep the original values
                simple_encoded_df[col] = df_copy[col]

            else:
                # Handle empty column
                encoded = np.full(len(df_copy), -1)
                categories = [-1, 0]
                category_map = {-1: 'nan', 0: '0'}
                probs = np.array([1.0])

                # Simple encoding for empty numerical column
                simple_encoded_df[col] = df_copy[col]

        else:
            # Handle categorical column - use standard encoding
            le = LabelEncoder()
            non_nan_vals = df_copy[col].dropna().unique()

            if len(non_nan_vals) > 0:
                # Convert all values to strings for consistent encoding
                str_vals = [str(x) for x in non_nan_vals]
                str_vals = list(dict.fromkeys(str_vals))  # Remove duplicates while preserving order

                le.fit(str_vals)
                encoded = np.full(len(df_copy), -1)

                mask = df_copy[col].notna()
                if mask.any():
                    encoded[mask] = le.transform([str(x) for x in df_copy[col][mask]])

                    # Simple encoding for categorical column
                    simple_encoded = np.full(len(df_copy), -1)
                    simple_encoded[mask] = le.transform([str(x) for x in df_copy[col][mask]])
                    simple_encoded_df[col] = simple_encoded

                categories = [-1] + list(range(len(str_vals)))
                category_map = {-1: 'nan', **{i: val for i, val in enumerate(str_vals)}}
                simple_category_maps[col] = category_map.copy()  # Store the mapping for simple encoded

                # Calculate categorical distribution
                counts = np.bincount(encoded[encoded >= 0].astype(int), minlength=len(str_vals))
                probs = counts / np.sum(counts) if np.sum(counts) > 0 else np.ones(len(str_vals)) / len(str_vals)

            else:
                # Handle empty categorical column
                encoded = np.full(len(df_copy), -1)
                categories = [-1, 0]
                category_map = {-1: 'nan', 0: 'empty'}
                probs = np.array([1.0])

                # Simple encoding for empty categorical column
                simple_encoded_df[col] = np.full(len(df_copy), -1)

        # Store results
        encoded_df[col] = encoded
        attr_categories.append(categories)
        category_maps[col] = category_map
        categorical_distribution[col] = probs.tolist()

    # Calculate correlation matrix optimized for SDV
    correlation_matrix = calculate_sdv_correlation_matrix(
        encoded_df, attr_columns, ensure_positive_definite
    )

    # Encode outcome column
    le = LabelEncoder()
    outcome_vals = df_copy['outcome'].astype(str).values
    encoded_df['outcome'] = le.fit_transform(outcome_vals)

    # Also encode outcome in simple encoded dataframe
    simple_encoded_df['outcome'] = encoded_df['outcome'].copy()

    # Create SDV metadata
    sdv_metadata = SingleTableMetadata()

    # Add attribute columns to metadata
    for attr_name in attr_columns:
        is_numeric_attr = is_numeric_type(df_copy[attr_name]) if attr_name in df_copy.columns else False
        sdv_metadata.add_column(attr_name, sdtype='numerical' if is_numeric_attr else 'categorical')

    # Add outcome column to metadata
    sdv_metadata.add_column('outcome', sdtype='categorical')

    # Create DataSchema object (without synthesizer initially)
    schema = DataSchema(
        attr_categories=attr_categories,
        protected_attr=[col in protected_columns for col in attr_columns],
        attr_names=attr_columns,
        categorical_distribution=categorical_distribution,
        correlation_matrix=correlation_matrix,
        gen_order=list(range(len(attr_columns))),
        category_maps=category_maps,
        column_mapping=new_cols_mapping,
        sdv_metadata=sdv_metadata,
        synthesizer=None,
    )

    # Fit a GaussianCopulaSynthesizer if requested
    if fit_synthesizer:
        # Prepare training data for the synthesizer
        training_data = encoded_df.copy()
        # Replace -1 (missing) values with NaN for SDV
        for col in attr_columns:
            training_data.loc[training_data[col] == -1, col] = np.nan

        # Create numerical distribution settings
        numerical_distributions = create_sdv_numerical_distributions(schema)

        # Set default synthesizer params
        default_params = {
            'enforce_min_max_values': True,
            'enforce_rounding': True,
            'numerical_distributions': numerical_distributions,
            'default_distribution': 'beta'
        }

        # Update with user-provided params if any
        if synthesizer_params:
            default_params.update(synthesizer_params)

        # Create and fit the synthesizer
        synthesizer = GaussianCopulaSynthesizer(sdv_metadata, **default_params)
        print("Fitting GaussianCopulaSynthesizer...")
        with suppress_stdout():
            synthesizer.fit(encoded_df)

        # Add the fitted synthesizer to the schema
        schema.synthesizer = synthesizer

    return schema, correlation_matrix, new_cols_mapping, encoded_df


def calculate_sdv_correlation_matrix(encoded_df, attr_columns, ensure_positive_definite=True):
    """
    Calculate a correlation matrix suitable for SDV from encoded data.

    Args:
        encoded_df: DataFrame with encoded values
        attr_columns: List of attribute column names
        ensure_positive_definite: Whether to ensure the matrix is positive definite

    Returns:
        Correlation matrix as numpy array
    """
    n_cols = len(attr_columns)
    correlation_matrix = np.zeros((n_cols, n_cols))

    # Fill the diagonal with 1.0
    np.fill_diagonal(correlation_matrix, 1.0)

    # Calculate pairwise correlations
    for i, col1 in enumerate(attr_columns):
        for j in range(i + 1, n_cols):  # Only need to calculate upper triangle
            col2 = attr_columns[j]

            # Filter out missing values (-1)
            mask = (encoded_df[col1] >= 0) & (encoded_df[col2] >= 0)

            if mask.sum() > 1:  # Need at least 2 points for correlation
                try:
                    # Use Spearman correlation as it works better with ordinal data
                    corr, _ = spearmanr(encoded_df[col1][mask], encoded_df[col2][mask])

                    # Handle NaN or infinity
                    if np.isnan(corr) or np.isinf(corr):
                        corr = 0.0

                    # Keep correlations in reasonable bounds
                    corr = np.clip(corr, -0.99, 0.99)

                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr  # Mirror to lower triangle
                except:
                    # If correlation calculation fails, use zero
                    correlation_matrix[i, j] = 0.0
                    correlation_matrix[j, i] = 0.0
            else:
                correlation_matrix[i, j] = 0.0
                correlation_matrix[j, i] = 0.0

    # Ensure the matrix is positive definite if requested
    if ensure_positive_definite:
        # Check eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)

        # If any eigenvalues are negative or very close to zero, adjust them
        if np.any(eigenvalues < 1e-6):
            # Set small eigenvalues to a small positive number
            eigenvalues[eigenvalues < 1e-6] = 1e-6

            # Reconstruct the correlation matrix
            correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

            # Normalize to ensure diagonal is 1
            d = np.sqrt(np.diag(correlation_matrix))
            correlation_matrix = correlation_matrix / np.outer(d, d)

            # Ensure it's symmetric (might have small numerical errors)
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2

            # Final check for diagonal elements
            np.fill_diagonal(correlation_matrix, 1.0)

    return correlation_matrix


def generate_from_real_data(dataset_name, use_cache=False, extra_rows=None, *args, **kwargs):
    """
    Generate synthetic discrimination data based on real-world datasets using SDV.

    Args:
        dataset_name: Name of the dataset ('adult', 'credit', 'bank')
        use_cache: Whether to use cached data if available
        extra_rows: Number of additional rows to generate outside of the group structure
        *args, **kwargs: Additional arguments for generate_data

    Returns:
        Tuple of (DiscriminationData, DataSchema)
    """
    # Try to load from cache first
    if use_cache:
        cache = DataCache()
        cache_params = {
            'dataset_name': dataset_name,
            'extra_rows': extra_rows,
            **{k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))}
        }
        cached_data = cache.load(cache_params)
        if cached_data is not None:
            print(f"Using cached data for {dataset_name}")
            return cached_data, cached_data.schema if hasattr(cached_data, 'schema') else None

    # Get the dataset and schema
    if dataset_name == 'adult':
        # Adult Income dataset
        adult = fetch_ucirepo(id=2)
        df = adult['data']['original']
        df.drop(columns=['fnlwgt'], inplace=True)

        # Get schema from the dataframe
        schema, correlation_matrix, column_mapping, enc_df = generate_schema_from_dataframe(
            df,
            protected_columns=['race', 'sex'],
            outcome_column='income',
            use_attr_naming_pattern=True
        )

    elif dataset_name == 'credit':
        # Credit dataset
        df = fetch_ucirepo(id=144)
        df = df['data']['original']

        # Get schema from the dataframe
        schema, correlation_matrix, column_mapping, enc_df = generate_schema_from_dataframe(
            df,
            protected_columns=['Attribute8', 'Attribute12'],
            outcome_column='Attribute20',
            use_attr_naming_pattern=True,
            ensure_positive_definite=True
        )

    elif dataset_name == 'bank':
        # Bank Marketing dataset
        bank_marketing = fetch_ucirepo(id=222)
        df = pd.concat([bank_marketing.data.features, bank_marketing.data.targets], axis=1)

        # Ensure all column names are valid Python identifiers
        df.columns = [col.replace('-', '_') for col in df.columns]

        # Get schema from the dataframe
        schema, correlation_matrix, column_mapping, enc_df = generate_schema_from_dataframe(
            df,
            protected_columns=['age', 'marital', 'education'],
            outcome_column='y',
            use_attr_naming_pattern=True,
            ensure_positive_definite=True
        )

    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    # Ensure schema attribute names are unique
    schema.attr_names = list(dict.fromkeys(schema.attr_names))

    # Create SDV metadata from schema
    metadata = schema.to_sdv_metadata()

    # Create numerical distribution settings
    numerical_distributions = create_sdv_numerical_distributions(schema)

    # Train an SDV synthesizer on the real data
    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        numerical_distributions=numerical_distributions,
        default_distribution='beta'
    )

    print(f"Training GaussianCopulaSynthesizer on {dataset_name} dataset...")

    # Prepare the training data (handle missing values)
    training_data = enc_df.copy()
    for col in schema.attr_names:
        if col in training_data.columns:
            # Replace -1 (missing) values with NaN for SDV
            training_data.loc[training_data[col] == -1, col] = np.nan
    with suppress_stdout():
        synthesizer.fit(training_data)

    schema.synthesizer = synthesizer

    # Generate data using our function that uses the trained synthesizer
    data = generate_data(
        gen_order=schema.gen_order,
        correlation_matrix=correlation_matrix,
        data_schema=schema,
        use_cache=False,  # We're already handling cache at this level
        extra_rows=extra_rows,
        *args, **kwargs
    )

    # Store schema with the data for convenience
    data.schema = schema

    # Cache the result if requested
    if use_cache:
        cache = DataCache()
        cache.save(data, cache_params)

    return data, schema


def get_real_data(
        dataset_name: str,
        protected_columns: Optional[List[str]] = None,
        outcome_column: Optional[str] = None,
        use_cache=False,
        *args, **kwargs
) -> Tuple:
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

    # Try to load from cache first
    if use_cache:
        cache = DataCache()
        cache_params = {
            'dataset_name': dataset_name,
            'protected_columns': str(config['protected_columns']),
            'outcome_column': config['outcome_column'],
            'real_data': True  # Flag to distinguish from synthetic data
        }
        cached_data = cache.load(cache_params)
        if cached_data is not None:
            print(f"Using cached real data for {dataset_name}")
            return cached_data, cached_data.schema if hasattr(cached_data, 'schema') else None

    # Fetch the dataset
    dataset = fetch_ucirepo(id=config['id'])
    df = dataset['data']['original']

    if dataset_name == 'adult':
        df['income'] = df['income'].apply(lambda x: x.replace('.', ''))

    # df = df.sample(60000, replace=True)

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

    # Create attr_possible_values mapping
    attr_possible_values = {
        attr_name: categories
        for attr_name, categories in zip(schema.attr_names, schema.attr_categories)
    }

    # Create DiscriminationData without generating synthetic data
    data = DiscriminationData(
        dataframe=enc_df,
        categorical_columns=list(schema.attr_names) + ['outcome'],
        attributes={k: v for k, v in zip(schema.attr_names, schema.protected_attr)},
        collisions=0,
        nb_groups=1,  # Just one "group" for the entire dataset
        max_group_size=len(enc_df),
        hiddenlayers_depth=0,
        outcome_column='outcome',
        attr_possible_values=attr_possible_values,
        schema=schema
    )

    # Cache the result if requested
    if use_cache:
        cache = DataCache()
        cache.save(data, cache_params)

    return data, schema


def create_sdv_numerical_distributions(data_schema):
    """Create numerical distributions configuration for SDV GaussianCopulaSynthesizer."""
    numerical_distributions = {}

    for attr_name, attr_cats in zip(data_schema.attr_names, data_schema.attr_categories):
        # Check if attribute is numerical
        if all(isinstance(c, (int, float)) for c in attr_cats if c != -1):
            # Use beta distribution for protected attributes and normal for non-protected
            is_protected = data_schema.protected_attr[data_schema.attr_names.index(attr_name)]
            if is_protected:
                numerical_distributions[attr_name] = 'beta'
            else:
                numerical_distributions[attr_name] = 'truncnorm'

    return numerical_distributions
