import copy
import datetime
import hashlib
import itertools
import json
import math
import pickle
import random
import contextlib
import io
import sys
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
from sklearn.tree import DecisionTreeClassifier
from sdv.sampling import Condition
from tqdm import tqdm

from path import HERE
from ucimlrepo import fetch_ucirepo
import logging
import warnings
from scipy.spatial.distance import jensenshannon
from uncertainty_quantification.main import UncertaintyRandomForest, AleatoricUncertainty, EpistemicUncertainty

# Suppress all warnings
warnings.filterwarnings('ignore')

# Set logging level to ERROR or higher (to suppress INFO, DEBUG, and WARNING)
logging.getLogger('sdv').setLevel(logging.ERROR)
logging.getLogger('SingleTableSynthesizer').setLevel(logging.ERROR)
logging.getLogger('copulas').setLevel(logging.ERROR)


def bin_array_values(array, num_bins):
    min_val = np.min(array)
    max_val = np.max(array)
    bins = np.linspace(min_val, max_val, num_bins + 1)
    binned_indices = np.digitize(array, bins) - 1
    return binned_indices


def calculate_subgroup_key_similarity_binary_overlap(data):
    """
    Calculate similarity based on binary representation overlap.
    Assumes subgroup keys can be converted to binary representations.
    """

    def calculate_group_similarity(group_data):
        subgroup_keys = group_data['subgroup_key'].unique()
        if len(subgroup_keys) != 2:
            return np.nan

        key1_parts = subgroup_keys[0].split('|')
        key2_parts = subgroup_keys[1].split('|')

        if len(key1_parts) != len(key2_parts):
            return np.nan

        matches = 0
        total_positions = len(key1_parts)

        for part1, part2 in zip(key1_parts, key2_parts):
            if part1 == part2:
                matches += 1

        return matches / total_positions

    return data.dataframe.groupby('group_key', group_keys=False).apply(calculate_group_similarity)


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
    if pd.api.types.is_numeric_dtype(y) and pd.Series(y).nunique() > 20:  # Heuristic for continuous
        from sklearn.ensemble import RandomForestRegressor
        regr = RandomForestRegressor(n_estimators=50, random_state=42)
        regr.fit(X, y)
        # For regressor, we can use prediction variance as a measure of uncertainty
        # This is a simplification; proper uncertainty quantification is more complex
        predictions = np.array([tree.predict(X) for tree in regr.estimators_])
        base_epistemic = np.std(predictions, axis=0)
        base_aleatoric = np.zeros_like(base_epistemic)  # Placeholder for aleatoric
    else:
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
        try:
            unique_subgroups = group_data['subgroup_key'].unique()
            subgroup1_outcome = group_data[group_data['subgroup_key'] == unique_subgroups[0]][data.outcome_column]
            subgroup2_outcome = group_data[group_data['subgroup_key'] == unique_subgroups[1]][data.outcome_column]
            res = (abs(avg_outcome - subgroup1_outcome.mean()) + abs(avg_outcome - subgroup2_outcome.mean())) / 2
        except Exception as e:
            print(e)
        return res

    return data.dataframe.groupby('group_key', group_keys=False).apply(calculate_group_diff)


def calculate_actual_metrics(data):
    """
    Calculate the actual metrics and relevance for each group.
    """
    actual_similarity = calculate_actual_similarity(data)
    actual_binary_similarity = calculate_subgroup_key_similarity_binary_overlap(data)
    actual_uncertainties = calculate_actual_uncertainties(data)
    actual_mean_diff_outcome = calculate_actual_mean_diff_outcome(data)
    actual_diff_outcome_from_avg = calculate_actual_diff_outcome_from_avg(data)

    # Merge these metrics into the main dataframe
    data.dataframe['calculated_similarity'] = data.dataframe['group_key'].map(actual_similarity)
    data.dataframe['calculated_binary_similarity'] = data.dataframe['group_key'].map(actual_binary_similarity)
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


class DataCache:
    def __init__(self):
        self.cache_dir = HERE.joinpath(".cache/discrimination_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a metadata file to track cache contents
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def generate_cache_key(self, params: dict) -> str:
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
        cache_key = self.generate_cache_key(params)
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
        cache_key = self.generate_cache_key(params)
        cache_path = self.get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cached data: {e}")
                return None
        return None


@dataclass
class GroupDefinition:
    group_size: int
    subgroup_bias: float
    similarity: float
    alea_uncertainty: float
    epis_uncertainty: float
    frequency: float
    diff_subgroup_size: float
    subgroup1: Dict[str, Any]
    subgroup2: Dict[str, Any]
    avg_diff_outcome: Optional[float] = None
    granularity: Optional[int] = None
    intersectionality: Optional[int] = None

    def __post_init__(self):
        """Validate the group definition after initialization."""
        if set(self.subgroup1.keys()) != set(self.subgroup2.keys()):
            raise ValueError("subgroups must have the same keys")

        if -1 in set(self.subgroup1.keys()) | set(self.subgroup2.keys()):
            raise ValueError("-1 is not a valide value")

        # Ensure subgroups have at least one attribute
        if not self.subgroup1 or not self.subgroup2:
            raise ValueError("Both subgroups must have at least one attribute defined")

        # Ensure parameters are within valid ranges
        if not (0.0 <= self.similarity <= 1.0):
            raise ValueError("Similarity must be between 0.0 and 1.0")
        if not (0.0 <= self.alea_uncertainty <= 1.0):
            raise ValueError("Aleatoric uncertainty must be between 0.0 and 1.0")
        if not (0.0 <= self.epis_uncertainty <= 1.0):
            raise ValueError("Epistemic uncertainty must be between 0.0 and 1.0")
        if not (0.0 <= self.frequency <= 1.0):
            raise ValueError("Frequency must be between 0.0 and 1.0")
        if not (0.0 <= self.diff_subgroup_size <= 1.0):
            raise ValueError("Difference in subgroup size must be between 0.0 and 1.0")

        # Ensure group_size is valid
        if self.group_size < 2:
            raise ValueError("Group size must be at least 2 to have both subgroups")

    def get_preset_values_for_subgroup1(self, attr_names):
        return [self.subgroup1.get(attr, None) for attr in attr_names]

    def get_preset_values_for_subgroup2(self, attr_names):
        return [self.subgroup2.get(attr, None) for attr in attr_names]

    def calculate_granularity_intersectionality(self, attr_names, protected_attr):
        # Get attributes used in either subgroup
        used_attrs = set(self.subgroup1.keys()) | set(self.subgroup2.keys())

        # Count protected and non-protected attributes
        granularity = 0
        intersectionality = 0

        for attr in used_attrs:
            if attr in attr_names:
                idx = attr_names.index(attr)
                if protected_attr[idx]:
                    intersectionality += 1
                else:
                    granularity += 1

        return granularity, intersectionality


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
    y_true_col: str
    y_pred_col: str
    outcome_column: str = 'outcome'
    relevance_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    attr_possible_values: Dict[str, List[int]] = field(default_factory=dict)
    generation_arguments: Dict[str, Any] = field(default_factory=dict)

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
    def sensitive_indices_dict(self):
        return {k: i for i, (k, v) in enumerate(self.attributes.items()) if v}

    @property
    def sensitive_indices(self):
        return [i for i, (k, v) in enumerate(self.attributes.items()) if v]

    @property
    def training_dataframe(self):
        return self.dataframe[list(self.attributes) + [self.outcome_column]]

    @property
    def training_dataframe_with_ypred(self):
        if self.y_pred_col in self.dataframe.columns:
            return self.dataframe[list(self.attributes) + [self.outcome_column, self.y_pred_col]]
        else:
            raise ValueError(f"Column {self.outcome_column} not found in dataframe")

    @property
    def xdf(self):
        return self.dataframe[list(self.attributes)]

    @property
    def ydf(self):
        return self.dataframe[self.outcome_column]

    @property
    def y_pred(self):
        if self.y_pred_col in self.dataframe.columns:
            return self.dataframe[self.y_pred_col]
        else:
            raise ValueError(f"Column {self.y_pred_col} not found in dataframe")

    @property
    def input_bounds(self):
        input_bounds = getattr(self, '_input_bounds', None)
        if input_bounds is None:
            self._input_bounds = []
            for col in list(self.attributes):
                min_val = max(math.floor(self.xdf[col].min()), 0)
                max_val = math.ceil(self.xdf[col].max())
                self._input_bounds.append([min_val, max_val])
        input_bounds = getattr(self, '_input_bounds')
        return input_bounds

    @property
    def nb_outcomes(self):
        return self.dataframe[self.outcome_column].unique().shape[0]

    def get_random_rows(self, n: int) -> pd.DataFrame:
        random_data = {}

        for i, attr in enumerate(self.attributes.keys()):
            min_val, max_val = self.input_bounds[i]

            if attr in self.categorical_columns:
                random_values = np.random.randint(min_val, max_val + 1, size=n)
            else:
                random_values = np.random.uniform(min_val, max_val, size=n)

            random_data[attr] = random_values

        random_df = pd.DataFrame(random_data)

        return random_df


class RobustGaussianCopulaSynthesizer(GaussianCopulaSynthesizer):
    """Enhanced GaussianCopulaSynthesizer with improved handling for conditional sampling.

    This class extends the standard GaussianCopulaSynthesizer to better handle conditional
    sampling by ensuring all possible value combinations are represented in the training data.

    Args:
        metadata (sdv.metadata.Metadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Whether to clip values to the min/max seen during fit. Defaults to True.
        enforce_rounding (bool):
            Whether to round values as in the original data. Defaults to True.
        locales (list or str):
            Default locale(s) for AnonymizedFaker transformers. Defaults to ['en_US'].
        numerical_distributions (dict):
            Dictionary mapping field names to distributions.
        default_distribution (str):
            Default distribution to use. Defaults to 'beta'.
        max_combinations_to_fit (int):
            Maximum number of attribute combinations to include in training data.
            If there are more possible combinations than this, a sample will be used.
            Defaults to 10000.
    """

    def __init__(
            self,
            metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            locales=['en_US'],
            numerical_distributions=None,
            default_distribution=None,
            max_combinations_to_fit=10000
    ):
        super().__init__(
            metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales,
            numerical_distributions=numerical_distributions,
            default_distribution=default_distribution,
        )
        self.max_combinations_to_fit = max_combinations_to_fit

    def _fit(self, processed_data):
        """Enhanced fit method that ensures the model learns all possible value combinations.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        # Get columns and their unique values
        columns_to_consider = []
        unique_values = {}

        for column in processed_data.columns:
            # Skip columns with too many unique values (likely numeric or IDs)
            unique_vals = processed_data[column].dropna().unique()
            if len(unique_vals) <= 20:  # Only include columns with reasonable cardinality
                columns_to_consider.append(column)
                unique_values[column] = unique_vals

        # Calculate the number of possible combinations
        total_combinations = 1
        for column in columns_to_consider:
            total_combinations *= len(unique_values[column])

        # Generate combinations for training
        combinations_data = None
        if total_combinations <= self.max_combinations_to_fit and columns_to_consider:
            print(f"Generating {total_combinations} combinations for robust training")

            # Generate all combinations of column values
            all_combinations = list(itertools.product(
                *[unique_values[col] for col in columns_to_consider]
            ))

            # Convert combinations to DataFrame
            combinations_dict = {
                col: [combo[i] for combo in all_combinations]
                for i, col in enumerate(columns_to_consider)
            }
            combinations_data = pd.DataFrame(combinations_dict)

            # Add random values for non-included columns
            for col in processed_data.columns:
                if col not in columns_to_consider:
                    if col in combinations_data:
                        continue

                    # Sample random values from the original column
                    combinations_data[col] = np.random.choice(
                        processed_data[col].dropna().values,
                        size=len(combinations_data),
                        replace=True
                    )

            # Combine with original data
            combined_data = pd.concat([processed_data, combinations_data], ignore_index=True)

            # Call the parent class _fit method with the enhanced data
            super()._fit(combined_data)
        else:
            # If too many combinations, use the original data
            if columns_to_consider:
                warnings.warn(
                    f"Too many possible combinations ({total_combinations}) to include all in training. "
                    f"Using original data only. This may affect conditional sampling performance."
                )
            super()._fit(processed_data)

    def sample_from_conditions(self, conditions, max_tries_per_batch=100, batch_size=None, output_file_path=None):
        """Sample from conditions with improved error handling.

        Args:
            conditions (list[sdv.sampling.Condition]):
                A list of conditions to sample from.
            max_tries_per_batch (int):
                Number of times to retry sampling. Defaults to 100.
            batch_size (int):
                Batch size for sampling. Defaults to None.
            output_file_path (str):
                Path to write samples to. Defaults to None.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        try:
            # Try the standard sampling method first
            return super().sample_from_conditions(
                conditions,
                max_tries_per_batch=max_tries_per_batch,
                batch_size=batch_size,
                output_file_path=output_file_path
            )
        except ValueError as e:
            error_msg = str(e)
            if "Unable to sample any rows for the given conditions" in error_msg:
                print("Attempting fallback sampling method...")

                # Fallback approach: Sample extra rows and filter them
                sampled_rows = []
                for condition in conditions:
                    column_values = condition.get_column_values()
                    num_rows = condition.get_num_rows()

                    # Sample 10x the needed rows
                    sample_size = min(num_rows * 10, 10000)
                    sample = self.sample(sample_size)

                    # Filter rows that match our condition
                    for col, val in column_values.items():
                        if col in sample.columns:
                            # For numerical columns, allow some tolerance
                            if pd.api.types.is_numeric_dtype(sample[col]):
                                # 1% tolerance for numeric values
                                tolerance = abs(val) * 0.01 if val != 0 else 0.01
                                sample = sample[
                                    (sample[col] >= val - tolerance) &
                                    (sample[col] <= val + tolerance)
                                    ]
                            else:
                                sample = sample[sample[col] == val]

                    # Take the needed number of rows
                    if len(sample) > 0:
                        # Fix exact values for the condition columns
                        for col, val in column_values.items():
                            if col in sample.columns:
                                sample[col] = val

                        sampled_rows.append(sample.head(min(len(sample), num_rows)))
                    else:
                        print(f"Couldn't generate data for condition: {column_values}")

                if sampled_rows:
                    return pd.concat(sampled_rows, ignore_index=True)

                # If fallback failed, raise the original error
                raise e
            else:
                # If it's a different error, re-raise it
                raise e


def generate_data_schema(min_number_of_classes, max_number_of_classes, nb_attributes,
                         prop_protected_attr, fit_synthesizer=True, n_samples=10000) -> DataSchema:
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
    for i, (attr_name, attr_cats, is_protected) in enumerate(zip(
            attr_names, attr_categories, protected_attr)):

        # Determine the SDV column type
        if -1 in attr_cats:  # If -1 is in categories, it handles missing values
            cats_without_missing = [c for c in attr_cats if c != -1]
        else:
            cats_without_missing = attr_cats

        # Check if this is a categorical or numerical column
        if all(isinstance(v, (int, float)) for v in cats_without_missing):
            # This is a numerical column
            sdv_metadata.add_column(attr_name, sdtype='numerical')
        else:
            # This is a categorical column
            sdv_metadata.add_column(attr_name, sdtype='categorical')

    # Add outcome column
    sdv_metadata.add_column('outcome', sdtype='categorical')

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

        # Add examples of all possible value combinations to ensure synthesizer can handle all requests
        # First, create a list of all possible values for each attribute
        all_possible_values = []
        for i, attr_name in enumerate(attr_names):
            # Skip the -1 value (missing), only include valid values
            valid_values = [v for v in attr_categories[i] if v != -1]
            all_possible_values.append(valid_values)

        # Generate combinations in a memory-efficient way
        # We'll generate a reasonable number of combinations if there are too many
        max_combinations = 1000  # Limit to prevent memory issues

        # Calculate total number of possible combinations
        total_combinations = 1
        for values in all_possible_values:
            total_combinations *= len(values)

        # If there are too many combinations, sample a subset
        if total_combinations > max_combinations:
            # Generate a diverse sample of combinations
            combination_samples = []
            for _ in range(max_combinations):
                sample = [random.choice(values) for values in all_possible_values]
                combination_samples.append(sample)
        else:
            # Generate all combinations if the number is manageable
            import itertools
            combination_samples = list(itertools.product(*all_possible_values))

        # Create dataframe with combinations
        combinations_data = {}
        for i, attr_name in enumerate(attr_names):
            combinations_data[attr_name] = [combo[i] for combo in combination_samples]

        # Add random outcomes
        combinations_df = pd.DataFrame(combinations_data)
        combinations_df['outcome'] = random.choices([0, 1], k=len(combinations_df))

        # Combine the random data with the combination data
        combined_df = pd.concat([initial_df, combinations_df], ignore_index=True)

        # Fit the synthesizer on the combined data
        print("Fitting initial GaussianCopulaSynthesizer...")
        synthesizer.fit(combined_df)

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
                                    preset_values=None, excluded_values=None,
                                    same_attributes=None, reference_values=None):
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
        same_attributes (List[int]): Indices of attributes that should have the same values as in reference_values.
                                   Default is None (no attributes need to match).
        reference_values (List): A list of values for all attributes, used as reference for attributes in same_attributes.
                               Default is None (no reference values).

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

    # Handle same_attributes and reference_values
    if same_attributes is not None and reference_values is not None:
        # For attributes in same_attributes, set preset_values to the specific reference value
        for idx in same_attributes:
            if 0 <= idx < len(attr_names) and idx < len(reference_values):
                # Use the reference value as a preset (forced) value for this attribute
                if preset_values[idx] is None:
                    preset_values[idx] = [reference_values[idx]]
                else:
                    # If there's already a preset list, make sure it includes the reference value
                    if reference_values[idx] not in preset_values[idx]:
                        # If reference value isn't in preset list, this is a conflict
                        # We prioritize the "same" constraint by replacing the preset list
                        preset_values[idx] = [reference_values[idx]]

    # Initialize dataframe to store generated data
    generated_data = defaultdict(list)

    for _ in range(n_samples):
        for i, attr_name in enumerate(attr_names):
            # Get the distribution for the current attribute
            valid_probs = distribution.get(attr_name, [])
            probs = distribution.get(attr_name, [])
            valid_categories = list(filter(lambda x: x != -1, attr_categories[i]))
            categories = list(filter(lambda x: x != -1, attr_categories[i]))

            # Handle case where this attribute should have the same value as reference
            if same_attributes is not None and i in same_attributes and reference_values is not None:
                # Simply use the reference value
                if i < len(reference_values) and reference_values[i] != -1:
                    generated_data[attr_name].append(reference_values[i])
                    continue
                # If reference value is -1 or invalid, fall through to normal processing

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


@contextlib.contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


@dataclass
class SubgroupDefinition:
    """Defines a single subgroup with specific attribute values."""
    subgroup_id: str
    attribute_values: Dict[str, Any]  # attribute_name -> value
    size: int
    bias: float = 0.0
    uncertainty_params: Dict[str, float] = None
    frequency: float = 1.0  # For compatibility with existing code

    def __post_init__(self):
        if self.uncertainty_params is None:
            self.uncertainty_params = {
                'epistemic': np.random.uniform(0.05, 0.3),
                'aleatoric': np.random.uniform(0.05, 0.3)
            }


class OptimalDataGenerator:
    """
    Improved data generation that first calculates optimal subgroups,
    then generates all possible group combinations.
    """

    def __init__(self, data_schema: 'DataSchema'):
        self.data_schema = data_schema
        self.subgroups: Dict[str, SubgroupDefinition] = {}
        self.groups: Dict[str, GroupDefinition] = {}
        self.generated_subgroup_data: Dict[str, pd.DataFrame] = {}

    def _generate_outcomes_with_analytical_predictions(self, df: pd.DataFrame,
                                                       subgroup: 'SubgroupDefinition',
                                                       W: np.ndarray, classifier=None) -> pd.DataFrame:
        """
        Generate outcomes with analytical predictions for y_true and y_pred.

        MODIFIED LOGIC: Bias is now added in the probability space to ensure a
        direct and controllable difference, avoiding the sigmoid dampening effect.
        """
        feature_columns = self.data_schema.attr_names
        X = df[feature_columns].values
        X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        weights = W[-1] if W.ndim > 1 else W

        # --- Step 1: Generate the base "unaware" prediction logit ---
        y_pred_logit = np.dot(X_std, weights)

        # --- Step 2: Add uncertainty/noise to the logit ---
        # This simulates the "wobble" or confidence of the unaware model.
        epistemic_noise = np.random.normal(0, subgroup.uncertainty_params['epistemic'], size=len(df))
        y_pred_logit_noisy = y_pred_logit + epistemic_noise

        # --- Step 3: Convert the noisy logit to a probability ---
        if classifier:
            X_pred = df[self.data_schema.attr_names]
            df['y_pred'] = classifier.predict(X_pred)
        else:
            # This is our final "y_pred".
            y_pred_prob = 1 / (1 + np.exp(-y_pred_logit_noisy))
            df['y_pred'] = y_pred_prob

        # --- Step 4: Directly apply the bias to the prediction probability to get the true outcome ---
        # This ensures the bias term has a direct, linear effect.
        y_true_prob_biased = y_pred_prob + subgroup.bias

        # --- Step 5: Add aleatoric (irreducible) noise to the true outcome ---
        aleatoric_noise = np.random.normal(0, subgroup.uncertainty_params['aleatoric'], size=len(df))
        y_true_prob_noisy = y_true_prob_biased + aleatoric_noise

        # --- Step 6: Clip the final true outcome to ensure it remains a valid probability [0, 1] ---
        df['y_true'] = np.clip(y_true_prob_noisy, 0, 1)

        # For compatibility, set the main 'outcome' column to the true outcome
        df['outcome'] = df['y_true']

        # Store the injected parameters for verification
        df['subgroup_bias_injected'] = subgroup.bias
        df['epistemic_param_injected'] = subgroup.uncertainty_params['epistemic']
        df['aleatoric_param_injected'] = subgroup.uncertainty_params['aleatoric']

        return df

    def calculate_optimal_subgroups(self,
                                    target_nb_groups: int,
                                    min_subgroups: int = None,
                                    max_subgroups: int = None,
                                    min_group_size: int = 10,
                                    max_group_size: int = 100) -> List[SubgroupDefinition]:
        """
        Calculate the optimal number and configuration of subgroups needed
        to generate the target number of groups.

        Args:
            target_nb_groups: Desired number of groups
            min_subgroups: Minimum number of subgroups to consider
            max_subgroups: Maximum number of subgroups to consider
            min_group_size: Minimum size for each group
            max_group_size: Maximum size for each group

        Returns:
            List of optimal subgroup definitions
        """
        # Calculate minimum number of subgroups needed
        # For n subgroups, we can create C(n,2) = n*(n-1)/2 groups
        min_needed = math.ceil((1 + math.sqrt(1 + 8 * target_nb_groups)) / 2)

        if min_subgroups is None:
            min_subgroups = min_needed
        if max_subgroups is None:
            max_subgroups = min_needed + 10  # Some buffer for optimization

        print(f"Target groups: {target_nb_groups}")
        print(f"Minimum subgroups needed: {min_needed}")
        print(f"Searching in range: {min_subgroups} to {max_subgroups}")

        # Find optimal number of subgroups
        best_config = None
        best_score = float('inf')

        for n_subgroups in range(min_subgroups, max_subgroups + 1):
            max_possible_groups = n_subgroups * (n_subgroups - 1) // 2

            if max_possible_groups >= target_nb_groups:
                # Score based on efficiency (how close to target without waste)
                efficiency = target_nb_groups / max_possible_groups
                waste_penalty = max_possible_groups - target_nb_groups
                score = waste_penalty * (1 - efficiency)

                if score < best_score:
                    best_score = score
                    best_config = {
                        'n_subgroups': n_subgroups,
                        'max_groups': max_possible_groups,
                        'efficiency': efficiency
                    }

        if best_config is None:
            raise ValueError(f"Cannot generate {target_nb_groups} groups with max {max_subgroups} subgroups")

        print(f"Optimal configuration: {best_config['n_subgroups']} subgroups "
              f"-> {best_config['max_groups']} possible groups "
              f"(efficiency: {best_config['efficiency']:.2%})")

        return self._generate_diverse_subgroups(best_config['n_subgroups'],
                                                min_group_size, max_group_size)

    def _generate_diverse_subgroups(self, n_subgroups: int,
                                    min_group_size: int, max_group_size: int) -> List[SubgroupDefinition]:
        """Generate diverse subgroups that maximize coverage of attribute space."""
        subgroups = []

        # Get all possible attribute combinations
        protected_attrs = [attr for attr, is_protected in zip(self.data_schema.attr_names,
                                                              self.data_schema.protected_attr) if is_protected]
        non_protected_attrs = [attr for attr, is_protected in zip(self.data_schema.attr_names,
                                                                  self.data_schema.protected_attr) if not is_protected]

        print(f"Protected attributes: {protected_attrs}")
        print(f"Non-protected attributes: {non_protected_attrs}")

        # Strategy: Create subgroups with different combinations of attribute constraints
        # to ensure good coverage of the attribute space

        used_combinations = set()
        attempt = 0
        max_attempts = n_subgroups * 20

        while len(subgroups) < n_subgroups and attempt < max_attempts:
            attempt += 1

            # Randomly select attributes to constrain for this subgroup
            n_protected = np.random.randint(1, min(len(protected_attrs) + 1, 4)) if protected_attrs else 0
            n_non_protected = np.random.randint(0, min(len(non_protected_attrs) + 1, 4)) if non_protected_attrs else 0

            # Ensure at least one attribute is selected
            if n_protected == 0 and n_non_protected == 0:
                if protected_attrs:
                    n_protected = 1
                elif non_protected_attrs:
                    n_non_protected = 1
                else:
                    continue

            selected_protected = []
            selected_non_protected = []

            if n_protected > 0 and protected_attrs:
                selected_protected = np.random.choice(protected_attrs,
                                                      min(n_protected, len(protected_attrs)),
                                                      replace=False)

            if n_non_protected > 0 and non_protected_attrs:
                selected_non_protected = np.random.choice(non_protected_attrs,
                                                          min(n_non_protected, len(non_protected_attrs)),
                                                          replace=False)

            # Generate attribute values for selected attributes
            attribute_values = {}

            for attr in selected_protected:
                attr_idx = self.data_schema.attr_names.index(attr)
                valid_values = [v for v in self.data_schema.attr_categories[attr_idx] if v != -1]
                if valid_values:
                    attribute_values[attr] = np.random.choice(valid_values)

            for attr in selected_non_protected:
                attr_idx = self.data_schema.attr_names.index(attr)
                valid_values = [v for v in self.data_schema.attr_categories[attr_idx] if v != -1]
                if valid_values:
                    attribute_values[attr] = np.random.choice(valid_values)

            # Skip if no valid attribute values were found
            if not attribute_values:
                continue

            # Create a signature for this combination
            signature = tuple(sorted(attribute_values.items()))

            if signature not in used_combinations:
                used_combinations.add(signature)

                # Calculate subgroup size (will be adjusted during group formation)
                base_size = np.random.randint(min_group_size // 2, max_group_size // 2)

                subgroup = SubgroupDefinition(
                    subgroup_id=f"subgroup_{len(subgroups):03d}",
                    attribute_values=attribute_values,
                    size=base_size,
                    bias=np.random.uniform(-0.5, 0.5),
                    uncertainty_params={
                        'epistemic': np.random.uniform(0.05, 0.3),
                        'aleatoric': np.random.uniform(0.05, 0.3)
                    }
                )

                subgroups.append(subgroup)
                self.subgroups[subgroup.subgroup_id] = subgroup

        if len(subgroups) < n_subgroups:
            print(f"Warning: Only generated {len(subgroups)} subgroups out of {n_subgroups} requested")

        return subgroups

    def generate_all_possible_groups(self,
                                     subgroups: List[SubgroupDefinition],
                                     target_nb_groups: int = None) -> List[GroupDefinition]:
        """Generate all possible group combinations from subgroups."""
        all_combinations = list(itertools.combinations(subgroups, 2))

        if target_nb_groups and target_nb_groups < len(all_combinations):
            # Select the most diverse/interesting groups
            selected_combinations = self._select_diverse_groups(all_combinations, target_nb_groups)
        else:
            selected_combinations = all_combinations

        groups = []
        for i, (sg1, sg2) in enumerate(selected_combinations):
            # Calculate similarity between subgroups
            similarity = self._calculate_subgroup_similarity(sg1, sg2)

            # Estimate expected outcome difference
            outcome_diff = abs(sg1.bias - sg2.bias)

            # Calculate granularity and intersectionality
            granularity, intersectionality = self._calculate_granularity_intersectionality(sg1, sg2)

            # Combine all attribute keys from both subgroups
            all_keys = set(sg1.attribute_values.keys()) | set(sg2.attribute_values.keys())

            # Create subgroup dictionaries with values and id
            subgroup1_dict = {
                'values': {k: sg1.attribute_values.get(k, None) for k in all_keys},
                'id': sg1.subgroup_id
            }
            subgroup2_dict = {
                'values': {k: sg2.attribute_values.get(k, None) for k in all_keys},
                'id': sg2.subgroup_id
            }

            group = GroupDefinition(
                group_size=2,  # Minimum size to have both subgroups
                subgroup_bias=abs(sg1.bias - sg2.bias),
                similarity=similarity,
                alea_uncertainty=0.0,  # These can be set later if needed
                epis_uncertainty=0.0,
                frequency=0.0,
                diff_subgroup_size=0.0,
                subgroup1=subgroup1_dict,
                subgroup2=subgroup2_dict,
                avg_diff_outcome=outcome_diff,
                granularity=granularity,
                intersectionality=intersectionality
            )

            group_id = f"group_{i:03d}"
            groups.append(group)
            self.groups[group_id] = group

        return groups

    def _calculate_granularity_intersectionality(self, sg1: SubgroupDefinition,
                                                 sg2: SubgroupDefinition) -> Tuple[int, int]:
        """Calculate granularity and intersectionality for a group."""
        all_attrs = set(sg1.attribute_values.keys()) | set(sg2.attribute_values.keys())

        granularity = 0
        intersectionality = 0

        for attr in all_attrs:
            attr_idx = self.data_schema.attr_names.index(attr)
            is_protected = self.data_schema.protected_attr[attr_idx]

            if is_protected:
                intersectionality += 1
            else:
                granularity += 1

        return granularity, intersectionality

    def _calculate_subgroup_similarity(self, sg1: SubgroupDefinition, sg2: SubgroupDefinition) -> float:
        """Calculate similarity between two subgroups based on their attribute values."""
        all_attrs = set(sg1.attribute_values.keys()) | set(sg2.attribute_values.keys())

        if not all_attrs:
            return 1.0  # Both have no constraints

        matching_attrs = 0
        for attr in all_attrs:
            val1 = sg1.attribute_values.get(attr, None)
            val2 = sg2.attribute_values.get(attr, None)

            if val1 == val2:
                matching_attrs += 1
            elif val1 is None or val2 is None:
                # One has constraint, other doesn't - partial similarity
                matching_attrs += 0.5

        return matching_attrs / len(all_attrs)

    def _select_diverse_groups(self, all_combinations: List[Tuple], target_count: int) -> List[Tuple]:
        """Select the most diverse subset of group combinations."""
        if target_count >= len(all_combinations):
            return all_combinations

        # Strategy: Select groups with diverse similarity scores and outcome differences
        scored_combinations = []

        for sg1, sg2 in all_combinations:
            similarity = self._calculate_subgroup_similarity(sg1, sg2)
            outcome_diff = abs(sg1.bias - sg2.bias)

            # Score favors diverse similarities and meaningful outcome differences
            diversity_score = abs(similarity - 0.5)  # Prefer not too similar, not too different
            outcome_score = min(outcome_diff, 1.0)  # Cap at 1.0

            total_score = diversity_score + outcome_score
            scored_combinations.append((total_score, (sg1, sg2)))

        # Sort by score and take top combinations
        scored_combinations.sort(key=lambda x: x[0], reverse=True)
        return [combo for _, combo in scored_combinations[:target_count]]

    def generate_subgroup_data(self, subgroup: SubgroupDefinition, W: np.ndarray, classifier=None) -> pd.DataFrame:
        """Generate data for a single subgroup using SDV synthesizer and uncertainty calculation."""
        # Check if we already generated this subgroup
        if subgroup.subgroup_id in self.generated_subgroup_data:
            return self.generated_subgroup_data[subgroup.subgroup_id].copy()

        # Use SDV synthesizer to generate data with conditions
        with suppress_stdout():
            if subgroup.attribute_values:
                conditions = [Condition(
                    num_rows=subgroup.size,
                    column_values=subgroup.attribute_values
                )]
                try:
                    subgroup_data = self.data_schema.synthesizer.sample_from_conditions(conditions)
                except Exception as e:
                    print(f"Error generating data for subgroup {subgroup.subgroup_id}: {e}")
                    # Fallback: generate unconstrained data and manually set values
                    subgroup_data = self.data_schema.synthesizer.sample(subgroup.size)
                    for attr, value in subgroup.attribute_values.items():
                        if attr in subgroup_data.columns:
                            subgroup_data[attr] = value
            else:
                subgroup_data = self.data_schema.synthesizer.sample(subgroup.size)

        # Fill any NaN values
        subgroup_data = self._fill_nan_values(subgroup_data)

        # Generate outcomes with uncertainty
        subgroup_data = self._generate_outcomes_with_uncertainty(
            subgroup_data, subgroup, W, classifier
        )

        # Add subgroup metadata
        subgroup_data['subgroup_id'] = subgroup.subgroup_id
        subgroup_data['subgroup_bias'] = subgroup.bias
        subgroup_data['epistemic_param'] = subgroup.uncertainty_params['epistemic']
        subgroup_data['aleatoric_param'] = subgroup.uncertainty_params['aleatoric']

        # Store for reuse
        self.generated_subgroup_data[subgroup.subgroup_id] = subgroup_data.copy()

        return subgroup_data

    def _fill_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values in a dataframe using smart defaults."""
        filled_df = df.copy()

        for col in filled_df.columns:
            if not filled_df[col].isna().any():
                continue

            if pd.api.types.is_numeric_dtype(filled_df[col]):
                median_val = filled_df[col].median()
                filled_df[col].fillna(0 if pd.isna(median_val) else median_val, inplace=True)
            else:
                if not filled_df[col].mode().empty:
                    mode_val = filled_df[col].mode().iloc[0]
                    filled_df[col].fillna(mode_val, inplace=True)
                else:
                    filled_df[col].fillna("unknown", inplace=True)

        return filled_df

    def _generate_outcomes_with_uncertainty(self, df: pd.DataFrame,
                                            subgroup: 'SubgroupDefinition',
                                            W: np.ndarray, classifier=None) -> pd.DataFrame:
        """Use the new analytical predictions method."""
        return self._generate_outcomes_with_analytical_predictions(df, subgroup, W, classifier)

    def generate_dataset(self, groups: List[GroupDefinition], W: np.ndarray,
                         min_group_size: int = 10, max_group_size: int = 100,
                         min_diff_subgroup_size: float = 0.0,
                         max_diff_subgroup_size: float = 0.5,
                         classifier=None) -> pd.DataFrame:
        """Generate the complete dataset with all groups."""
        all_dataframes = []

        print("Generating dataset from groups...")

        for group in tqdm(groups, desc="Processing groups"):
            sg1 = self.subgroups[group.subgroup1['id']]
            sg2 = self.subgroups[group.subgroup2['id']]

            # Calculate group size and subgroup size difference
            total_group_size = np.random.randint(min_group_size, max_group_size + 1)
            diff_percentage = np.random.uniform(min_diff_subgroup_size, max_diff_subgroup_size)

            # Calculate individual subgroup sizes
            diff_size = int(total_group_size * diff_percentage)
            sg1_size = max(1, (total_group_size + diff_size) // 2)
            sg2_size = max(1, total_group_size - sg1_size)

            # Update subgroup sizes for this group
            sg1_copy = copy.deepcopy(sg1)
            sg2_copy = copy.deepcopy(sg2)
            sg1_copy.size = sg1_size
            sg2_copy.size = sg2_size

            # Generate data for each subgroup
            sg1_data = self.generate_subgroup_data(sg1_copy, W, classifier)
            sg2_data = self.generate_subgroup_data(sg2_copy, W, classifier)

            # Take only the required number of rows
            sg1_data = sg1_data.head(sg1_size).copy()
            sg2_data = sg2_data.head(sg2_size).copy()

            # Create subgroup keys based on ordered attribute values (like original code)
            sg1_vals = []
            sg2_vals = []

            # Get ordered attribute values for all attributes (using -1 for unconstrained)
            for attr_name in self.data_schema.attr_names:
                sg1_vals.append(sg1.attribute_values.get(attr_name, -1))
                sg2_vals.append(sg2.attribute_values.get(attr_name, -1))

            # Create keys using the same format as original code
            subgroup1_key = '|'.join(list(map(lambda x: '*' if x == -1 else str(x), sg1_vals)))
            subgroup2_key = '|'.join(list(map(lambda x: '*' if x == -1 else str(x), sg2_vals)))
            group_key = subgroup1_key + '-' + subgroup2_key

            # Assign the keys to the dataframes
            sg1_data['subgroup_key'] = subgroup1_key
            sg2_data['subgroup_key'] = subgroup2_key

            sg1_data['group_key'] = group_key
            sg2_data['group_key'] = group_key

            # Add individual keys (same as original)
            sg1_data['indv_key'] = sg1_data[self.data_schema.attr_names].apply(
                lambda x: '|'.join(list(x.astype(int).astype(str))), axis=1)
            sg2_data['indv_key'] = sg2_data[self.data_schema.attr_names].apply(
                lambda x: '|'.join(list(x.astype(int).astype(str))), axis=1)

            # Combine subgroups into group
            group_data = pd.concat([sg1_data, sg2_data], ignore_index=True)

            # Add group-level metadata
            group_data['similarity_param'] = group.similarity
            group_data['expected_outcome_diff'] = group.avg_diff_outcome
            group_data['granularity_param'] = group.granularity
            group_data['intersectionality_param'] = group.intersectionality
            group_data['group_size'] = len(group_data)
            group_data['diff_subgroup_size'] = diff_percentage
            group_data['frequency_param'] = 1.0  # All groups have equal frequency in this approach

            all_dataframes.append(group_data)

        # Combine all groups
        complete_dataset = pd.concat(all_dataframes, ignore_index=True)

        # Sort by group and individual keys
        complete_dataset = complete_dataset.sort_values(['group_key', 'indv_key'])

        return complete_dataset


def generate_optimal_discrimination_data(
        data_schema: Optional['DataSchema'] = None,
        nb_groups: int = 100,
        nb_attributes: int = 20,
        min_number_of_classes: int = None,
        max_number_of_classes: int = None,
        prop_protected_attr: float = 0.2,
        min_group_size: int = 10,
        max_group_size: int = 100,
        min_diff_subgroup_size: float = 0.0,
        max_diff_subgroup_size: float = 0.5,
        W: Optional[np.ndarray] = None,
        categorical_outcome: bool = True,
        nb_categories_outcome: int = 6,
        classifier=None,
        use_cache: bool = True,
        **kwargs
) -> 'DiscriminationData':
    """
    Generate discrimination data using the improved optimal architecture.

    Args:
        data_schema: Optional DataSchema with fitted synthesizer. If None, will generate one.
        nb_groups: Target number of groups to generate
        nb_attributes: Number of attributes (used only if data_schema is None)
        min_number_of_classes: Min classes per attribute (used only if data_schema is None)
        max_number_of_classes: Max classes per attribute (used only if data_schema is None)
        prop_protected_attr: Proportion of protected attributes (used only if data_schema is None)
        min_group_size: Minimum size for each group
        max_group_size: Maximum size for each group
        min_diff_subgroup_size: Minimum difference between subgroup sizes
        max_diff_subgroup_size: Maximum difference between subgroup sizes
        W: Weight matrix for outcome generation
        categorical_outcome: Whether to use categorical outcomes
        nb_categories_outcome: Number of categories for outcome if categorical
        **kwargs: Additional arguments for compatibility

    Returns:
        DiscriminationData object with generated data
    """

    # Generate DataSchema if not provided
    if data_schema is None:
        print("No DataSchema provided, generating a new one...")

        # Estimate min/max classes if not provided
        if min_number_of_classes is None or max_number_of_classes is None:
            est_min, est_max = estimate_min_attributes_and_classes(nb_groups)
            min_number_of_classes = min_number_of_classes or est_min
            max_number_of_classes = max_number_of_classes or int(est_max * 1.5)

        # Generate the schema with fitted synthesizer
        data_schema = generate_data_schema(
            min_number_of_classes=min_number_of_classes,
            max_number_of_classes=max_number_of_classes,
            nb_attributes=nb_attributes,
            prop_protected_attr=prop_protected_attr,
            fit_synthesizer=True,
            n_samples=10000
        )
        print(f"Generated DataSchema with {len(data_schema.attr_names)} attributes")
        print(f"Protected attributes: {sum(data_schema.protected_attr)}")
        print(f"Non-protected attributes: {len(data_schema.attr_names) - sum(data_schema.protected_attr)}")

    # Validate that the schema has a fitted synthesizer
    if not hasattr(data_schema, 'synthesizer') or data_schema.synthesizer is None:
        raise ValueError(
            "DataSchema must have a fitted synthesizer. Set fit_synthesizer=True when creating the schema.")

    # Generate weights if not provided
    if W is None:
        hiddenlayers_depth = 3
        nb_attributes_actual = len(data_schema.attr_names)
        W = np.random.uniform(low=0.0, high=1.0, size=(hiddenlayers_depth, nb_attributes_actual))

    # Initialize the optimal generator
    generator = OptimalDataGenerator(data_schema)

    # Step 1: Calculate optimal subgroups
    print("Step 1: Calculating optimal subgroups...")
    subgroups = generator.calculate_optimal_subgroups(
        nb_groups, min_group_size=min_group_size, max_group_size=max_group_size
    )

    # Step 2: Generate all possible groups
    print("Step 2: Generating group combinations...")
    groups = generator.generate_all_possible_groups(subgroups, nb_groups)

    # Step 3: Generate the complete dataset
    print("Step 3: Generating complete dataset...")
    dataset = generator.generate_dataset(
        groups, W, min_group_size, max_group_size,
        min_diff_subgroup_size, max_diff_subgroup_size,
        classifier=classifier
    )

    # ### NEW: Process both y_true and y_pred if needed
    if categorical_outcome:
        # Create bins based on the distribution of the "true" outcome
        min_val = dataset['y_true'].min()
        max_val = dataset['y_true'].max()
        bins = np.linspace(min_val, max_val, nb_categories_outcome + 1)

        # Bin both columns using the same bins for consistency
        dataset['y_true_cat'] = np.digitize(dataset['y_true'], bins) - 1
        dataset['y_pred_cat'] = np.digitize(dataset['y_pred'], bins) - 1

        # The main 'outcome' column will be the categorical version of the true label
        dataset['outcome'] = dataset['y_true_cat']
    else:
        # For continuous outcomes, just use the probabilities directly
        dataset['outcome'] = dataset['y_true']

    # Step 4: Process outcome column
    outcome_column = 'outcome'
    if categorical_outcome:
        dataset[outcome_column] = bin_array_values(dataset[outcome_column], nb_categories_outcome)
    else:
        dataset[outcome_column] = ((dataset[outcome_column] - dataset[outcome_column].min()) /
                                   (dataset[outcome_column].max() - dataset[outcome_column].min()))

    # Step 5: Create DiscriminationData object
    protected_attr = {k: v for k, v in zip(data_schema.attr_names, data_schema.protected_attr)}
    attr_possible_values = {attr_name: values for attr_name, values in
                            zip(data_schema.attr_names, data_schema.attr_categories)}

    # Create the DiscriminationData object
    data = DiscriminationData(
        dataframe=dataset,
        categorical_columns=list(data_schema.attr_names) + [outcome_column],
        attributes=protected_attr,
        collisions=0,  # No collisions with this approach
        nb_groups=len(groups),
        max_group_size=max_group_size,
        hiddenlayers_depth=W.shape[0],
        outcome_column=outcome_column,
        attr_possible_values=attr_possible_values,
        schema=data_schema,
        y_true_col='y_true_cat' if categorical_outcome else 'y_true',
        y_pred_col='y_pred_cat' if categorical_outcome else 'y_pred',
        generation_arguments={
            'nb_groups': nb_groups,
            'nb_subgroups': len(subgroups),
            'nb_attributes': len(data_schema.attr_names),
            'min_number_of_classes': min_number_of_classes,
            'max_number_of_classes': max_number_of_classes,
            'prop_protected_attr': prop_protected_attr,
            'min_group_size': min_group_size,
            'max_group_size': max_group_size,
            'categorical_outcome': categorical_outcome,
            'nb_categories_outcome': nb_categories_outcome,
            'approach': 'optimal_subgroup_generation_with_analytical_predictions'  # Updated
        }
    )

    # Step 6: Calculate the missing actual metrics (this is the key addition!)
    print("Step 4: Calculating actual metrics...")
    data = calculate_actual_metrics(data)

    print(f"\nGeneration Summary:")
    print(f"- Schema: {'Auto-generated' if 'schema_generated' not in locals() else 'User-provided'}")
    print(f"- Attributes: {len(data_schema.attr_names)} ({sum(data_schema.protected_attr)} protected)")
    print(f"- Generated {len(subgroups)} unique subgroups")
    print(f"- Created {len(groups)} groups from subgroup combinations")
    print(f"- Final dataset contains {len(dataset)} individuals")
    print(f"- Efficiency: {len(groups) / len(subgroups):.1f} groups per subgroup")

    return data


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
        data_schema: 'DataSchema' = None,
        predefined_groups: List['GroupDefinition'] = None,
        classifier=None
) -> 'DiscriminationData':
    """
    Drop-in replacement for your original generate_data function using optimal approach.

    This function provides the same interface as your original generate_data but uses
    the new optimal subgroup generation approach internally.

    Args:
        Similar to your original generate_data function arguments

    Returns:
        DiscriminationData object generated using optimal approach
    """
    # Note: Some parameters like min_similarity, max_similarity etc. are not directly
    # used in the optimal approach but are kept for compatibility

    return generate_optimal_discrimination_data(
        data_schema=data_schema,
        nb_groups=nb_groups,
        nb_attributes=nb_attributes,
        min_number_of_classes=min_number_of_classes,
        max_number_of_classes=max_number_of_classes,
        prop_protected_attr=prop_protected_attr,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        min_diff_subgroup_size=min_diff_subgroup_size,
        max_diff_subgroup_size=max_diff_subgroup_size,
        W=W,
        categorical_outcome=categorical_outcome,
        nb_categories_outcome=nb_categories_outcome,
        classifier=classifier
    )


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
        synthesizer = RobustGaussianCopulaSynthesizer(sdv_metadata, **default_params)
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


def generate_from_real_data(dataset_name, use_cache=False, predefined_groups=None, *args, **kwargs):
    """
    Generate synthetic discrimination data based on real-world datasets using SDV.

    Args:
        dataset_name: Name of the dataset ('adult', 'credit', 'bank')
        use_cache: Whether to use cached data if available
        *args, **kwargs: Additional arguments for generate_data

    Returns:
        Tuple of (DiscriminationData, DataSchema)
    """
    # Try to load from cache first
    if use_cache:
        cache = DataCache()
        cache_params = {
            'dataset_name': dataset_name,
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

        df = df.drop(columns='age')

        # Get schema from the dataframe
        schema, correlation_matrix, column_mapping, enc_df = generate_schema_from_dataframe(
            df,
            protected_columns=['marital', 'education'],
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
    synthesizer = RobustGaussianCopulaSynthesizer(
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

    # Train a classifier on the real data to be used for generating predictions
    X_real = enc_df[schema.attr_names]
    y_real = enc_df['outcome']
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_real, y_real)

    # Generate data using our function that uses the trained synthesizer
    data = generate_data(
        gen_order=schema.gen_order,
        correlation_matrix=correlation_matrix,
        data_schema=schema,
        use_cache=False,  # We're already handling cache at this level
        predefined_groups=predefined_groups,
        classifier=dt_classifier,  # Pass the trained classifier
        *args, **kwargs
    )

    # Store schema with the data for convenience
    data.schema = schema
    data.generation_arguments = {'real_dataset': dataset_name}

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
            'protected_columns': ['race', 'sex',
                                  # 'age'
                                  ] if protected_columns is None else protected_columns,
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

    if dataset_name == 'bank':
        df = df.drop(columns='age')
        config['protected_columns'] = ['marital', 'education']

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

    enc_df['indv_key'] = enc_df[schema.attr_names].apply(
        lambda x: '|'.join(list(x.astype(str))), axis=1
    )

    # Train a classifier and add predictions
    X = enc_df[schema.attr_names]
    y = enc_df['outcome']
    dt = DecisionTreeClassifier()
    dt.fit(X, y)
    enc_df['y_pred'] = dt.predict(X)

    # Create DiscriminationData without generating synthetic data
    data = DiscriminationData(
        dataframe=enc_df,
        categorical_columns=list(schema.attr_names) + ['outcome', 'y_pred'],
        attributes={k: v for k, v in zip(schema.attr_names, schema.protected_attr)},
        collisions=0,
        nb_groups=1,  # Just one "group" for the entire dataset
        max_group_size=len(enc_df),
        hiddenlayers_depth=0,
        y_true_col='outcome',
        y_pred_col='y_pred',
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


def save_discrimination_data(db_path: str, data_obj: 'DiscriminationData', analysis_id: str = None) -> str:
    """
    Save a DiscriminationData object to an SQLite database.
    
    Args:
        db_path: Path to the SQLite database file
        data_obj: DiscriminationData object to save
        analysis_id: Optional analysis ID. If not provided, a new UUID will be generated
        
    Returns:
        str: The analysis_id used to save the data
    """
    import sqlite3
    import pickle
    import uuid
    from datetime import datetime

    if analysis_id is None:
        analysis_id = f"data_{str(uuid.uuid4())}"

    # Create or connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table for discrimination data if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS discrimination_data (
            analysis_id TEXT PRIMARY KEY,
            data BLOB,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            nb_groups INTEGER,
            nb_attributes INTEGER,
            categorical_columns TEXT,
            protected_attributes TEXT
        )
    ''')

    # Serialize the data object
    data_binary = pickle.dumps(data_obj)

    # Extract metadata for easier querying
    categorical_columns = ','.join(data_obj.categorical_columns)
    protected_attributes = ','.join(data_obj.protected_attributes())

    # Save the data with metadata
    cursor.execute('''
        INSERT INTO discrimination_data 
        (analysis_id, data, nb_groups, nb_attributes, categorical_columns, protected_attributes)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        analysis_id,
        data_binary,
        data_obj.nb_groups,
        len(data_obj.attributes),
        categorical_columns,
        protected_attributes
    ))

    conn.commit()
    conn.close()

    return analysis_id


def load_discrimination_data(db_path: str, analysis_id: str) -> Optional['DiscriminationData']:
    """
    Load a DiscriminationData object from an SQLite database.
    
    Args:
        db_path: Path to the SQLite database file
        analysis_id: The analysis ID to load
        
    Returns:
        Optional[DiscriminationData]: The loaded data object, or None if not found
    """
    import sqlite3
    import pickle

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT data FROM discrimination_data WHERE analysis_id = ?', (analysis_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return pickle.loads(result[0])
    return None


if __name__ == '__main__':
    # Generate test data
    data = generate_optimal_discrimination_data(nb_groups=10, nb_attributes=5)

    # Verify new columns exist
    assert 'y_true' in data.dataframe.columns
    assert 'y_pred' in data.dataframe.columns
    assert hasattr(data, 'y_true_col')
    assert hasattr(data, 'y_pred_col')

    print("Integration successful!")
