import warnings
from functools import lru_cache
import traceback
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
import statsmodels.api as sm
import pandas as pd
from typing import List, Dict, Any, Optional, Set, Tuple
import uuid
import logging
from enum import Enum
import re
from pathlib import Path

from data_generator.main import generate_from_real_data, generate_data, get_real_data
from path import HERE
import sqlite3
import json

warnings.filterwarnings("ignore")
# Ignore specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# %%

def get_column_values(df):
    return [set(df[col].unique()) for col in df.columns]


def is_couple_part_of_a_group(couple_key, group_key_list, res_pattern):
    res = []

    couple_key_elems = couple_key.split('-')
    if len(couple_key_elems) != 2:
        print(f"Warning: Unexpected couple key format: {couple_key}")
        return res

    opt1 = f"{couple_key_elems[0]}-{couple_key_elems[1]}"
    opt2 = f"{couple_key_elems[1]}-{couple_key_elems[0]}"

    grp_res_pattern = f"{res_pattern}-{res_pattern}"

    for grp_key in group_key_list:
        if matches_pattern(grp_key, opt1, grp_res_pattern) or matches_pattern(grp_key, opt2, grp_res_pattern):
            res.append(grp_key)
    return res


@lru_cache(maxsize=4096)
def matches_pattern(pattern_string: str, test_string: str, data_schema: str) -> bool:
    """Cached pattern matching."""

    def _compile_pattern(pattern_string: str, data_schema: str):
        res_pattern = []
        for k, v in zip(pattern_string.split('|'), data_schema.split('|')):
            if k == '*':
                res_pattern.append(f"[{v}]")
            else:
                res_pattern.append(k)
        res_pattern = "\|".join(res_pattern)
        return res_pattern

    if '-' in pattern_string:
        if '-' not in data_schema:
            data_schema = f"{data_schema}-{data_schema}"
        result_pat = []
        for pat1, pat2 in zip(pattern_string.split('-'), data_schema.split('-')):
            result_pat.append(_compile_pattern(pat1, pat2))

        pattern = re.compile('\-'.join(result_pat))
        res = bool(pattern.match(test_string))
        return res

    else:
        pattern = re.compile(_compile_pattern(pattern_string, data_schema))
        return bool(pattern.match(test_string))


class Method(Enum):
    AEQUITAS = "aequitas"
    BIASSCAN = "biasscan"
    EXPGA = "expga"
    MLCHECK = "mlcheck"


@dataclass
class ExperimentConfig:
    # Required parameters
    nb_attributes: int
    prop_protected_attr: float
    nb_groups: int
    max_group_size: int
    min_group_size: int = 10
    nb_categories_outcome: int = 4
    methods: Set[Method] = None
    min_number_of_classes: int = 2
    max_number_of_classes: int = 4
    min_granularity: int = 1
    max_granularity: int = None
    min_intersectionality: int = 1
    max_intersectionality: int = None

    # Real dataset parameters
    use_real_data: bool = False
    real_dataset_name: Optional[str] = None
    dataset_name: Optional[str] = None
    correlation_matrix: Optional[np.ndarray] = None
    data_schema: Optional[str] = None
    data_generation_method: Optional[str] = None  # Can be 'get_real_data' or 'generate_from_real_data'

    # Method specific parameters
    # Aequitas
    aequitas_model_type: str = "RandomForest"
    aequitas_threshold: float = 0.0
    aequitas_perturbation_unit: float = 0.5
    aequitas_global_iteration_limit: int = 200
    aequitas_local_iteration_limit: int = 20

    # BiassScan
    bias_scan_test_size: float = 0.3
    bias_scan_random_state: int = 42
    bias_scan_n_estimators: int = 200
    bias_scan_num_iters: int = 100
    bias_scan_scoring: str = 'Poisson'
    bias_scan_favorable_value: str = 'high'
    bias_scan_mode: str = 'ordinal'

    # ExpGA
    expga_threshold_rank: float = 0.5
    expga_threshold: float = 0.2  # Lower threshold to match papers
    expga_max_global: int = 100  # Increase exploration
    expga_max_local: int = 100  # Increase local refinement

    # MLCheck
    mlcheck_iteration_no: int = 1

    def __post_init__(self):
        if self.methods is None:
            self.methods = {Method.AEQUITAS, Method.BIASSCAN, Method.EXPGA, Method.MLCHECK}
        elif isinstance(self.methods, list):
            self.methods = set(self.methods)

        # Validate real dataset configuration
        if self.use_real_data:
            if self.real_dataset_name is None:
                raise ValueError("real_dataset_name must be specified when use_real_data is True")
            if self.real_dataset_name not in ['adult', 'credit']:
                raise ValueError(
                    f"Unsupported dataset: {self.real_dataset_name}. Supported datasets are: 'adult', 'credit'")
            if self.data_generation_method not in ['get_real_data', 'generate_from_real_data', None]:
                raise ValueError(
                    f"Invalid data_generation_method: {self.data_generation_method}. Must be 'get_real_data' or 'generate_from_real_data'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, handling complex types"""
        config_dict = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, np.ndarray):
                config_dict[field] = value.tolist() if value is not None else None
            elif isinstance(value, set) and field == 'methods':
                config_dict[field] = [m.value for m in value] if value is not None else None
            else:
                config_dict[field] = value
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary, handling complex types"""
        # Convert lists back to numpy arrays for correlation matrix
        if config_dict.get('correlation_matrix') is not None:
            config_dict['correlation_matrix'] = np.array(config_dict['correlation_matrix'])

        # Convert method list back to set
        if config_dict.get('methods') is not None:
            config_dict['methods'] = {Method(m) for m in config_dict['methods']}

        return cls(**config_dict)


def evaluate_discrimination_detection(
        ge,
        results_df: pd.DataFrame,
        experiment_id: str,
        method: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate discrimination detection results with optimized performance.
    """
    synthetic_data = ge.dataframe

    # Update calculated properties to match new column names
    calculated_properties = [
        'calculated_epistemic_random_forest', 'calculated_aleatoric_random_forest',
        'calculated_aleatoric_entropy', 'calculated_aleatoric_probability_margin',
        'calculated_aleatoric_label_smoothing', 'calculated_epistemic_ensemble',
        'calculated_epistemic_mc_dropout', 'calculated_epistemic_evidential',
        'calculated_epistemic_group', 'calculated_aleatoric_group', 'calculated_magnitude',
        'calculated_group_size', 'calculated_granularity', 'calculated_intersectionality',
        'calculated_uncertainty_group', 'calculated_similarity', 'calculated_subgroup_ratio'
    ]

    # Convert types only for required columns
    synthetic_data = synthetic_data.copy()
    synthetic_data[['indv_key', 'group_key', 'subgroup_key']] = (
        synthetic_data[['indv_key', 'group_key', 'subgroup_key']].astype(str)
    )

    # Update attribute column detection
    attr_columns = [col for col in synthetic_data.columns if
                    col.startswith(('Attr1_', 'Attr2_', 'Attr3_', 'Attr4_', 'Attr5_', 'Attr6_', 'Attr7_'))]

    if ge.attr_possible_values is None:
        data_schema = get_column_values(synthetic_data[attr_columns].drop_duplicates())
        data_schema = '|'.join([''.join(list(map(str, e))) for e in data_schema])
    else:
        data_schema = ge.schema

    # Pre-compute group data
    unique_groups = synthetic_data['group_key'].unique()
    group_data_dict = {
        group: synthetic_data[synthetic_data['group_key'] == group]
        for group in unique_groups
    }

    if results_df.empty:
        # Optimize empty results case
        group_analysis_data = []

        for group_key, group_data in group_data_dict.items():
            # Calculate stats efficiently using vectorized operations
            group_individuals = group_data['indv_key'].unique()
            calc_stats = group_data[calculated_properties].agg(['mean', 'min', 'max', 'median'])

            group_stats = {
                'group_key': group_key,
                'synthetic_group_size': len(group_data),
                'nb_unique_indv': len(group_individuals),
                'individuals_part_of_original_data': [],
                'couples_part_of_original_data': [],
                'new_individuals_part_of_a_group_regex': [],
                'new_couples_part_of_a_group_regex': [],
                'num_exact_individual_matches': 0,
                'num_exact_couple_matches': 0,
                'num_new_group_individuals': 0,
                'num_new_group_couples': 0,
                **{prop: calc_stats.loc['mean', prop] for prop in calculated_properties},
                **{
                    f"{prop}_{stat}": calc_stats.loc[stat, prop]
                    for prop in ['calculated_epistemic_random_forest', 'calculated_aleatoric_random_forest']
                    for stat in ['min', 'max', 'median']
                }
            }
            group_analysis_data.append(group_stats)

        # Create empty DataFrames efficiently
        empty_metrics = pd.DataFrame([{
            'experiment_id': experiment_id,
            'method': method,
            'group_key': group_key,
            'synthetic_group_size': len(group_data),
            'num_exact_individual_matches': 0,
            'num_exact_couple_matches': 0,
            'num_new_group_individuals': 0,
            'num_new_group_couples': 0
        } for group_key, group_data in group_data_dict.items()])

        return (
            pd.DataFrame(group_analysis_data),
            pd.DataFrame(columns=[
                'indv_key', 'couple_key', 'is_original_data',
                'is_couple_part_of_a_group',
                'matching_groups'
            ]),
            empty_metrics
        )

    # Process non-empty results case
    results_df = results_df.copy()
    results_df[['indv_key', 'couple_key']] = results_df[['indv_key', 'couple_key']].astype(str)

    # Pre-compute unique values
    unique_indv_keys = set(results_df['indv_key'].unique())
    unique_couple_keys = set(results_df['couple_key'].unique())

    # Pre-compute group patterns
    group_patterns = {
        group_key: group_key.split('-')
        for group_key in unique_groups
    }

    # Pre-compute individual sets for each group
    group_individuals = {
        group_key: set(group_data['indv_key'].unique())
        for group_key, group_data in group_data_dict.items()
    }

    group_analysis_data = []

    for group_key, group_data in group_data_dict.items():
        pattern1, pattern2 = group_patterns[group_key]
        current_group_individuals = group_individuals[group_key]

        # Efficient set operations for matches
        exact_indv_matches = list(unique_indv_keys.intersection(current_group_individuals))

        # Optimize couple matching using sets
        exact_couple_matches = [
            couple_key for couple_key in unique_couple_keys
            if all(indv in current_group_individuals for indv in couple_key.split('-'))
        ]

        # Cache results of pattern matching
        new_group_indv = [
            key for key in unique_indv_keys
            if key not in exact_indv_matches and
               (matches_pattern(pattern1, key, data_schema) or matches_pattern(pattern2, key, data_schema))
        ]

        new_group_couples = [
            key for key in unique_couple_keys
            if key not in exact_couple_matches and
               is_couple_part_of_a_group(key, [group_key], data_schema)
        ]

        # Calculate stats efficiently using vectorized operations
        calc_stats = group_data[calculated_properties].agg(['mean', 'min', 'max', 'median'])

        group_stats = {
            'group_key': group_key,
            'synthetic_group_size': len(group_data),
            'nb_unique_indv': len(current_group_individuals),
            'individuals_part_of_original_data': exact_indv_matches,
            'couples_part_of_original_data': exact_couple_matches,
            'new_individuals_part_of_a_group_regex': new_group_indv,
            'new_couples_part_of_a_group_regex': new_group_couples,
            'num_exact_individual_matches': len(exact_indv_matches),
            'num_exact_couple_matches': len(exact_couple_matches),
            'num_new_group_individuals': len(new_group_indv),
            'num_new_group_couples': len(new_group_couples),
            **{prop: calc_stats.loc['mean', prop] for prop in calculated_properties},
            **{
                f"{prop}_{stat}": calc_stats.loc[stat, prop]
                for prop in ['calculated_epistemic_random_forest', 'calculated_aleatoric_random_forest']
                for stat in ['min', 'max', 'median']
            }
        }
        group_analysis_data.append(group_stats)

    group_analysis_df = pd.DataFrame(group_analysis_data)

    if 'TSN' in results_df.columns:
        group_analysis_df['TSN'] = results_df.iloc[0]['TSN']
    else:
        group_analysis_df['TSN'] = 0

    if 'DSN' in results_df.columns:
        group_analysis_df['DSN'] = results_df.iloc[0]['DSN']
    else:
        group_analysis_df['DSN'] = 0

    if 'DSS' in results_df.columns:
        group_analysis_df['DSS'] = results_df.iloc[0]['DSS']
    else:
        group_analysis_df['DSS'] = 0

    if 'SUR' in results_df.columns:
        group_analysis_df['SUR'] = results_df.iloc[0]['SUR']
    else:
        group_analysis_df['SUR'] = 0

    # Optimize results processing using vectorized operations where possible
    def process_row(row):
        indv_key = row['indv_key']
        couple_key = row['couple_key']

        # Use pre-computed sets for faster lookups
        is_original = any(indv_key in group_set for group_set in group_individuals.values())

        # Use cached pattern matching
        matching_groups = [
            group_key for group_key, patterns in group_patterns.items()
            if
            matches_pattern(patterns[0], indv_key, data_schema) or matches_pattern(patterns[1], indv_key, data_schema)
        ]

        return pd.Series({
            'is_original_data': is_original,
            'is_couple_part_of_a_group': len(matching_groups) > 0,
            'matching_groups': matching_groups
        })

    # Process results in chunks for better memory usage
    chunk_size = 1000
    results_additions = []
    for i in range(0, len(results_df), chunk_size):
        chunk = results_df.iloc[i:i + chunk_size]
        results_additions.append(chunk.apply(process_row, axis=1))
    results_additions = pd.concat(results_additions)
    results_df = pd.concat([results_df, results_additions], axis=1)

    # Create metrics DataFrame efficiently
    metrics_columns = [
        'group_key', 'synthetic_group_size', 'num_exact_individual_matches',
        'num_exact_couple_matches', 'num_new_group_individuals', 'num_new_group_couples',
        'TSN', 'DSN', 'DSS', 'SUR'
    ]
    metrics_df = group_analysis_df[metrics_columns].copy()
    metrics_df.insert(0, 'experiment_id', experiment_id)
    metrics_df.insert(1, 'method', method)

    return group_analysis_df, results_df, metrics_df


def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Any Python object that might contain numpy data types

    Returns:
        obj with all numpy types converted to native Python types
    """
    import numpy as np

    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class ExperimentRunner:
    def __init__(self, db_path: str = "experiments.db", output_dir: str = "output_dir"):
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.setup_database()

    def setup_database(self):
        """Setup database tables with updated schema for new discrimination metrics"""
        # Create empty DataFrames with base schema
        experiments_df = pd.DataFrame(columns=[
            'experiment_id', 'config', 'methods', 'status',
            'start_time', 'end_time', 'error'
        ])

        results_df = pd.DataFrame(columns=[
            'result_id', 'experiment_id', 'method_name',
            'result_data', 'metrics', 'execution_time'
        ])

        analysis_metadata_df = pd.DataFrame(columns=[
            'analysis_id', 'experiment_id', 'method_name', 'created_at'
        ])

        augmented_results_df = pd.DataFrame(columns=[
            'analysis_id', 'indv_key', 'couple_key', 'is_original_data',
            'is_couple_part_of_a_group',
            'matching_groups', 'data'
        ])

        # Updated schema for evaluated_results with new discrimination metrics
        evaluated_results_df = pd.DataFrame(columns=[
            'analysis_id',
            'group_key',
            'synthetic_group_size',
            'nb_unique_indv',
            'individuals_part_of_original_data',
            'couples_part_of_original_data',
            'new_individuals_part_of_a_group_regex',
            'new_couples_part_of_a_group_regex',
            'num_exact_individual_matches',
            'num_exact_couple_matches',
            'num_new_group_individuals',
            'num_new_group_couples',
            # New discrimination metrics
            'TSN',  # Total Sample Number
            'DSN',  # Discrimination Sample Number
            'DSS',  # Discrimination Sample Score
            'dsr',  # Discrimination Sample Ratio
            # Existing uncertainty metrics
            'calculated_epistemic_random_forest',
            'calculated_aleatoric_random_forest',
            'calculated_aleatoric_entropy',
            'calculated_aleatoric_probability_margin',
            'calculated_aleatoric_label_smoothing',
            'calculated_epistemic_ensemble',
            'calculated_epistemic_mc_dropout',
            'calculated_epistemic_evidential',
            'calculated_epistemic_group',
            'calculated_aleatoric_group',
            'calculated_magnitude',
            'calculated_uncertainty_group',
            'calculated_intersectionality',
            'calculated_granularity',
            'calculated_group_size',
            'calculated_similarity',
            'calculated_subgroup_ratio',
            # Min/max/median for main uncertainties
            'calculated_epistemic_random_forest_min',
            'calculated_epistemic_random_forest_max',
            'calculated_epistemic_random_forest_median',
            'calculated_aleatoric_random_forest_min',
            'calculated_aleatoric_random_forest_max',
            'calculated_aleatoric_random_forest_median'
        ])

        synthetic_data_df = pd.DataFrame(columns=[
            'synthetic_data_id', 'experiment_id',
            'training_data', 'full_data',
            'protected_attributes', 'outcome_column',
            'created_at', 'attr_possible_values'
        ])

        # Create tables using to_sql
        with sqlite3.connect(self.db_path) as conn:
            experiments_df.to_sql('experiments', conn, if_exists='append', index=False)
            results_df.to_sql('results', conn, if_exists='append', index=False)
            analysis_metadata_df.to_sql

    def experiment_exists(self, config: ExperimentConfig) -> Optional[str]:
        """Check if an experiment with the same configuration exists"""
        config_dict = config.to_dict()  # Use the new to_dict method
        config_json = json.dumps(config_dict, sort_keys=True)

        query = "SELECT experiment_id FROM experiments WHERE config = ? AND status = 'completed'"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(config_json,))

        return df.iloc[0]['experiment_id'] if not df.empty else None

    def get_completed_methods(self, experiment_id: str) -> Set[Method]:
        """Get completed methods using pandas DataFrame"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT DISTINCT method_name FROM results WHERE experiment_id = ?",
                conn,
                params=(experiment_id,)
            )

        return {Method(row['method_name']) for _, row in df.iterrows()}

    def resume_experiments(self, use_cache: bool = True):
        """Resume experiments using pandas operations"""
        with sqlite3.connect(self.db_path) as conn:
            incomplete_experiments = pd.read_sql_query(
                "SELECT experiment_id, config, methods FROM experiments WHERE status = 'running'",
                conn
            )

        for _, row in incomplete_experiments.iterrows():
            config_dict = json.loads(row['config'])
            methods = json.loads(row['methods'])
            config_dict['methods'] = {Method(m) for m in methods}
            config = ExperimentConfig(**config_dict)

            completed_methods = self.get_completed_methods(row['experiment_id'])
            remaining_methods = config.methods - completed_methods

            if remaining_methods:
                logger.info(f"Resuming experiment {row['experiment_id']} for methods: {remaining_methods}")
                config.methods = remaining_methods
                try:
                    self.run_experiment(config, use_cache=use_cache)
                except Exception as e:
                    logger.error(f"Failed to resume experiment {row['experiment_id']}: {str(e)}")
                    continue

    def run_method(self, method: Method, ge, config: ExperimentConfig) -> tuple:
        """
        Run a discrimination detection method with timeout and error handling.

        Parameters:
        -----------
        method : Method
            The method to run (Aequitas, BiassScan, etc.)
        ge : object
            Generated data object
        config : ExperimentConfig
            Configuration for the experiment

        Returns:
        --------
        tuple: (results_df, metrics, execution_time)
        """
        import time
        import signal
        from contextlib import contextmanager
        from functools import partial
        from data_generator.main import generate_data
        from methods.aequitas.algo import run_aequitas
        from methods.biasscan.algo import run_bias_scan
        from methods.exp_ga.algo import run_expga
        from methods.ml_check.algo import run_mlcheck

        @contextmanager
        def timeout(seconds):
            def signal_handler(signum, frame):
                raise TimeoutError(f"Method execution timed out after {seconds} seconds")

            # Register the signal function handler
            signal.signal(signal.SIGALRM, signal_handler)

            # Set the alarm
            if hasattr(signal, 'SIGALRM'):  # Only on Unix-like systems
                signal.alarm(seconds)

            try:
                yield
            finally:
                if hasattr(signal, 'SIGALRM'):  # Only on Unix-like systems
                    signal.alarm(0)  # Disable the alarm

        start_time = time.time()
        results_df = pd.DataFrame()
        metrics = {}

        try:
            # Set method-specific timeouts
            if method == Method.AEQUITAS:
                timeout_seconds = 300  # 5 minutes for Aequitas
            else:
                timeout_seconds = 1800  # 30 minutes for other methods

            # Only use timeout on Unix-like systems where signal.SIGALRM is available
            if hasattr(signal, 'SIGALRM'):
                with timeout(timeout_seconds):
                    if method == Method.AEQUITAS:
                        results_df, metrics = run_aequitas(ge.training_dataframe, model_type=config.aequitas_model_type)
                    elif method == Method.BIASSCAN:
                        results_df, metrics = run_bias_scan(
                            ge,
                            test_size=config.bias_scan_test_size,
                            random_state=config.bias_scan_random_state,
                            n_estimators=config.bias_scan_n_estimators,
                            bias_scan_num_iters=config.bias_scan_num_iters,
                            bias_scan_scoring=config.bias_scan_scoring,
                            bias_scan_favorable_value=config.bias_scan_favorable_value,
                            bias_scan_mode=config.bias_scan_mode
                        )
                    elif method == Method.EXPGA:
                        results_df, metrics = run_expga(ge, threshold_rank=config.expga_threshold_rank,
                                                        max_global=config.expga_max_global,
                                                        max_local=config.expga_max_local,
                                                        threshold=config.expga_threshold)
                    elif method == Method.MLCHECK:
                        results_df, metrics = run_mlcheck(
                            ge,
                            iteration_no=config.mlcheck_iteration_no
                        )
            else:
                # On Windows or systems without SIGALRM, run without timeout
                if method == Method.AEQUITAS:
                    results_df, metrics = run_aequitas(ge.training_dataframe, model_type=config.aequitas_model_type)
                elif method == Method.BIASSCAN:
                    results_df, metrics = run_bias_scan(
                        ge,
                        test_size=config.bias_scan_test_size,
                        random_state=config.bias_scan_random_state,
                        n_estimators=config.bias_scan_n_estimators,
                        bias_scan_num_iters=config.bias_scan_num_iters,
                        bias_scan_scoring=config.bias_scan_scoring,
                        bias_scan_favorable_value=config.bias_scan_favorable_value,
                        bias_scan_mode=config.bias_scan_mode
                    )
                elif method == Method.EXPGA:
                    results_df, metrics = run_expga(ge, threshold_rank=config.expga_threshold_rank,
                                                    max_global=config.expga_max_global,
                                                    max_local=config.expga_max_local, threshold=config.expga_threshold)
                elif method == Method.MLCHECK:
                    results_df, metrics = run_mlcheck(
                        ge,
                        iteration_no=config.mlcheck_iteration_no
                    )

        except TimeoutError as e:
            logger.error(f"Method {method.value} timed out: {str(e)}")
            metrics = {'error': 'timeout'}
        except KeyboardInterrupt:
            logger.error(f"Method {method.value} was interrupted by user")
            metrics = {'error': 'interrupted'}
        except Exception as e:
            logger.error(f"Error running method {method.value}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            metrics = {'error': str(e)}

        execution_time = time.time() - start_time

        # Ensure results_df is a DataFrame even if empty
        if not isinstance(results_df, pd.DataFrame):
            results_df = pd.DataFrame()

        return results_df, metrics, execution_time

    def run_experiment(self, config: ExperimentConfig, use_cache: bool = True) -> str:
        """Run experiment with synthetic data storage"""
        if existing_id := self.experiment_exists(config):
            logger.info(f"Experiment already completed with ID: {existing_id}")
            return existing_id

        experiment_id = str(uuid.uuid4())
        config_dict = config.to_dict()  # Use the new to_dict method

        self.save_experiment(experiment_id, config_dict, 'running')

        try:
            # Generate data based on configuration
            if config.data_generation_method == 'generate_from_real_data':
                ge, schema = generate_from_real_data(
                    config.real_dataset_name,
                    nb_groups=config.nb_groups,
                    max_group_size=config.max_group_size,
                    min_number_of_classes=config.min_number_of_classes,
                    max_number_of_classes=config.max_number_of_classes,
                    use_cache=use_cache
                )
            elif config.data_generation_method == 'get_real_data':
                ge, schema = get_real_data(
                    dataset_name=config.real_dataset_name
                )
            else:
                ge = generate_data(
                    nb_attributes=config.nb_attributes,
                    min_number_of_classes=config.min_number_of_classes,
                    max_number_of_classes=config.max_number_of_classes,
                    min_granularity=config.min_granularity,
                    max_granularity=config.max_granularity,
                    min_intersectionality=config.min_intersectionality,
                    max_intersectionality=config.max_intersectionality,
                    prop_protected_attr=config.prop_protected_attr,
                    nb_groups=config.nb_groups,
                    max_group_size=config.max_group_size,
                    categorical_outcome=True,
                    nb_categories_outcome=config.nb_categories_outcome,
                    use_cache=use_cache
                )

            # Save synthetic data
            self.save_synthetic_data(experiment_id, ge)
            logger.info(f"Saved synthetic data for experiment: {experiment_id}")

            for method in config.methods:
                logger.info(f"Running method: {method.value}")
                try:
                    results_df, metrics, execution_time = self.run_method(method, ge, config)
                    self.save_result(experiment_id, method.value, results_df, metrics, execution_time)
                    logger.info(f"Completed method: {method.value}")
                except Exception as e:
                    logger.error(f"Failed to run method {method.value}: {str(e)}")
                    raise e

            self.update_experiment_status(experiment_id, 'completed')
            logger.info(f"Experiment completed successfully: {experiment_id}")

            logger.info(f"Analyzing results for experiment: {experiment_id}")
            self.analyze_experiment_results(experiment_id, ge, config)

            logger.info(f"Experiment and analysis completed successfully: {experiment_id}")

        except Exception as e:
            self.update_experiment_status(experiment_id, 'failed', str(e))
            logger.error(f"Experiment failed: {str(e)}")
            raise e

        return experiment_id

    def save_experiment(self, experiment_id: str, config: Dict, status: str):
        """Save experiment using pandas DataFrame"""
        experiment_data = pd.DataFrame([{
            'experiment_id': experiment_id,
            'config': json.dumps(config, sort_keys=True),
            'methods': json.dumps([m for m in config['methods']]),  # methods are already strings from to_dict()
            'status': status,
            'start_time': datetime.now(),
            'end_time': None,
            'error': None
        }])

        with sqlite3.connect(self.db_path) as conn:
            experiment_data.to_sql('experiments', conn, if_exists='append', index=False)

    def save_result(self, experiment_id: str, method_name: str,
                    result_df: pd.DataFrame, metrics: Dict[str, Any],
                    execution_time: float):
        """Save results using pandas DataFrame with numpy type conversion"""

        # Convert numpy types in metrics
        converted_metrics = convert_numpy_types(metrics)

        result_data = pd.DataFrame([{
            'result_id': str(uuid.uuid4()),
            'experiment_id': experiment_id,
            'method_name': method_name,
            'result_data': result_df.reset_index(drop=True).to_json(),
            'metrics': json.dumps(converted_metrics),
            'execution_time': execution_time
        }])

        with sqlite3.connect(self.db_path) as conn:
            result_data.to_sql('results', conn, if_exists='append', index=False)

    def update_experiment_status(self, experiment_id: str, status: str, error: str = None):
        """Update experiment status using pandas DataFrame"""
        with sqlite3.connect(self.db_path) as conn:
            # Read ALL data
            current_data = pd.read_sql_query("SELECT * FROM experiments", conn)

            if not current_data.empty:
                # Update the relevant fields
                mask = current_data['experiment_id'] == experiment_id
                current_data.loc[mask, ['status', 'end_time', 'error']] = [status, datetime.now(), error]

                # Write back to database
                current_data.to_sql('experiments', conn, if_exists='replace', index=False)

    def analyze_experiment_results(self, experiment_id: str, ge,
                                   config: Optional[ExperimentConfig] = None):
        """Analyze results using pandas operations"""

        if config is None:
            with sqlite3.connect(self.db_path) as conn:
                config_df = pd.read_sql_query(
                    "SELECT config FROM experiments WHERE experiment_id = ?",
                    conn,
                    params=(experiment_id,)
                )
                if not config_df.empty:
                    config_dict = json.loads(config_df.iloc[0]['config'])
                    config_dict['methods'] = {Method(m) for m in config_dict['methods']}
                    config = ExperimentConfig(**config_dict)
                else:
                    logger.warning(f"Could not find configuration for experiment {experiment_id}")

        with sqlite3.connect(self.db_path) as conn:
            results_df = pd.read_sql_query(
                "SELECT method_name, result_data, metrics FROM results WHERE experiment_id = ?",
                conn,
                params=(experiment_id,)
            )

        for _, row in results_df.iterrows():
            method_name = row['method_name']
            result_df = pd.read_json(row['result_data'])

            try:
                # Run evaluation
                group_analysis_df, augmented_results_df, metrics_df = evaluate_discrimination_detection(
                    ge, result_df, experiment_id, method_name
                )

                # Save analysis results using the new method
                self.save_analysis_results(
                    experiment_id,
                    method_name,
                    group_analysis_df,
                    augmented_results_df,
                    metrics_df
                )

                logger.info(f"Analysis completed for experiment {experiment_id}, method {method_name}")

            except Exception as e:
                logger.error(f"Error analyzing results for method {method_name}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue

    def save_analysis_results(
            self,
            experiment_id: str,
            method_name: str,
            group_analysis_df: pd.DataFrame,
            results_df: pd.DataFrame,
            metrics_df: pd.DataFrame
    ):
        """
        Save analysis results with list columns converted to JSON strings.
        """
        try:
            # Generate a unique analysis ID
            analysis_id = str(uuid.uuid4())
            current_time = datetime.now()

            # Save analysis metadata
            metadata = pd.DataFrame([{
                'analysis_id': analysis_id,
                'experiment_id': experiment_id,
                'method_name': method_name,
                'created_at': current_time
            }])

            # Convert list columns to JSON strings for evaluated results
            evaluated_results = group_analysis_df.copy()
            evaluated_results['analysis_id'] = analysis_id
            evaluated_results['individuals_part_of_original_data'] = evaluated_results[
                'individuals_part_of_original_data'].apply(json.dumps)
            evaluated_results['couples_part_of_original_data'] = evaluated_results[
                'couples_part_of_original_data'].apply(json.dumps)
            evaluated_results['new_individuals_part_of_a_group_regex'] = evaluated_results[
                'new_individuals_part_of_a_group_regex'].apply(json.dumps)
            evaluated_results['new_couples_part_of_a_group_regex'] = evaluated_results[
                'new_couples_part_of_a_group_regex'].apply(json.dumps)

            # Convert list columns to JSON strings for augmented results
            augmented_results = pd.DataFrame([{
                'analysis_id': analysis_id,
                'indv_key': row['indv_key'] if 'indv_key' in row else None,
                'couple_key': row['couple_key'] if 'couple_key' in row else None,
                'is_original_data': row['is_original_data'] if 'is_original_data' in row else None,
                'is_couple_part_of_a_group': row[
                    'is_couple_part_of_a_group'] if 'is_couple_part_of_a_group' in row else None,
                'matching_groups': json.dumps(row['matching_groups']) if 'matching_groups' in row else None,
                'data': json.dumps(row[
                                       (list(filter(lambda x: x.startswith('Attr') or x == 'outcome',
                                                    results_df.columns.tolist())))].to_dict()),
            } for _, row in results_df.iterrows()])

            # Save metrics
            metrics_data = metrics_df.copy()
            metrics_data['analysis_id'] = analysis_id

            # Save all DataFrames to the database
            with sqlite3.connect(self.db_path) as conn:
                metadata.to_sql('analysis_metadata', conn, if_exists='append', index=False)

                if not evaluated_results.empty:
                    evaluated_results.to_sql('evaluated_results', conn, if_exists='append', index=False)

                if not augmented_results.empty:
                    augmented_results.to_sql('augmented_results', conn, if_exists='append', index=False)

                if not metrics_data.empty:
                    metrics_data.to_sql('analysis_metrics', conn, if_exists='append', index=False)

            logger.info(f"Successfully saved analysis results for experiment {experiment_id}, method {method_name}")

        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def save_synthetic_data(self, experiment_id: str, ge):
        """Save synthetic dataset to database"""
        synthetic_data = pd.DataFrame([{
            'synthetic_data_id': str(uuid.uuid4()),
            'experiment_id': experiment_id,
            'training_data': ge.training_dataframe.to_json(),
            'full_data': ge.dataframe.to_json(),
            'protected_attributes': json.dumps(ge.protected_attributes),
            'outcome_column': ge.outcome_column,
            'created_at': datetime.now(),
            'attr_possible_values': ge.schema
        }])

        with sqlite3.connect(self.db_path) as conn:
            synthetic_data.to_sql('synthetic_data', conn, if_exists='append', index=False)


def create_method_configurations(methods: Set[Method] = None, include_real_data: bool = True,
                                 only_real_data: bool = False) -> List[ExperimentConfig]:
    """
    Creates comprehensive test configurations optimized for specific discrimination detection methods.

    Args:
        methods: Set of Method enums to create configurations for. If None, creates for all methods.
        include_real_data: Whether to include configurations for real datasets
        only_real_data: If True, only generate configurations for real datasets

    Returns:
        List of ExperimentConfig objects tailored for specified methods
    """
    # Parameter ranges optimized per method
    param_ranges = {
        'aequitas': {
            'nb_attributes': [5, 8, 10, 12, 15, 20],
            'prop_protected_attr': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'nb_groups': [10, 20, 35, 50, 75, 100, 150],
            'max_group_size': [100, 200, 350, 500, 750, 1000, 1500],
            'configs': {
                'ultralight': {
                    'aequitas_perturbation_unit': 0.25,
                    'aequitas_global_iteration_limit': 25,
                    'aequitas_local_iteration_limit': 3,
                    'aequitas_threshold': 0.2,
                    'aequitas_model_type': 'RandomForest'
                },
                'lightweight': {
                    'aequitas_perturbation_unit': 0.5,
                    'aequitas_global_iteration_limit': 50,
                    'aequitas_local_iteration_limit': 5,
                    'aequitas_threshold': 0.1,
                    'aequitas_model_type': 'RandomForest'
                },
                'default': {
                    'aequitas_perturbation_unit': 1.0,
                    'aequitas_global_iteration_limit': 100,
                    'aequitas_local_iteration_limit': 10,
                    'aequitas_threshold': 0.0,
                    'aequitas_model_type': 'RandomForest'
                }
            }
        },
        'biasscan': {
            'nb_attributes': [5, 10, 15, 20],
            'prop_protected_attr': [0.1, 0.2, 0.3, 0.4],
            'nb_groups': [25, 50, 75, 100],
            'max_group_size': [250, 500, 750, 1000],
            'configs': {
                'default': {
                    'bias_scan_test_size': 0.3,
                    'bias_scan_random_state': 42,
                    'bias_scan_n_estimators': 200,
                    'bias_scan_num_iters': 100,
                    'bias_scan_scoring': 'Poisson',
                    'bias_scan_favorable_value': 'high',
                    'bias_scan_mode': 'ordinal'
                }
            }
        },
        'expga': {
            'nb_attributes': [5, 8, 10, 12],
            'prop_protected_attr': [0.1, 0.2, 0.3, 0.4],
            'nb_groups': [10, 20, 35, 50],
            'max_group_size': [100, 200, 350, 500],
            'configs': {
                'default': {
                    'expga_threshold': 0.5,
                    'expga_threshold_rank': 0.5,
                    'expga_max_global': 50,
                    'expga_max_local': 50
                }
            }
        },
        'mlcheck': {
            'nb_attributes': [5, 8, 10, 12],
            'prop_protected_attr': [0.1, 0.2, 0.3, 0.4],
            'nb_groups': [10, 20, 35, 50],
            'max_group_size': [100, 200, 350, 500],
            'configs': {
                'default': {
                    'mlcheck_iteration_no': 1
                }
            }
        }
    }

    # Base configuration options
    base_config = {
        'min_group_size': 10,
        'nb_categories_outcome': 4,
        'min_number_of_classes': 2,
        'max_number_of_classes': 4,
        'min_granularity': 1,
        'min_intersectionality': 1,
        'use_real_data': False,
        'real_dataset_name': None,
        'dataset_name': None,
        'correlation_matrix': None,
        'data_schema': None,
        'data_generation_method': None,
        'max_granularity': None,
        'max_intersectionality': None
    }

    # Real dataset configurations
    real_dataset_configs = {
        'adult_config1': {
            'nb_attributes': 14,
            'prop_protected_attr': 0.14,
            'nb_groups': 5,
            'max_group_size': 500,
            'use_real_data': True,
            'real_dataset_name': 'adult',
            'dataset_name': 'adult',
            'min_number_of_classes': 2,
            'max_number_of_classes': 3,
            'nb_categories_outcome': 2
        },
        'adult_config2': {
            'nb_attributes': 14,
            'prop_protected_attr': 0.14,
            'nb_groups': 13,
            'max_group_size': 1000,
            'use_real_data': True,
            'real_dataset_name': 'adult',
            'dataset_name': 'adult',
            'min_number_of_classes': 2,
            'max_number_of_classes': 3,
            'nb_categories_outcome': 2
        },
        'credit_config1': {
            'nb_attributes': 20,
            'prop_protected_attr': 0.1,
            'nb_groups': 5,
            'max_group_size': 500,
            'use_real_data': True,
            'real_dataset_name': 'credit',
            'dataset_name': 'credit',
            'min_number_of_classes': 2,
            'max_number_of_classes': 3,
            'nb_categories_outcome': 2
        },
        'credit_config2': {
            'nb_attributes': 20,
            'prop_protected_attr': 0.1,
            'nb_groups': 10,
            'max_group_size': 1000,
            'use_real_data': True,
            'real_dataset_name': 'credit',
            'dataset_name': 'credit',
            'min_number_of_classes': 2,
            'max_number_of_classes': 3,
            'nb_categories_outcome': 2
        }
    }

    # Method mapping
    method_map = {
        'aequitas': Method.AEQUITAS,
        'biasscan': Method.BIASSCAN,
        'expga': Method.EXPGA,
        'mlcheck': Method.MLCHECK
    }

    if methods is None:
        methods = set(Method)

    configurations = []

    # Generate synthetic data configurations
    if not only_real_data:
        for method_name, method_enum in method_map.items():
            if method_enum not in methods:
                continue

            method_params = param_ranges[method_name]
            standard_combo = {
                'nb_attributes': method_params['nb_attributes'][2],
                'prop_protected_attr': method_params['prop_protected_attr'][2],
                'nb_groups': method_params['nb_groups'][2],
                'max_group_size': method_params['max_group_size'][2]
            }

            for intensity, method_specific_params in method_params['configs'].items():
                config_dict = {
                    **base_config,
                    **standard_combo,
                    **method_specific_params,
                    'methods': {method_enum}
                }
                configurations.append(ExperimentConfig(**config_dict))

    # Generate real data configurations
    if include_real_data or only_real_data:
        for config_name, dataset_config in real_dataset_configs.items():
            # Get real data parameters using get_real_data

            # Update configuration with actual data parameters
            get_real_config = {
                **base_config,
                **dataset_config,
                'data_generation_method': 'get_real_data'
            }

            for method_name, method_enum in method_map.items():
                if method_enum not in methods:
                    continue

                method_params = param_ranges[method_name]['configs']['default']
                config_dict = {
                    **get_real_config,
                    **method_params,
                    'methods': {method_enum}
                }
                configurations.append(ExperimentConfig(**config_dict))

            # Generate configuration using generate_from_real_data

            gen_real_config = {
                **base_config,
                **dataset_config,
                'data_generation_method': 'generate_from_real_data'
            }

            for method_name, method_enum in method_map.items():
                if method_enum not in methods:
                    continue

                method_params = param_ranges[method_name]['configs']['default']
                config_dict = {
                    **gen_real_config,
                    **method_params,
                    'methods': {method_enum}
                }
                configurations.append(ExperimentConfig(**config_dict))

    return configurations


def run_experiments(configs: List[ExperimentConfig], methods: Set[Method] = None,
                    db_path: str = "experiments.db", use_cache=True):
    """Run experiments with duplicate prevention and support for real datasets"""
    runner = ExperimentRunner(db_path)

    # Set methods for each config if specified
    if methods:
        for config in configs:
            config.methods = methods

    # Resume any incomplete experiments
    runner.resume_experiments(use_cache=use_cache)

    # Run new experiments, skipping duplicates
    for config in configs:
        existing_id = runner.experiment_exists(config)
        if existing_id:
            logger.info(f"Skipping duplicate experiment (ID: {existing_id})")
            continue

        try:
            experiment_id = runner.run_experiment(config, use_cache=use_cache)
            logger.info(f"Completed experiment: {experiment_id}")
        except Exception as e:
            logger.error(f"Failed to run experiment with config: {config}")
            logger.error(f"Error: {str(e)}")
            raise e


# %%
DB_PATH = HERE.joinpath("experiments/discrimination_detection_results18.db").as_posix()
FIGURES_PATH = HERE.joinpath("experiments/figures").as_posix()

# %%
for meth in [Method.AEQUITAS, Method.EXPGA]:
    methods = {meth}
    configs = create_method_configurations(methods=methods, only_real_data=True)
    run_experiments(configs, methods=methods, db_path=DB_PATH, use_cache=False)
