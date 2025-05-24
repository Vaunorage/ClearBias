import ast
import multiprocessing
import signal
import time
import warnings
from functools import lru_cache
from multiprocessing import freeze_support
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from io import StringIO

import matplotlib.pyplot as plt
import patsy
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
import doubleml as dml

from methods.aequitas.algo import run_aequitas
from methods.biasscan.algo import run_bias_scan
from methods.exp_ga.algo import run_expga
from methods.ml_check.algo import run_mlcheck
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
    nb_categories_outcome: int = 4
    methods: Set[Method] = None  # Methods to run
    min_number_of_classes: int = 2
    max_number_of_classes: int = 4

    # Method specific parameters
    # Aequitas
    aequitas_global_iteration_limit: int = 100
    aequitas_local_iteration_limit: int = 10
    aequitas_model_type: str = "RandomForest"
    aequitas_perturbation_unit: float = 1.0
    aequitas_threshold: float = 0.0

    # BiassScan
    bias_scan_test_size: float = 0.3
    bias_scan_random_state: int = 42
    bias_scan_n_estimators: int = 200
    bias_scan_num_iters: int = 100
    bias_scan_scoring: str = 'Poisson'
    bias_scan_favorable_value: str = 'high'
    bias_scan_mode: str = 'ordinal'

    # ExpGA
    expga_threshold: float = 0.5
    expga_threshold_rank: float = 0.5
    expga_max_global: int = 50
    expga_max_local: int = 50

    # MLCheck
    mlcheck_iteration_no: int = 1

    def __post_init__(self):
        if self.methods is None:
            self.methods = {Method.AEQUITAS, Method.BIASSCAN, Method.EXPGA, Method.MLCHECK}
        elif isinstance(self.methods, list):
            self.methods = set(self.methods)

    def __hash__(self):

        return hash((
            self.nb_attributes,
            self.prop_protected_attr,
            self.nb_groups,
            self.max_group_size,
            self.nb_categories_outcome,
            self.min_number_of_classes,
            self.max_number_of_classes,
            self.global_iteration_limit,
            self.local_iteration_limit,
            self.model_type,
            self.perturbation_unit,
            self.threshold,
            self.test_size,
            self.random_state,
            self.n_estimators,
            self.bias_scan_num_iters,
            self.bias_scan_scoring,
            self.bias_scan_favorable_value,
            self.bias_scan_mode,
            self.expga_threshold,
            self.expga_threshold_rank,
            self.max_global,
            self.max_local,
            self.mlcheck_iteration_no
        ))


@lru_cache(maxsize=4096)
def matches_pattern(pattern: str, value: str) -> bool:
    """Convert the pattern with * wildcards to regex and match against value."""
    import re
    # Escape special regex characters except *
    regex_pattern = re.escape(pattern).replace('\\*', '.*')
    return bool(re.match(f'^{regex_pattern}$', value))


def is_individual_part_of_the_original_indv(indv_key, indv_key_list):
    return indv_key in indv_key_list


def is_couple_part_of_a_group(couple_key, group_key_list):
    res = []

    couple_key_elems = couple_key.split('-')
    if len(couple_key_elems) != 2:
        print(f"Warning: Unexpected couple key format: {couple_key}")
        return res

    opt1 = f"{couple_key_elems[0]}-{couple_key_elems[1]}"
    opt2 = f"{couple_key_elems[1]}-{couple_key_elems[0]}"

    for grp_key in group_key_list:
        if matches_pattern(grp_key, opt1) or matches_pattern(grp_key, opt2):
            res.append(grp_key)
    return res


def evaluate_discrimination_detection(
        synthetic_data: pd.DataFrame,
        results_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Optimized evaluation of discrimination detection with fixed metrics calculation."""
    if results_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Convert to string type once
    synthetic_data = synthetic_data.astype({
        'indv_key': str,
        'group_key': str,
        'subgroup_key': str
    })

    results_df = results_df.astype({
        'indv_key': str,
        'couple_key': str
    })

    # Create a DataFrame for group-level analysis
    unique_groups = synthetic_data['group_key'].unique()

    def analyze_group(group_key: str) -> pd.Series:
        # Get individuals in this group from synthetic data
        group_synthetic_indv = set(
            synthetic_data[synthetic_data['group_key'] == group_key]['indv_key']
        )

        # Split group key into its two patterns
        group_patterns = group_key.split('-')
        pattern1, pattern2 = group_patterns[0], group_patterns[1]

        # Find exact individual matches in results
        exact_indv_matches = [
            key for key in results_df['indv_key'].unique()
            if is_individual_part_of_the_original_indv(key, group_synthetic_indv)
        ]

        # For couples, we need to check if both individuals in the couple are exact matches
        exact_couple_matches = []
        for couple_key in results_df['couple_key'].unique():
            indv1, indv2 = couple_key.split('-')
            if (is_individual_part_of_the_original_indv(indv1, group_synthetic_indv) and
                    is_individual_part_of_the_original_indv(indv2, group_synthetic_indv)):
                exact_couple_matches.append(couple_key)

        # Find new individuals matching either group pattern but not in original data
        new_group_indv = [
            key for key in results_df['indv_key'].unique()
            if (matches_pattern(pattern1, key) or matches_pattern(pattern2, key))
               and key not in exact_indv_matches
        ]

        # Find new couples matching group pattern but not in exact matches
        new_group_couples = [
            key for key in results_df['couple_key'].unique()
            if is_couple_part_of_a_group(key, [group_key])
               and key not in exact_couple_matches
        ]

        return pd.Series({
            'group_key': group_key,
            'synthetic_group_size': len(group_synthetic_indv),
            'individuals_part_of_original_data': exact_indv_matches,
            'couples_part_of_original_data': exact_couple_matches,
            'new_individuals_part_of_a_group_regex': new_group_indv,
            'new_couples_part_of_a_group_regex': new_group_couples,
            'num_exact_individual_matches': len(exact_indv_matches),
            'num_exact_couple_matches': len(exact_couple_matches),
            'num_new_group_individuals': len(new_group_indv),
            'num_new_group_couples': len(new_group_couples)
        })

    # Create group analysis DataFrame
    group_analysis_df = pd.DataFrame([
        analyze_group(group_key) for group_key in unique_groups
    ])

    # Process results additions
    def process_results_row(row):
        # Check if individual is an exact match in any group
        is_original = any(
            is_individual_part_of_the_original_indv(row['indv_key'],
                                                    synthetic_data[synthetic_data['group_key'] == group_key][
                                                        'indv_key'])
            for group_key in unique_groups
        )

        # Check which groups the individual matches patterns for
        individual_groups = []
        for group_key in unique_groups:
            pattern1, pattern2 = group_key.split('-')
            if matches_pattern(pattern1, row['indv_key']) or matches_pattern(pattern2, row['indv_key']):
                individual_groups.append(group_key)

        # Check which groups the couple matches patterns for
        couple_groups = [
            group_key for group_key in unique_groups
            if is_couple_part_of_a_group(row['couple_key'], [group_key])
        ]

        return pd.Series({
            'is_original_data': is_original,
            'is_individual_part_of_a_group': len(individual_groups) > 0,
            'is_couple_part_of_a_group': len(couple_groups) > 0,
            'matching_groups': individual_groups
        })

    # Apply the processing to results data
    results_additions = results_df.apply(process_results_row, axis=1)
    results_df = pd.concat([results_df, results_additions], axis=1)

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'total_synthetic_records': len(synthetic_data),
        'total_result_records': len(results_df),
        'total_groups': len(unique_groups),
        'avg_group_size': group_analysis_df['synthetic_group_size'].mean(),
        'avg_exact_matches_per_group': group_analysis_df['num_exact_individual_matches'].mean(),
        'avg_new_matches_per_group': group_analysis_df['num_new_group_individuals'].mean()
    }, index=[0])

    return group_analysis_df, results_df, summary_df


def timeout_handler(signum, frame):
    raise TimeoutError("Method execution timed out")


def run_method_with_timeout(method, ge, config):
    """Wrapper function to run in separate process without using signals"""
    try:
        start_time = time.time()

        if method == Method.AEQUITAS:
            results_df, metrics = run_aequitas(ge.training_dataframe, model_type=config.aequitas_model_type)
        elif method == Method.BIASSCAN:
            results_df, metrics = run_bias_scan(
                ge,
                test_size=config.bias_scan_test_size,
                random_state=config.random_state,
                n_estimators=config.bias_scan_n_estimators,
                bias_scan_num_iters=config.bias_scan_num_iters,
                bias_scan_scoring=config.bias_scan_scoring,
                bias_scan_favorable_value=config.bias_scan_favorable_value,
                bias_scan_mode=config.bias_scan_mode
            )
        elif method == Method.EXPGA:
            results_df, metrics = run_expga(ge, threshold_rank=config.expga_threshold_rank,
                                            max_global=config.max_global, max_local=config.max_local,
                                            threshold=config.expga_threshold)
        elif method == Method.MLCHECK:
            results_df, metrics = run_mlcheck(
                ge,
                iteration_no=config.mlcheck_iteration_no
            )

        execution_time = time.time() - start_time
        return results_df, metrics, execution_time

    except Exception as e:
        logger.error(f"Error in method execution: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), {"error": str(e)}, 0


class ExperimentRunner:
    def __init__(self, db_path: str = "experiments.db", output_dir: str = "output_dir",
                 max_workers: int = None, method_timeout: int = 5):  # 10 minutes default timeout
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.setup_database()
        self.max_workers = max_workers if max_workers is not None else max(1, multiprocessing.cpu_count() - 1)
        self.method_timeout = method_timeout

    def setup_database(self):
        """Setup database tables including synthetic data storage"""
        # Create empty DataFrames with correct schema
        experiments_df = pd.DataFrame(columns=[
            'experiment_id', 'config', 'methods', 'status',
            'start_time', 'end_time', 'error'
        ])

        results_df = pd.DataFrame(columns=[
            'result_id', 'experiment_id', 'method_name',
            'result_data', 'metrics', 'execution_time'
        ])

        analysis_df = pd.DataFrame(columns=[
            'analysis_id', 'experiment_id', 'method_name',
            'evaluated_results', 'group_individual_table',
            'group_detections', 'analysis_metrics', 'created_at'
        ])

        synthetic_data_df = pd.DataFrame(columns=[
            'synthetic_data_id', 'experiment_id',
            'training_data', 'full_data',
            'protected_attributes', 'outcome_column',
            'created_at'
        ])

        # Create tables using to_sql
        with sqlite3.connect(self.db_path) as conn:
            experiments_df.to_sql('experiments', conn, if_exists='append', index=False)
            results_df.to_sql('results', conn, if_exists='append', index=False)
            analysis_df.to_sql('analysis_results', conn, if_exists='append', index=False)
            synthetic_data_df.to_sql('synthetic_data', conn, if_exists='append', index=False)

    def experiment_exists(self, config: ExperimentConfig) -> Optional[str]:
        """Check if an experiment with the same configuration exists using pandas"""
        config_dict = vars(config).copy()
        config_dict['methods'] = sorted([m.value for m in config_dict['methods']])
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

    def resume_experiments(self, continue_when_error=True):
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
                    self.run_experiment(config, row['experiment_id'])
                except Exception as e:
                    logger.error(f"Failed to resume experiment {row['experiment_id']}: {str(e)}")
                    if not continue_when_error:
                        raise e
                    else:
                        continue

    def run_method(self, method: Method, ge, config: ExperimentConfig) -> tuple:
        """Run method with timeout using ProcessPoolExecutor"""
        ctx = multiprocessing.get_context('spawn')  # Use spawn context for Windows compatibility

        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
            future = executor.submit(run_method_with_timeout, method, ge, config)

            try:
                # Wait for the result with timeout
                results_df, metrics, execution_time = future.result(timeout=self.method_timeout)
                return results_df, metrics, execution_time

            except TimeoutError:
                logger.warning(f"Method {method.value} timed out after {self.method_timeout} seconds")
                # Cancel the future if possible
                future.cancel()
                return pd.DataFrame(), {"error": "timeout"}, self.method_timeout

            except Exception as e:
                logger.error(f"Error running method {method.value}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return pd.DataFrame(), {"error": str(e)}, 0

    def run_experiment(self, config: ExperimentConfig, existing_id: str = None, continue_when_error=False):
        """Run experiment with timeout handling"""
        if existing_id is None:
            existing_id = self.experiment_exists(config)
            if existing_id:
                logger.info(f"Experiment already completed with ID: {existing_id}")
                return existing_id

            experiment_id = str(hash(config))
        else:
            experiment_id = existing_id

        config_dict = vars(config).copy()
        config_dict['methods'] = [m.value for m in config_dict['methods']]

        if existing_id is None:
            self.save_experiment(experiment_id, config_dict, 'running')

        try:
            from data_generator.main import generate_data

            ge = generate_data(
                nb_attributes=config.nb_attributes,
                min_number_of_classes=config.min_number_of_classes,
                max_number_of_classes=config.max_number_of_classes,
                prop_protected_attr=config.prop_protected_attr,
                nb_groups=config.nb_groups,
                max_group_size=config.max_group_size,
                categorical_outcome=True,
                nb_categories_outcome=config.nb_categories_outcome
            )

            # Save synthetic data
            self.save_synthetic_data(experiment_id, ge)
            logger.info(f"Saved synthetic data for experiment: {experiment_id}")

            for method in config.methods:
                logger.info(f"Running method: {method.value}")
                try:
                    results_df, metrics, execution_time = self.run_method(method, ge, config)

                    if isinstance(metrics, dict) and metrics.get("error") == "timeout":
                        logger.warning(f"Method {method.value} timed out - moving to next method")
                        if not continue_when_error:
                            self.update_experiment_status(experiment_id, 'timeout', f"Method {method.value} timed out")
                            continue
                    else:
                        self.save_result(experiment_id, method.value, results_df, metrics, execution_time)
                        logger.info(f"Completed method: {method.value}")

                except Exception as e:
                    error_msg = f"Failed to run method {method.value}: {str(e)}"
                    logger.error(error_msg)
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    if not continue_when_error:
                        self.update_experiment_status(experiment_id, 'failed', error_msg)
                        raise e

            self.update_experiment_status(experiment_id, 'completed')
            logger.info(f"Experiment completed successfully: {experiment_id}")

            logger.info(f"Analyzing results for experiment: {experiment_id}")
            self.analyze_experiment_results(experiment_id, ge.dataframe)

            logger.info(f"Experiment and analysis completed successfully: {experiment_id}")

        except Exception as e:
            error_msg = f"Experiment failed: {str(e)}"
            self.update_experiment_status(experiment_id, 'failed', error_msg)
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            if not continue_when_error:
                raise e

        return experiment_id

    def save_experiment(self, experiment_id: str, config: Dict, status: str):
        """Save experiment using pandas DataFrame"""
        experiment_data = pd.DataFrame([{
            'experiment_id': experiment_id,
            'config': json.dumps(config, sort_keys=True),
            'methods': json.dumps([m for m in config['methods']]),
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
        """Save results using pandas DataFrame"""
        result_data = pd.DataFrame([{
            'result_id': str(uuid.uuid4()),
            'experiment_id': experiment_id,
            'method_name': method_name,
            'result_data': result_df.reset_index(drop=True).to_json(),
            'metrics': json.dumps(metrics),
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

    def analyze_experiment_results(self, experiment_id: str, synthetic_data: pd.DataFrame,
                                   continue_when_error=True):
        """Analyze results using pandas operations"""

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
                # Run evaluation with config
                evaluated_results, group_individual_table, group_detections = evaluate_discrimination_detection(
                    synthetic_data, result_df
                )

                # Calculate any additional analysis metrics
                analysis_metrics = {
                    'total_evaluated': len(evaluated_results),
                    'total_groups': len(group_detections),
                    'total_individuals': len(group_individual_table)
                    # Add any other relevant metrics
                }

                # Save the analysis results
                self.save_analysis_results(
                    experiment_id,
                    method_name,
                    evaluated_results,
                    group_individual_table,
                    group_detections,
                    analysis_metrics
                )

            except Exception as e:
                error_msg = f"Failed to analyze results for method {method_name}: {str(e)}"
                logger.error(error_msg)
                if not continue_when_error:
                    raise e

    def save_analysis_results(self, experiment_id: str, method_name: str,
                              evaluated_results: pd.DataFrame,
                              group_individual_table: pd.DataFrame,
                              group_detections: pd.DataFrame,
                              analysis_metrics: Dict):
        try:
            # Create analysis data entry
            analysis_data = pd.DataFrame([{
                'analysis_id': str(uuid.uuid4()),
                'experiment_id': experiment_id,
                'method_name': method_name,
                'evaluated_results': evaluated_results.to_json(
                    date_format='iso') if not evaluated_results.empty else None,
                'group_individual_table': group_individual_table.to_json(
                    date_format='iso') if not group_individual_table.empty else None,
                'group_detections': group_detections.to_json(date_format='iso') if not group_detections.empty else None,
                'analysis_metrics': json.dumps(analysis_metrics) if analysis_metrics else None,
                'created_at': datetime.now()
            }])

            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                analysis_data.to_sql('analysis_results', conn, if_exists='append', index=False)

            logger.info(f"Saved analysis results for experiment {experiment_id}, method {method_name}")

        except Exception as e:
            logger.error(
                f"Error saving analysis results for experiment {experiment_id}, method {method_name}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def save_synthetic_data(self, experiment_id: str, ge):
        """Save synthetic dataset to database"""
        synthetic_data = pd.DataFrame([{
            'synthetic_data_id': str(uuid.uuid4()),
            'experiment_id': experiment_id,
            'training_data': ge.training_dataframe.to_json(),
            'full_data': ge.dataframe.to_json(),
            'protected_attributes': json.dumps(ge.protected_attributes),
            'outcome_column': ge.outcome_column,
            'created_at': datetime.now()
        }])

        with sqlite3.connect(self.db_path) as conn:
            synthetic_data.to_sql('synthetic_data', conn, if_exists='append', index=False)

    def get_synthetic_data(self, experiment_id: str) -> tuple:
        """Retrieve synthetic dataset for an experiment"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM synthetic_data WHERE experiment_id = ?"
            df = pd.read_sql_query(query, conn, params=(experiment_id,))

        if df.empty:
            return None, None, None, None

        row = df.iloc[0]
        training_data = pd.read_json(row['training_data'])
        full_data = pd.read_json(row['full_data'])
        protected_attributes = json.loads(row['protected_attributes'])
        outcome_column = row['outcome_column']

        return training_data, full_data, protected_attributes, outcome_column

    def get_experiment_data(self, experiment_id: str, tables: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        available_tables = {
            'experiments', 'results', 'analysis_results', 'synthetic_data'
        }

        if tables is None:
            tables = available_tables
        else:
            invalid_tables = set(tables) - available_tables
            if invalid_tables:
                raise ValueError(f"Invalid table names: {invalid_tables}. Valid options are: {available_tables}")

        result = {}

        with sqlite3.connect(self.db_path) as conn:
            # Extract experiments table
            if 'experiments' in tables:
                experiments_df = pd.read_sql_query(
                    "SELECT * FROM experiments WHERE experiment_id = ?",
                    conn,
                    params=(experiment_id,)
                )

                if not experiments_df.empty:
                    # Expand config and methods JSON
                    experiments_df['config'] = experiments_df['config'].apply(json.loads)
                    experiments_df['methods'] = experiments_df['methods'].apply(json.loads)

                    # Extract nested config fields
                    config_df = pd.json_normalize(experiments_df['config'])
                    experiments_df = pd.concat([
                        experiments_df.drop('config', axis=1),
                        config_df
                    ], axis=1)

                result['experiments'] = experiments_df

            # Extract results table
            if 'results' in tables:
                results_df = pd.read_sql_query(
                    "SELECT * FROM results WHERE experiment_id = ?",
                    conn,
                    params=(experiment_id,)
                )

                if not results_df.empty:
                    # Expand result_data and metrics JSON
                    results_df['result_data'] = results_df['result_data'].apply(
                        lambda x: pd.read_json(x) if pd.notna(x) else pd.DataFrame()
                    )
                    results_df['metrics'] = results_df['metrics'].apply(json.loads)

                    # Create separate DataFrames for each result
                    expanded_results = []
                    for idx, row in results_df.iterrows():
                        result_data = row['result_data']
                        if not result_data.empty:
                            # Add method info to result data
                            result_data['result_id'] = row['result_id']
                            result_data['method_name'] = row['method_name']
                            result_data['execution_time'] = row['execution_time']
                            # Add metrics as columns
                            for metric_name, metric_value in row['metrics'].items():
                                result_data[f'metric_{metric_name}'] = metric_value
                            expanded_results.append(result_data)

                    if expanded_results:
                        results_df = pd.concat(expanded_results, ignore_index=True)
                    else:
                        results_df = pd.DataFrame()

                result['results'] = results_df

            # Extract analysis_results table
            if 'analysis_results' in tables:
                analysis_df = pd.read_sql_query(
                    "SELECT * FROM analysis_results WHERE experiment_id = ?",
                    conn,
                    params=(experiment_id,)
                )

                if not analysis_df.empty:
                    # Expand JSON columns
                    analysis_df['evaluated_results'] = analysis_df['evaluated_results'].apply(
                        lambda x: pd.read_json(x) if pd.notna(x) else pd.DataFrame()
                    )
                    analysis_df['group_individual_table'] = analysis_df['group_individual_table'].apply(
                        lambda x: pd.read_json(x) if pd.notna(x) else pd.DataFrame()
                    )
                    analysis_df['group_detections'] = analysis_df['group_detections'].apply(
                        lambda x: pd.read_json(x) if pd.notna(x) else pd.DataFrame()
                    )
                    analysis_df['analysis_metrics'] = analysis_df['analysis_metrics'].apply(json.loads)

                    # Create separate DataFrames for each type of result
                    analysis_results = {
                        'evaluated_results': {},
                        'group_individual_table': {},
                        'group_detections': {},
                        'metrics': {}
                    }

                    for _, row in analysis_df.iterrows():
                        method = row['method_name']
                        analysis_results['evaluated_results'][method] = row['evaluated_results']
                        analysis_results['group_individual_table'][method] = row['group_individual_table']
                        analysis_results['group_detections'][method] = row['group_detections']
                        analysis_results['metrics'][method] = row['analysis_metrics']

                    result['analysis_results'] = analysis_results

            # Extract synthetic_data table
            if 'synthetic_data' in tables:
                synthetic_df = pd.read_sql_query(
                    "SELECT * FROM synthetic_data WHERE experiment_id = ?",
                    conn,
                    params=(experiment_id,)
                )

                if not synthetic_df.empty:
                    # Expand JSON columns
                    synthetic_df['training_data'] = synthetic_df['training_data'].apply(
                        lambda x: pd.read_json(x) if pd.notna(x) else pd.DataFrame()
                    )
                    synthetic_df['full_data'] = synthetic_df['full_data'].apply(
                        lambda x: pd.read_json(x) if pd.notna(x) else pd.DataFrame()
                    )
                    synthetic_df['protected_attributes'] = synthetic_df['protected_attributes'].apply(json.loads)

                    result['synthetic_data'] = {
                        'training_data': synthetic_df.iloc[0]['training_data'],
                        'full_data': synthetic_df.iloc[0]['full_data'],
                        'protected_attributes': synthetic_df.iloc[0]['protected_attributes'],
                        'outcome_column': synthetic_df.iloc[0]['outcome_column']
                    }

        return result

    def analyze_metrics_influence(self) -> Dict[str, Any]:  # Changed return type to include adequacy analysis
        detections_df = self.consolidate_group_detections()

        # Alternative power calculation if statsmodels gives issues
        def calculate_power_alternative(effect_size, sample_size, alpha=0.05):
            """
            Calculate statistical power for correlation using approximation
            Based on Cohen's power tables for correlation
            """
            try:
                import scipy.stats as stats
                z = np.arctanh(abs(effect_size))
                se = 1.0 / np.sqrt(sample_size - 3)
                crit = stats.norm.ppf(1 - alpha / 2)
                ncp = z / se
                power = 1 - stats.norm.cdf(crit - ncp) + stats.norm.cdf(-crit - ncp)

                return power
            except Exception as e:
                logger.error(f"Error calculating power: {str(e)}")
                return np.nan

        def calculate_partial_correlation(data, x, y, control_vars):
            """Calculate partial correlation between x and y controlling for other variables"""
            try:
                scaler = StandardScaler()
                X = scaler.fit_transform(data[[x] + control_vars])
                y_scaled = scaler.fit_transform(data[[y]])

                # Residualize x and y with respect to control variables
                X_control = X[:, 1:]
                x_resid = X[:, 0] - LinearRegression().fit(X_control, X[:, 0]).predict(X_control)
                y_resid = y_scaled.ravel() - LinearRegression().fit(X_control, y_scaled.ravel()).predict(X_control)

                return stats.pearsonr(x_resid, y_resid)  # Return both correlation and p-value
            except Exception as e:
                logger.error(f"Error calculating correlation for {x} vs {y}: {str(e)}")
                return np.nan, np.nan

        def assess_correlation_stability(correlations_df):
            """Assess stability of correlations using bootstrap"""
            bootstrap_correlations = []
            n_bootstrap = 1000
            sample_size = len(correlations_df)

            for _ in range(n_bootstrap):
                bootstrap_sample = correlations_df.sample(n=sample_size, replace=True)
                corrs = bootstrap_sample.groupby('Detection_Metric')['Partial_Correlation'].mean()
                bootstrap_correlations.append(corrs.values)

            ci_lower = np.percentile(bootstrap_correlations, 2.5, axis=0)
            ci_upper = np.percentile(bootstrap_correlations, 97.5, axis=0)

            return ci_lower, ci_upper

        try:
            detection_metrics = [
                'nb_indv_detected', 'nb_couple_detected',
                'indv_detection_rate', 'couple_detection_rate'
            ]

            all_metrics = {
                'calculated': [
                    'calculated_epistemic', 'calculated_aleatoric', 'relevance',
                    'calculated_magnitude', 'calculated_group_size', 'calculated_granularity',
                    'calculated_intersectionality', 'calculated_uncertainty',
                    'calculated_similarity', 'calculated_subgroup_ratio'
                ],
                'config': [
                    'nb_attributes', 'prop_protected_attr', 'nb_groups', 'max_group_size'
                ]
            }

            # Combine all metrics for control variables
            all_metric_list = all_metrics['calculated'] + all_metrics['config']

            results = {
                'correlations': [],
                'method_specific': {},
                'adequacy_metrics': {}  # New section for adequacy analysis
            }

            # Analyze full dataset
            for det_metric in detection_metrics:
                for metric_type, metrics in all_metrics.items():
                    for metric in metrics:
                        if det_metric in detections_df.columns and metric in detections_df.columns:
                            # Define control variables
                            control_vars = [m for m in all_metric_list if m != metric
                                            and m in detections_df.columns]

                            # Calculate partial correlation
                            partial_corr, p_value = calculate_partial_correlation(
                                detections_df,
                                metric,
                                det_metric,
                                control_vars
                            )

                            if np.isnan(partial_corr):
                                continue

                            # Calculate statistical power
                            power = calculate_power_alternative(partial_corr, len(detections_df))

                            # Prepare data for regression
                            X = detections_df[control_vars + [metric]]
                            y = detections_df[det_metric]
                            X = sm.add_constant(X)

                            try:
                                # Fit regression model
                                model = sm.OLS(y, X).fit()

                                # Get statistics
                                coef = model.params[metric]
                                p_value = model.pvalues[metric]

                                # Calculate unique variance explained
                                full_r2 = model.rsquared
                                reduced_X = X.drop(columns=[metric])
                                reduced_model = sm.OLS(y, reduced_X).fit()
                                unique_r2 = full_r2 - reduced_model.rsquared

                                # Calculate standardized coefficient
                                X_std = StandardScaler().fit_transform(detections_df[[metric]])
                                y_std = StandardScaler().fit_transform(detections_df[[det_metric]])
                                beta = LinearRegression().fit(X_std, y_std).coef_[0]

                                results['correlations'].append({
                                    'Detection_Metric': det_metric,
                                    'Metric_Type': metric_type,
                                    'Metric': metric,
                                    'Partial_Correlation': partial_corr,
                                    'Regression_Coefficient': coef,
                                    'Standardized_Beta': beta,
                                    'P_Value': p_value,
                                    'Statistical_Power': power,
                                    'Unique_Variance_Explained': unique_r2,
                                    'Total_R2': full_r2,
                                    'Sample_Size': len(detections_df)
                                })

                            except Exception as e:
                                logger.error(f"Error in regression analysis for {metric}: {str(e)}")
                                continue

            # Convert to DataFrame
            results['correlations'] = pd.DataFrame(results['correlations'])

            # Calculate adequacy metrics
            if not results['correlations'].empty:
                # Sample size analysis
                sample_size = len(detections_df)
                unique_experiments = len(detections_df['experiment_id'].unique())

                # Effect size analysis
                effect_sizes = results['correlations']['Partial_Correlation'].abs()
                mean_effect = effect_sizes.mean()
                max_effect = effect_sizes.max()

                # Statistical power analysis
                mean_power = results['correlations']['Statistical_Power'].mean()
                min_power = results['correlations']['Statistical_Power'].min()

                # Correlation stability
                ci_lower, ci_upper = assess_correlation_stability(results['correlations'])
                ci_width = ci_upper - ci_lower

                # R² analysis
                mean_r2 = results['correlations']['Total_R2'].mean()

                results['adequacy_metrics'] = {
                    'sample_metrics': {
                        'total_samples': sample_size,
                        'unique_experiments': unique_experiments,
                        'samples_per_detection_metric': sample_size / len(detection_metrics)
                    },
                    'effect_metrics': {
                        'mean_effect_size': mean_effect,
                        'max_effect_size': max_effect,
                        'mean_statistical_power': mean_power,
                        'min_statistical_power': min_power
                    },
                    'stability_metrics': {
                        'mean_ci_width': np.mean(ci_width),
                        'max_ci_width': np.max(ci_width),
                        'ci_lower': ci_lower.tolist(),
                        'ci_upper': ci_upper.tolist()
                    },
                    'variance_metrics': {
                        'mean_r2': mean_r2
                    }
                }

                # Generate recommendations
                recommendations = []

                if mean_power < 0.8:
                    recommendations.append(
                        f"Need more samples to achieve adequate statistical power. "
                        f"Current mean power: {mean_power:.2f}, target: 0.8"
                    )

                if mean_r2 < 0.2:
                    recommendations.append(
                        f"Low R² ({mean_r2:.3f}) suggests need for additional metrics or "
                        f"different experimental conditions"
                    )

                if np.mean(ci_width) > 0.1:
                    recommendations.append(
                        f"Wide confidence intervals (mean width: {np.mean(ci_width):.3f}) "
                        f"suggest need for more samples to improve stability"
                    )

                if unique_experiments < 30:
                    recommendations.append(
                        f"Only {unique_experiments} unique experiments. Consider running "
                        f"more experiments with different configurations"
                    )

                results['adequacy_metrics']['recommendations'] = recommendations

            # Sort results by absolute partial correlation
            results['correlations'] = results['correlations'].sort_values(
                ['Detection_Metric', 'Partial_Correlation'],
                key=lambda x: abs(x) if x.name == 'Partial_Correlation' else x,
                ascending=[True, False]
            )

            return results

        except Exception as e:
            logger.error(f"Error analyzing metric influence: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'correlations': pd.DataFrame(),
                'method_specific': {},
                'adequacy_metrics': {}
            }

    def analyze_discrimination_metrics_cate(self, methods=None) -> Dict[str, Any]:
        """
        Analyze discrimination metrics using CATE with improved error handling and index alignment.
        """

        def analyze_cate(data, det_metric, treatment_metric, metric_type):
            """
            Perform CATE analysis with proper index alignment and robust error handling.
            """
            if len(data) < 50:
                return None

            try:
                # Prepare data for CATE analysis - ensure index alignment
                covariates = [m for m in metrics_config['calculated'] + metrics_config['config']
                              if m != treatment_metric and m in data.columns]

                # Select relevant columns and remove missing values
                analysis_cols = [det_metric, treatment_metric] + covariates
                analysis_data = data[analysis_cols].copy().dropna()

                # Reset index to ensure alignment
                analysis_data = analysis_data.reset_index(drop=True)

                # Define treatment groups using loc to ensure proper alignment
                median_treatment = analysis_data[treatment_metric].median()
                T = (analysis_data[treatment_metric] > median_treatment).astype(int)

                # Verify sufficient samples in each group
                min_group_size = min(T.sum(), len(T) - T.sum())
                if min_group_size < 10:
                    return None

                # Prepare outcome and covariates with aligned indices
                Y = analysis_data[det_metric].values
                X = analysis_data[covariates]

                if X.empty or len(X.columns) == 0:
                    return None

                # Standardize covariates
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

                # Split data by treatment using boolean indexing with aligned indices
                mask_treated = T == 1
                mask_control = T == 0

                X_treated = X_scaled[mask_treated]
                Y_treated = Y[mask_treated]
                X_control = X_scaled[mask_control]
                Y_control = Y[mask_control]

                # Configure cross-validation
                n_splits = min(5, min(len(Y_treated), len(Y_control)) // 2)
                if n_splits < 2:
                    return None

                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

                # Fit models and get predictions
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                y_pred_treated = cross_val_predict(rf, X_treated, Y_treated, cv=cv)
                y_pred_control = cross_val_predict(rf, X_control, Y_control, cv=cv)

                # Calculate CATE
                cate = np.mean(y_pred_treated) - np.mean(y_pred_control)

                # Bootstrap for uncertainty estimation
                n_bootstrap = 1000
                bootstrap_cates = []

                for _ in range(n_bootstrap):
                    treated_idx = np.random.choice(len(y_pred_treated), size=len(y_pred_treated))
                    control_idx = np.random.choice(len(y_pred_control), size=len(y_pred_control))

                    bootstrap_cate = (np.mean(y_pred_treated[treated_idx]) -
                                      np.mean(y_pred_control[control_idx]))
                    bootstrap_cates.append(bootstrap_cate)

                # Calculate statistics
                std_error = np.std(bootstrap_cates)
                t_stat = cate / std_error if std_error > 0 else 0
                p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
                ci_lower, ci_upper = np.percentile(bootstrap_cates, [2.5, 97.5])

                # Calculate relative effect
                control_mean = np.mean(y_pred_control)
                relative_effect = cate / control_mean if control_mean != 0 else np.nan

                return {
                    'Detection_Metric': det_metric,
                    'Metric_Type': metric_type,
                    'Metric': treatment_metric,
                    'CATE': cate,
                    'P_Value': p_value,
                    'CI_Lower': ci_lower,
                    'CI_Upper': ci_upper,
                    'Std_Error': std_error,
                    'Relative_Effect': relative_effect,
                    'Sample_Size': len(analysis_data),
                    'Treated_Size': T.sum(),
                    'Control_Size': len(T) - T.sum(),
                    'N_Splits': n_splits
                }

            except Exception as e:
                logger.warning(f"Error in CATE analysis for {det_metric} and {treatment_metric}: {str(e)}")
                return None

        try:
            # Get consolidated group detections with proper index
            detections_df = self.consolidate_group_detections()
            if detections_df.empty:
                return {'error': 'No group detection data found'}

            # Reset index to ensure alignment
            detections_df = detections_df.reset_index(drop=True)

            # Define metrics to analyze
            metrics_config = {
                'detection': [
                    'indv_detection_rate', 'couple_detection_rate'
                ],
                'calculated': [
                    'calculated_epistemic', 'calculated_aleatoric',
                    'calculated_magnitude', 'calculated_group_size', 'calculated_granularity',
                    'calculated_intersectionality',
                    'calculated_similarity', 'calculated_subgroup_ratio'
                ],
                'config': [
                    'nb_attributes', 'prop_protected_attr', 'nb_groups', 'max_group_size'
                ]
            }

            # Get unique methods
            if methods is None:
                methods = detections_df['method'].unique()

            # Initialize results storage
            overall_results = []
            method_specific_results = {method: [] for method in methods}

            # Calculate total number of iterations for progress bar
            total_iterations = 0
            for det_metric in metrics_config['detection']:
                if det_metric in detections_df.columns:
                    for metric_type in ['calculated', 'config']:
                        for treatment_metric in metrics_config[metric_type]:
                            if treatment_metric in detections_df.columns:
                                # Add 1 for overall analysis and 1 for each method-specific analysis
                                total_iterations += 1 + len(methods)

            # Create progress bar
            pbar = tqdm(total=total_iterations, desc="Analyzing CATE metrics")

            # Analyze each combination of detection and treatment metrics
            for det_metric in metrics_config['detection']:
                if det_metric not in detections_df.columns:
                    continue

                for metric_type in ['calculated', 'config']:
                    for treatment_metric in metrics_config[metric_type]:
                        if treatment_metric not in detections_df.columns:
                            continue

                        # Overall analysis
                        result = analyze_cate(detections_df, det_metric, treatment_metric, metric_type)
                        if result:
                            overall_results.append(result)
                        pbar.update(1)

                        # Per-method analysis
                        for method in methods:
                            method_data = detections_df[detections_df['method'] == method].reset_index(drop=True)
                            result = analyze_cate(method_data, det_metric, treatment_metric, metric_type)
                            if result:
                                result['Method'] = method
                                method_specific_results[method].append(result)
                            pbar.update(1)

            pbar.close()

            # Convert results to DataFrames with proper index handling
            overall_df = pd.DataFrame(overall_results) if overall_results else pd.DataFrame()
            method_dfs = {
                method: pd.DataFrame(results) if results else pd.DataFrame()
                for method, results in method_specific_results.items()
            }

            # Calculate adequacy metrics
            def calculate_adequacy_metrics(df):
                if df.empty:
                    return None

                metrics = {
                    'sample_metrics': {
                        'total_samples': len(df),
                        'mean_treated_size': df['Treated_Size'].mean() if 'Treated_Size' in df else None,
                        'mean_control_size': df['Control_Size'].mean() if 'Control_Size' in df else None,
                        'mean_cv_splits': df['N_Splits'].mean() if 'N_Splits' in df else None
                    },
                    'effect_metrics': {
                        'mean_cate': df['CATE'].abs().mean() if 'CATE' in df else None,
                        'median_cate': df['CATE'].abs().median() if 'CATE' in df else None,
                        'max_cate': df['CATE'].abs().max() if 'CATE' in df else None,
                        'significant_effects': sum(df['P_Value'] < 0.05) if 'P_Value' in df else None
                    },
                    'precision_metrics': {
                        'mean_std_error': df['Std_Error'].mean() if 'Std_Error' in df else None,
                        'mean_ci_width': ((df['CI_Upper'] - df['CI_Lower']).mean()
                                          if 'CI_Upper' in df and 'CI_Lower' in df else None),
                        'mean_p_value': df['P_Value'].mean() if 'P_Value' in df else None
                    }
                }

                # Remove None values
                metrics = {k: {k2: v2 for k2, v2 in v.items() if v2 is not None}
                           for k, v in metrics.items()}

                return metrics

            overall_adequacy = calculate_adequacy_metrics(overall_df)
            method_adequacy = {method: calculate_adequacy_metrics(df)
                               for method, df in method_dfs.items()}

            return {
                'overall_analysis': {
                    'cate_results': overall_df,
                    'adequacy_metrics': overall_adequacy
                },
                'method_specific': {
                    method: {
                        'cate_results': df,
                        'adequacy_metrics': method_adequacy[method]
                    }
                    for method, df in method_dfs.items()
                }
            }

        except Exception as e:
            logger.error(f"Error in CATE analysis: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'error': str(e)}

    def visualize_detection_rates(self) -> None:
        """
        Create distribution plots showing detection rates conditioned on calculated attributes for each method.
        Excludes the relevance metric.
        """
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt

            # Get consolidated group detections
            detections_df = self.consolidate_group_detections()

            if detections_df.empty:
                logger.warning("No detection data available for visualization")
                return

            # Define calculated metrics to analyze (excluding relevance)
            calculated_metrics = [
                'calculated_epistemic', 'calculated_aleatoric',
                'calculated_magnitude', 'calculated_group_size', 'calculated_granularity',
                'calculated_intersectionality',
                'calculated_similarity', 'calculated_subgroup_ratio'
            ]

            # Create plots for each method
            for method in detections_df['method'].unique():
                method_data = detections_df[detections_df['method'] == method]

                # Create individual detection rate plot
                fig, axes = plt.subplots(4, 2, figsize=(20, 25))
                fig.suptitle(f'{method} - Individual Detection Rate by Calculated Attributes', fontsize=16)
                axes = axes.flatten()

                for i, metric in enumerate(calculated_metrics):
                    if metric in method_data.columns:
                        sns.scatterplot(data=method_data, x=metric, y='indv_detection_rate', ax=axes[i])
                        axes[i].set_title(f'{metric}')

                plt.tight_layout()
                plt.subplots_adjust(top=0.95)

                # Create output directory if it doesn't exist
                output_dir = self.output_dir.joinpath('detection_rate_plots')
                output_dir.mkdir(exist_ok=True)

                # Save individual detection plot
                plt.savefig(output_dir.joinpath(f'{method}_individual_detection_by_attributes.png'),
                            bbox_inches='tight', dpi=300)
                plt.close()

                # Create couple detection rate plot
                fig, axes = plt.subplots(4, 2, figsize=(20, 25))
                fig.suptitle(f'{method} - Couple Detection Rate by Calculated Attributes', fontsize=16)
                axes = axes.flatten()

                for i, metric in enumerate(calculated_metrics):
                    if metric in method_data.columns:
                        sns.scatterplot(data=method_data, x=metric, y='couple_detection_rate', ax=axes[i])
                        axes[i].set_title(f'{metric}')

                plt.tight_layout()
                plt.subplots_adjust(top=0.95)

                # Save couple detection plot
                plt.savefig(output_dir.joinpath(f'{method}_couple_detection_by_attributes.png'),
                            bbox_inches='tight', dpi=300)
                plt.close()

            logger.info("Detection rate visualizations created successfully")

        except Exception as e:
            logger.error(f"Error creating detection rate visualizations: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def plot_discriminatory_individuals(self,
                                        experiment_id: str,
                                        method_name: str = None,
                                        n_neighbors: int = 15,
                                        min_dist: float = 0.1,
                                        figsize: tuple = (12, 8),
                                        background_alpha: float = 0.2,
                                        point_size: int = 80) -> plt.Figure:
        """
        Plot discriminatory individuals from experiment results overlaid on synthetic data embedding.
        Optimized version using vectorized operations instead of iterrows().
        """
        try:
            from umap import UMAP
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("Please install umap-learn and scikit-learn: pip install umap-learn scikit-learn")

        # Get experiment data
        experiment_data = self.get_experiment_data(
            experiment_id,
            tables=['synthetic_data', 'analysis_results']
        )

        if not experiment_data:
            raise ValueError(f"No data found for experiment {experiment_id}")

        # Get synthetic data and results
        synthetic_data = experiment_data['synthetic_data']['full_data']
        analysis_results = experiment_data['analysis_results']

        # Get all attribute columns (both protected and unprotected)
        attr_columns = [col for col in synthetic_data.columns if col.startswith('Attr')]
        if not attr_columns:
            raise ValueError("No attribute columns found in synthetic data")

        # Prepare the feature matrix for UMAP
        X = synthetic_data[attr_columns].values

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Create and fit UMAP
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42,
            verbose=True
        )

        background_embedding = umap_model.fit_transform(X_scaled)

        # Create the visualization
        fig, ax = plt.subplots(figsize=figsize)

        # Plot background points
        ax.scatter(
            background_embedding[:, 0],
            background_embedding[:, 1],
            c='lightgrey',
            alpha=background_alpha,
            s=point_size // 2,
            label='Background Data'
        )

        # Determine methods to process
        if method_name:
            methods_to_process = [method_name]
        else:
            methods_to_process = analysis_results['evaluated_results'].keys()

        # Extract discriminatory individuals using vectorized operations
        all_discriminatory_indv = set()
        for method in methods_to_process:
            if method not in analysis_results['evaluated_results']:
                continue

            eval_results = analysis_results['evaluated_results'][method]

            # Filter for valid couple keys
            valid_couples = eval_results[eval_results['couple_key'].str.contains('-', na=False)]

            # Split all couple keys at once
            couples_split = valid_couples['couple_key'].str.split('-', expand=True)

            # Add all individuals to the set using vectorized operations
            all_discriminatory_indv.update(couples_split[0].unique())
            all_discriminatory_indv.update(couples_split[1].unique())

        if all_discriminatory_indv:
            # Create mask for discriminatory individuals
            disc_mask = synthetic_data['indv_key'].isin(all_discriminatory_indv)
            disc_indv_data = synthetic_data[disc_mask]

            # Get features for discriminatory individuals
            disc_features = disc_indv_data[attr_columns].values

            # Scale and embed discriminatory individuals
            disc_features_scaled = scaler.transform(disc_features)
            disc_embedding = umap_model.transform(disc_features_scaled)

            # Create mask for original vs new individuals
            original_mask = disc_indv_data['indv_key'].isin(synthetic_data['indv_key'])

            # Plot original discriminatory individuals
            if original_mask.any():
                ax.scatter(
                    disc_embedding[original_mask, 0],
                    disc_embedding[original_mask, 1],
                    c='green',
                    alpha=0.8,
                    s=point_size,
                    label='Original Discriminatory Individuals'
                )

            # Plot new discriminatory individuals
            new_mask = ~original_mask
            if new_mask.any():
                ax.scatter(
                    disc_embedding[new_mask, 0],
                    disc_embedding[new_mask, 1],
                    c='red',
                    alpha=0.8,
                    s=point_size,
                    label='New Discriminatory Individuals'
                )

            # Set title and labels
            title = 'Discriminatory Individuals in Synthetic Data Space'
            if method_name:
                title += f'\nMethod: {method_name}'
            plt.title(title, pad=20)
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')

            # Style the plot
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_facecolor('#f8f9fa')

            # Add UMAP parameters
            param_text = (f'UMAP Parameters:\n'
                          f'n_neighbors={n_neighbors}, '
                          f'min_dist={min_dist}')
            plt.text(0.02, 0.98, param_text,
                     transform=ax.transAxes,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Add legend with counts
            n_original = original_mask.sum()
            n_new = new_mask.sum()
            legend_title = (f'Individual Counts:\n'
                            f'Original: {n_original}\n'
                            f'New: {n_new}')
            plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

        return fig


def create_test_configurations() -> List[ExperimentConfig]:
    base_config = {
        "min_number_of_classes": 2,
        "max_number_of_classes": 10,
        "nb_categories_outcome": 4
    }

    test_variations = [

        {"nb_attributes": 10, "prop_protected_attr": 0.3, "nb_groups": 1, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.3, "nb_groups": 1, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.3, "nb_groups": 2, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.3, "nb_groups": 2, "max_group_size": 400},

        # Varying number of discriminatory groups (nb_groups)
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 5, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 10, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 25, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 75, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 100, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 150, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 200, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 300, "max_group_size": 400},

        # Varying proportion of protected attributes (prop_protected_attr)
        {"nb_attributes": 10, "prop_protected_attr": 0.05, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.1, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.15, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.25, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.3, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.4, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.5, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.6, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.8, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 1.0, "nb_groups": 50, "max_group_size": 400},

        # Varying number of discriminatory groups (nb_groups)
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 1, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 5, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 10, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 25, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 75, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 100, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 150, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 200, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 300, "max_group_size": 400},

        # Varying proportion of protected attributes (prop_protected_attr)
        {"nb_attributes": 10, "prop_protected_attr": 0.05, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.1, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.15, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.25, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.3, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.4, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.5, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.6, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 0.8, "nb_groups": 50, "max_group_size": 400},
        {"nb_attributes": 10, "prop_protected_attr": 1.0, "nb_groups": 50, "max_group_size": 400},

        # # Varying number of attributes (nb_attributes)
        # {"nb_attributes": 3, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        # {"nb_attributes": 5, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        # {"nb_attributes": 8, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        # {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        # {"nb_attributes": 15, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        # {"nb_attributes": 20, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        # {"nb_attributes": 30, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        # {"nb_attributes": 50, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        # {"nb_attributes": 75, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        # {"nb_attributes": 100, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        #
        # # Varying max group size (max_group_size)
        # {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 10},
        # {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 20},
        # {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 50},
        # {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        # {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 200},
        # {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 500},
        # {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 1000},
        # {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 2000},
        # {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 5000},
        #
        # # Combined variations
        # # High complexity scenarios
        # {"nb_attributes": 50, "prop_protected_attr": 0.5, "nb_groups": 200, "max_group_size": 1000},
        # {"nb_attributes": 75, "prop_protected_attr": 0.4, "nb_groups": 300, "max_group_size": 2000},
        # {"nb_attributes": 100, "prop_protected_attr": 0.3, "nb_groups": 500, "max_group_size": 5000},
        #
        # # Low complexity scenarios
        # {"nb_attributes": 3, "prop_protected_attr": 0.1, "nb_groups": 5, "max_group_size": 20},
        # {"nb_attributes": 5, "prop_protected_attr": 0.15, "nb_groups": 10, "max_group_size": 50},
        # {"nb_attributes": 8, "prop_protected_attr": 0.2, "nb_groups": 15, "max_group_size": 75},
        #
        # # Balanced scenarios
        # {"nb_attributes": 15, "prop_protected_attr": 0.3, "nb_groups": 75, "max_group_size": 250},
        # {"nb_attributes": 25, "prop_protected_attr": 0.35, "nb_groups": 100, "max_group_size": 500},
        # {"nb_attributes": 35, "prop_protected_attr": 0.4, "nb_groups": 150, "max_group_size": 750},
        #
        # # Edge cases
        # {"nb_attributes": 2, "prop_protected_attr": 0.01, "nb_groups": 1, "max_group_size": 10},
        # # Minimal configuration
        # {"nb_attributes": 100, "prop_protected_attr": 1.0, "nb_groups": 1000, "max_group_size": 5000},
        # # Maximal configuration
        # {"nb_attributes": 10, "prop_protected_attr": 0.5, "nb_groups": 50, "max_group_size": 100},
        # Perfectly balanced protected attributes
    ]

    configurations = []
    for variation in test_variations:
        config_dict = {**base_config, **variation}
        configurations.append(ExperimentConfig(**config_dict))

    return configurations


def run_experiments(configs: List[ExperimentConfig], methods: Set[Method] = None,
                    db_path: str = "experiments.db", continue_when_error=True):
    """Run experiments with duplicate prevention"""
    runner = ExperimentRunner(db_path)

    # Set methods for each config if specified
    if methods:
        for config in configs:
            config.methods = methods

    # Resume any incomplete experiments
    runner.resume_experiments(continue_when_error=continue_when_error)

    # Run new experiments, skipping duplicates
    for config in configs:
        existing_id = runner.experiment_exists(config)
        if existing_id:
            logger.info(f"Skipping duplicate experiment (ID: {existing_id})")
            continue

        experiment_id = runner.run_experiment(config, continue_when_error=continue_when_error)
        logger.info(f"Completed experiment: {experiment_id}")


# %%
DB_PATH = HERE.joinpath("experiments/discrimination_detection_results5.db").as_posix()
FIGURES_PATH = HERE.joinpath("experiments/figures").as_posix()


# %%
def main():
    """Main function to run experiments"""
    multiprocessing.freeze_support()

    configs = create_test_configurations()
    methods = {Method.AEQUITAS}

    DB_PATH = HERE.joinpath("experiments/discrimination_detection_results5.db").as_posix()
    FIGURES_PATH = HERE.joinpath("experiments/figures").as_posix()

    run_experiments(configs, methods=methods, db_path=DB_PATH, continue_when_error=False)


if __name__ == '__main__':
    main()

# %%
# runner = ExperimentRunner(DB_PATH, FIGURES_PATH)
# res = runner.get_experiment_data('4e2a04c4-27e7-44c7-ab14-c1121105a007')
# re2s = runner.generate_performance_report()
# re4s = runner.summarize_experiment_performance()
# re3s = runner.summarize_method_performance()
# runner.visualize_detection_rates()
# %%
# print(re4s['new_individuals_found'].head())

# %%
# Run analysis
# metric_results_corr = runner.analyze_metrics_influence()
# metric_results = runner.analyze_discrimination_metrics_cate(methods=['aequitas'])

# %%
# runner.plot_discriminatory_individuals('45bafd30-7c65-4c9c-b7c4-17b9d7cfc705')

# %%
# class ParallelExperimentRunner(ExperimentRunner):
#     def __init__(self, db_path: str = "experiments.db", max_workers: Optional[int] = None):
#         super().__init__(db_path)
#         self.max_workers = max_workers or multiprocessing.cpu_count()
#
#     def _run_single_experiment(self, config: ExperimentConfig) -> Optional[str]:
#         """Run a single experiment with proper error handling"""
#         try:
#             return self.run_experiment(config)
#         except Exception as e:
#             logger.error(f"Error running experiment: {str(e)}\n{traceback.format_exc()}")
#             return None
#
#     def run_experiments_parallel(self, configs: List[ExperimentConfig], methods: Set[Method] = None):
#         """Run multiple experiments in parallel"""
#         if methods:
#             for config in configs:
#                 config.methods = methods
#
#         # Filter out configs that already have completed experiments
#         new_configs = []
#         for config in configs:
#             existing_id = self.experiment_exists(config)
#             if existing_id:
#                 logger.info(f"Skipping duplicate experiment (ID: {existing_id})")
#             else:
#                 new_configs.append(config)
#
#         if not new_configs:
#             logger.info("No new experiments to run")
#             return
#
#         logger.info(f"Running {len(new_configs)} experiments with {self.max_workers} workers")
#
#         # Create a process pool and run experiments
#         with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
#             # Submit all experiments
#             future_to_config = {
#                 executor.submit(self._run_single_experiment, config): config
#                 for config in new_configs
#             }
#
#             # Process completed experiments
#             for future in future_to_config:
#                 config = future_to_config[future]
#                 try:
#                     experiment_id = future.result()
#                     if experiment_id:
#                         logger.info(f"Completed experiment {experiment_id} for config: {config}")
#                     else:
#                         logger.error(f"Failed to complete experiment for config: {config}")
#                 except Exception as e:
#                     logger.error(f"Exception running experiment: {str(e)}")
#
#
# def run_parallel_experiments(configs: List[ExperimentConfig],
#                              methods: Set[Method] = None,
#                              db_path: str = "experiments.db",
#                              max_workers: Optional[int] = None):
#     """Main function to run experiments in parallel"""
#     runner = ParallelExperimentRunner(db_path, max_workers)
#
#     # Resume any incomplete experiments first
#     runner.resume_experiments()
#
#     # Run new experiments in parallel
#     runner.run_experiments_parallel(configs, methods)
#
#
# # %%
# def main():
#     DB_PATH = HERE.joinpath("experiments/discrimination_detection_results5.db").as_posix()
#
#     # Get optimized configurations
#     configs = create_test_configurations()
#
#     # Set number of parallel workers (optional, defaults to CPU count)
#     max_workers = multiprocessing.cpu_count() - 1  # Leave one CPU free
#
#     # Run experiments in parallel
#     methods = {Method.MLCHECK, Method.BIASSCAN, Method.EXPGA, Method.AEQUITAS}
#     run_parallel_experiments(
#         configs,
#         methods=methods,
#         db_path=DB_PATH,
#         max_workers=max_workers
#     )
#
#
# if __name__ == '__main__':
#     freeze_support()  # Required for Windows
#     main()
# %%

# %%
