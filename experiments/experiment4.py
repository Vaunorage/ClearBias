from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any, Optional, Set
import uuid
import logging
from enum import Enum
import re
from pathlib import Path
from paths import HERE
import sqlite3
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def matches_pattern(pattern_string: str, test_string: str) -> bool:
    regex_pattern = pattern_string.replace("*", r"\d").replace("|", r"\|").replace("-", r"\-")
    pattern = re.compile(f"^{regex_pattern}$")
    return bool(pattern.match(test_string))


def is_individual_part_of_the_original_indv(indv_key, indv_key_list):
    return indv_key in indv_key_list


def is_individual_part_of_a_group(indv_key, group_key_list):
    res = []
    for grp in group_key_list:
        opt1, opt2 = grp.split('-')
        if matches_pattern(opt1, indv_key) or matches_pattern(opt2, indv_key):
            res.append(grp)
    return res


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


def count_individuals_per_group(aequitas_results: pd.DataFrame, all_group_keys: set) -> dict:
    group_counts = {group: 0 for group in all_group_keys}

    for _, row in aequitas_results.iterrows():
        groups = row['couple_part_of_a_group']
        for group in groups:
            if group in group_counts:
                group_counts[group] += 1

    return group_counts


def create_group_individual_table(aequitas_results: pd.DataFrame, synthetic_data: pd.DataFrame,
                                  all_group_keys: set) -> pd.DataFrame:
    calculated_properties = [
        'calculated_epistemic', 'calculated_aleatoric', 'relevance',
        'calculated_magnitude', 'calculated_group_size', 'calculated_granularity',
        'calculated_intersectionality', 'calculated_uncertainty',
        'calculated_similarity', 'calculated_subgroup_ratio'
    ]

    # Create a dictionary of groups with their individuals, metrics, and couples
    group_data = {group: {'individuals': set(), 'metrics': [], 'couples': set()} for group in all_group_keys}

    # Process synthetic data
    for _, row in synthetic_data.iterrows():
        group = row['group_key']
        if group in group_data:
            group_data[group]['individuals'].add(row['indv_key'])
            group_data[group]['metrics'].append({prop: row[prop] for prop in calculated_properties})

    # Process Aequitas results
    detected_groups = set()
    for _, row in aequitas_results.iterrows():
        for group in row['couple_part_of_a_group']:
            if group in group_data:
                detected_groups.add(group)
                indv1, indv2 = row['couple_key'].split('-')
                group_data[group]['individuals'].update([indv1, indv2])
                group_data[group]['couples'].add(row['couple_key'])

    # Create result data
    result_data = []
    for group, data in group_data.items():
        base_row = {
            'group_key': group,
            'individuals': list(data['individuals']),
            'num_individuals': len(data['individuals']),
            'detected': group in detected_groups,
            'couple_keys': list(data['couples'])
        }

        if data['metrics']:
            for metrics in data['metrics']:
                row = base_row.copy()
                row.update(metrics)
                result_data.append(row)
        else:
            row = base_row.copy()
            row.update({prop: None for prop in calculated_properties})
            result_data.append(row)

    return pd.DataFrame(result_data)


def evaluate_discrimination_detection(synthetic_data: pd.DataFrame, results_df: pd.DataFrame) -> tuple:
    if results_df.empty:
        return pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames

    synthetic_data['indv_key'] = synthetic_data['indv_key'].astype(str)
    synthetic_data['group_key'] = synthetic_data['group_key'].astype(str)
    synthetic_data['subgroup_key'] = synthetic_data['subgroup_key'].astype(str)

    results_df['indv_key'] = results_df['indv_key'].astype(str)
    results_df['couple_key'] = results_df['couple_key'].astype(str)

    all_indv_keys = set(synthetic_data['indv_key'])
    all_group_keys = set(synthetic_data['group_key'])

    results_df['indv_in_original_indv'] = results_df['indv_key'].apply(
        lambda x: is_individual_part_of_the_original_indv(x, all_indv_keys))

    # Check if 'indv_part_of_group' column exists, if not, create it
    if 'indv_part_of_group' not in results_df.columns:
        results_df['indv_part_of_group'] = results_df['indv_key'].apply(
            lambda x: is_individual_part_of_a_group(x, all_group_keys))

    results_df['couple_part_of_a_group'] = results_df['couple_key'].apply(
        lambda x: is_couple_part_of_a_group(x, all_group_keys))

    results_df['correct_detection'] = results_df['couple_part_of_a_group'].apply(lambda x: len(x) > 0)

    # Calculate evaluation metrics
    total_couples = len(results_df)
    true_positives = results_df['correct_detection'].sum()
    false_positives = total_couples - true_positives

    # Proportion of Original Individuals
    total_individuals = len(set(results_df['indv_key']))
    original_individuals = sum(results_df['indv_in_original_indv'])
    p_original = original_individuals / total_individuals if total_individuals > 0 else 0

    # Proportion of New Individuals
    p_new = 1 - p_original

    # Proportion of original individuals that belong to a group
    original_in_group = sum(results_df[results_df['indv_in_original_indv']]['indv_part_of_group'].apply(len) > 0)
    p_group_original = original_in_group / original_individuals if original_individuals > 0 else 0

    # Proportion of new individuals that belong to a group
    new_individuals = total_individuals - original_individuals
    new_in_group = sum(
        results_df[~results_df['indv_in_original_indv']]['indv_part_of_group'].apply(len) > 0)
    p_group_new = new_in_group / new_individuals if new_individuals > 0 else 0

    # Correct Couple Detection Rate (Precision)
    r_correct = true_positives / total_couples if total_couples > 0 else 0

    # Proportion of Groups Detected
    detected_groups = set()
    for groups in results_df['couple_part_of_a_group']:
        detected_groups.update(groups)
    p_groups = len(detected_groups) / len(all_group_keys) if len(all_group_keys) > 0 else 0

    # Calculate average values for metrics from synthetic data
    calculated_properties = [
        'calculated_epistemic', 'calculated_aleatoric', 'relevance',
        'calculated_magnitude', 'calculated_group_size', 'calculated_granularity',
        'calculated_intersectionality', 'calculated_uncertainty',
        'calculated_similarity', 'calculated_subgroup_ratio'
    ]
    avg_metrics = synthetic_data[calculated_properties].mean().to_dict()
    avg_metrics = {f"Average {k.replace('calculated_', '').capitalize()}": v for k, v in avg_metrics.items()}

    # Prepare the evaluation metrics
    metrics = {
        "Proportion of Original Individuals": p_original,
        "Proportion of New Individuals": p_new,
        "Proportion of original individuals that belong to a group": p_group_original,
        "Proportion of new individuals that belong to a group": p_group_new,
        "Correct Couple Detection Rate": r_correct,
        "Total Couples Detected": total_couples,
        "True Positives": true_positives,
        "False Positives": false_positives,
        "Proportion of Groups Detected": p_groups,
        **avg_metrics
    }

    # Add metrics to results_df
    for key, value in metrics.items():
        results_df[key] = value

    # Create group-individual table
    group_individual_table = create_group_individual_table(results_df, synthetic_data, all_group_keys)

    return results_df, group_individual_table


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
    global_iteration_limit: int = 100
    local_iteration_limit: int = 10
    model_type: str = "RandomForest"
    perturbation_unit: float = 1.0
    threshold: float = 0.0

    # BiassScan
    test_size: float = 0.3
    random_state: int = 42
    n_estimators: int = 200
    bias_scan_num_iters: int = 100
    bias_scan_scoring: str = 'Poisson'
    bias_scan_favorable_value: str = 'high'
    bias_scan_mode: str = 'ordinal'

    # ExpGA
    expga_threshold: float = 0.5
    expga_threshold_rank: float = 0.5
    max_global: int = 50
    max_local: int = 50

    # MLCheck
    mlcheck_iteration_no: int = 1

    def __post_init__(self):
        if self.methods is None:
            self.methods = {Method.AEQUITAS, Method.BIASSCAN, Method.EXPGA, Method.MLCHECK}
        elif isinstance(self.methods, list):
            self.methods = set(self.methods)


class ExperimentRunner:
    def __init__(self, db_path: str = "experiments.db"):
        # Convert db_path to Path object
        self.db_path = Path(db_path)

        # Create parent directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.setup_database()

    def setup_database(self):
        """Setup database tables using pandas DataFrame structures"""
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
            'analysis_metrics', 'created_at'
        ])

        # Create tables using to_sql
        with sqlite3.connect(self.db_path) as conn:
            experiments_df.to_sql('experiments', conn, if_exists='append', index=False)
            results_df.to_sql('results', conn, if_exists='append', index=False)
            analysis_df.to_sql('analysis_results', conn, if_exists='append', index=False)

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

    def resume_experiments(self):
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
                    continue

    def run_method(self, method: Method, ge, config: ExperimentConfig) -> tuple:
        import time
        from data_generator.main2 import generate_data
        from methods.aequitas.algo import run_aequitas
        from methods.biasscan.algo import run_bias_scan
        from methods.exp_ga.algo import run_expga
        from methods.ml_check.algo import run_mlcheck

        start_time = time.time()

        if method == Method.AEQUITAS:
            results_df, metrics = run_aequitas(
                ge.training_dataframe,
                col_to_be_predicted=ge.outcome_column,
                sensitive_param_name_list=ge.protected_attributes,
                perturbation_unit=config.perturbation_unit,
                model_type=config.model_type,
                threshold=config.threshold,
                global_iteration_limit=config.global_iteration_limit,
                local_iteration_limit=config.local_iteration_limit
            )
        elif method == Method.BIASSCAN:
            results_df, metrics = run_bias_scan(
                ge,
                test_size=config.test_size,
                random_state=config.random_state,
                n_estimators=config.n_estimators,
                bias_scan_num_iters=config.bias_scan_num_iters,
                bias_scan_scoring=config.bias_scan_scoring,
                bias_scan_favorable_value=config.bias_scan_favorable_value,
                bias_scan_mode=config.bias_scan_mode
            )
        elif method == Method.EXPGA:
            results_df, metrics = run_expga(
                ge,
                threshold=config.expga_threshold,
                threshold_rank=config.expga_threshold_rank,
                max_global=config.max_global,
                max_local=config.max_local
            )
        elif method == Method.MLCHECK:
            results_df, metrics = run_mlcheck(
                ge,
                iteration_no=config.mlcheck_iteration_no
            )

        execution_time = time.time() - start_time
        return results_df, metrics, execution_time

    def run_experiment(self, config: ExperimentConfig, existing_id: str = None):
        """Run experiment with duplicate checking"""
        if existing_id is None:  # New experiment
            # Check if experiment already exists
            existing_id = self.experiment_exists(config)
            if existing_id:
                logger.info(f"Experiment already completed with ID: {existing_id}")
                return existing_id

            experiment_id = str(uuid.uuid4())
        else:
            experiment_id = existing_id

        config_dict = vars(config).copy()
        config_dict['methods'] = [m.value for m in config_dict['methods']]

        if existing_id is None:  # Only save new experiment if it doesn't exist
            self.save_experiment(experiment_id, config_dict, 'running')

        try:
            from data_generator.main2 import generate_data

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

            for method in config.methods:
                logger.info(f"Running method: {method.value}")
                try:
                    results_df, metrics, execution_time = self.run_method(method, ge, config)
                    self.save_result(experiment_id, method.value, results_df, metrics, execution_time)
                    logger.info(f"Completed method: {method.value}")
                except Exception as e:
                    logger.error(f"Failed to run method {method.value}: {str(e)}")
                    continue

            self.update_experiment_status(experiment_id, 'completed')
            logger.info(f"Experiment completed successfully: {experiment_id}")

            logger.info(f"Analyzing results for experiment: {experiment_id}")
            self.analyze_experiment_results(experiment_id, ge.dataframe)  # Add synthetic data here

            logger.info(f"Experiment and analysis completed successfully: {experiment_id}")


        except Exception as e:
            self.update_experiment_status(experiment_id, 'failed', str(e))
            logger.error(f"Experiment failed: {str(e)}")
            raise

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
            'result_data': result_df.to_json(),
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

    def save_analysis_results(self, experiment_id: str, method_name: str,
                              evaluated_results: pd.DataFrame,
                              group_individual_table: pd.DataFrame,
                              analysis_metrics: Dict):
        """Save analysis results using pandas DataFrame"""
        analysis_data = pd.DataFrame([{
            'analysis_id': str(uuid.uuid4()),
            'experiment_id': experiment_id,
            'method_name': method_name,
            'evaluated_results': evaluated_results.to_json(),
            'group_individual_table': group_individual_table.to_json(),
            'analysis_metrics': json.dumps(analysis_metrics),
            'created_at': datetime.now()
        }])

        with sqlite3.connect(self.db_path) as conn:
            analysis_data.to_sql('analysis_results', conn, if_exists='append', index=False)

    def analyze_experiment_results(self, experiment_id: str, synthetic_data: pd.DataFrame):
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
                # Run evaluation
                evaluated_results, group_individual_table = evaluate_discrimination_detection(
                    synthetic_data, result_df
                )

                # Extract metrics from evaluated_results
                if not evaluated_results.empty:
                    metrics_columns = [
                        "Proportion of Original Individuals",
                        "Proportion of New Individuals",
                        "Proportion of original individuals that belong to a group",
                        "Proportion of new individuals that belong to a group",
                        "Correct Couple Detection Rate",
                        "Total Couples Detected",
                        "True Positives",
                        "False Positives",
                        "Proportion of Groups Detected"
                    ]
                    analysis_metrics = evaluated_results.iloc[0][metrics_columns].to_dict()
                else:
                    analysis_metrics = {}

                # Save analysis results
                self.save_analysis_results(
                    experiment_id,
                    method_name,
                    evaluated_results,
                    group_individual_table,
                    analysis_metrics
                )

                logger.info(f"Analysis completed for experiment {experiment_id}, method {method_name}")
                logger.info(f"Analysis metrics: {analysis_metrics}")

            except Exception as e:
                logger.error(f"Error analyzing results for method {method_name}: {str(e)}")
                continue


def create_test_configurations() -> List[ExperimentConfig]:
    base_config = {
        "min_number_of_classes": 2,
        "max_number_of_classes": 10,
        "nb_categories_outcome": 6
    }

    test_variations = [
        # Varying number of attributes (nb_attributes)
        {"nb_attributes": 3, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 5, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 8, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 15, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 20, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 30, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 50, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 75, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 100, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},

        # Varying proportion of protected attributes (prop_protected_attr)
        {"nb_attributes": 10, "prop_protected_attr": 0.05, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.1, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.15, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.25, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.3, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.4, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.5, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.6, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.8, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 1.0, "nb_groups": 50, "max_group_size": 100},

        # Varying number of discriminatory groups (nb_groups)
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 5, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 10, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 25, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 75, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 100, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 150, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 200, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 300, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 500, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 1000, "max_group_size": 100},

        # Varying max group size (max_group_size)
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 10},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 20},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 50},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 100},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 200},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 500},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 1000},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 2000},
        {"nb_attributes": 10, "prop_protected_attr": 0.2, "nb_groups": 50, "max_group_size": 5000},

        # Combined variations
        # High complexity scenarios
        {"nb_attributes": 50, "prop_protected_attr": 0.5, "nb_groups": 200, "max_group_size": 1000},
        {"nb_attributes": 75, "prop_protected_attr": 0.4, "nb_groups": 300, "max_group_size": 2000},
        {"nb_attributes": 100, "prop_protected_attr": 0.3, "nb_groups": 500, "max_group_size": 5000},

        # Low complexity scenarios
        {"nb_attributes": 3, "prop_protected_attr": 0.1, "nb_groups": 5, "max_group_size": 20},
        {"nb_attributes": 5, "prop_protected_attr": 0.15, "nb_groups": 10, "max_group_size": 50},
        {"nb_attributes": 8, "prop_protected_attr": 0.2, "nb_groups": 15, "max_group_size": 75},

        # Balanced scenarios
        {"nb_attributes": 15, "prop_protected_attr": 0.3, "nb_groups": 75, "max_group_size": 250},
        {"nb_attributes": 25, "prop_protected_attr": 0.35, "nb_groups": 100, "max_group_size": 500},
        {"nb_attributes": 35, "prop_protected_attr": 0.4, "nb_groups": 150, "max_group_size": 750},

        # Edge cases
        {"nb_attributes": 2, "prop_protected_attr": 0.01, "nb_groups": 1, "max_group_size": 10},
        # Minimal configuration
        {"nb_attributes": 100, "prop_protected_attr": 1.0, "nb_groups": 1000, "max_group_size": 5000},
        # Maximal configuration
        {"nb_attributes": 10, "prop_protected_attr": 0.5, "nb_groups": 50, "max_group_size": 100},
        # Perfectly balanced protected attributes
    ]

    configurations = []
    for variation in test_variations:
        config_dict = {**base_config, **variation}
        configurations.append(ExperimentConfig(**config_dict))

    return configurations


def run_experiments(configs: List[ExperimentConfig], methods: Set[Method] = None,
                    db_path: str = "experiments.db"):
    """Run experiments with duplicate prevention"""
    runner = ExperimentRunner(db_path)

    # Set methods for each config if specified
    if methods:
        for config in configs:
            config.methods = methods

    # Resume any incomplete experiments
    runner.resume_experiments()

    # Run new experiments, skipping duplicates
    for config in configs:
        existing_id = runner.experiment_exists(config)
        if existing_id:
            logger.info(f"Skipping duplicate experiment (ID: {existing_id})")
            continue

        experiment_id = runner.run_experiment(config)
        logger.info(f"Completed experiment: {experiment_id}")


# %%

DB_PATH = HERE.joinpath("experiments/discrimination_detection_results4.db").as_posix()
configs = create_test_configurations()
methods = {Method.EXPGA, Method.BIASSCAN, Method.AEQUITAS, Method.MLCHECK}  # Only run Aequitas and BiasScan
run_experiments(configs, methods=methods, db_path=DB_PATH)

# %%
import pandas as pd
import sqlite3
from pathlib import Path
import json
from typing import Optional, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


class ResultsAnalyzer:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)

    def get_all_experiments(self) -> pd.DataFrame:
        """Get summary of all experiments"""
        with sqlite3.connect(self.db_path) as conn:
            experiments = pd.read_sql_query("""
                SELECT 
                    experiment_id,
                    config,
                    methods,
                    status,
                    start_time,
                    end_time,
                    error
                FROM experiments
                """, conn)

            # Parse config JSON for better readability
            experiments['config'] = experiments['config'].apply(json.loads)

            # Add key configuration parameters as separate columns
            experiments['nb_attributes'] = experiments['config'].apply(lambda x: x.get('nb_attributes'))
            experiments['nb_groups'] = experiments['config'].apply(lambda x: x.get('nb_groups'))
            experiments['prop_protected_attr'] = experiments['config'].apply(lambda x: x.get('prop_protected_attr'))
            experiments['max_group_size'] = experiments['config'].apply(lambda x: x.get('max_group_size'))

        return experiments

    def get_experiment_results(self, experiment_id: str) -> Dict:
        """Get detailed results for a specific experiment"""
        with sqlite3.connect(self.db_path) as conn:
            # Get experiment details
            experiment = pd.read_sql_query(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                conn,
                params=(experiment_id,)
            )

            # Get results
            results = pd.read_sql_query(
                "SELECT * FROM results WHERE experiment_id = ?",
                conn,
                params=(experiment_id,)
            )

            # Get analysis results
            analysis = pd.read_sql_query(
                "SELECT * FROM analysis_results WHERE experiment_id = ?",
                conn,
                params=(experiment_id,)
            )

            # Parse JSON data
            if not experiment.empty:
                experiment['config'] = experiment['config'].apply(json.loads)
                experiment['methods'] = experiment['methods'].apply(json.loads)

            if not results.empty:
                results['metrics'] = results['metrics'].apply(json.loads)
                results['result_data'] = results['result_data'].apply(json.loads)
                results['result_data'] = results['result_data'].apply(pd.read_json)

            if not analysis.empty:
                analysis['analysis_metrics'] = analysis['analysis_metrics'].apply(json.loads)
                analysis['evaluated_results'] = analysis['evaluated_results'].apply(json.loads)
                analysis['evaluated_results'] = analysis['evaluated_results'].apply(pd.read_json)
                analysis['group_individual_table'] = analysis['group_individual_table'].apply(json.loads)
                analysis['group_individual_table'] = analysis['group_individual_table'].apply(pd.read_json)

        return {
            'experiment': experiment,
            'results': results,
            'analysis': analysis
        }

    def get_analysis_summary(self) -> pd.DataFrame:
        """Get summary of analysis results for all experiments"""
        with sqlite3.connect(self.db_path) as conn:
            # Get experiments with their configs
            experiments = pd.read_sql_query("""
                SELECT experiment_id, config 
                FROM experiments 
                WHERE status = 'completed'
                """, conn)

            # Get analysis results
            analysis = pd.read_sql_query("""
                SELECT * FROM analysis_results
                """, conn)

            # Check if we have any data
            if analysis.empty:
                logger.warning("No analysis results found in the database")
                return pd.DataFrame()

            if experiments.empty:
                logger.warning("No completed experiments found in the database")
                return pd.DataFrame()

            # Parse JSON data
            experiments['config'] = experiments['config'].apply(json.loads)
            analysis['analysis_metrics'] = analysis['analysis_metrics'].apply(json.loads)

            # Create a summary DataFrame
            summary_rows = []
            for _, row in analysis.iterrows():
                metrics = row['analysis_metrics']
                metrics['experiment_id'] = row['experiment_id']
                metrics['method_name'] = row['method_name']
                summary_rows.append(metrics)

            if not summary_rows:
                logger.warning("No metrics data found in analysis results")
                return pd.DataFrame()

            summary = pd.DataFrame(summary_rows)

            # Merge with experiment configurations
            try:
                summary = summary.merge(experiments, on='experiment_id', how='left')

                # Extract config parameters
                summary['nb_attributes'] = summary['config'].apply(lambda x: x.get('nb_attributes'))
                summary['nb_groups'] = summary['config'].apply(lambda x: x.get('nb_groups'))
                summary['prop_protected_attr'] = summary['config'].apply(lambda x: x.get('prop_protected_attr'))
                summary['max_group_size'] = summary['config'].apply(lambda x: x.get('max_group_size'))

            except Exception as e:
                logger.error(f"Error processing summary data: {str(e)}")
                logger.info(f"Summary columns: {summary.columns}")
                logger.info(f"Experiments columns: {experiments.columns}")
                return pd.DataFrame()

            return summary

    def plot_analysis_results(self, summary: Optional[pd.DataFrame] = None):
        """Plot analysis results with different parameters"""
        if summary is None:
            summary = self.get_analysis_summary()

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Analysis Results Overview', fontsize=16)

        # Plot 1: Detection Rate vs Number of Attributes
        sns.scatterplot(
            data=summary,
            x='nb_attributes',
            y='Correct Couple Detection Rate',
            hue='method_name',
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Detection Rate vs Number of Attributes')

        # Plot 2: Detection Rate vs Number of Groups
        sns.scatterplot(
            data=summary,
            x='nb_groups',
            y='Correct Couple Detection Rate',
            hue='method_name',
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('Detection Rate vs Number of Groups')

        # Plot 3: Detection Rate vs Protected Attributes Proportion
        sns.scatterplot(
            data=summary,
            x='prop_protected_attr',
            y='Correct Couple Detection Rate',
            hue='method_name',
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Detection Rate vs Protected Attributes Proportion')

        # Plot 4: Groups Detected vs Max Group Size
        sns.scatterplot(
            data=summary,
            x='max_group_size',
            y='Proportion of Groups Detected',
            hue='method_name',
            ax=axes[1, 1]
        )
        axes[1, 1].set_title('Groups Detected vs Max Group Size')

        plt.tight_layout()
        return fig


def check_database_content(db_path: str):
    """Check the content of the database tables"""
    with sqlite3.connect(db_path) as conn:
        # Check experiments table
        experiments = pd.read_sql_query("SELECT * FROM experiments", conn)
        print("\nExperiments table:")
        print(f"Total rows: {len(experiments)}")
        if not experiments.empty:
            print("Columns:", experiments.columns.tolist())
            print("\nStatus counts:")
            print(experiments['status'].value_counts())

        # Check results table
        results = pd.read_sql_query("SELECT * FROM results", conn)
        print("\nResults table:")
        print(f"Total rows: {len(results)}")
        if not results.empty:
            print("Columns:", results.columns.tolist())

        # Check analysis_results table
        analysis = pd.read_sql_query("SELECT * FROM analysis_results", conn)
        print("\nAnalysis results table:")
        print(f"Total rows: {len(analysis)}")
        if not analysis.empty:
            print("Columns:", analysis.columns.tolist())


# Usage example
def analyze_results(db_path: str):
    """Analyze results with better error handling"""
    analyzer = ResultsAnalyzer(db_path)

    # Get overall experiment summary
    print("\n=== Experiment Summary ===")
    experiments = analyzer.get_all_experiments()
    print(f"Total experiments: {len(experiments)}")
    print(f"Completed experiments: {len(experiments[experiments['status'] == 'completed'])}")
    print(f"Failed experiments: {len(experiments[experiments['status'] == 'failed'])}")

    # Get analysis summary
    print("\n=== Analysis Summary ===")
    summary = analyzer.get_analysis_summary()

    if summary.empty:
        print("No analysis results available yet.")
        return {
            'experiments': experiments,
            'summary': summary,
            'detailed_results': None,
            'plots': None
        }

    print("\nAverage metrics across all experiments:")
    metrics_columns = [col for col in [
        "Proportion of Original Individuals",
        "Proportion of New Individuals",
        "Correct Couple Detection Rate",
        "Proportion of Groups Detected"
    ] if col in summary.columns]

    if metrics_columns:
        print(summary[metrics_columns].mean())

    # Plot results only if we have data
    fig = None
    if not summary.empty and all(
            col in summary.columns for col in ['nb_attributes', 'nb_groups', 'prop_protected_attr', 'max_group_size']):
        try:
            fig = analyzer.plot_analysis_results(summary)
        except Exception as e:
            logger.error(f"Error creating plots: {str(e)}")

    # Get detailed results for a specific experiment (first completed experiment)
    detailed_results = None
    completed_exps = experiments[experiments['status'] == 'completed']
    if not completed_exps.empty:
        exp_id = completed_exps.iloc[0]['experiment_id']
        print(f"\n=== Detailed Results for Experiment {exp_id} ===")
        detailed_results = analyzer.get_experiment_results(exp_id)

        # Print some key metrics
        if detailed_results and 'analysis' in detailed_results and not detailed_results['analysis'].empty:
            analysis = detailed_results['analysis'].iloc[0]
            print("\nAnalysis Metrics:")
            print(json.dumps(analysis['analysis_metrics'], indent=2))

    return {
        'experiments': experiments,
        'summary': summary,
        'detailed_results': detailed_results,
        'plots': fig
    }

# Run the analysis
# DB_PATH = HERE.joinpath("experiments/discrimination_detection_results4.db").as_posix()
# check_database_content(DB_PATH)
# results = analyze_results(DB_PATH)
