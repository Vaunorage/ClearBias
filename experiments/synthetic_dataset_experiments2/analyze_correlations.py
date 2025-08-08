#!/usr/bin/env python3
import sqlite3
import json
from io import StringIO
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from path import HERE
from tqdm import tqdm

# Database files
DATABASE_FILE = HERE.joinpath('experiments/synthetic_dataset_experiments2/synthetic_experiments_optuna.db')
STUDIES_DATABASE_FILE = HERE.joinpath('experiments/synthetic_dataset_experiments2/optuna_studies.db')
ANALYSIS_DB_FILE = HERE.joinpath('experiments/synthetic_dataset_experiments2/analysis_results.db')
OUTPUT_DIR = HERE.joinpath('experiments/synthetic_dataset_experiments2/group_attributes_analysis')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define method types based on available tables in the database
METHOD_TYPES = {
    'adf': 'algorithm', 'aequitas': 'algorithm', 'bias_scan': 'subgroup', 'divexplorer': 'subgroup',
    'expga': 'algorithm', 'fliptest': 'algorithm', 'gerryfair': 'algorithm', 'kosei': 'algorithm',
    'limi': 'algorithm', 'naive_bayes': 'algorithm', 'sg': 'algorithm', 'slicefinder': 'subgroup',
    'sliceline': 'subgroup', 'verifair': 'subgroup'
}
ALL_METHODS = list(METHOD_TYPES.keys())


def add_group_means_and_drop_originals(df_data):
    """
    Add group-level means for epistemic and aleatoric columns and drop the original columns.

    Args:
        df_data: DataFrame with epistemic and aleatoric columns

    Returns:
        DataFrame with group means added and original columns dropped
    """
    # Identify columns that contain 'epistemic' or 'aleatoric'
    epistemic_aleatoric_cols = [col for col in df_data.columns
                                if 'epistemic' in col.lower() or 'aleatoric' in col.lower()]

    print(f"Found {len(epistemic_aleatoric_cols)} epistemic/aleatoric columns: {epistemic_aleatoric_cols}")

    # Calculate group means for these columns
    group_means = df_data.groupby('group_key')[epistemic_aleatoric_cols].mean()

    # Create new column names with '_group_mean' suffix
    group_mean_cols = {col: f"{col}_group_mean" for col in epistemic_aleatoric_cols}
    group_means = group_means.rename(columns=group_mean_cols)

    # Merge the group means back to the original dataframe
    df_data = df_data.merge(group_means, left_on='group_key', right_index=True, how='left')

    # Drop the original epistemic and aleatoric columns
    df_data = df_data.drop(columns=epistemic_aleatoric_cols)

    return df_data


# Modified load_trial_data function with the group mean calculation
def load_trial_data(method_name):
    """Load all trial data for a specific method from the database."""
    conn = sqlite3.connect(DATABASE_FILE)
    analysis_conn = sqlite3.connect(ANALYSIS_DB_FILE)

    # Check if the table exists before querying
    table_name = f"{method_name}_results"
    check_query = f"""
    SELECT name FROM sqlite_master 
    WHERE type='table' AND name='{table_name}';
    """

    table_exists = pd.read_sql_query(check_query, conn).shape[0] > 0

    if not table_exists:
        print(f"Table '{table_name}' does not exist in the database. Skipping.")
        conn.close()
        analysis_conn.close()
        return None

    query = f"""
    SELECT 
        trial_number, trial_params, trial_value, trial_state, all_metrics, dataset_params, dataset_content
    FROM 
        "{table_name}"
    """

    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        print(f"Error querying table '{table_name}': {str(e)}")
        conn.close()
        analysis_conn.close()
        return None

    if df.empty:
        print(f"No data found for method: {method_name}")
        analysis_conn.close()
        return None

    # Load analysis data for this method
    query_analysis = f"""
        SELECT * 
        FROM detailed_pattern_matches
        WHERE method = '{method_name}'
    """
    df_analysis = pd.read_sql_query(query_analysis, analysis_conn)
    analysis_conn.close()

    datasets = []
    for idx, data_row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {method_name} trials"):
        df_data = pd.read_json(StringIO(data_row['dataset_content']), orient='split')
        df_data = df_data[['group_key', 'subgroup_key', 'indv_key'] +
                          list(filter(lambda x: 'calculated' in x, list(df_data.columns)))].drop_duplicates()

        # Parse the JSON strings
        algo_params = json.loads(data_row['trial_params'])
        res_metrics = json.loads(data_row['all_metrics'])
        dataset_params = json.loads(data_row['dataset_params'])

        # Add algo_params as columns (prefix with 'algo_')
        for key, value in algo_params.items():
            df_data[f'algo_{key}'] = value

        # Add res_metrics as columns (prefix with 'metric_')
        for key, value in res_metrics.items():
            df_data[f'metric_{key}'] = value

        # Add dataset_params as columns (prefix with 'dataset_')
        for key, value in dataset_params.items():
            df_data[f'dataset_{key}'] = value

        # Add trial_number to df_data for merging
        df_data['trial_number'] = data_row['trial_number']

        # ADD GROUP MEANS AND DROP ORIGINAL COLUMNS
        df_data = add_group_means_and_drop_originals(df_data)

        # Filter analysis data for this specific trial
        trial_analysis = df_analysis[df_analysis['trial_index'] == data_row['trial_number']].copy()

        trial_analysis_unique = trial_analysis[
            ['matched_ground_truth', 'num_matches', 'total_ground_truth', 'pattern_type']].drop_duplicates(
            subset=['matched_ground_truth'])

        # Left merge df_data with analysis data
        if not trial_analysis.empty:
            df_data = df_data.merge(
                trial_analysis_unique,
                left_on='group_key',
                right_on='matched_ground_truth',
                how='left',
                suffixes=('', '_analysis')
            )

        # Ensure required columns exist even if df_analysis is empty
        required_columns = ['matched_ground_truth', 'num_matches', 'total_ground_truth', 'pattern_type']
        for col in required_columns:
            if col not in df_data.columns:
                df_data[col] = 0

        # Replace NaN values with 0 for the specified columns
        for col in required_columns:
            if col in df_data.columns:
                df_data[col] = df_data[col].fillna(0)

        datasets.append(df_data)

    return datasets


def analyze_correlations(df_data, result_columns, output_prefix, method_name=None):
    """Analyze correlations between calculated_ columns and specified result columns.

    Args:
        df_data: DataFrame containing calculated_ columns and result columns
        result_columns: List of result column names to analyze correlations with
        output_prefix: Prefix for output file names
        method_name: Name of the method being analyzed

    Returns:
        Dictionary containing correlation DataFrames
    """
    if df_data is None or df_data.empty:
        print("No data available for correlation analysis")
        return None

    # Get all calculated_ columns
    calculated_cols = [col for col in df_data.columns if col.startswith('calculated_')]

    if not calculated_cols:
        print("No calculated_ columns found in the data")
        return None

    # Check if result columns exist in the dataframe
    existing_result_cols = [col for col in result_columns if col in df_data.columns]
    if not existing_result_cols:
        print(f"None of the specified result columns {result_columns} found in the data")
        return None

    print(
        f"Analyzing correlations between {len(calculated_cols)} calculated columns and {len(existing_result_cols)} result columns")

    # Calculate correlations using both Pearson and Spearman methods
    pearson_correlations = pd.DataFrame()
    spearman_correlations = pd.DataFrame()

    for result_col in existing_result_cols:
        # Calculate Pearson correlation
        pearson_col_correlations = df_data[calculated_cols].corrwith(df_data[result_col], method='pearson')
        pearson_correlations[result_col] = pearson_col_correlations

        # Calculate Spearman correlation
        spearman_col_correlations = df_data[calculated_cols].corrwith(df_data[result_col], method='spearman')
        spearman_correlations[result_col] = spearman_col_correlations

    # Sort by absolute correlation values
    pearson_correlations['max_abs_corr'] = pearson_correlations.abs().max(axis=1)
    pearson_correlations = pearson_correlations.sort_values('max_abs_corr', ascending=False)
    pearson_correlations = pearson_correlations.drop('max_abs_corr', axis=1)

    spearman_correlations['max_abs_corr'] = spearman_correlations.abs().max(axis=1)
    spearman_correlations = spearman_correlations.sort_values('max_abs_corr', ascending=False)
    spearman_correlations = spearman_correlations.drop('max_abs_corr', axis=1)

    # Save correlation results to SQLite database
    analysis_conn = sqlite3.connect(ANALYSIS_DB_FILE)

    # Save Pearson correlations
    pearson_df = pearson_correlations.reset_index()
    pearson_df.rename(columns={'index': 'feature'}, inplace=True)
    pearson_df['method'] = method_name if method_name else 'combined'
    pearson_df['correlation_type'] = 'pearson'
    pearson_df['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')  # Convert to string format

    # Save Spearman correlations
    spearman_df = spearman_correlations.reset_index()
    spearman_df.rename(columns={'index': 'feature'}, inplace=True)
    spearman_df['method'] = method_name if method_name else 'combined'
    spearman_df['correlation_type'] = 'spearman'
    spearman_df['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')  # Convert to string format

    # Create table if it doesn't exist
    analysis_conn.execute('''
                          CREATE TABLE IF NOT EXISTS feature_correlations
                          (
                              id
                              INTEGER
                              PRIMARY
                              KEY
                              AUTOINCREMENT,
                              method
                              TEXT,
                              feature
                              TEXT,
                              correlation_type
                              TEXT,
                              timestamp
                              TIMESTAMP,
                              correlation_data
                              JSON
                          )
                          ''')

    # Insert Pearson correlations
    for _, row in pearson_df.iterrows():
        feature = row['feature']
        method = row['method']
        corr_type = row['correlation_type']
        timestamp = row['timestamp']

        # Convert correlation data to JSON
        corr_data = {}
        for col in existing_result_cols:
            corr_data[col] = row[col]

        corr_json = json.dumps(corr_data)

        analysis_conn.execute('''
                              INSERT INTO feature_correlations (method, feature, correlation_type, timestamp, correlation_data)
                              VALUES (?, ?, ?, ?, ?)
                              ''', (method, feature, corr_type, timestamp, corr_json))

    # Insert Spearman correlations
    for _, row in spearman_df.iterrows():
        feature = row['feature']
        method = row['method']
        corr_type = row['correlation_type']
        timestamp = row['timestamp']

        # Convert correlation data to JSON
        corr_data = {}
        for col in existing_result_cols:
            corr_data[col] = row[col]

        corr_json = json.dumps(corr_data)

        analysis_conn.execute('''
                              INSERT INTO feature_correlations (method, feature, correlation_type, timestamp, correlation_data)
                              VALUES (?, ?, ?, ?, ?)
                              ''', (method, feature, corr_type, timestamp, corr_json))

    analysis_conn.commit()
    print(f"Correlation results saved to database table: feature_correlations")

    # Create and save heatmaps for better visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(pearson_correlations, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'Pearson Correlation between Calculated Features and {", ".join(existing_result_cols)}')
    plt.tight_layout()
    pearson_heatmap_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_pearson_heatmap.png")
    plt.savefig(pearson_heatmap_path)
    plt.close()

    plt.figure(figsize=(12, 10))
    sns.heatmap(spearman_correlations, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'Spearman Correlation between Calculated Features and {", ".join(existing_result_cols)}')
    plt.tight_layout()
    spearman_heatmap_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_spearman_heatmap.png")
    plt.savefig(spearman_heatmap_path)
    plt.close()

    print(f"Correlation heatmaps saved to: {pearson_heatmap_path} and {spearman_heatmap_path}")
    analysis_conn.close()

    return {
        'pearson': pearson_correlations,
        'spearman': spearman_correlations
    }


def calculate_treatment_effect(df, treatment, outcome_col):
    """Calculate the Average Treatment Effect (ATE) of a treatment on an outcome.

    Args:
        df: DataFrame containing treatment and outcome columns
        treatment: Name of the treatment column
        outcome_col: Name of the outcome column

    Returns:
        Dictionary with ATE and related statistics
    """
    treatment_median = df[treatment].median()
    T = (df[treatment] > treatment_median).astype(int)
    y = df[outcome_col]
    treated_mean = y[T == 1].mean()
    control_mean = y[T == 0].mean()
    ate = treated_mean - control_mean
    return {
        'treatment': treatment,
        'outcome': outcome_col,
        'ATE': ate,
        'treated_mean': treated_mean,
        'control_mean': control_mean,
        'treatment_median': treatment_median
    }


def calculate_cate(df_data, treatment_col, outcome_col, covariates=None):
    """Calculate Conditional Average Treatment Effect (CATE) for a treatment.


    Args:
        df_data: DataFrame with treatment and outcome columns
        treatment_col: Name of the treatment column
        outcome_col: Name of the outcome column
        covariates: List of covariate column names to condition on

    Returns:
        DataFrame with CATE values for different covariate strata
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    # If no covariates provided, use all calculated_ columns except the treatment
    if covariates is None:
        covariates = [col for col in df_data.columns
                      if col.startswith('calculated_') and col != treatment_col]

    # Create binary treatment indicator (above/below median)
    treatment_median = df_data[treatment_col].median()
    df_data['T'] = (df_data[treatment_col] > treatment_median).astype(int)

    # Prepare data for the model
    X = df_data[covariates].copy()
    y = df_data[outcome_col]
    T = df_data['T']

    # Split data for training and evaluation
    X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(
        X, y, T, test_size=0.3, random_state=42)

    # Train models for treated and control groups
    model_treated = RandomForestRegressor(n_estimators=100, random_state=42)
    model_control = RandomForestRegressor(n_estimators=100, random_state=42)

    model_treated.fit(X_train[T_train == 1], y_train[T_train == 1])
    model_control.fit(X_train[T_train == 0], y_train[T_train == 0])

    # Predict outcomes for all test instances under both treatment conditions
    y_pred_treated = model_treated.predict(X_test)
    y_pred_control = model_control.predict(X_test)

    # Calculate individual treatment effects
    ite = y_pred_treated - y_pred_control

    # Create a DataFrame with covariates and ITE
    cate_df = X_test.copy()
    cate_df['ITE'] = ite
    cate_df['actual_outcome'] = y_test.values
    cate_df['actual_treatment'] = T_test.values

    # Calculate feature importance for treatment effect heterogeneity
    feature_importance = pd.DataFrame()
    feature_importance['feature'] = covariates
    feature_importance['treated_importance'] = model_treated.feature_importances_
    feature_importance['control_importance'] = model_control.feature_importances_
    feature_importance['diff_importance'] = abs(model_treated.feature_importances_ -
                                                model_control.feature_importances_)
    feature_importance = feature_importance.sort_values('diff_importance', ascending=False)

    # Calculate CATE for different strata of the most important features
    top_features = feature_importance.head(min(3, len(feature_importance)))['feature'].tolist()
    cate_by_strata = []

    for feature in top_features:
        # Create quartiles for the feature
        try:
            # Try to create quartiles, handling duplicate values
            df_data['strata'] = pd.qcut(df_data[feature], 4, labels=False, duplicates='drop')
        except ValueError as e:
            # If we still have issues, try a different approach with equal-width bins
            print(f"Warning: Could not create quartiles for {feature}: {str(e)}")
            print(f"Using equal-width bins instead of quantiles for {feature}")
            df_data['strata'] = pd.cut(df_data[feature], 4, labels=False)
        strata_cates = []

        for strata in range(4):
            strata_data = df_data[df_data['strata'] == strata]
            if len(strata_data) > 10:  # Ensure enough data in each stratum
                treated = strata_data[strata_data['T'] == 1][outcome_col].mean()
                control = strata_data[strata_data['T'] == 0][outcome_col].mean()
                cate = treated - control
                strata_cates.append({
                    'feature': feature,
                    'strata': strata,
                    'strata_range': f"Q{strata + 1}",
                    'cate': cate,
                    'treated_mean': treated,
                    'control_mean': control,
                    'sample_size': len(strata_data)
                })

        cate_by_strata.extend(strata_cates)

    return {
        'treatment': treatment_col,
        'outcome': outcome_col,
        'feature_importance': feature_importance,
        'cate_by_strata': pd.DataFrame(cate_by_strata),
        'individual_treatment_effects': cate_df
    }


def analyze_cate(df_data, treatment_cols, outcome_cols, output_prefix, method_name=None):
    """Analyze Conditional Average Treatment Effects between treatment and outcome columns.

    Args:
        df_data: DataFrame containing treatment and outcome columns
        treatment_cols: List of treatment column names (calculated_ columns)
        outcome_cols: List of outcome column names (result columns)
        output_prefix: Prefix for output file names
        method_name: Name of the method being analyzed

    Returns:
        Dictionary containing CATE analysis results
    """
    if df_data is None or df_data.empty:
        print("No data available for CATE analysis")
        return None

    # Check if columns exist in the dataframe
    existing_treatment_cols = [col for col in treatment_cols if col in df_data.columns]
    existing_outcome_cols = [col for col in outcome_cols if col in df_data.columns]

    if not existing_treatment_cols or not existing_outcome_cols:
        print(f"Missing treatment or outcome columns for CATE analysis")
        return None

    print(
        f"Analyzing CATE between {len(existing_treatment_cols)} treatment columns and {len(existing_outcome_cols)} outcome columns")

    # Connect to the database
    analysis_conn = sqlite3.connect(ANALYSIS_DB_FILE)

    # Create tables if they don't exist
    analysis_conn.execute('''
                          CREATE TABLE IF NOT EXISTS treatment_effects
                          (
                              id
                              INTEGER
                              PRIMARY
                              KEY
                              AUTOINCREMENT,
                              method
                              TEXT,
                              treatment
                              TEXT,
                              outcome
                              TEXT,
                              effect_type
                              TEXT,
                              ate
                              REAL,
                              treated_mean
                              REAL,
                              control_mean
                              REAL,
                              treatment_median
                              REAL,
                              timestamp
                              TIMESTAMP
                          )
                          ''')

    analysis_conn.execute('''
                          CREATE TABLE IF NOT EXISTS cate_strata
                          (
                              id
                              INTEGER
                              PRIMARY
                              KEY
                              AUTOINCREMENT,
                              method
                              TEXT,
                              treatment
                              TEXT,
                              outcome
                              TEXT,
                              feature
                              TEXT,
                              strata
                              INTEGER,
                              strata_range
                              TEXT,
                              cate
                              REAL,
                              treated_mean
                              REAL,
                              control_mean
                              REAL,
                              sample_size
                              INTEGER,
                              timestamp
                              TIMESTAMP
                          )
                          ''')

    analysis_conn.execute('''
                          CREATE TABLE IF NOT EXISTS feature_importance
                          (
                              id
                              INTEGER
                              PRIMARY
                              KEY
                              AUTOINCREMENT,
                              method
                              TEXT,
                              treatment
                              TEXT,
                              outcome
                              TEXT,
                              feature
                              TEXT,
                              treated_importance
                              REAL,
                              control_importance
                              REAL,
                              diff_importance
                              REAL,
                              timestamp
                              TIMESTAMP
                          )
                          ''')

    # Store results
    ate_results = []
    cate_results = {}
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')  # Convert to string format

    # For each treatment-outcome pair, calculate ATE and CATE
    for treatment_col in existing_treatment_cols:
        for outcome_col in existing_outcome_cols:
            print(f"Analyzing treatment effect of {treatment_col} on {outcome_col}...")

            # Skip if either column has too many NaN values
            if df_data[treatment_col].isna().sum() > 0.5 * len(df_data) or \
                    df_data[outcome_col].isna().sum() > 0.5 * len(df_data):
                print(f"Skipping due to too many NaN values")
                continue

            # Calculate ATE
            ate = calculate_treatment_effect(df_data, treatment_col, outcome_col)
            ate_results.append(ate)

            # Save ATE to database
            analysis_conn.execute('''
                                  INSERT INTO treatment_effects
                                  (method, treatment, outcome, effect_type, ate, treated_mean, control_mean,
                                   treatment_median, timestamp)
                                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                  ''', (
                                      method_name if method_name else 'combined',
                                      treatment_col,
                                      outcome_col,
                                      'ATE',
                                      ate['ATE'],
                                      ate['treated_mean'],
                                      ate['control_mean'],
                                      ate['treatment_median'],
                                      timestamp
                                  ))

            # Calculate CATE if ATE is significant enough
            if abs(ate['ATE']) > 0.05:  # Arbitrary threshold, adjust as needed
                try:
                    cate = calculate_cate(df_data, treatment_col, outcome_col)
                    cate_results[f"{treatment_col}_{outcome_col}"] = cate

                    # Save feature importance to database
                    for _, row in cate['feature_importance'].iterrows():
                        analysis_conn.execute('''
                                              INSERT INTO feature_importance
                                              (method, treatment, outcome, feature, treated_importance,
                                               control_importance, diff_importance, timestamp)
                                              VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                              ''', (
                                                  method_name if method_name else 'combined',
                                                  treatment_col,
                                                  outcome_col,
                                                  row['feature'],
                                                  row['treated_importance'],
                                                  row['control_importance'],
                                                  row['diff_importance'],
                                                  timestamp
                                              ))

                    # Save CATE by strata to database
                    for _, row in cate['cate_by_strata'].iterrows():
                        analysis_conn.execute('''
                                              INSERT INTO cate_strata
                                              (method, treatment, outcome, feature, strata, strata_range, cate,
                                               treated_mean, control_mean, sample_size, timestamp)
                                              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                              ''', (
                                                  method_name if method_name else 'combined',
                                                  treatment_col,
                                                  outcome_col,
                                                  row['feature'],
                                                  row['strata'],
                                                  row['strata_range'],
                                                  row['cate'],
                                                  row['treated_mean'],
                                                  row['control_mean'],
                                                  row['sample_size'],
                                                  timestamp
                                              ))

                    # Create visualization of CATE by strata
                    if not cate['cate_by_strata'].empty:
                        plt.figure(figsize=(12, 8))
                        sns.barplot(x='strata_range', y='cate', hue='feature',
                                    data=cate['cate_by_strata'])
                        plt.title(f'CATE of {treatment_col} on {outcome_col} by Feature Strata')
                        plt.ylabel('Conditional Average Treatment Effect')
                        plt.xlabel('Feature Quartile')
                        plt.tight_layout()
                        cate_plot_path = os.path.join(OUTPUT_DIR,
                                                      f"{output_prefix}_cate_{treatment_col}_{outcome_col}_plot.png")
                        plt.savefig(cate_plot_path)
                        plt.close()

                        print(f"CATE analysis for {treatment_col} on {outcome_col} saved to database")
                except Exception as e:
                    print(f"Error calculating CATE for {treatment_col} on {outcome_col}: {str(e)}")

    # Commit all database changes
    analysis_conn.commit()
    analysis_conn.close()

    print(f"Treatment effects saved to database tables: treatment_effects, cate_strata, feature_importance")
    print(f"CATE analysis completed for {len(cate_results)} treatment-outcome pairs")

    # Convert ATE results to DataFrame for return value
    ate_df = pd.DataFrame(ate_results)

    return {
        'ate_results': ate_df,
        'cate_results': cate_results
    }


def main():
    """Main function to run the analysis."""
    # Get list of available tables from the database
    conn = sqlite3.connect(DATABASE_FILE)
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_results';"
    available_tables = pd.read_sql_query(query, conn)['name'].tolist()
    available_tables = ['bias_scan_results']
    conn.close()

    # Extract method names from table names (remove _results suffix)
    methods = [table_name.replace('_results', '') for table_name in available_tables]
    print(f"Found {len(methods)} available methods in the database: {methods}")

    datasets = {}

    # Make sure we have the json module imported
    import json

    # Load data for each available method
    for method in methods:
        print(f"Loading data for method: {method}")
        datasets[method] = load_trial_data(method)

    # Combine all datasets - each dataset is a list of DataFrames, so we need to flatten the list
    all_dataframes = []
    for method_data in datasets.values():
        if method_data is not None:  # Check if the method data exists
            all_dataframes.extend(method_data)  # Extend the list with all DataFrames from this method

    # Now concatenate all the DataFrames
    if all_dataframes:
        all_data = pd.concat(all_dataframes)
    else:
        print("No data available for analysis.")
        return

    # Define result columns to analyze correlations with
    result_columns = ['metric_DSN', 'metric_TSN', 'total_ground_truth', 'num_matches']

    # Analyze correlations for each method
    for method, method_data in datasets.items():
        if method_data is not None and len(method_data) > 0:
            # Concatenate all DataFrames for this method
            method_df = pd.concat(method_data)
            method_df = method_df.head(100000)

            print(f"\nAnalyzing correlations for method: {method}")
            correlations = analyze_correlations(method_df, result_columns, f"{method}", method_name=method)

            if correlations:
                # Get top correlated features for CATE analysis
                top_features = correlations['pearson'].index[:5].tolist()  # Top 5 features by Pearson correlation

                # Run CATE analysis on top correlated features
                print(f"\nRunning CATE analysis for method: {method}")
                analyze_cate(method_df, top_features, result_columns, f"{method}", method_name=method)

    # Analyze correlations for combined dataset
    print("\nAnalyzing correlations for combined dataset")
    correlations = analyze_correlations(all_data, result_columns, "combined", method_name="combined")

    if correlations:
        # Get top correlated features for CATE analysis
        top_features = correlations['pearson'].index[:5].tolist()  # Top 5 features by Pearson correlation

        # Run CATE analysis on top correlated features for combined dataset
        print("\nRunning CATE analysis for combined dataset")
        analyze_cate(all_data, top_features, result_columns, "combined", method_name="combined")

        # For backward compatibility, also run the analysis on all methods combined using the old approach
        all_datasets = []
        for method in ALL_METHODS:
            method_data = load_trial_data(method)
            if method_data and len(method_data) > 0:
                all_datasets.extend(method_data)

        if all_datasets:
            combined_all_df = pd.concat(all_datasets, ignore_index=True)
            print("\nAnalyzing correlations across all methods combined...")
            correlations = analyze_correlations(combined_all_df, result_columns, "all_methods_combined",
                                                method_name="all_methods_combined")

            if correlations:
                # Perform CATE analysis on combined data
                print("\nPerforming CATE analysis across all methods combined...")
                # Get top correlated features based on correlation analysis
                top_treatments = correlations['pearson'].index[:min(5, len(correlations['pearson']))].tolist()
                analyze_cate(combined_all_df, top_treatments, result_columns, "all_methods_combined",
                             method_name="all_methods_combined")

        print("\nAnalysis complete. Results saved to:", OUTPUT_DIR)
    else:
        print("Failed to build complete dataframe. Analysis aborted.")


if __name__ == "__main__":
    main()
