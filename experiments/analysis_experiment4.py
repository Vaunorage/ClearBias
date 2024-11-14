#%%
from sqlalchemy import create_engine
import pandas as pd

from path import HERE

DB_PATH = HERE.joinpath('experiments/discrimination_detection_results5.db')
conn = create_engine(f'sqlite:///{DB_PATH}')
#%%
query = """
SELECT ex.experiment_id,
       am.analysis_id,
       am.method_name,
       ex.status,
       json_extract(ex.config, '$.aequitas_global_iteration_limit') as aequitas_global_iteration_limit,
       json_extract(ex.config, '$.aequitas_local_iteration_limit')  as aequitas_local_iteration_limit,
       json_extract(ex.config, '$.aequitas_model_type')             as aequitas_model_type,
       json_extract(ex.config, '$.aequitas_perturbation_unit')      as aequitas_perturbation_unit,
       json_extract(ex.config, '$.aequitas_threshold')              as aequitas_threshold,
       json_extract(ex.config, '$.bias_scan_favorable_value')       as bias_scan_favorable_value,
       json_extract(ex.config, '$.bias_scan_mode')                  as bias_scan_mode,
       json_extract(ex.config, '$.bias_scan_n_estimators')          as bias_scan_n_estimators,
       json_extract(ex.config, '$.bias_scan_num_iters')             as bias_scan_num_iters,
       json_extract(ex.config, '$.bias_scan_random_state')          as bias_scan_random_state,
       json_extract(ex.config, '$.bias_scan_scoring')               as bias_scan_scoring,
       json_extract(ex.config, '$.bias_scan_test_size')             as bias_scan_test_size,
       json_extract(ex.config, '$.expga_max_global')                as expga_max_global,
       json_extract(ex.config, '$.expga_max_local')                 as expga_max_local,
       json_extract(ex.config, '$.expga_threshold')                 as expga_threshold,
       json_extract(ex.config, '$.expga_threshold_rank')            as expga_threshold_rank,
       json_extract(ex.config, '$.max_group_size')                  as max_group_size,
       json_extract(ex.config, '$.max_number_of_classes')           as max_number_of_classes,
       json_extract(ex.config, '$.min_number_of_classes')           as min_number_of_classes,
       json_extract(ex.config, '$.mlcheck_iteration_no')            as mlcheck_iteration_no,
       json_extract(ex.config, '$.nb_attributes')                   as nb_attributes,
       json_extract(ex.config, '$.nb_categories_outcome')           as nb_categories_outcome,
       json_extract(ex.config, '$.nb_groups')                       as nb_groups,
       json_extract(ex.config, '$.prop_protected_attr')             as prop_protected_attr,

       er.calculated_aleatoric,
       er.calculated_aleatoric_max,
       er.calculated_aleatoric_median,
       er.calculated_aleatoric_min,

       er.calculated_epistemic,
       er.calculated_epistemic_max,
       er.calculated_epistemic_median,
       er.calculated_epistemic_min,

       er.calculated_granularity,
       er.calculated_group_size,
       er.calculated_intersectionality,
       er.calculated_magnitude,
       er.calculated_similarity,
       er.calculated_subgroup_ratio,
       er.calculated_uncertainty,

       er.synthetic_group_size,
       er.nb_unique_indv,

       er.num_exact_couple_matches,
       er.num_new_group_couples

FROM experiments ex
         join main.analysis_metadata am on ex.experiment_id = am.experiment_id
         join main.evaluated_results er on am.analysis_id = er.analysis_id
WHERE status = 'completed'
"""

df = pd.read_sql_query(query, conn)

df['num_exact_couple_matches'] = pd.to_numeric(df['num_exact_couple_matches'])
df['num_new_group_couples'] = pd.to_numeric(df['num_new_group_couples'])

#%%
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


def filter_columns_by_method(df):
    """
    Filter columns based on the method name, removing irrelevant method-specific columns.

    Parameters:
    df (pandas.DataFrame): Input DataFrame

    Returns:
    pandas.DataFrame: DataFrame with filtered columns
    """
    method_prefixes = {
        'aequitas': ['aequitas_'],
        'ml_check': ['mlcheck_'],
        'bias_scan': ['bias_scan_'],
        'expga': ['expga_']
    }

    # Create a copy of the DataFrame
    filtered_df = df.copy()

    # Process each row based on its method
    for idx in filtered_df.index:
        current_method = filtered_df.at[idx, 'method_name']

        # Get columns to drop for this method
        columns_to_drop = []
        for method, prefixes in method_prefixes.items():
            if method != current_method:
                # Add columns that start with other method prefixes to drop list
                for prefix in prefixes:
                    columns_to_drop.extend([col for col in filtered_df.columns if col.startswith(prefix)])

        # Set irrelevant method-specific columns to NaN for this row
        filtered_df.loc[idx, columns_to_drop] = np.nan

    return filtered_df


def analyze_correlations(df, target_column='num_exact_couple_matches'):
    """
    Analyze correlations between all columns and a target column,
    handling both numerical and categorical data.

    Parameters:
    df (pandas.DataFrame): Input DataFrame
    target_column (str): Column to correlate against

    Returns:
    tuple: (numerical correlations DataFrame, categorical correlations DataFrame)
    """
    # Initialize empty DataFrames for results
    numerical_df = pd.DataFrame()
    categorical_df = pd.DataFrame()

    # Filter columns based on method first
    df = filter_columns_by_method(df)

    # Remove columns that are all NaN after filtering
    df = df.dropna(axis=1, how='all')

    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    # Initialize results dictionaries
    numerical_correlations = {}
    categorical_correlations = {}

    # Calculate correlations for numerical columns
    if target_column in numerical_cols:
        for col in numerical_cols:
            if col != target_column:
                # Skip columns with all NaN values
                if df[col].notna().any():
                    try:
                        # Calculate Pearson correlation
                        correlation, p_value = stats.pearsonr(
                            df[col].dropna(),
                            df[target_column][df[col].notna()]  # Use matching indices
                        )
                        numerical_correlations[col] = {
                            'correlation': correlation,
                            'p_value': p_value
                        }
                    except ValueError:
                        continue  # Skip if correlation can't be calculated

        # Convert to DataFrame and sort by absolute correlation
        if numerical_correlations:
            numerical_df = pd.DataFrame.from_dict(numerical_correlations, orient='index')
            numerical_df = numerical_df.sort_values(
                by='correlation',
                key=abs,
                ascending=False
            )

    # Calculate correlations for categorical columns using Cramer's V
    for col in categorical_cols:
        if col != target_column:
            # Skip columns with all NaN values
            if df[col].notna().any():
                try:
                    # Create contingency table
                    contingency = pd.crosstab(df[col], df[target_column])

                    # Calculate Chi-square test
                    chi2, p_value = stats.chi2_contingency(contingency)[:2]

                    # Calculate Cramer's V
                    n = contingency.sum().sum()
                    min_dim = min(contingency.shape) - 1
                    cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

                    categorical_correlations[col] = {
                        'cramers_v': cramer_v,
                        'p_value': p_value
                    }
                except ValueError:
                    continue  # Skip if correlation can't be calculated

    # Convert to DataFrame and sort by Cramer's V
    if categorical_correlations:
        categorical_df = pd.DataFrame.from_dict(categorical_correlations, orient='index')
        categorical_df = categorical_df.sort_values(by='cramers_v', ascending=False)

    return numerical_df, categorical_df


def plot_correlations(numerical_df, categorical_df, target_column, method=None):
    """
    Create visualizations for correlations.

    Parameters:
    numerical_df (pandas.DataFrame): DataFrame with numerical correlations
    categorical_df (pandas.DataFrame): DataFrame with categorical correlations
    target_column (str): Name of the target column
    method (str, optional): If specified, add method name to plot title
    """
    method_str = f" ({method})" if method else ""

    # Plot numerical correlations
    if not numerical_df.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=numerical_df.index, y='correlation', data=numerical_df)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Numerical Correlations with {target_column}{method_str}')
        plt.tight_layout()
        plt.show()

    # Plot categorical correlations
    if not categorical_df.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=categorical_df.index, y='cramers_v', data=categorical_df)
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Categorical Correlations (Cramer's V) with {target_column}{method_str}")
        plt.tight_layout()
        plt.show()

#%%

def plot_correlations(numerical_df, categorical_df, target_column, method=None):
    """
    Create visualizations for correlations with enhanced direction indication.
    """
    method_str = f" ({method})" if method else ""

    # Plot numerical correlations with color-coding
    if not numerical_df.empty:
        # Debug print
        print("\nDebug - Correlation values:")
        print(numerical_df['correlation'])

        plt.figure(figsize=(15, 8))

        # Get correlation values and indices
        correlations = numerical_df['correlation'].values
        indices = range(len(correlations))

        # Create color list based on correlation values
        colors = ['forestgreen' if c > 0 else 'crimson' for c in correlations]

        # Create the bar plot
        bars = plt.bar(indices, correlations, color=colors)

        # Set x-axis labels
        plt.xticks(indices, numerical_df.index, rotation=45, ha='right')

        # Add a horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, zorder=1)

        # Add title and labels
        plt.title(f'Numerical Correlations with {target_column}{method_str}')
        plt.ylabel('Correlation Coefficient')

        # Add value labels on top of each bar
        for i, v in enumerate(correlations):
            label_color = 'black'
            if abs(v) < 0.2:  # If the bar is very short, use black for better visibility
                label_color = 'black'
            plt.text(i, v, f'{v:.3f}',
                     ha='center',
                     va='bottom' if v > 0 else 'top',
                     fontsize=9,
                     color=label_color)

        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='forestgreen', label='Positive Correlation'),
            Patch(facecolor='crimson', label='Negative Correlation')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        # Adjust layout and display
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Plot categorical correlations
    if not categorical_df.empty:
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(categorical_df)), categorical_df['cramers_v'], color='blue')
        plt.xticks(range(len(categorical_df)), categorical_df.index, rotation=45, ha='right')
        plt.title(f"Categorical Correlations (Cramer's V) with {target_column}{method_str}")

        # Add value labels on top of each bar
        for i, v in enumerate(categorical_df['cramers_v']):
            plt.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()


# Test the function with a sample of your data
# Example usage:
methods = ['aequitas', 'ml_check', 'bias_scan', 'expga']
for method in methods:
    method_df = df[df['method_name'] == method].copy()

    if not method_df.empty:
        print(f"\nAnalyzing correlations for {method}:")
        numerical_correlations, categorical_correlations = analyze_correlations(method_df)

        if not numerical_correlations.empty:
            print("\nNumerical Correlations with directions:")
            print(numerical_correlations[['correlation', 'p_value']].sort_values(
                by='correlation', key=abs, ascending=False))

        plot_correlations(numerical_correlations, categorical_correlations,
                          'num_exact_couple_matches', method)

#%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df = df[[c for c in list(df.columns) if not c.startswith('expga') and not c.startswith('mlcheck') and not c.startswith('bias_scan')]]
correlations = df[numeric_cols].corr()['num_exact_couple_matches'].sort_values()

#%% Remove self-correlation
correlations = correlations.drop('num_exact_couple_matches')

# Create a figure with a larger size
plt.figure(figsize=(12, 8))

# Create horizontal bar plot
correlations.plot(kind='barh')

# Customize the plot
plt.title('Correlation with num_exact_couple_matches')
plt.xlabel('Correlation Coefficient')
plt.grid(True, alpha=0.3)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()