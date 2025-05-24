import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from experiments.synthetic_datasets_experiments.analysis import get_result_and_synthetic_matching_res, analyze_treatment_effects, \
    plot_treatment_effects
from path import HERE

DB_PATH = HERE.joinpath("methods/optimization/fairness_test_results2.db")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

run_ids = pd.read_sql_query("""SELECT DISTINCT run_id FROM fairness_test_results""", conn)

synth_df, res_df = [], []

synth_cols = ['outcome', 'epis_uncertainty', 'alea_uncertainty', 'subgroup_key', 'indv_key', 'group_key',
              'granularity_param', 'intersectionality_param', 'similarity_param', 'epis_uncertainty_param',
              'alea_uncertainty_param', 'frequency_param', 'group_size', 'diff_subgroup_size', 'collisions',
              'diff_outcome', 'diff_variation', 'calculated_epistemic_random_forest',
              'calculated_aleatoric_random_forest', 'calculated_aleatoric_entropy',
              'calculated_aleatoric_probability_margin', 'calculated_aleatoric_label_smoothing',
              'calculated_epistemic_ensemble', 'calculated_epistemic_mc_dropout', 'calculated_epistemic_evidential',
              'calculated_similarity', 'calculated_epistemic_group', 'calculated_aleatoric_group',
              'calculated_magnitude', 'calculated_mean_demographic_disparity', 'calculated_uncertainty_group',
              'calculated_intersectionality', 'calculated_granularity', 'calculated_group_size',
              'calculated_subgroup_ratio',
              'indv_matches', 'unique_indv_matches', 'couple_matches',
              'unique_couple_matches',
              'method']

res_cols = ['indv_key', 'outcome', 'couple_key', 'diff_outcome', 'case_id', 'TSN', 'DSN', 'SUR', 'DSS',
            'indv_matching_subgroups', 'indv_matching_groups', 'couple_matching_groups', 'method']

for run_id in run_ids['run_id'].tolist():
    res_with_groups, synth_with_groups = get_result_and_synthetic_matching_res(experiment_id=run_id)
    res_df1 = res_with_groups[res_cols]
    synth_df1 = synth_with_groups[synth_cols]

    res_df1['run_id'] = run_id
    synth_df1['run_id'] = run_id

    res_df.append(res_df1)
    synth_df.append(synth_df1)

synth_df, res_df = pd.concat(synth_df), pd.concat(res_df)

numeric_cols = synth_df.select_dtypes(include=[np.number]).columns
numeric_cols = [e for e in numeric_cols if e.startswith('calculated_')] + ['unique_couple_matches']

df = synth_df[numeric_cols+['method']]

# %%
# Select only numeric columns
correlations = df[numeric_cols].corr()['unique_couple_matches'].sort_values()

# Remove self-correlation
correlations = correlations.drop('unique_couple_matches')

print(correlations)

# Create a figure with a larger size
plt.figure(figsize=(12, 8))

# Create horizontal bar plot
correlations.plot(kind='barh')

# Customize the plot
plt.title('Correlation with unique_couple_matches')
plt.xlabel('Correlation Coefficient')
plt.grid(True, alpha=0.3)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()


# %% Filter columns based on method
def filter_columns_by_method(df):
    method_prefixes = {
        'aequitas': ['aequitas_'],
        'sg': ['sg_'],
        'adf': ['adf_'],
        'expga': ['expga_']
    }

    filtered_df = df.copy()

    # Iterate through rows using iterrows() instead of accessing by index
    for idx, row in filtered_df.iterrows():
        current_method = row['method']  # This gets the scalar value from the row

        columns_to_drop = []
        for method, prefixes in method_prefixes.items():
            if method != current_method:  # Now comparing strings, not a Series
                for prefix in prefixes:
                    columns_to_drop.extend([col for col in filtered_df.columns if col.startswith(prefix)])

        # Set the values in the specified columns to NaN
        filtered_df.loc[idx, columns_to_drop] = np.nan

    return filtered_df


# Analyze correlations
def analyze_correlations(df, target_column='unique_couple_matches'):
    numerical_df = pd.DataFrame()
    categorical_df = pd.DataFrame()

    df = filter_columns_by_method(df)

    df = df.dropna(axis=1, how='all')

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    numerical_correlations = {}
    categorical_correlations = {}

    if target_column in numerical_cols:
        for col in numerical_cols:
            if col != target_column:
                if df[col].notna().any():
                    try:
                        correlation, p_value = stats.pearsonr(
                            df[col].dropna(),
                            df[target_column][df[col].notna()]
                        )
                        numerical_correlations[col] = {
                            'correlation': correlation,
                            'p_value': p_value
                        }
                    except ValueError:
                        continue

        numerical_df = pd.DataFrame.from_dict(numerical_correlations, orient='index')
        numerical_df = numerical_df.sort_values(by='correlation', key=abs, ascending=False)

    for col in categorical_cols:
        if col != target_column:
            if df[col].notna().any():
                try:
                    contingency = pd.crosstab(df[col], df[target_column])

                    chi2, p_value = stats.chi2_contingency(contingency)[:2]

                    n = contingency.sum().sum()
                    min_dim = min(contingency.shape) - 1
                    cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

                    categorical_correlations[col] = {
                        'cramers_v': cramer_v,
                        'p_value': p_value
                    }
                except ValueError:
                    continue

    if categorical_correlations:
        categorical_df = pd.DataFrame.from_dict(categorical_correlations, orient='index')
        categorical_df = categorical_df.sort_values(by='cramers_v', ascending=False)

    return numerical_df, categorical_df


# Plot correlations
def plot_correlations(numerical_df, categorical_df, target_column, method=None):
    method_str = f" ({method})" if method else ""

    if not numerical_df.empty:
        plt.figure(figsize=(15, 8))

        correlations = numerical_df['correlation'].values
        indices = range(len(correlations))

        colors = ['forestgreen' if c > 0 else 'crimson' for c in correlations]

        bars = plt.bar(indices, correlations, color=colors)

        plt.xticks(indices, numerical_df.index, rotation=45, ha='right')

        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, zorder=1)

        plt.title(f'Numerical Correlations with {target_column}{method_str}')
        plt.ylabel('Correlation Coefficient')

        for i, v in enumerate(correlations):
            label_color = 'black'
            if abs(v) < 0.2:
                label_color = 'black'
            plt.text(i, v, f'{v:.3f}',
                     ha='center',
                     va='bottom' if v > 0 else 'top',
                     fontsize=9,
                     color=label_color)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='forestgreen', label='Positive Correlation'),
            Patch(facecolor='crimson', label='Negative Correlation')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    if not categorical_df.empty:
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(categorical_df)), categorical_df['cramers_v'], color='blue')
        plt.xticks(range(len(categorical_df)), categorical_df.index, rotation=45, ha='right')
        plt.title(f"Categorical Correlations (Cramer's V) with {target_column}{method_str}")

        for i, v in enumerate(categorical_df['cramers_v']):
            plt.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()


# %% Test the function with a sample of your data
methods = ['aequitas', 'sg', 'adf', 'expga']
for method in methods:
    method_df = df[df['method'] == method].copy()

    if not method_df.empty:
        print(f"\nAnalyzing correlations for {method}:")
        numerical_correlations, categorical_correlations = analyze_correlations(method_df)

        if not numerical_correlations.empty:
            print("\nNumerical Correlations with directions:")
            print(numerical_correlations[['correlation', 'p_value']].sort_values(
                by='correlation', key=abs, ascending=False))

        plot_correlations(numerical_correlations, categorical_correlations,
                          'unique_couple_matches', method)

# %%
te_res = analyze_treatment_effects(synth_df)

# %%
plot_treatment_effects(te_res)
plt.show()
