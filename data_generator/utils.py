from typing import Dict, List
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind  # Correctly imported for statistical testing
from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap


def scale_dataframe(df, reverse=False, min_values=None, max_values=None):
    if not reverse:
        min_values = df.min()
        max_values = df.max()

        range_values = max_values - min_values
        range_values[range_values == 0] = 1

        scaled_df = (df - min_values) / range_values

        return scaled_df, min_values, max_values
    else:
        if min_values is None or max_values is None:
            raise ValueError("min_values and max_values must be provided to reverse scaling.")

        range_values = max_values - min_values
        range_values[range_values == 0] = 1

        original_df = df * range_values + min_values
        return original_df


def create_parallel_coordinates_plot(data):
    group_properties = data.groupby('group_key').agg({
        'calculated_group_size': 'mean',  # Added group size
        'calculated_granularity': 'mean',
        'calculated_intersectionality': 'mean',
        'calculated_subgroup_ratio': 'mean',
        'calculated_similarity': 'mean',
        'epis_uncertainty': 'mean',
        'alea_uncertainty': 'mean',
        'calculated_magnitude': 'mean'
    }).reset_index().copy()

    for column in group_properties.columns:
        if column != 'group_key':
            group_properties[column] = pd.to_numeric(group_properties[column], errors='coerce')

    # Remove any rows with NaN values
    group_properties = group_properties.dropna()

    # Normalize the data to a 0-1 range for each property
    columns_to_plot = [
        'calculated_group_size',  # Added group size
        'calculated_granularity',
        'calculated_intersectionality',
        'calculated_subgroup_ratio',
        'calculated_similarity',
        'alea_uncertainty',
        'epis_uncertainty',
        'calculated_magnitude'
    ]
    normalized_data = group_properties[columns_to_plot].copy()
    for column in columns_to_plot:
        min_val = normalized_data[column].min()
        max_val = normalized_data[column].max()
        if min_val != max_val:
            normalized_data[column] = (normalized_data[column] - min_val) / (max_val - min_val)
        else:
            normalized_data[column] = 0.5  # Set to middle value if all values are the same

    # Create the plot with increased height for better readability
    fig, ax = plt.subplots(figsize=(15, 8))

    # Create x-coordinates for each property
    x = list(range(len(columns_to_plot)))

    # Create colormap
    norm = Normalize(vmin=group_properties['calculated_magnitude'].min(),
                     vmax=group_properties['calculated_magnitude'].max())
    cmap = plt.get_cmap('viridis')

    # Plot each group
    for i, row in normalized_data.iterrows():
        y = row[columns_to_plot].values
        color = cmap(norm(group_properties.loc[i, 'calculated_magnitude']))
        ax.plot(x, y, c=color, alpha=0.5)

    # Add mean line with different style
    mean_values = normalized_data.mean()
    ax.plot(x, mean_values, 'r--', linewidth=2, label='Mean', alpha=0.8)

    # Customize the plot
    ax.set_xticks(x)
    ax.set_xticklabels(columns_to_plot, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.set_title('Parallel Coordinates Plot of Discrimination Metrics')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Normalized Values')

    # Add gridlines
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Add value ranges in the grid
    for i, column in enumerate(columns_to_plot):
        min_val = group_properties[column].min()
        max_val = group_properties[column].max()
        ax.text(i, -0.1, f'Min: {min_val:.2f}\nMax: {max_val:.2f}',
                ha='center', va='top', fontsize=8)

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Calculated Magnitude')

    # Add legend
    ax.legend()

    plt.tight_layout()
    plt.show()


def visualize_df(df, columns, outcome_col, figure_path):
    fig = px.parallel_coordinates(
        df,
        color=outcome_col,
        labels={e: e for e in columns + [outcome_col]},
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=df[outcome_col].max() / 2)

    fig.update_layout(title="",
                      plot_bgcolor='white', coloraxis_showscale=True, font=dict(size=18))

    # Set axes to start from 0 and end at 1
    for dimension in fig.data[0]['dimensions']:
        if dimension['label'] in ['frequency', 'similarity', 'alea_uncertainty', 'epis_uncertainty', 'magnitude']:
            dimension['range'] = [0, 1]

    # fig.write_image(figure_path)
    return fig


def plot_distribution_comparison(schema, data, figsize=(15, 10)):
    """
    Plot distribution comparison between schema and generated data.
    """
    n_attrs = len(schema.attr_names)
    n_rows = (n_attrs + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten()

    for idx, (attr_name, protected) in enumerate(zip(schema.attr_names, schema.protected_attr)):
        ax = axes[idx]

        # Check if the column is numeric and has numeric categories
        try:
            # Try to convert first non-nan value to float to check if truly numeric
            sample_val = next((float(v) for k, v in schema.category_maps[attr_name].items()
                               if k != -1 and v != 'nan'), None)
            is_numeric = sample_val is not None
        except (ValueError, TypeError):
            is_numeric = False

        if is_numeric:
            # For numeric columns, use KDE plots
            sns.kdeplot(data=data.dataframe, x=attr_name, ax=ax, label='Generated', color='green')

            # Get schema distribution
            theo_probs = schema.categorical_distribution[attr_name]
            category_map = schema.category_maps[attr_name]

            # Create x values from category map, excluding 'nan'
            x_values = []
            y_values = []
            for k, v in category_map.items():
                if k != -1 and v != 'nan':
                    try:
                        x_values.append(float(v))
                        y_values.append(theo_probs[k])
                    except (ValueError, IndexError):
                        continue

            # Sort points by x value for proper line plotting
            points = sorted(zip(x_values, y_values))
            if points:
                x_values, y_values = zip(*points)
                ax.plot(x_values, y_values, label='Schema', color='blue')
        else:
            # Original categorical plotting code
            theo_values = [val for val in schema.attr_categories[idx] if val != -1]
            theo_probs = schema.categorical_distribution[attr_name]

            actual_dist = data.dataframe[attr_name].value_counts(normalize=True)
            x = np.arange(len(theo_values))
            width = 0.35

            ax.bar(x - width / 2, theo_probs, width, label='Schema', alpha=0.6, color='blue')
            ax.bar(x + width / 2, [actual_dist.get(val, 0) for val in theo_values],
                   width, label='Generated', alpha=0.6, color='green')
            ax.set_xticks(x)
            ax.set_xticklabels(theo_values)

        ax.set_title(f'{attr_name} {"(Protected)" if protected else ""}')
        ax.set_ylabel('Probability')
        ax.legend()

    # Remove empty subplots
    for idx in range(len(schema.attr_names), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    return fig


# Modified summary statistics function
def print_distribution_stats(schema, data):
    """Print summary statistics comparing distributions"""
    print("\nDistribution Comparison Statistics:")
    print("-" * 50)

    for idx, attr_name in enumerate(schema.attr_names):
        possible_values = schema.attr_categories[idx]
        theo_dist = pd.Series({val: 1 / len([v for v in possible_values if v != -1])
                               for val in possible_values if val != -1})
        actual_dist = data.dataframe[attr_name].value_counts(normalize=True)

        # Calculate KL divergence
        kl_div = np.sum(theo_dist * np.log(theo_dist / actual_dist.reindex(theo_dist.index).fillna(1e-10)))

        print(f"\n{attr_name}:")
        print(f"KL Divergence: {kl_div:.4f}")


def plot_and_print_metric_distributions(data, num_bins=10):
    metrics = [
        'calculated_granularity', 'calculated_intersectionality', 'calculated_subgroup_ratio', 'calculated_similarity',
        'calculated_aleatoric_group', 'calculated_epistemic_group', 'calculated_magnitude'
    ]

    group_properties = data.groupby('group_key').agg({metric: 'mean' for metric in metrics}).reset_index()

    # Create a 2x3 subplot layout
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), tight_layout=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        # Remove NaN values
        clean_data = group_properties[metric].dropna()

        if clean_data.empty:
            print(f"\nWarning: All values for {metric} are NaN. Skipping this metric.")
            ax.text(0.5, 0.5, f"No valid data for {metric}", ha='center', va='center')
            continue

        try:
            # Determine the number of bins
            unique_values = clean_data.nunique()
            actual_bins = min(num_bins, unique_values)

            # Create histogram
            if actual_bins == unique_values:
                bins = np.sort(clean_data.unique())
            else:
                bins = actual_bins

            n, bins, patches = ax.hist(clean_data, bins=bins, edgecolor='black')

            # Add labels and title
            ax.set_xlabel(metric, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'Distribution of {metric}', fontsize=12, pad=10)

            # Add percentage labels on top of each bar
            total_count = len(clean_data)
            for j, rect in enumerate(patches):
                height = rect.get_height()
                percentage = height / total_count * 100
                ax.text(rect.get_x() + rect.get_width() / 2., height,
                        f'{percentage:.1f}%',
                        ha='center', va='bottom', rotation=90, fontsize=8)

            # Adjust y-axis to make room for percentage labels
            ax.set_ylim(top=ax.get_ylim()[1] * 1.2)

            # Add mean and median lines
            mean_value = clean_data.mean()
            median_value = clean_data.median()
            ax.axvline(mean_value, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_value:.2f}')
            ax.axvline(median_value, color='green', linestyle=':', alpha=0.7, label=f'Median: {median_value:.2f}')
            ax.legend(fontsize=8)

        except Exception as e:
            print(f"\nError processing {metric}: {str(e)}")
            ax.text(0.5, 0.5, f"Error processing {metric}", ha='center', va='center')

    # Add a main title to the figure
    fig.suptitle('Distribution of Discrimination Properties for the groups', fontsize=14, y=1.02)

    plt.show()


def unique_individuals_ratio(data: pd.DataFrame, individual_col: str, attr_possible_values: Dict[str, List[int]]):
    unique_individuals_count = data[individual_col].nunique()

    # Calculate the total possible unique individuals by taking the product of possible values for each attribute
    possible_unique_individuals = np.prod([len(values) for values in attr_possible_values.values()])

    if possible_unique_individuals == 0:
        return 0, 0  # To handle division by zero if no data or attributes

    # Calculate the number of duplicates
    total_individuals = data.shape[0]
    duplicates_count = total_individuals - unique_individuals_count

    # Calculate the ratio
    ratio = unique_individuals_count / total_individuals

    return ratio, duplicates_count, total_individuals


def group_summary_table(data: pd.DataFrame, individual_col: str, group_col: str) -> pd.DataFrame:
    """
    Creates a summary table showing the total number of rows and unique individuals per group.

    Parameters:
    data (pd.DataFrame): Input DataFrame
    individual_col (str): Name of the column containing individual IDs
    group_col (str): Name of the column containing group IDs

    Returns:
    pd.DataFrame: Summary table with group statistics
    """
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'total_rows': data.groupby(group_col).size(),
        'unique_individuals': data.groupby(group_col)[individual_col].nunique()
    }).reset_index()

    # Sort by total_rows in descending order
    summary = summary.sort_values('total_rows', ascending=False)

    # Add percentage of unique individuals
    summary['pct_unique'] = (summary['unique_individuals'] / summary['total_rows'] * 100).round(2)

    # Rename columns for clarity
    summary.columns = [group_col, 'Total Rows', 'Unique Individuals', '% Unique']

    return summary


def individuals_in_multiple_groups(data: pd.DataFrame, individual_col: str, group_col: str) -> int:
    # Display the summary table first
    print("\nGroup Summary Table:")
    summary_table = group_summary_table(data, individual_col, group_col)
    print(summary_table.to_string(index=False))
    print("\nDistribution of Individuals Across Groups:")

    # Create histogram
    group_counts = data.groupby(individual_col)[group_col].nunique()
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(group_counts, bins=range(1, group_counts.max() + 2), edgecolor='black',
                                     align='left')

    # Add text annotations on top of each bar
    for count, patch in zip(counts, patches):
        plt.text(patch.get_x() + patch.get_width() / 2, count, f'{int(count)}', ha='center', va='bottom')

    plt.title('Histogram of Individuals Belonging to Multiple Groups with Counts')
    plt.xlabel('Number of Groups')
    plt.ylabel('Number of Individuals')
    plt.xticks(range(1, group_counts.max() + 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


from scipy.spatial.distance import cosine


def plot_correlation_matrices(input_correlation_matrix, generated_data, figsize=(30, 10)):
    # Get attribute columns from the generated data
    attr_columns = [col for col in generated_data.dataframe.columns if col.startswith('Attr')]
    generated_correlation_matrix = generated_data.dataframe[attr_columns].corr(method='spearman')

    # Convert input correlation matrix to DataFrame if it's a numpy array
    if isinstance(input_correlation_matrix, np.ndarray):
        # Create column names that match the size of the input matrix
        input_cols = [f'Attr{i + 1}_X' if i < input_correlation_matrix.shape[0] - sum(
            generated_data.schema.protected_attr) else f'Attr{i - input_correlation_matrix.shape[0] + sum(generated_data.schema.protected_attr) + 1}_T'
                      for i in range(input_correlation_matrix.shape[0])]
        input_correlation_matrix = pd.DataFrame(input_correlation_matrix, columns=input_cols, index=input_cols)

    # Get common columns between input and generated matrices
    common_cols = sorted(set(attr_columns) & set(input_correlation_matrix.columns))
    if not common_cols:
        raise ValueError("No common attributes found between input and generated correlation matrices")

    # Filter both matrices to use only common columns
    input_correlation_matrix = input_correlation_matrix.loc[common_cols, common_cols]
    generated_correlation_matrix = generated_correlation_matrix.loc[common_cols, common_cols]

    if isinstance(input_correlation_matrix, np.ndarray):
        input_correlation_matrix = pd.DataFrame(input_correlation_matrix, columns=attr_columns, index=attr_columns)

    # Calculate similarity metrics
    frobenius = np.linalg.norm(input_correlation_matrix - generated_correlation_matrix)
    cosine_sim = 1 - cosine(input_correlation_matrix.values.flatten(),
                            generated_correlation_matrix.values.flatten())
    mse = np.mean((input_correlation_matrix - generated_correlation_matrix) ** 2)
    correlation = np.corrcoef(input_correlation_matrix.values.flatten(),
                              generated_correlation_matrix.values.flatten())[0, 1]

    # Calculate the absolute difference matrix
    abs_diff_matrix = np.abs(input_correlation_matrix - generated_correlation_matrix)

    # Create custom colormap
    colors = ['#053061', '#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
    n_bins = 256
    custom_cmap = LinearSegmentedColormap.from_list('custom_blue_red', colors, N=n_bins)

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    def plot_heatmap(data, ax, title, cmap='coolwarm', vmin=-1, vmax=1):
        sns.heatmap(data, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, center=0,
                    annot=True, fmt='.2f', square=True, cbar=False,
                    annot_kws={'size': 11}, linewidths=0.5)
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

    # Plot matrices
    plot_heatmap(input_correlation_matrix, ax1, 'Input Correlation Matrix')
    plot_heatmap(generated_correlation_matrix, ax2, 'Generated Data Correlation Matrix')
    plot_heatmap(abs_diff_matrix, ax3, 'Absolute Difference Matrix', cmap=custom_cmap, vmin=0, vmax=1)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=10)

    # Adjust layout and display
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.show()

    # Print all similarity metrics
    print("\nMatrix Similarity Metrics:")
    print(f"Mean absolute difference: {np.mean(abs_diff_matrix):.4f}")
    print(f"Maximum absolute difference: {np.max(abs_diff_matrix):.4f}")
    print(f"Frobenius norm: {frobenius:.4f}")
    print(f"Cosine similarity: {cosine_sim:.4f}")
    print(f"Mean squared error: {mse:.4f}")
    print(f"Correlation coefficient: {correlation:.4f}")


def visualize_injected_discrimination(
        data: DiscriminationData,
        sample_size: int = 5000,
        top_n_biased_groups: int = 4,
        p_value_threshold: float = 0.05
):
    """
    Creates a set of visualizations to show injected discrimination by empirically
    detecting bias from the difference between 'y_true' and 'y_pred'.

    This function is self-contained and identifies bias without needing access
    to the generator's internal parameters.

    Args:
        data (DiscriminationData): The generated data object.
        sample_size (int): Number of data points to sample for plotting.
        top_n_biased_groups (int): Number of most biased groups to highlight.
        p_value_threshold (float): The statistical significance level to determine if a
                                   group's prediction error is different from zero.
    """
    df = data.dataframe.copy()  # Use a copy to avoid modifying the original object
    try:
        y_true_col = data.y_true_col
        y_pred_col = data.y_pred_col
    except AttributeError:
        print("Error: `DiscriminationData` object must have `y_true_col` and `y_pred_col` attributes.")
        return

    # --- 1. EMPIRICALLY IDENTIFY BIASED GROUPS ---

    # Calculate the prediction error for each individual
    df['prediction_error'] = df[y_true_col] - df[y_pred_col]

    # For each group, determine if the mean prediction error is statistically
    # different from zero. A t-test is perfect for this.
    def check_group_bias(group_df: pd.DataFrame) -> pd.Series:
        # We need at least a few samples to run a meaningful test
        if len(group_df) < 10:
            return pd.Series({'is_biased': False, 'mean_error': 0.0})

        # Perform an independent t-test comparing the group's error distribution
        # to a zero-error distribution. This is statistically equivalent to a
        # one-sample t-test against a population mean of 0.
        ttest_result = ttest_ind(group_df['prediction_error'], [0] * len(group_df), equal_var=False)

        is_biased = ttest_result.pvalue < p_value_threshold
        mean_error = group_df['prediction_error'].mean()

        return pd.Series({'is_biased': is_biased, 'mean_error': mean_error})

    print("Empirically identifying biased groups by testing if mean(y_true - y_pred) is non-zero...")
    group_stats = df.groupby('group_key').apply(check_group_bias)

    # Map the results back to the main dataframe
    df['is_group_biased'] = df['group_key'].map(group_stats['is_biased'])

    # Find the top N most biased groups for highlighting
    # We use the absolute mean error to find the most biased, regardless of direction
    biased_stats = group_stats[group_stats['is_biased']].copy()
    biased_stats['abs_error'] = biased_stats['mean_error'].abs()
    top_biased_groups = biased_stats.nlargest(top_n_biased_groups, 'abs_error').index.tolist()

    # Create a categorical column for plotting
    def get_group_category(row: pd.Series) -> str:
        if row['group_key'] in top_biased_groups:
            return f"Biased Group (Top {top_n_biased_groups})"
        elif row['is_group_biased']:
            return "Biased Group (Other)"
        else:
            return "Unbiased Group"

    df['group_category'] = df.apply(get_group_category, axis=1)

    # --- DIAGNOSTIC PRINTS ---
    print("\nValue counts in the full dataset before sampling:")
    print(df['group_category'].value_counts())

    # --- Stratified Sampling ---
    if len(df) > sample_size and 'group_category' in df.columns and df['group_category'].nunique() > 1:
        try:
            plot_df = df.groupby('group_category', group_keys=False).apply(
                lambda x: x.sample(int(np.ceil(sample_size * len(x) / len(df))) if len(x) > 0 else 0),
                include_groups=False
            ).sample(frac=1).reset_index(drop=True)
        except Exception as e:
            print(f"Stratified sampling failed: {e}. Falling back to random sampling.")
            plot_df = df.sample(n=sample_size, random_state=42)
    else:
        plot_df = df.copy()

    if plot_df.empty:
        print("No data available for plotting after sampling. Exiting.")
        return

    print("\nValue counts in the sampled data for plotting:")
    print(plot_df['group_category'].value_counts())

    # --- 2. Create Visualizations ---
    sns.set_theme(style="whitegrid")

    # ----- PLOT 1: Distribution of Prediction Errors -----
    plt.figure(figsize=(12, 7))
    sns.kdeplot(data=plot_df, x='prediction_error', hue='is_group_biased', fill=True, common_norm=False,
                palette={True: "coral", False: "steelblue"})
    plt.title('Distribution of Prediction Error (y_true - y_pred)\nfor Empirically Biased vs. Unbiased Groups',
              fontsize=16)
    plt.xlabel('Prediction Error', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.axvline(0, color='k', linestyle='--', label='No Error')
    plt.legend(title='Is Group Biased?')
    plt.show()

    # ----- PLOT 2: Violin Plot of True vs. Predicted Outcomes -----
    plot_df_melted = pd.melt(
        plot_df,
        id_vars=['group_category'],
        value_vars=[y_true_col, y_pred_col],
        var_name='Outcome Type',
        value_name='Outcome Value'
    )
    plot_df_melted['Outcome Type'] = plot_df_melted['Outcome Type'].map(
        {y_true_col: 'Ground Truth', y_pred_col: 'Prediction'})

    plt.figure(figsize=(14, 8))
    sns.violinplot(
        data=plot_df_melted,
        x='group_category',
        y='Outcome Value',
        hue='Outcome Type',
        split=True,
        inner="quart",
        palette={"Ground Truth": "skyblue", "Prediction": "lightcoral"},
        order=["Unbiased Group", f"Biased Group (Top {top_n_biased_groups})", "Biased Group (Other)"]
    )
    plt.title('Comparison of Ground Truth vs. Predicted Outcomes by Subgroup Type', fontsize=16)
    plt.xlabel('Subgroup Category', fontsize=12)
    plt.ylabel('Outcome', fontsize=12)
    plt.xticks(rotation=10)
    plt.legend(title='Outcome Type', loc='upper left')
    plt.show()

    # ----- PLOT 3: Scatter Plot of Prediction vs. Ground Truth -----
    g = sns.FacetGrid(plot_df, col="group_category", hue="group_category",
                      col_wrap=3, height=5,
                      col_order=["Unbiased Group", f"Biased Group (Top {top_n_biased_groups})", "Biased Group (Other)"])
    g.map(sns.scatterplot, y_pred_col, y_true_col, alpha=0.6)
    for ax in g.axes.flatten():
        if ax is not None:
            min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
            max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([min_val, max_val], [min_val, max_val], ls="--", c=".3")
            ax.set_xlabel("Prediction", fontsize=11)
            ax.set_ylabel("Ground Truth", fontsize=11)

    g.fig.suptitle('Ground Truth vs. Prediction for Different Subgroup Types', y=1.03, fontsize=16)
    g.add_legend()
    plt.show()

# if __name__ == '__main__':
#     # ==================================================================
#     # This block demonstrates how to use the function.
#     # It creates a sample DataFrame that mimics your generator's output.
#     # ==================================================================
#     print("Creating a sample dataset for demonstration...")
#
#     # Define some group keys
#     group_keys = ['unbiased_A', 'unbiased_B', 'biased_pos_strong', 'biased_neg_weak', 'biased_pos_weak']
#
#     # Define biases for each group
#     biases = {
#         'unbiased_A': 0.0,
#         'unbiased_B': 0.0,
#         'biased_pos_strong': 0.3,
#         'biased_neg_weak': -0.15,
#         'biased_pos_weak': 0.1
#     }
#
#     # Generate sample data
#     records = []
#     for group in group_keys:
#         for i in range(1000):  # 1000 individuals per group
#             pred = np.random.rand()  # A random prediction between 0 and 1
#             bias = biases[group]
#             noise = np.random.normal(0, 0.05)  # Add some noise
#
#             true_val = pred + bias + noise
#             # Clip values to be within [0, 1] range
#             true_val = np.clip(true_val, 0, 1)
#
#             records.append({
#                 'group_key': group,
#                 'y_pred': pred,
#                 'y_true': true_val
#             })
#
#     sample_df = pd.DataFrame(records)
#
#     # Create the wrapper object, similar to your DiscriminationData
#     mock_data_obj = DiscriminationData(
#         dataframe=sample_df,
#         y_true_col='y_true',
#         y_pred_col='y_pred'
#     )
#
#     print("\nSample dataset created. Now running visualization...")
#     print("-" * 50)
#
#     # Call the visualization function with the mock data
#     visualize_injected_discrimination(
#         data=mock_data_obj,
#         sample_size=4000,
#         top_n_biased_groups=2,
#         p_value_threshold=0.01  # Use a stricter p-value for clean separation
#     )
