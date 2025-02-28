from typing import Dict, List

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
    attr_columns = [col for col in generated_data.dataframe.columns if col.startswith('Attr')]
    generated_correlation_matrix = generated_data.dataframe[attr_columns].corr(method='spearman')

    assert input_correlation_matrix.shape == generated_correlation_matrix.shape, "Correlation matrices have different shapes"

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
