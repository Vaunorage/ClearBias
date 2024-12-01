from typing import Dict, List
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from data_generator.main import generate_data, generate_valid_correlation_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score

nb_attributes = 10
correlation_matrix = generate_valid_correlation_matrix(nb_attributes)


data = generate_data(
    nb_attributes=nb_attributes,
    correlation_matrix=correlation_matrix,
    min_number_of_classes=2,
    max_number_of_classes=9,
    prop_protected_attr=0.4,
    nb_groups=10,
    max_group_size=400,
    max_granularity=2,
    max_intersectionality=2,
    categorical_outcome=True,
    nb_categories_outcome=4,
    use_cache=False,
    corr_matrix_randomness=0.6)

print(f"Generated {len(data.dataframe)} samples in {data.nb_groups} groups")
print(f"Collisions: {data.collisions}")

#
# %%

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


def individuals_in_multiple_groups(data: pd.DataFrame, individual_col: str, group_col: str) -> int:
    group_counts = data.groupby(individual_col)[group_col].nunique()

    # Create the histogram
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


# Example usage:
individual_col = 'indv_key'
group_col = 'group_key'

unique_ratio, duplicates_count, total = unique_individuals_ratio(data.dataframe, 'indv_key', data.attr_possible_values)
individuals_in_multiple_groups_count = individuals_in_multiple_groups(data.dataframe, individual_col, group_col)

print(f"Unique Individuals Ratio: {unique_ratio}, duplicate : {duplicates_count}, total: {total}")
print(f"Individuals in Multiple Groups: {individuals_in_multiple_groups_count}")


# %%
def create_parallel_coordinates_plot(data):
    group_properties = data.groupby('group_key').agg({
        'group_size': 'mean',  # Added group size
        'granularity': 'mean',
        'intersectionality': 'mean',
        'diff_subgroup_size': 'mean',
        'actual_similarity': 'mean',
        'actual_alea_uncertainty': 'mean',
        'actual_epis_uncertainty': 'mean',
        'actual_mean_diff_outcome': 'mean',
        'relevance': 'mean'
    }).reset_index().copy()

    group_properties.rename(columns={
        'actual_similarity': 'similarity',
        'actual_alea_uncertainty': 'alea_uncertainty',
        'actual_epis_uncertainty': 'epis_uncertainty',
        'actual_mean_diff_outcome': 'diff_outcome'
    }, inplace=True)

    for column in group_properties.columns:
        if column != 'group_key':
            group_properties[column] = pd.to_numeric(group_properties[column], errors='coerce')

    # Remove any rows with NaN values
    group_properties = group_properties.dropna()

    # Normalize the data to a 0-1 range for each property
    columns_to_plot = [
        'group_size',  # Added group size
        'granularity',
        'intersectionality',
        'diff_subgroup_size',
        'similarity',
        'alea_uncertainty',
        'epis_uncertainty',
        'diff_outcome'
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
    norm = Normalize(vmin=group_properties['relevance'].min(), vmax=group_properties['relevance'].max())
    cmap = plt.get_cmap('viridis')

    # Plot each group
    for i, row in normalized_data.iterrows():
        y = row[columns_to_plot].values
        color = cmap(norm(group_properties.loc[i, 'relevance']))
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
    cbar = plt.colorbar(sm, ax=ax, label='Relevance')

    # Add legend
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nSummary Statistics:")
    for column in columns_to_plot:
        print(f"\n{column}:")
        print(f"Mean: {group_properties[column].mean():.2f}")
        print(f"Median: {group_properties[column].median():.2f}")
        print(f"Std Dev: {group_properties[column].std():.2f}")
        print(f"Min: {group_properties[column].min():.2f}")
        print(f"Max: {group_properties[column].max():.2f}")


# Usage
create_parallel_coordinates_plot(data.dataframe)


# %%

def plot_and_print_metric_distributions(data, num_bins=10):
    metrics = [
        'granularity', 'intersectionality', 'diff_subgroup_size', 'actual_similarity',
        'actual_alea_uncertainty', 'actual_epis_uncertainty'
    ]

    group_properties = data.groupby('group_key').agg({metric: 'mean' for metric in metrics}).reset_index()

    # Create a 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), tight_layout=True)
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

            # Print statistics
            print(f"\nDistribution of {metric}:")
            print(f"Total data points: {total_count}")
            print(f"Mean: {mean_value:.2f}")
            print(f"Median: {median_value:.2f}")
            print(f"Std Dev: {clean_data.std():.2f}")
            print(f"Number of bins: {actual_bins}")
            print("\nBin ranges, counts, and percentages:")
            for k in range(len(n)):
                bin_start = bins[k]
                bin_end = bins[k + 1] if k < len(bins) - 1 else bin_start
                count = n[k]
                percentage = (count / total_count) * 100
                print(f"Bin {k + 1}: {bin_start:.2f} to {bin_end:.2f}")
                print(f"  Count: {count}")
                print(f"  Percentage: {percentage:.1f}%")

        except Exception as e:
            print(f"\nError processing {metric}: {str(e)}")
            ax.text(0.5, 0.5, f"Error processing {metric}", ha='center', va='center')

    # Add a main title to the figure
    fig.suptitle('Distribution of Discrimination Metrics', fontsize=14, y=1.02)

    plt.show()


# Usage
plot_and_print_metric_distributions(data.dataframe)

# %%

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import time
from datetime import datetime
from tqdm.auto import tqdm


def test_models_on_generated_data(data, n_runs=10, test_size=0.2, random_seed=42):
    """
    Comprehensive testing framework for evaluating multiple models across multiple runs.
    """

    # Initialize results storage
    results_list = []

    # Define models to test
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'SVM': SVC(probability=True)
    }

    # Define metrics to evaluate
    def safe_roc_auc(y_true, y_pred_proba):
        try:
            # For binary classification
            if len(np.unique(y_true)) == 2:
                return roc_auc_score(y_true, y_pred_proba[:, 1])
            # For multiclass
            return roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        except:
            return np.nan

    metrics = {
        'accuracy': accuracy_score,
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
        'roc_auc': safe_roc_auc
    }

    print("Preparing data...")
    # Extract features and target
    X = data.dataframe[data.feature_names].copy()
    y = data.dataframe[data.outcome_column].copy()

    # Initialize preprocessing objects
    label_encoders = {}
    scaler = StandardScaler()

    # Encode categorical variables
    for column in X.columns:
        if X[column].dtype == 'object':
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le

    # Encode target variable if categorical
    if y.dtype == 'object':
        print("Encoding target variable...")
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Run multiple tests
    run_pbar = tqdm(range(n_runs), desc="Running test iterations")
    for run in run_pbar:
        current_seed = random_seed + run
        run_results = {
            'run_id': run + 1,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'seed': current_seed
        }

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=current_seed
        )

        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for model_name, model in models.items():
            start_time = time.time()

            try:
                # Train model
                model.fit(X_train_scaled, y_train)

                # Get predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None

                # Calculate metrics

                for metric_name, metric_func in metrics.items():
                    try:
                        if metric_name == 'roc_auc' and y_pred_proba is not None:
                            score = metric_func(y_test, y_pred_proba)
                        else:
                            score = metric_func(y_test, y_pred)
                        run_results[f'{model_name}_{metric_name}'] = score
                    except Exception as e:
                        print(f"\nWarning: Could not calculate {metric_name} for {model_name}: {str(e)}")
                        run_results[f'{model_name}_{metric_name}'] = np.nan

                # Get feature importance if available
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    top_features = pd.Series(importance, index=X.columns).nlargest(5).index.tolist()
                    run_results[f'{model_name}_top_features'] = ','.join(top_features)

                # Calculate training time
                run_results[f'{model_name}_train_time'] = time.time() - start_time

            except Exception as e:
                print(f"\nError with {model_name}: {str(e)}")
                # Fill with NaN values for failed model
                for metric_name in metrics.keys():
                    run_results[f'{model_name}_{metric_name}'] = np.nan
                run_results[f'{model_name}_train_time'] = np.nan

        results_list.append(run_results)

        # Update run progress bar description with current accuracies
        acc_desc = " | ".join([f"{model}: {run_results.get(f'{model}_accuracy', 0):.3f}"
                               for model in models.keys()])
        run_pbar.set_description(f"Run {run + 1}/{n_runs} | {acc_desc}")

    print("\nComputing summary statistics...")
    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)

    # Calculate summary statistics
    summary_stats = {}
    for column in results_df.columns:
        if column not in ['run_id', 'timestamp', 'seed'] and not column.endswith('top_features'):
            summary_stats[f'{column}_mean'] = results_df[column].mean()
            summary_stats[f'{column}_std'] = results_df[column].std()
            summary_stats[f'{column}_min'] = results_df[column].min()
            summary_stats[f'{column}_max'] = results_df[column].max()

    summary_df = pd.DataFrame([summary_stats])

    return results_df, summary_df


def visualize_model_comparison(results_df, summary_df):
    """
    Create visualizations to compare model performance.
    """
    print("Generating visualizations...")
    # Create performance comparison plot
    metric_names = ['accuracy', 'f1', 'precision', 'recall']
    model_names = ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metric_names):
        ax = axes[i]
        data = []
        labels = []

        for model in model_names:
            col_name = f'{model}_{metric}'
            if col_name in results_df.columns:
                data.append(results_df[col_name].values)
                labels.append(model)

        if data:  # Only create plot if we have data
            ax.boxplot(data, labels=labels)
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(summary_df.round(4).to_string())

    return fig


def run_model_comparison(data, n_runs=10):
    """
    Run the complete model comparison analysis.
    """
    print(f"Starting model comparison with {n_runs} runs...")
    results_df, summary_df = test_models_on_generated_data(data, n_runs=n_runs)

    visualize_model_comparison(results_df, summary_df)

    return results_df, summary_df


results_df, summary_df = run_model_comparison(data, n_runs=2)

# To see detailed results
print("\nDetailed Results:")
print(results_df.describe())


# %%

def plot_correlation_matrices(input_correlation_matrix, generated_data, figsize=(30, 10)):
    attr_columns = [col for col in generated_data.dataframe.columns if col.startswith('Attr')]

    generated_correlation_matrix = generated_data.dataframe[attr_columns].corr(method='spearman')

    assert input_correlation_matrix.shape == generated_correlation_matrix.shape, "Correlation matrices have different shapes"

    if isinstance(input_correlation_matrix, np.ndarray):
        input_correlation_matrix = pd.DataFrame(input_correlation_matrix, columns=attr_columns, index=attr_columns)

    # Calculate the absolute difference matrix
    abs_diff_matrix = np.abs(input_correlation_matrix - generated_correlation_matrix)

    # Create a custom colormap for the absolute difference (blue to white to red)
    colors = ['#053061', '#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
    n_bins = 256  # Increase for smoother gradient
    custom_cmap = LinearSegmentedColormap.from_list('custom_blue_red', colors, N=n_bins)

    # Create a figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Function to plot heatmap with adjusted parameters
    def plot_heatmap(data, ax, title, cmap='coolwarm', vmin=-1, vmax=1):
        sns.heatmap(data, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, center=0,
                    annot=True, fmt='.2f', square=True, cbar=False,
                    annot_kws={'size': 11}, linewidths=0.5)
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

    # Plot input correlation matrix
    plot_heatmap(input_correlation_matrix, ax1, 'Input Correlation Matrix')

    # Plot generated correlation matrix
    plot_heatmap(generated_correlation_matrix, ax2, 'Generated Data Correlation Matrix')

    # Plot absolute difference matrix with custom colormap
    plot_heatmap(abs_diff_matrix, ax3, 'Absolute Difference Matrix', cmap=custom_cmap, vmin=0, vmax=1)

    # Add a color bar for the absolute difference matrix
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=10)

    # Adjust layout and display the plot
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.show()

    # Calculate and print summary statistics
    mean_diff = np.mean(abs_diff_matrix)
    max_diff = np.max(abs_diff_matrix)
    print(f"Mean absolute difference between matrices: {mean_diff:.4f}")
    print(f"Maximum absolute difference between matrices: {max_diff:.4f}")


plot_correlation_matrices(correlation_matrix, data)

# %%

input_correlation_matrices, generated_data_list = [], []
nb_attributes = 20
for da in range(3):
    correlation_matrix = generate_valid_correlation_matrix(nb_attributes)

    data = generate_data(
        nb_attributes=nb_attributes,
        correlation_matrix=correlation_matrix,
        min_number_of_classes=8,
        max_number_of_classes=10,
        prop_protected_attr=0.4,
        nb_groups=200,
        max_group_size=100,
        categorical_outcome=True,
        nb_categories_outcome=4)

    input_correlation_matrices.append(correlation_matrix)
    generated_data_list.append(data)


# %%


def plot_aggregate_correlation_matrices(input_correlation_matrices, generated_data_list, figsize=(30, 10)):
    # Get all unique column names across all datasets
    all_columns = set()
    for data in generated_data_list:
        all_columns.update(data.feature_names)
    all_columns = sorted(list(all_columns))

    # Initialize a dictionary to store difference matrices
    diff_matrices = {col1: {col2: [] for col2 in all_columns} for col1 in all_columns}

    # Calculate the difference matrices and store them
    for input_corr, generated_data in zip(input_correlation_matrices, generated_data_list):
        feature_names = generated_data.feature_names
        generated_corr = generated_data.dataframe[feature_names].corr(method='spearman')

        if isinstance(input_corr, np.ndarray):
            input_corr = pd.DataFrame(input_corr, columns=feature_names, index=feature_names)

        for col1 in feature_names:
            for col2 in feature_names:
                diff = abs(input_corr.loc[col1, col2] - generated_corr.loc[col1, col2])
                diff_matrices[col1][col2].append(diff)

    # Calculate aggregate statistics
    aggregate_diff_matrix = pd.DataFrame({col1: {col2: np.mean(values) if values else np.nan
                                                 for col2, values in col_dict.items()}
                                          for col1, col_dict in diff_matrices.items()})

    variance_matrix = pd.DataFrame({col1: {col2: np.var(values) if len(values) > 1 else np.nan
                                           for col2, values in col_dict.items()}
                                    for col1, col_dict in diff_matrices.items()})

    # Calculate summary statistics
    mean_diff = np.nanmean(aggregate_diff_matrix.values)
    max_diff = np.nanmax(aggregate_diff_matrix.values)
    median_diff = np.nanmedian(aggregate_diff_matrix.values)
    mean_variance = np.nanmean(variance_matrix.values)
    max_variance = np.nanmax(variance_matrix.values)

    # Create a custom colormap for the absolute difference (blue to white to red)
    colors = ['#053061', '#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
    n_bins = 256
    custom_cmap = LinearSegmentedColormap.from_list('custom_blue_red', colors, N=n_bins)

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot aggregate difference matrix
    sns.heatmap(aggregate_diff_matrix, ax=ax1, cmap=custom_cmap, vmin=0, vmax=1, center=0,
                annot=True, fmt='.2f', square=True, cbar=True,
                annot_kws={'size': 8}, linewidths=0.5)
    ax1.set_title('Aggregate Absolute Difference Matrix', fontsize=16, pad=20)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='right', fontsize=8)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=8)

    # Plot variance matrix
    sns.heatmap(variance_matrix, ax=ax2, cmap='viridis', vmin=0, center=0,
                annot=True, fmt='.2f', square=True, cbar=True,
                annot_kws={'size': 8}, linewidths=0.5)
    ax2.set_title('Variance of Absolute Differences', fontsize=16, pad=20)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, ha='right', fontsize=8)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"Mean absolute difference across all matrices: {mean_diff:.4f}")
    print(f"Maximum absolute difference across all matrices: {max_diff:.4f}")
    print(f"Median absolute difference across all matrices: {median_diff:.4f}")
    print(f"Mean variance of differences: {mean_variance:.4f}")
    print(f"Maximum variance of differences: {max_variance:.4f}")


# Example usage:
plot_aggregate_correlation_matrices(input_correlation_matrices, generated_data_list, figsize=(30, 10))


# %%

def plot_metric_distributions(data_list, num_bins=10):
    metrics = [
        'granularity', 'intersectionality', 'diff_subgroup_size', 'actual_similarity',
        'actual_alea_uncertainty', 'actual_epis_uncertainty', 'actual_mean_diff_outcome', 'relevance'
    ]

    fig, axes = plt.subplots(4, 2, figsize=(15, 20), tight_layout=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        all_data = []
        for data in data_list:
            group_properties = data.dataframe.groupby('group_key').agg({metric: 'mean'}).reset_index()
            all_data.extend(group_properties[metric].dropna())

        if not all_data:
            print(f"\nWarning: All values for {metric} are NaN. Skipping this metric.")
            ax.text(0.5, 0.5, f"No valid data for {metric}", ha='center', va='center')
            continue

        try:
            # Create histogram
            counts, bin_edges = np.histogram(all_data, bins=num_bins)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            # Calculate error (assuming Poisson distribution for bin counts)
            errors = np.sqrt(counts)

            # Plot histogram without error bars
            ax.bar(bin_centers, counts, width=np.diff(bin_edges), alpha=0.7)

            # Add error bars separately
            ax.errorbar(bin_centers, counts, yerr=errors, fmt='none', ecolor='black', capsize=3)

            # Add labels and title
            ax.set_xlabel(metric)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {metric}')

            # Print statistics
            print(f"\nDistribution of {metric}:")
            print(f"Total data points: {len(all_data)}")
            print(f"Number of bins: {num_bins}")
            print("\nBin ranges, counts, and errors:")
            for j in range(len(counts)):
                print(f"Bin {j + 1}: {bin_edges[j]:.2f} to {bin_edges[j + 1]:.2f}")
                print(f"  Count: {counts[j]}")
                print(f"  Error: ±{errors[j]:.2f}")

        except Exception as e:
            print(f"\nError processing {metric}: {str(e)}")
            ax.text(0.5, 0.5, f"Error processing {metric}", ha='center', va='center')

    plt.show()


plot_metric_distributions(generated_data_list)


# %%


def test_models_on_multiple_datasets(datasets):
    all_results = []

    for data in datasets:
        # Extract features and target
        X = data.dataframe[data.feature_names]
        y = data.dataframe[data.outcome_column]

        # Encode categorical variables
        label_encoders = {}
        for column in X.columns:
            if X[column].dtype == 'object':
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column])
                label_encoders[column] = le

        # Encode target variable if it's categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize models
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train and evaluate Random Forest
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_predictions)
        rf_f1 = f1_score(y_test, rf_predictions, average='weighted')
        rf_recall = recall_score(y_test, rf_predictions, average='weighted')

        # Feature importance
        rf_feature_importance = rf_model.feature_importances_

        # Sort feature importances
        rf_sorted_idx = np.argsort(rf_feature_importance)
        rf_top_features = X.columns[rf_sorted_idx][-5:][::-1]

        # Store results
        all_results.append({
            'rf_accuracy': rf_accuracy,
            'rf_f1': rf_f1,
            'rf_recall': rf_recall,
            'rf_top_features': rf_top_features
        })

    # Calculate mean and variance for each metric
    metrics = ['rf_accuracy', 'rf_f1', 'rf_recall']
    summary = {}

    for metric in metrics:
        values = [result[metric] for result in all_results]
        summary[f'{metric}_mean'] = np.mean(values)
        summary[f'{metric}_variance'] = np.var(values)

    # Summarize top features
    all_top_features = [result['rf_top_features'] for result in all_results]
    feature_counts = {}
    for top_features in all_top_features:
        for feature in top_features:
            if feature in feature_counts:
                feature_counts[feature] += 1
            else:
                feature_counts[feature] = 1

    most_common_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    summary['most_common_top_features'] = [feature for feature, count in most_common_features]

    # Print results
    print("Summary of Results:")
    for metric in metrics:
        print(f"{metric} - Mean: {summary[f'{metric}_mean']:.4f}, Variance: {summary[f'{metric}_variance']:.4f}")
    print("Most common top features:", ", ".join(summary['most_common_top_features']))

    return summary, all_results


# Example usage:
results_summary, individual_results = test_models_on_multiple_datasets(generated_data_list)
