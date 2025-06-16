import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
from itertools import combinations
import colorsys
import warnings

from data_generator.main import DiscriminationData

warnings.filterwarnings('ignore')


def generate_distinct_colors(n):
    """Generate n visually distinct colors using golden ratio method"""
    colors = []
    for i in range(n):
        hue = i * 0.618033988749895 % 1
        saturation = 0.7 + np.random.random() * 0.3
        value = 0.8 + np.random.random() * 0.2
        colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
    return colors


def calculate_inverse_sizes(group_sizes, min_size=100, max_size=800):
    """Calculate inverse sizes for groups - smaller groups get bigger markers"""
    inverse_sizes = 1 / group_sizes
    scaled_sizes = (inverse_sizes - inverse_sizes.min()) / (inverse_sizes.max() - inverse_sizes.min())
    scaled_sizes = scaled_sizes * (max_size - min_size) + min_size
    return scaled_sizes


def estimate_optimal_clusters(X, max_clusters=20):
    """Estimate optimal number of clusters using elbow method"""
    inertias = []
    K = range(1, min(max_clusters + 1, len(X)))
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Calculate the elbow point
    diffs = np.diff(inertias)
    diffs_r = diffs[1:] / diffs[:-1]
    elbow_point = np.argmin(diffs_r) + 2
    return min(elbow_point, max_clusters)


def visualize_enhanced_clustering(data):
    combinations_df = DiscriminationData.generate_individual_synth_combinations(data.dataframe, drop_duplicates=False)

    if len(combinations_df) == 0:
        print("No valid combinations found!")
        return None, None

    # Prepare features
    feature_cols = []
    for feat in data.feature_names:
        feature_cols.extend([f'{feat}_1', f'{feat}_2'])

    X = combinations_df[feature_cols].copy()

    # Encode categorical variables
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # t-SNE for dimensionality reduction
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(X) - 1),
        random_state=42,
        learning_rate='auto',
        init='pca',
        n_iter=1000
    )
    embedding = tsne.fit_transform(X_scaled)

    # Estimate optimal number of clusters
    n_clusters = estimate_optimal_clusters(embedding)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embedding)

    # Calculate cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Create plot with dark background
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_facecolor('#1a1a1a')
    fig.set_facecolor('#1a1a1a')

    # Generate distinct colors
    unique_groups = combinations_df['group_key'].unique()
    colors = generate_distinct_colors(len(unique_groups))
    color_dict = dict(zip(unique_groups, colors))

    # Calculate group sizes and inverse size scaling
    group_sizes = data.dataframe.groupby('group_key').apply(
        lambda x: len(x[data.feature_names].drop_duplicates())
    )
    size_dict = calculate_inverse_sizes(group_sizes).to_dict()
    groups_by_size = sorted(unique_groups, key=lambda x: group_sizes[x], reverse=True)

    # Plot with enhanced visual separation
    for group in groups_by_size:
        mask = combinations_df['group_key'] == group
        group_embedding = embedding[mask]
        color = color_dict[group]
        size = size_dict[group]

        # Plot cluster regions
        group_clusters = cluster_labels[mask]
        unique_clusters = np.unique(group_clusters)

        for cluster in unique_clusters:
            cluster_mask = group_clusters == cluster
            cluster_points = group_embedding[cluster_mask]

            # Add cluster boundary glow
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c=[color],
                s=size * 2,
                alpha=0.1,
                label=None
            )

        # Add glow effect for points
        plt.scatter(
            group_embedding[:, 0],
            group_embedding[:, 1],
            c=[color],
            s=size * 1.5,
            alpha=0.2,
            label=None
        )

        # Main points
        plt.scatter(
            group_embedding[:, 0],
            group_embedding[:, 1],
            c=[color],
            label=f'Group {group}\n({group_sizes[group]} unique)',
            alpha=0.8,
            s=size,
            edgecolor='white',
            linewidth=0.5
        )

    # Plot cluster centers
    plt.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        c='white',
        marker='x',
        s=100,
        linewidths=2,
        label='Cluster Centers'
    )

    plt.title('Enhanced Clustering of Individual Combinations\n(with K-Means clustering and t-SNE)',
              fontsize=16,
              color='white',
              pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=12, color='white')
    plt.ylabel('t-SNE Dimension 2', fontsize=12, color='white')

    # Add statistics
    stats_text = (
        f'Total Unique Individuals: {sum(group_sizes):,}\n'
        f'Total Combinations: {len(combinations_df):,}\n'
        f'Number of Groups: {len(unique_groups)}\n'
        f'Group Size Range: {min(group_sizes)} - {max(group_sizes)} unique\n'
        f'Number of Clusters: {n_clusters}'
    )

    plt.text(
        0.02, 0.98,
        stats_text,
        transform=plt.gca().transAxes,
        bbox=dict(
            facecolor='black',
            alpha=0.8,
            edgecolor='white',
            boxstyle='round,pad=0.5'
        ),
        color='white',
        verticalalignment='top',
        fontsize=10
    )

    # Enhanced legend
    if len(unique_groups) > 20:
        top_groups = sorted(unique_groups, key=lambda x: group_sizes[x])[:20]
        handles, labels = plt.gca().get_legend_handles_labels()
        handle_dict = dict(zip(labels, handles))
        legend = plt.legend(
            [handle_dict[f'Group {g}\n({group_sizes[g]} unique)'] for g in top_groups],
            [f'Group {g}\n({group_sizes[g]} unique)' for g in top_groups],
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            title='Smallest 20 Groups',
            fontsize=8,
            title_fontsize=10
        )
    else:
        legend = plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            title='Groups (by size)',
            fontsize=8,
            title_fontsize=10
        )

    legend.get_frame().set_facecolor('#2a2a2a')
    legend.get_frame().set_edgecolor('white')
    plt.setp(legend.get_title(), color='white')

    plt.grid(True, alpha=0.1)
    plt.tight_layout()

    return plt.gcf(), combinations_df, cluster_labels, cluster_centers


if __name__ == "__main__":
    from main import generate_data

    data = generate_data(nb_groups=5, nb_attributes=10, min_group_size=90, max_group_size=100)
    fig, combinations_df, cluster_labels, cluster_centers = visualize_enhanced_clustering(data)
    plt.show()