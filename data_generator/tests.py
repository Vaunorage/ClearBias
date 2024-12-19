from data_generator.main import generate_data, generate_data_schema, generate_from_real_data

data, schema = generate_from_real_data('adult', use_cache=False)

# %%
# nb_attributes = 20
#
# schema = generate_data_schema(min_number_of_classes=2,
#                               max_number_of_classes=9,
#                               prop_protected_attr=0.4,
#                               nb_attributes=nb_attributes)
#
# data = generate_data(
#     nb_attributes=nb_attributes,
#     nb_groups=100,
#     max_group_size=100,
#     categorical_outcome=True,
#     nb_categories_outcome=4,
#     corr_matrix_randomness=1,
#     categorical_influence=1,
#     data_schema=schema,
#     use_cache=False
# )
#
# print(f"Generated {len(data.dataframe)} samples in {data.nb_groups} groups")

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_distribution_comparison(schema, data, figsize=(15, 10)):
    """
    Plot distribution comparison between schema and generated data.
    """
    # Calculate number of rows needed for subplots
    n_attrs = len(schema.attr_names)
    n_rows = (n_attrs + 2) // 3  # 3 plots per row

    # Create figure
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten()

    # For each attribute
    for idx, (attr_name, protected) in enumerate(zip(schema.attr_names, schema.protected_attr)):
        ax = axes[idx]

        # Get theoretical distribution from schema
        # Get possible values from attr_categories
        # Create theoretical distribution (uniform for now - modify based on actual schema)
        theo_values = [val for val in schema.attr_categories[idx] if val != -1]
        theo_probs = schema.categorical_distribution[attr_name]

        # Calculate actual distribution from data
        if hasattr(data, 'dataframe'):
            actual_dist = data.dataframe[attr_name].value_counts(normalize=True)
        else:
            actual_dist = data[attr_name].value_counts(normalize=True)
        for k, v in zip(theo_values, theo_probs):
            if k not in actual_dist:
                actual_dist[k] = 0

        # Plot bars
        x = np.arange(len(theo_values))
        width = 0.35

        ax.bar(x - width / 2, theo_probs, width, label='Schema', alpha=0.6, color='blue')
        ax.bar(x + width / 2, [actual_dist.get(val, 0) for val in theo_values],
               width, label='Generated', alpha=0.6, color='green')

        # Customize plot
        ax.set_title(f'{attr_name} {"(Protected)" if protected else ""}')
        ax.set_xticks(x)
        ax.set_xticklabels(theo_values)
        ax.set_ylabel('Probability')
        ax.legend()

    # Remove empty subplots if any
    for idx in range(len(schema.attr_names), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    return fig


# Create and show the plot
fig = plot_distribution_comparison(schema, data)
plt.show()


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


# Print statistics
print_distribution_stats(schema, data)
