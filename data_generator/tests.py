from data_generator.main import generate_data, generate_data_schema, generate_from_real_data, get_real_data, \
    is_numeric_column

ll = generate_from_real_data('adult')
data, schema = get_real_data('adult')

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
