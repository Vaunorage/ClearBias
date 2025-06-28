import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_generator.main import get_real_data
from methods.subgroup.slicefinder.slice_finder import SliceFinder, Slice
from methods.subgroup.slicefinder.decision_tree import DecisionTree

# ========== DATA PREPARATION ==========
# Load your dataset (using Adult dataset as an example)
data_obj, schema = get_real_data('adult', use_cache=True)

# Drop missing values
data = data_obj.dataframe.dropna()

#%% Split features and target
X = data_obj.xdf
y = data_obj.ydf

#%% Train a model
model = RandomForestClassifier(max_depth=5, n_estimators=10)
model.fit(X, y)

# ========== APPROACH 1: LATTICE SEARCH ==========
print("\n===== USING LATTICE SEARCH APPROACH =====")

# Initialize SliceFinder with your model and data
slice_finder = SliceFinder(model, (X, y))

# Find interesting slices
# Parameters:
# - k: Number of slices to return
# - epsilon: Minimum effect size threshold
# - degree: Maximum complexity of slice filters
# - max_workers: Number of parallel workers
lattice_slices = slice_finder.find_slice(k=5, epsilon=0.3, degree=2, max_workers=4)

#%% Display results
print("\nTop interesting slices found using lattice search:")
for i, s in enumerate(lattice_slices):
    print(f"\n----- Slice {i + 1} -----")
    print("Slice description:")
    for attr, values in s.filters.items():
        # Decode categorical values if needed
        if attr in encoders:
            le = encoders[attr]
            decoded_values = []
            for v in values:
                if len(v) > 1:  # Range
                    try:
                        low = le.inverse_transform([v[0]])[0]
                        high = le.inverse_transform([v[1]])[0]
                        decoded_values.append(f"{low} to {high}")
                    except:
                        decoded_values.append(f"{v[0]} to {v[1]}")
                else:  # Single value
                    try:
                        decoded_values.append(le.inverse_transform([v[0]])[0])
                    except:
                        decoded_values.append(str(v[0]))
            print(f"{attr}: {decoded_values}")
        else:
            # For numerical attributes
            for v in values:
                if len(v) > 1:  # Range
                    print(f"{attr}: {v[0]} to {v[1]}")
                else:  # Single value
                    print(f"{attr}: {v[0]}")

    print(f"Effect size: {s.effect_size:.4f}")
    print(f"Average metric: {s.metric:.4f}")
    print(f"Slice size: {s.size} samples")

#%% ========== APPROACH 2: DECISION TREE ==========
print("\n\n===== USING DECISION TREE APPROACH =====")

# Initialize DecisionTree with your data and model
dt_finder = DecisionTree((X, y), model)

# Build the decision tree
# Parameters:
# - max_depth: Maximum depth of the tree
# - min_size: Minimum number of samples required to split a node
dt_finder = dt_finder.fit(max_depth=3, min_size=100)

# Find interesting slices
# Parameters:
# - k: Number of slices to return
# - min_effect_size: Minimum effect size threshold
dt_slices = dt_finder.recommend_slices(k=5, min_effect_size=0.3)

#%% Display results
print("\nTop interesting slices found using decision tree:")
for i, node in enumerate(dt_slices):
    print(f"\n----- Slice {i + 1} -----")

    # Get the path from root to this node
    ancestry = node.__ancestry__()
    if ancestry:
        print("Path from root:", " → ".join(ancestry))

    # Display the node description
    print(f"Node: {node}")

    # Display effect size and slice size
    print(f"Effect size: {node.eff_size:.4f}")
    print(f"Slice size: {node.size} samples")

# ========== COMPARING THE APPROACHES ==========
print("\n\n===== COMPARISON OF APPROACHES =====")
print(f"Lattice search found {len(lattice_slices)} slices")
print(f"Decision tree found {len(dt_slices)} slices")

# Compare average effect sizes
if lattice_slices:
    avg_effect_lattice = sum(s.effect_size for s in lattice_slices) / len(lattice_slices)
    print(f"Average effect size (lattice): {avg_effect_lattice:.4f}")

if dt_slices:
    avg_effect_dt = sum(node.eff_size for node in dt_slices) / len(dt_slices)
    print(f"Average effect size (decision tree): {avg_effect_dt:.4f}")