### Title
Automated Data Slicing for Model Validation: A Big data - Al Integration Approach

### Metric
The method identifies problematic slices by analyzing the model's performance, primarily using a **loss function** (e.g., log loss). A slice is considered problematic if it meets two criteria based on this loss:

1.  **High Effect Size (φ):** The magnitude of the difference in the average loss between a slice and its counterpart (the rest of the data) must be large. The paper uses Cohen's d to quantify this, indicating how many standard deviations separate the two loss distributions.
2.  **Statistical Significance:** The difference in loss must be statistically significant. This is verified using a hypothesis test (**Welch's t-test**) to ensure the observed poor performance is not due to random chance.

### Granularity and Intersectionality
The method focuses on **subgroup discrimination**. It is designed to find underperforming slices at various levels of granularity:

*   **Group Discrimination:** It can identify simple groups where the model underperforms, defined by a single feature value (e.g., `Sex = Male`).
*   **Intersectionality (Subgroup):** The core strength of the method is finding intersectional subgroups where performance degrades. The search algorithms (Lattice Search and Decision Tree) explicitly construct slices as conjunctions of multiple feature-value pairs (e.g., `Marital Status ≠ Married-civ-spouse` AND `Capital Gain < 7298` AND `Age < 28`).

The method does not analyze discrimination at the individual level.

### Location
The method identifies discrimination within the **model's performance on a validation dataset**. It is a model validation tool that analyzes a trained model's predictions. By "slicing data to identify subsets of the validation data where the model performs poorly," it traces poor aggregate performance metrics back to specific, interpretable cohorts in the data.

### What They Find
The goal is to automatically discover and present to the user a set of **large, interpretable, and problematic data slices**.

*   **Problematic:** Slices where the model's loss is significantly higher than on the rest of the data, as determined by effect size and statistical significance.
*   **Interpretable:** Slices are defined by a simple and understandable predicate (a conjunction of a few feature-value conditions), making it easy for a human to understand the specific demographic or data cohort that is affected. This is contrasted with non-interpretable clusters.
*   **Large:** The method prioritizes larger slices, as these have a greater impact on the overall model quality and are less likely to be statistical noise.

### Data Structure Returned
The method returns a **ranked list of the top-k problematic data slices**. Each element in the list represents one slice and contains:
1.  **A Predicate:** A conjunction of feature-value conditions that defines the slice (e.g., `Marital Status = Married-civ-spouse`).
2.  **Associated Metrics:** Key statistics for the slice, including its **size** (number of data points), average **log loss**, and calculated **effect size**.

This output is presented in an interactive UI with a scatter plot and a sortable table for exploration (Figure 3).

### Performance Evaluation
The performance of the proposed methods, **Lattice Search (LS)** and **Decision Tree (DT)**, was evaluated against a **Clustering (CL)** baseline.

*   **Evaluation Method:**
    *   **Accuracy:** Since the ground truth for problematic slices is unknown in real data, the authors injected known problematic slices by randomly flipping labels for certain subgroups in synthetic and real (UCI Census) datasets. Performance was then measured using **precision, recall, and accuracy (harmonic mean)** in identifying these injected slices.
    *   **Scalability:** Runtimes were measured against increasing dataset sample sizes, number of parallel workers, and number of recommendations requested (`k`).
    *   **Slice Quality:** The average effect size and average slice size of the recommended slices were compared.
    *   **False Discovery:** The effectiveness of their `α-investing` technique was compared against standard Bonferroni and Benjamini-Hochberg procedures.

*   **Results:**
    *   **Accuracy:** LS and DT significantly **outperformed** the clustering baseline. LS was generally more accurate than DT because it can find overlapping problematic slices.
    *   **Slice Quality:** LS and DT found slices with much **higher effect sizes** compared to clustering, which tended to find large but non-problematic groups.
    *   **Scalability:** The methods scaled **linearly with data size** and were effective even on very small data samples (~1% of the data), demonstrating efficiency. LS also showed improved runtime with parallelization.
    *   **Interpretability:** The slices produced by LS and DT were shown to be easily interpretable, defined by a small number of feature conditions (Table 2).

# SliceFinder Algorithm

## Implementation

SliceFinder is a bias detection algorithm that identifies subgroups (slices) in a dataset where a model performs differently compared to its overall performance. It uses two approaches: lattice search and decision tree-based search to find these interesting slices.

## Input Parameters

The main function `run_slicefinder` accepts the following parameters:

### Core Parameters

- **`data_obj`**: Object
  - Contains the dataset information with attributes:
    - `dataframe`: The complete dataset
    - `xdf`: Features dataframe
    - `ydf`: Target variable dataframe

- **`approach`**: String, default="both"
  - Which approach to use for finding slices:
    - `"lattice"`: Only use lattice search approach
    - `"decision_tree"`: Only use decision tree approach
    - `"both"`: Use both approaches

- **`model`**: Object, optional
  - Pre-trained model to use for slice finding
  - If None, a RandomForestClassifier will be trained automatically

### Model Training Parameters

- **`max_depth`**: int, default=5
  - Maximum depth for RandomForestClassifier
- **`n_estimators`**: int, default=10
  - Number of estimators for RandomForestClassifier

### Common Slice Parameters

- **`k`**: int, default=5
  - Number of slices to return
- **`epsilon`**: float, default=0.3
  - Minimum effect size threshold for considering a slice interesting

### Lattice Search Specific Parameters

- **`degree`**: int, default=2
  - Maximum complexity of slice filters (number of conditions combined)
- **`max_workers`**: int, default=4
  - Number of parallel workers for lattice search

### Decision Tree Specific Parameters

- **`dt_max_depth`**: int, default=3
  - Maximum depth for decision tree approach
- **`min_size`**: int, default=100
  - Minimum number of samples required to split a node
- **`min_effect_size`**: float, default=0.3
  - Minimum effect size threshold for decision tree slices

### Display Options

- **`verbose`**: bool, default=True
  - Whether to print detailed results during execution
- **`drop_na`**: bool, default=True
  - Whether to drop missing values from data

## Output

### Lattice Slices in Detail

The `lattice_slices` output is generated by the lattice search approach of SliceFinder. Each lattice slice represents a subgroup of the data where the model performs differently compared to its overall performance.

#### Structure of a Lattice Slice

Each `Slice` object in the lattice approach contains:

- **`filters`**: A dictionary mapping attributes to conditions
  - Format: `{attribute_name: [[[condition_value, condition_upper_bound]]]}`
  - Example: `{'age': [[[30, 45]]]}` means "30 <= age < 45"
  - Example: `{'gender': [['Male']]}` means "gender = Male"

- **`data_idx`**: The indices of data points that belong to this slice

- **`size`**: Number of data points in the slice

- **`effect_size`**: The difference in model performance for this slice compared to overall performance
  - Higher values indicate larger performance differences
  - Calculated using statistical methods comparing slice metrics to reference metrics

- **`metric`**: The actual performance metric value for this slice
  - For classification, this could be log loss, accuracy, etc.

#### Lattice Slices DataFrame

When converted to a DataFrame using `lattice_slices_to_dataframe()`, each row represents a slice with these columns:

- **`slice_index`**: Index of the slice in the original list
- **`slice_size`**: Number of samples in the slice
- **`effect_size`**: Magnitude of performance difference (higher = more interesting)
- **`metric`**: Actual performance metric value for this slice
- **Feature columns**: For each feature in the dataset:
  - If the feature is used in the slice filter: Shows the condition (e.g., "≥ 30")
  - If not used: Shows None

#### Concrete Example of Lattice Slices DataFrame

```python
# Example lattice_slices DataFrame with real-world values
lattice_slices_df = pd.DataFrame({
    'slice_index': [0, 1, 2],
    'slice_size': [1200, 800, 500],
    'effect_size': [0.42, 0.38, 0.31],
    'metric': [0.85, 0.62, 0.71],
    'age': ['≥ 30', None, '≥ 45'],
    'education': [None, '≤ Bachelor', None],
    'income': ['< 50000', '< 50000', None],
    'gender': [None, 'Female', 'Male'],
    'occupation': [None, None, 'Professional'],
    'marital_status': ['Married', None, None]
})
```

In this example:

- **Slice 0**: People who are 30 or older, married, with income less than $50,000
- **Slice 1**: Women with education up to Bachelor's degree and income less than $50,000
- **Slice 2**: Men who are 45 or older in professional occupations

### Decision Tree Slices in Detail

The `dt_slices` output is generated by the decision tree approach of SliceFinder. This approach builds a decision tree to identify regions of the feature space where model performance differs significantly.

#### Structure of Decision Tree Slices

Each decision tree slice is represented by a Node object with:

- **`desc`**: Description of the split at this node (feature and threshold)
  - Example: `['age', 30]` means a split on the 'age' feature at value 30

- **`size`**: Number of samples in this node

- **`eff_size`**: Effect size (performance difference) at this node

- **`ancestry`**: Method that returns the path from root to this node

#### Decision Tree Slices DataFrame

When converted to a DataFrame using `create_dataframe_from_nodes_for_tree_method()`, each row represents a decision rule with these columns:

- **`case`**: Case identifier (e.g., "case 1")
- **`feature`**: Feature used for the split
- **`feature_name`**: Name of the feature (same as feature)
- **`threshold`**: Threshold value for the split
- **`operator`**: Comparison operator (e.g., "<", "≥")
- **`val_diff_outcome`**: Effect size (difference in performance)
- **`order`**: Order of the split in the path (1 for root split, 2 for next level, etc.)

#### Concrete Example of Decision Tree Slices DataFrame

```python
# Example dt_slices DataFrame with real-world values
dt_slices_df = pd.DataFrame({
    'case': ['case 1', 'case 1', 'case 2', 'case 2', 'case 3'],
    'feature': ['age', 'income', 'education', 'gender', 'occupation'],
    'feature_name': ['age', 'income', 'education', 'gender', 'occupation'],
    'threshold': [30, 50000, 'Bachelor', 'Female', 'Professional'],
    'operator': ['≥', '<', '≤', '=', '='],
    'val_diff_outcome': [0.42, 0.38, 0.31, 0.29, 0.27],
    'order': [1, 2, 1, 2, 1]
})
```

In this example:

- **Case 1**: Represents a path in the decision tree where age ≥ 30 AND income < 50000
- **Case 2**: Represents a path where education ≤ Bachelor AND gender = Female
- **Case 3**: Represents a path where occupation = Professional

## Key Differences Between Lattice and Decision Tree Slices

### Representation
- **Lattice slices** represent each subgroup as a single row with all feature conditions
- **Decision tree slices** represent each decision rule as a separate row, with multiple rows forming a path

### Structure
- **Lattice slices** can have arbitrary combinations of features
- **Decision tree slices** follow hierarchical paths in a tree structure

### Interpretation
- **Lattice slices** are more intuitive for understanding complete subgroup definitions
- **Decision tree slices** better show the hierarchical importance of features

### Complexity Control
- **Lattice search** controls complexity via the `degree` parameter (max conditions per slice)
- **Decision tree approach** controls complexity via `dt_max_depth` (max depth of tree)

Both approaches aim to identify subgroups with significant performance differences, but they use different algorithms to explore the feature space and may find different interesting slices in the data.