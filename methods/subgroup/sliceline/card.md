### **Title**
SliceLine: Fast, Linear-Algebra-based Slice Finding for ML Model Debugging

### **Metric**
The method relies on a custom, flexible **Scoring Function** to find problematic data slices. This score quantifies how much worse a model performs on a specific slice compared to its average performance, while also considering the size of the slice.

The score `sc` for a slice `S` is defined as:
`sc = α * ( (se / |S|) / ē - 1 ) - (1 - α) * ( n / |S| - 1 )`

Where:
*   `se / |S|` is the average error on the slice.
*   `ē` is the average error on the entire dataset.
*   `|S|` is the size (number of rows) of the slice.
*   `n` is the size of the entire dataset.
*   `α` is a user-defined weight parameter (between 0 and 1) that balances the importance of the slice's error versus its size. A higher `α` prioritizes slices with high error, even if they are small.

A score `sc > 0` indicates that the model performs worse on the slice than on the overall dataset. The goal is to find slices that maximize this score.

### **Individual discrimination, group, subgroup discrimination**
SliceLine is designed to find **subgroup discrimination** with a high degree of **granularity and intersectionality**.

*   It does not focus on individual discrimination (i.e., comparing one individual to another).
*   It identifies **groups** (or "slices") defined by the conjunction of multiple feature predicates. For example, it can find that a model underperforms for the subgroup where `gender = female` AND `degree = PhD`.
*   By searching through combinations of features, it inherently uncovers intersectional biases that might be missed when looking at single features in isolation.
*   **The implementation specifically targets two types of errors**: False Positive Rate (FPR) and False Negative Rate (FNR) discrimination, running separate analyses for each error type.

### **Location**
The method tries to find discrimination by analyzing a **trained model's performance on the data**. It does not modify the model or the training process itself. It operates as a post-hoc debugging tool that takes a model's predictions (and resulting errors) on a dataset and searches for problematic subsets within that data.

**Implementation Details:**
- Uses a Random Forest model (`model_type='rf'`) as the base classifier when training is needed
- Operates on the training dataframe with predicted outcomes (`training_dataframe_with_ypred`)
- Calculates error vectors for both False Positives and False Negatives separately
- Runs two independent SliceFinder instances, one for each error type

### **What they find**
The method finds the **top-K problematic data slices** for both False Positive and False Negative errors. A "slice" is a subset of the data defined by a conjunction of predicates on its features. A "problematic" slice is one where a trained ML model performs significantly worse (i.e., has a higher error rate) than its average performance across the entire dataset, according to the scoring function.

**Specific Error Types:**
- **False Positive slices**: Subgroups where the model incorrectly predicts positive outcomes (y_true=0, y_pred=1)
- **False Negative slices**: Subgroups where the model incorrectly predicts negative outcomes (y_true=1, y_pred=0)

The method aims to find discriminated groups (subgroups) relative to the average performance, not to find discriminated individuals.

### **What does the method return in terms of data structure?**
The algorithm returns a **pandas DataFrame** containing the combined top-K slices from both FPR and FNR analyses, along with a **metrics dictionary**.

**DataFrame Structure:**
1. **Feature columns**: Values indicating the conditions that define each slice
   - For categorical features: specific values (e.g., "female", "PhD")
   - For numerical features: threshold values (e.g., ">30", "<25")
   - `None` values indicate "don't care" features not part of the slice definition

2. **Statistical columns** added by the algorithm:
   - **score**: The calculated score for the slice based on the scoring function
   - **error**: Total error in the slice (sum of errors)
   - **error_rate**: Average error rate in the slice (error / slice size)
   - **size**: Number of rows in the slice
   - **metric**: Either 'fpr' (False Positive Rate) or 'fnr' (False Negative Rate)
   - **subgroup_key**: String representation of the slice conditions
   - **diff_outcome**: Difference from mean outcome
   - **slice**: Human-readable description of the slice conditions

**Metrics Dictionary:**
- **RUNTIME**: Total execution time in seconds
- **TSN**: Total Sample Number (total input samples tested)
- **DSN**: Discriminatory Sample Number (samples in found slices)
- **SUR**: Success Rate (DSN/TSN ratio)
- **DSS**: Discriminatory Sample Search time (RUNTIME/DSN ratio)

### **Performance**
The performance was evaluated on its effectiveness (pruning), efficiency (runtime), and scalability using a variety of real-world datasets (Adult, Covtype, KDD98, US Census, Criteo).

**Implementation Performance Characteristics:**
*   **Default Parameters**: K=5 (top slices), alpha=0.95 (error-focused), max_l=3 (max predicates), max_runtime_seconds=60
*   **Error Handling**: Gracefully handles cases where no significant slices are found
*   **Caching**: Supports model training cache to improve repeated runs
*   **Memory Management**: Uses pandas DataFrames for efficient data manipulation
*   **Timeout Protection**: Built-in maximum runtime limit to prevent infinite execution

*   **Evaluation Environment**: Experiments were run on a powerful scale-up server (112 virtual cores, 768 GB RAM) and a scale-out cluster. The method was implemented in Apache SystemDS, which leverages efficient sparse linear algebra.
*   **Pruning Effectiveness**: The paper shows that the combination of size pruning, score pruning, and handling of parent-child relationships in the search lattice is crucial. Without these techniques, the enumeration of slices becomes computationally infeasible even on small datasets.
*   **End-to-end Runtime & Scalability**:
    *   **Local Performance**: The method is very fast. On the Adult dataset, it completed in **5.6 seconds**, which is significantly faster than the >100s reported for the original SliceFinder work on the same dataset.
    *   **Scalability with Rows**: The method showed excellent scalability with the number of rows, demonstrating near-linear performance degradation as the USCensus dataset was replicated up to 10 times (to ~25M rows).
    *   **Scalability with Columns**: On the massive Criteo dataset (192M rows, 76M one-hot encoded columns), SliceLine was able to effectively enumerate slices and completed in under 45 minutes in a distributed environment, demonstrating its ability to handle extremely high-dimensional, sparse data.

### **Result DataFrame Example**

Here's an example of the actual output DataFrame structure returned by SliceLine:

```csv
   Attr1_X  Attr2_X  Attr3_X  Attr4_X  Attr5_X  Attr6_X  Attr7_X  Attr8_T  Attr9_T  Attr10_X  Attr11_X  Attr12_X  Attr13_X  slice_score        sum_slice_error  max_slice_error  slice_size  slice_average_error  metric  subgroup_key                      outcome  diff_outcome  slice
0     None     None     None       14     None     None        0     None     None      None      None        67      None          NaN                NaN              NaN         NaN                  NaN     fpr  *|*|*|14|*|*|0|*|*|*|*|67|*           0        -0.125  Attr4_X=14, Attr7_X=0, Attr12_X=67
1     None     None        9     None     None     None        0     None     None      None      None        67      None          NaN                NaN              NaN         NaN                  NaN     fpr  *|*|9|*|*|*|0|*|*|*|*|67|*            0        -0.125  Attr3_X=9, Attr7_X=0, Attr12_X=67
2      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN       NaN       NaN       NaN       NaN    4.748395               66.0              1.0      1470.0            0.044898     fpr  *|*|*|*|*|*|*|*|*|*|*|*|*             0        -0.125  slice_score=4.75, sum_slice_error=66.0
3      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN       NaN       NaN       NaN       NaN    4.748395               66.0              1.0      1470.0            0.044898     fpr  *|*|*|*|*|*|*|*|*|*|*|*|*             0        -0.125  slice_score=4.75, sum_slice_error=66.0
4     None     None     None        7     None     None        0     None     None      None      None        67      None          NaN                NaN              NaN         NaN                  NaN     fnr  *|*|*|7|*|*|0|*|*|*|*|67|*            1         0.875  Attr4_X=7, Attr7_X=0, Attr12_X=67
```

**Key aspects of the result DataFrame:**

- **Feature columns (Attr1_X - Attr13_X)**: Show the specific conditions defining each slice
  - `None`: Feature not part of the slice definition (don't care)
  - `NaN`: Used when no specific slice conditions are found
  - Specific values (e.g., `14`, `9`, `0`, `67`): Exact conditions for the slice

- **Statistical columns**:
  - `slice_score`: The calculated discrimination score (higher = more problematic)
  - `sum_slice_error`: Total number of errors in the slice
  - `max_slice_error`: Maximum error value in the slice
  - `slice_size`: Number of data points in the slice
  - `slice_average_error`: Average error rate within the slice

- **Metadata columns**:
  - `metric`: Either 'fpr' (False Positive Rate) or 'fnr' (False Negative Rate)
  - `subgroup_key`: Pipe-separated representation of slice conditions (`*` = don't care)
  - `outcome`: Predicted outcome for this slice (0 or 1)
  - `diff_outcome`: Difference from the overall mean outcome
  - `slice`: Human-readable description of the slice conditions

**Interpretation:**
- Rows 0, 1: Specific slices with precise feature conditions but NaN scores (may indicate insufficient data)
- Rows 2, 3: General slices with high discrimination scores (4.75) affecting large populations (1470 samples)
- Row 4: False Negative slice with different feature conditions

### **Usage Example**
```python
from data_generator.main import get_real_data

# Load data
data_obj, schema = get_real_data('adult', use_cache=False)

# Run sliceline
results_df, metrics = run_sliceline(
    data=data_obj, 
    K=5,                    # Number of top slices to find
    alpha=0.95,             # Weight parameter (0-1)
    max_l=3,                # Max predicates per slice
    max_runtime_seconds=60, # Timeout limit
    random_state=42,        # Reproducibility
    use_cache=True,         # Enable model caching
    logger=logger           # Optional logging
)

if not results_df.empty:
    print("Top Slices found by Sliceline:")
    print(results_df[['slice', 'slice_score', 'slice_average_error', 'slice_size', 'metric']])
    print(f"Metrics: {metrics}")
```