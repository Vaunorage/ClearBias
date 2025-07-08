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

### **Location**
The method tries to find discrimination by analyzing a **trained model's performance on the data**. It does not modify the model or the training process itself. It operates as a post-hoc debugging tool that takes a model's predictions (and resulting errors) on a dataset and searches for problematic subsets within that data.

### **What they find**
The method finds the **top-K problematic data slices**. A "slice" is a subset of the data defined by a conjunction of predicates on its features. A "problematic" slice is one where a trained ML model performs significantly worse (i.e., has a higher error rate) than its average performance across the entire dataset, according to the scoring function.

The method aims to find discriminated groups (subgroups) relative to the average performance, not to find discriminated individuals.

### **What does the method return in terms of data structure?**
The algorithm returns two main data structures:
1.  **`TS`**: A `K × m` integer-encoded matrix representing the top-K slices found. Each row corresponds to a slice, and the values indicate the feature predicates that define it (with zeros representing "don't care" features).
2.  **`TR`**: A matrix containing the corresponding statistics for each of the top-K slices. This includes the calculated score, total error, average error, and size of each slice.

### **Performance**
The performance was evaluated on its effectiveness (pruning), efficiency (runtime), and scalability using a variety of real-world datasets (Adult, Covtype, KDD98, US Census, Criteo).

*   **Evaluation Environment**: Experiments were run on a powerful scale-up server (112 virtual cores, 768 GB RAM) and a scale-out cluster. The method was implemented in Apache SystemDS, which leverages efficient sparse linear algebra.
*   **Pruning Effectiveness**: The paper shows that the combination of size pruning, score pruning, and handling of parent-child relationships in the search lattice is crucial. Without these techniques, the enumeration of slices becomes computationally infeasible even on small datasets.
*   **End-to-end Runtime & Scalability**:
    *   **Local Performance**: The method is very fast. On the Adult dataset, it completed in **5.6 seconds**, which is significantly faster than the >100s reported for the original SliceFinder work on the same dataset.
    *   **Scalability with Rows**: The method showed excellent scalability with the number of rows, demonstrating near-linear performance degradation as the USCensus dataset was replicated up to 10 times (to ~25M rows).
    *   **Scalability with Columns**: On the massive Criteo dataset (192M rows, 76M one-hot encoded columns), SliceLine was able to effectively enumerate slices and completed in under 45 minutes in a distributed environment, demonstrating its ability to handle extremely high-dimensional, sparse data.

# SliceLine Algorithm Documentation

## Implementation

### Input Parameters

The main function for running the SliceLine algorithm is `run_sliceline()` in the `main.py` file. Here are the input parameters:

**data_obj: DiscriminationData** - This is the primary input parameter, which is an instance of the `DiscriminationData` class containing:
- **dataframe**: The dataset containing features, true outcomes, and predicted outcomes
- **attr_columns**: Feature columns used for slice finding
- **outcome_column**: Column name containing the true outcome values (0/1)
- **y_pred_col**: Column name containing the predicted outcome values (0/1)
- **categorical_columns**: List of categorical feature names
- **attributes**: Dictionary mapping attribute names to boolean values (True if protected)

**K=5** - The number of top slices to find (default: 5)
- This parameter determines how many problematic slices the algorithm will return

**alpha=0.95** - A weight parameter between 0 and 1 (default: 0.95)
- Controls the balance between slice error rate and slice size in the scoring function
- Higher values prioritize finding slices with high error rates, even if they are small
- Lower values prioritize finding larger slices with moderately high error rates

**max_l=3** - Maximum number of predicates in a slice (default: 3)
- Limits the complexity of the slices by restricting how many feature conditions can be combined
- For example, with max_l=3, a slice could have at most 3 conditions like gender=female AND age>30 AND education=PhD

## How the Algorithm Works

SliceLine works by:

1. **Calculating error vectors for False Positives (FP) and False Negatives (FN):**
   - False Positives: Cases where y_true=0 and y_pred=1
   - False Negatives: Cases where y_true=1 and y_pred=0

2. **Using the Slicefinder class to find slices with high error rates for both error types:**
   - For FP errors, it finds slices where the model has a high False Positive Rate
   - For FN errors, it finds slices where the model has a high False Negative Rate

3. **The scoring function used by SliceLine is:**
   ```
   score = α * ((slice_error_rate / overall_error_rate) - 1) - (1 - α) * ((total_rows / slice_size) - 1)
   ```
   where:
   - `slice_error_rate` is the average error on the slice
   - `overall_error_rate` is the average error on the entire dataset
   - `slice_size` is the number of rows in the slice
   - `total_rows` is the total number of rows in the dataset
   - `α` is the weight parameter

## Output

The output of the `run_sliceline()` function is a pandas DataFrame containing the top-K problematic slices found, with the following columns:

### Feature columns
Values indicating the conditions that define each slice:
- **For categorical features**: The specific value that defines the slice condition
- **For numerical features**: The threshold value for the condition
- **None values** indicate "don't care" features (not part of the slice definition)

### Statistical columns
Added by the algorithm:
- **score**: The calculated score for the slice based on the scoring function
- **error**: Total error in the slice (sum of errors)
- **error_rate**: Average error rate in the slice (error / slice size)
- **size**: Number of rows in the slice
- **metric**: Either 'fpr' (False Positive Rate) or 'fnr' (False Negative Rate) indicating the error type

## Example Output

Here's an example of what the output DataFrame might look like:

```python
# Example output DataFrame
"""
   age  education  gender  occupation  ...  score  error  error_rate  size  metric
0   >30      PhD  female     manager  ...   0.87   45.0        0.75    60     fpr
1  None  Masters    male   engineer  ...   0.65   32.0        0.64    50     fpr
2   <25      PhD    None  professor  ...   0.58   28.0        0.70    40     fnr
3  None     None  female     doctor  ...   0.52   38.0        0.63    60     fnr
4   >50  HighSch    male     driver  ...   0.45   25.0        0.62    40     fpr
"""
```

In this example:

- **Row 0** represents a slice where age > 30 AND education = PhD AND gender = female AND occupation = manager, which has a high False Positive Rate
- **Row 2** represents a slice where age < 25 AND education = PhD AND occupation = professor (gender doesn't matter), which has a high False Negative Rate
- The **score** column indicates how problematic each slice is according to the scoring function
- The **error_rate** shows what percentage of predictions in the slice are incorrect
- The **size** column shows how many data points are in each slice

This output allows data scientists to identify specific subgroups where the model is underperforming, which can guide targeted improvements to the model or data collection to address bias and fairness issues.