# DivExplorer Algorithm Implementation

## Input Parameters

The DivExplorer algorithm is implemented in the `run_divexplorer` function in `main.py`. Here are the key input parameters:

**data**: A `DiscriminationData` object that contains:
- `training_dataframe_with_ypred`: DataFrame containing the dataset with both true labels and predicted labels
- `outcome_column`: Name of the column containing the true class labels
- `y_pred_col`: Name of the column containing the predicted class labels
- `attributes`: List of attribute names in the dataset
- `protected_attributes`: List of protected/sensitive attribute names
- `attr_columns`: List of all attribute column names

**K**: Integer (default=5) - The number of top divergent patterns to return for each metric (FPR and FNR)

**max_runtime_seconds**: Integer (default=60) - Maximum execution time in seconds before timeout

**min_support**: Float (default=0.05) - The minimum support threshold for frequent pattern mining. Only patterns that appear in at least this fraction of the dataset will be considered.

**random_state**: Integer (default=42) - Random seed for reproducibility

**use_cache**: Boolean (default=True) - Whether to use cached model training results

### Internal Parameters (used by the underlying algorithm)

When the function calls `FP_DivergenceExplorer`, it uses these parameters:

- **th_redundancy**: Integer (default=0) - Threshold for redundancy when selecting top-K patterns. Used to filter out similar patterns.

## How the Algorithm Works

1. **Model Training**: The algorithm first trains a Random Forest model using `train_sklearn_model` on the training data to generate predictions if not already available.

2. **Binary Classification Check**: It verifies that the problem is a binary classification task. If not, it returns empty results.

3. **Frequent Pattern Mining**: It initializes the `FP_DivergenceExplorer` with the dataset containing true and predicted class labels, then uses frequent pattern mining to identify patterns (subgroups) in the data with the `getFrequentPatternDivergence` method.

4. **Divergence Calculation**: For each pattern, it calculates divergence metrics:
   - **d_fpr**: Divergence in False Positive Rate - how much the FPR in this subgroup differs from the global FPR
   - **d_fnr**: Divergence in False Negative Rate - how much the FNR in this subgroup differs from the global FNR

5. **Top-K Selection**: It selects the top-K patterns with the highest absolute divergence for each metric using `getDivergenceTopKDf`.

6. **Result Processing**: It processes the results into a standardized format, filling missing attribute values with medians and generating subgroup keys.

7. **Timeout Handling**: The algorithm includes timeout functionality to prevent excessive runtime, logging a timeout message if the time limit is exceeded.

## Output

The function returns a tuple containing:

1. **result_df**: A pandas DataFrame containing the identified discriminatory subgroups with the following columns:
   - **Attribute columns**: One column for each attribute in the dataset, with values indicating the specific attribute value for this subgroup, or filled with median values if not part of the pattern
   - **outcome_column**: Predicted outcomes for the subgroup
   - **subgroup_key**: A string identifier for each subgroup using '|' as separator and '*' for wildcard values
   - **diff_outcome**: Absolute difference between subgroup outcome and mean outcome

2. **metrics**: A dictionary containing performance metrics:
   - **TSN** (Total Sample Number): Total number of instances in the dataset
   - **DSN** (Discriminatory Subgroup Number): Number of unique discriminatory subgroups found
   - **DSR** (Discriminatory Subgroup Ratio): Ratio of DSN to TSN
   - **DSS** (Discriminatory Subgroup Speed): Average time per discriminatory subgroup found

## Example

Let's say we have a dataset about loan applications with attributes like 'age', 'income', 'education', and a binary outcome 'loan_approved':

### Input Example:

```python
from data_generator.main import get_real_data, DiscriminationData
from methods.subgroup.divexplorer.main import run_divexplorer

# Get a dataset (e.g., the adult dataset)
data_obj, schema = get_real_data('adult', use_cache=False)

# Run DivExplorer with K=5 (top-5 patterns for each metric)
results_df, metrics = run_divexplorer(data_obj, K=5, max_runtime_seconds=60)
```

### Output Example:

```python
# results_df might look like:
   age  sex  education  income  marital_status  occupation  loan_approved  subgroup_key      diff_outcome
0   45    1         12   50000              1           5              1      45|1|12|50000|1|5      0.23
1   25    0          8   25000              0           3              0      25|0|8|25000|0|3       0.18
2   35    1         16   75000              1           2              1      35|1|16|75000|1|2      0.15

# metrics might look like:
{
    'TSN': 32561,      # Total samples in dataset
    'DSN': 150,        # Number of discriminatory subgroups found
    'DSR': 0.0046,     # Ratio of subgroups to total samples
    'DSS': 0.4         # Average seconds per subgroup found
}
```

## Key Features

- **Timeout Protection**: Prevents infinite runtime with configurable timeout
- **Binary Classification Focus**: Only processes binary classification problems
- **Frequent Pattern Mining**: Uses efficient pattern mining to identify subgroups
- **Dual Metric Analysis**: Analyzes both False Positive Rate and False Negative Rate divergence
- **Standardized Output**: Provides consistent output format with performance metrics
- **Robust Error Handling**: Includes fallback mechanisms and logging for debugging

This algorithm is particularly useful for identifying specific subgroups where a machine learning model exhibits biased behavior, helping to detect algorithmic discrimination in binary classification tasks.