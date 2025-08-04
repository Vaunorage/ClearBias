### **Title**
Identifying Significant Predictive Bias in Classifiers using BiasScan

### **Metric**
The method relies on a statistical measure of **predictive bias**, which quantifies the discrepancy between a classifier's predicted outcomes and the observed outcomes for subgroups. The implementation uses the **MDSS (Multidimensional Subset Scanning)** detector with configurable scoring functions.

The core approach compares:
*   **Observations:** Actual outcomes from the dataset (`data.outcome_column`)
*   **Expectations:** Model predictions on the same data using a trained RandomForest classifier

The method performs two complementary bias scans:
1. **Overprediction Scan:** Identifies subgroups where the model predicts higher outcomes than observed
2. **Underprediction Scan:** Identifies subgroups where the model predicts lower outcomes than observed

### **Discrimination Granularity**
*   **Type:** The method is designed to find **subgroup discrimination** through multidimensional subset scanning.
*   **Granularity & Intersectionality:** The algorithm analyzes all possible intersectional combinations of attribute values across the feature space. It can identify complex, multi-dimensional subgroups defined by specific combinations of feature values (e.g., "individuals with attribute1=1, attribute2=0, attribute3=1"). The `make_products_df` function generates Cartesian products of attribute combinations to create comprehensive subgroup representations.

### **Location of Discrimination**
The method locates discrimination within the **model's predictions** by comparing predicted outcomes against actual outcomes. It performs model checking by:
- Training a RandomForest classifier on the data
- Generating predictions for all data points
- Using MDSS to identify regions where predictions systematically deviate from observations
- Scanning both overprediction and underprediction patterns

### **What They Find**
The method identifies **subgroups where the classifier exhibits systematic prediction bias**. Specifically, it finds:
1. **Overpredicted subgroups:** Where model predictions are consistently higher than actual outcomes
2. **Underpredicted subgroups:** Where model predictions are consistently lower than actual outcomes

The algorithm generates subgroup representations and calculates outcome differences from the mean to quantify bias magnitude.

### **Output Data Structure**
The method returns a tuple containing:

#### result_df (DataFrame)
Contains identified biased subgroups with columns:
- **Attribute columns:** Values defining each subgroup (from `data.attributes`)
- **outcome:** Predicted outcome for the subgroup (from `data.outcome_column`)
- **subgroup_key:** String representation of subgroup attributes (pipe-separated values)
- **diff_outcome:** Absolute difference between subgroup outcome and dataset mean outcome

#### metrics (Dictionary)
Performance metrics including:
- **TSN (Total Sample Number):** Total number of instances in the dataset
- **DSN (Detected Subgroup Number):** Number of unique subgroups identified
- **DSR (Detection Success Rate):** Ratio of detected subgroups to total samples (DSN/TSN)
- **DSS (Detection Speed Score):** Time per detected subgroup (total_time/DSN)

### **Performance**
The algorithm's performance is measured through:

*   **Computational Efficiency:**
    - Memory-efficient chunk processing in `make_products_df` (default chunk_size=10000)
    - Configurable number of iterations (`bias_scan_num_iters`, default=50)
    - Optional runtime limits (`max_runtime_seconds`)

*   **Detection Capabilities:**
    - Identifies both overprediction and underprediction bias patterns
    - Handles various data types through configurable modes ('binary', 'continuous', 'nominal', 'ordinal')
    - Supports multiple scoring functions ('Bernoulli', 'BerkJones', 'Gaussian', 'Poisson')

*   **Model Integration:**
    - Uses RandomForest classifier with configurable parameters
    - Leverages existing model training utilities (`train_sklearn_model`)
    - Supports caching for improved performance (`use_cache=True`)

### **Input Parameters**

The `run_bias_scan` function accepts:
- **data:** DiscriminationData object containing dataset, attributes, and outcome information
- **random_state:** Seed for reproducibility (default=42)
- **bias_scan_num_iters:** Number of bias scan iterations (default=50)
- **bias_scan_scoring:** Scoring function ('Poisson', 'Bernoulli', 'BerkJones', 'Gaussian')
- **bias_scan_favorable_value:** Definition of favorable outcomes ('high', 'low', or specific value)
- **bias_scan_mode:** Data type handling ('ordinal', 'binary', 'continuous', 'nominal')
- **max_runtime_seconds:** Optional runtime limit
- **use_cache:** Enable caching for performance optimization

### **Example Output**

Here's an example of what the algorithm returns:

#### result_df Example:
```
   attr1  attr2  attr3  outcome  subgroup_key  diff_outcome
0    1.0    0.0    1.0        1      1|0|1|1           0.25
1    0.0    1.0    0.0        0      0|1|0|0           0.18
2    1.0    1.0    0.0        1      1|1|0|1           0.32
3    0.0    0.0    1.0        0      0|0|1|0           0.15
4    1.0    0.0    0.0        1      1|0|0|1           0.28
```

#### metrics Example:
```python
{
    'TSN': 1000,        # Total number of instances in dataset
    'DSN': 45,          # Number of unique subgroups detected
    'DSR': 0.045,       # Detection success rate (45/1000)
    'DSS': 0.022        # Detection speed: 0.022 seconds per subgroup
}
```

**Interpretation:**
- Each row in `result_df` represents a biased subgroup identified by the algorithm
- `subgroup_key` provides a human-readable identifier for each subgroup (pipe-separated attribute values)
- `diff_outcome` shows how much each subgroup's outcome deviates from the dataset mean (higher values indicate stronger bias)
- The metrics help assess the algorithm's performance: detecting 45 biased subgroups from 1000 total instances in approximately 1 second total runtime

### **Implementation Notes**
- The algorithm processes subgroups in memory-efficient chunks to handle large combinatorial spaces
- Handles missing values by filling with median values from training data
- Removes duplicates from generated subgroup combinations
- Calculates absolute deviations from mean outcomes to quantify bias magnitude
- Supports both overprediction and underprediction bias detection in a single run

```python
from data_generator.main import generate_data
from methods.subgroup.biasscan.algo import run_bias_scan

# Generate synthetic data with controlled bias
ge = generate_data(
    nb_attributes=6,
    min_number_of_classes=2,
    max_number_of_classes=4,
    prop_protected_attr=0.1,
    nb_groups=100,
    max_group_size=100,
    categorical_outcome=True,
    nb_categories_outcome=4,
    use_cache=True
)

# Run the BiasScan algorithm
result_df, report = run_bias_scan(ge, test_size=0.3, random_state=42, n_estimators=200, bias_scan_num_iters=100,
                                  bias_scan_scoring='Poisson', bias_scan_favorable_value='high',
                                  bias_scan_mode='ordinal')

# Print results
print(result_df)
print(f"Classification Report:\n{report}")
```

The BiasScan algorithm is particularly useful for identifying specific subgroups in the data where a model exhibits biased behavior, allowing for targeted bias mitigation strategies.