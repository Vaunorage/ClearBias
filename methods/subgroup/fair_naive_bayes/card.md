### Title: Learning Fair Naive Bayes Classifiers by Discovering and Eliminating Discrimination Patterns

### Metric: How discrimination is measured

The algorithm uses a unified metric to detect discrimination at various levels of granularity, implemented through probabilistic inference on a trained Naive Bayes model.

*   **Core Metric (Degree of Discrimination):** The fundamental metric is the **degree of discrimination**, defined for a specific context. It measures how the probability of a positive decision `d` changes for an individual when their sensitive attributes `x` are observed, compared to when they are not (i.e., when the individual is only identified by their non-sensitive attributes `y`).
    *   Formula: `Δ(x, y) = P(d|xy) – P(d|y)`
    *   A **discrimination pattern** exists if the absolute value of this degree, `|Δ(x, y)|`, exceeds a user-defined threshold `δ` (default: 0.01).

*   **Granularity (Individual, Group, Subgroup):** This single metric can capture different fairness notions depending on the context `y`:
    *   **Group Discrimination (Statistical Parity):** If the set of non-sensitive attributes `y` is empty, the metric approximates statistical parity (`P(d|x) ≈ P(d)`).
    *   **Individual Discrimination:** If `y` includes all non-sensitive attributes, the metric captures a form of individual fairness where individuals with identical non-sensitive features should have similar outcomes regardless of their sensitive features.
    *   **Subgroup Discrimination:** By allowing `y` to be any subset of non-sensitive attributes, the method can detect discrimination within arbitrary subgroups (e.g., for a specific combination of occupation and education level).

*   **Implementation Details:** The algorithm computes probabilities `pDXY` (probability of unfavorable outcome for sensitive group) and `pD_XY` (probability of unfavorable outcome for non-sensitive group) using maximum likelihood estimation from binarized data.

### Location: Where discrimination is found

The method finds discrimination through a two-stage process combining model analysis and prediction validation.

*   **Primary Analysis:** The approach analyzes a trained **Naive Bayes classifier** built from binarized data, where all attributes are converted to binary values using median-based thresholding.
*   **Model Integration:** Unlike pure model-centric approaches, this implementation also integrates with external classifiers (Random Forest by default) to validate and predict outcomes for discovered patterns.
*   **Data Processing:** All continuous attributes are automatically binarized using median splits, and the algorithm works with the resulting binary probability distributions.
*   **Probabilistic Inference:** Discrimination patterns are identified by performing probabilistic inference on the Naive Bayes model parameters (`root_params` and `leaf_params`).

### What they find: The output of the method

The method discovers and returns specific, interpretable discrimination patterns with enhanced contextual information.

*   **What it finds:** The algorithm identifies situations where individuals receive different classification outcomes solely because their sensitive attribute was observed, along with predictions for these subgroups.
*   **Data Structure Returned:** The output is a **DataFrame containing discrimination patterns** with the following structure:
    *   `case_id`: Unique identifier for each discrimination case
    *   `discrimination_score`: Quantified measure of discrimination (pattern.score)
    *   `p_unfavorable_sensitive`: Probability of unfavorable outcome for the sensitive group
    *   `p_unfavorable_others`: Probability of unfavorable outcome for the non-sensitive group
    *   **Feature columns**: Specific assignments to both sensitive and non-sensitive attributes (None for irrelevant attributes)
    *   `nature`: Indicates whether the row represents 'base' (non-sensitive context) or 'sensitive' (sensitive attribute context)
    *   `outcome`: Predicted outcome using the trained external model
    *   `subgroup_key`: String representation of the subgroup pattern using '|' separator and '*' for wildcards
    *   `diff_outcome`: Absolute difference from mean outcome for the subgroup

### Performance: Evaluation and Results

The implementation provides comprehensive performance metrics and efficient pattern discovery.

*   **Discovery Performance (Efficiency):**
    *   **Search Efficiency:** Uses a branch-and-bound search algorithm through `PatternFinder` that explores only a fraction of the total search space
    *   **Runtime Control:** Supports configurable maximum runtime limits (`max_runtime_seconds`) for time-bounded execution
    *   **Node Tracking:** Monitors the number of nodes visited during pattern search (`nodes_visited`)

*   **Algorithm Performance Metrics:**
    *   **TSN (Total Searched Nodes):** Proxy for total sample number, representing computational effort
    *   **DSN (Discriminatory Sample Number):** Count of discovered discriminating patterns (top k patterns, default k=5)
    *   **SUR (Success Rate):** Ratio of discriminatory patterns found to total nodes searched (DSN/TSN)
    *   **DSS (Discriminatory Sample Search time):** Average time per discriminatory pattern discovery
    *   **Total Runtime:** Complete execution time from start to finish

*   **Integration Benefits:**
    *   **Model Compatibility:** Works with external ML models (Random Forest, etc.) for outcome prediction
    *   **Caching Support:** Supports model training cache for repeated experiments
    *   **Flexible Thresholds:** Configurable discrimination threshold (delta) and pattern count (k)

## Implementation

### Inputs

The Fair Naive Bayes algorithm takes a `DiscriminationData` object as its primary input, along with configuration parameters:

1. **Primary Data (`data: DiscriminationData`)**:
   - `data.dataframe`: pandas DataFrame with features, target, and protected attributes
   - `data.attr_columns`: List of feature column names
   - `data.protected_attributes`: List of sensitive/protected attribute names
   - `data.sensitive_indices`: Indices of sensitive attributes
   - `data.training_dataframe`: Subset used for model training
   - `data.outcome_column`: Name of the target/outcome column

2. **Configuration Parameters**:
   - `delta`: Discrimination threshold (default: 0.01)
   - `k`: Number of top discriminating patterns to find (default: 5)
   - `max_runtime_seconds`: Optional time limit for execution
   - `random_state`: Seed for reproducibility (default: 42)
   - `use_cache`: Enable caching for model training (default: True)

### Processing Steps

1. **External Model Training**:
   - Trains a Random Forest classifier on the original (non-binarized) data
   - Uses cross-validation and caching for efficiency
   - Model used later for outcome prediction on discovered patterns

2. **Data Binarization**:
   - Converts all attributes and target to binary values (0 or 1)
   - Uses median-based thresholding for continuous variables
   - Preserves binary attributes as-is

3. **Naive Bayes Parameter Learning**:
   - Calculates maximum likelihood parameters from binarized data
   - Creates probability dictionaries using `get_params_dict()` and `maximum_likelihood_from_data()`
   - Converts to root and leaf parameters for pattern finding

4. **Pattern Discovery**:
   - Uses `PatternFinder` with configurable parameters
   - Implements branch-and-bound search with optional time limits
   - Tracks search efficiency metrics

5. **Result Enhancement**:
   - Predicts outcomes for discovered patterns using the external model
   - Generates subgroup keys for pattern identification
   - Calculates outcome differences from population mean

### Outputs

The algorithm returns two main outputs:

1. **Enhanced Discriminating Patterns DataFrame (`res_df`)**:
   - `case_id`: Pattern identifier (grouped patterns share same ID)
   - `discrimination_score`: Quantified discrimination measure
   - `p_unfavorable_sensitive` & `p_unfavorable_others`: Probability comparisons
   - **All feature columns**: Attribute values (None for non-relevant attributes)
   - `nature`: 'base' or 'sensitive' pattern type
   - `outcome`: Predicted outcome from external model
   - `subgroup_key`: String representation (e.g., "1|0|*|1" where * = wildcard)
   - `diff_outcome`: Absolute deviation from mean outcome
   - **Performance metrics**: TSN, DSN, SUR, DSS, total_time, nodes_visited

2. **Comprehensive Metrics Dictionary (`metrics`)**:
   - `TSN`: Total Searched Nodes (computational effort proxy)
   - `DSN`: Discriminatory Sample Number (patterns found)
   - `SUR`: Success Rate (efficiency ratio)
   - `DSS`: Average time per discriminatory pattern
   - `total_time`: Total execution time
   - `nodes_visited`: Search space exploration count

### Example

For a dataset with attributes like age, income, education, and gender (protected), the algorithm:

1. **Preprocessing**: Trains Random Forest on original data, then binarizes all attributes
2. **Parameter Learning**: Calculates Naive Bayes probabilities from binary data
3. **Pattern Search**: Discovers contexts where gender affects outcomes significantly
4. **Validation**: Predicts outcomes for patterns using the Random Forest model

#### Sample Output

```
--- Discrimination Results ---
   case_id  discrimination_score  p_unfavorable_sensitive  p_unfavorable_others  age  income  education  gender  nature  outcome  subgroup_key  diff_outcome
1        1                 0.152                    0.723                 0.571    1       0          1    None   base         1       1|0|1|*        0.23
2        1                 0.152                    0.723                 0.571  None    None       None       1   sensitive     0       *|*|*|1        0.45
3        2                 0.143                    0.698                 0.555    0       1          1    None   base         1       0|1|1|*        0.18
4        2                 0.143                    0.698                 0.555  None    None       None       1   sensitive     0       *|*|*|1        0.45

--- Summary Metrics ---
TSN: 128
DSN: 2
SUR: 0.0156
DSS: 0.8745
total_time: 1.75
nodes_visited: 128
```

**Interpretation:**
- **Case 1**: High-age, low-income, high-education individuals show 15.2% discrimination based on gender
- **Pattern Keys**: "1|0|1|*" represents age=1, income=0, education=1, gender=any
- **Outcome Validation**: External model predictions confirm differential treatment
- **Efficiency**: Found 2 discriminatory patterns by exploring only 128 of potentially millions of combinations