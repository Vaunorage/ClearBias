### Title: Probabilistic Verification of Fairness Properties via Concentration

### **Metric: What metric does the method rely on to find discrimination?**

The method uses a **flexible specification language** that can express various fairness definitions through arithmetic and logical constraints. Based on the implementation, the core metric is:

* **Ratio-based Fairness Assessment:** The algorithm computes the ratio of positive outcomes between two demographic groups and compares it against a fairness threshold `c` (default: 0.15 or 15%).
* **Demographic Parity:** The primary implementation focuses on demographic parity, verifying that minority members receive favorable outcomes at a rate within an acceptable threshold of the majority group rate.
* **Mutual Information Scoring:** The algorithm uses mutual information scores between protected attributes and outcomes to prioritize which attributes to analyze first, focusing on those most likely to exhibit discrimination.

The method supports additional fairness metrics through its specification language framework, including Equal Opportunity, Path-Specific Causal Fairness, and Individual Fairness.

### **Location: Where does this method try to find discrimination?**

The method finds discrimination in **trained machine learning models** when applied to test datasets. The algorithm operates on:

1. **Trained Scikit-learn Models**: The implementation uses a `SklearnModelSampler` wrapper that makes any trained scikit-learn classifier compatible with VeriFair's verification process.
2. **Test Dataset Subgroups**: The algorithm partitions the test data based on protected attribute values (e.g., Male vs Female for gender) and analyzes the model's behavior on each subgroup.
3. **All Protected Attribute Combinations**: The method systematically examines all possible pairwise combinations of values within each protected attribute, ordered by their mutual information scores with the outcome variable.

The verification process treats the ML model as a black box and focuses on its decision-making behavior across different demographic groups.

### **What they find: What exactly does this method try to find?**

The method performs **systematic fairness verification** across all combinations of protected attributes:

1. **Prioritized Analysis**: Uses mutual information scores to rank protected attributes by their likelihood of exhibiting discrimination, analyzing the most suspicious attributes first.
2. **Pairwise Group Comparisons**: For each protected attribute, compares all possible pairs of attribute values (e.g., Male vs Female, White vs Black vs Asian for all combinations).
3. **Binary Fairness Determination**: For each comparison, determines whether the model violates the specified fairness threshold between the two groups.
4. **Comprehensive Coverage**: Analyzes all protected attributes and their value combinations rather than requiring pre-specified groups of interest.

### **What does the method return in terms of data structure?**

The method returns a **pandas DataFrame** where each row represents a demographic subgroup from the pairwise comparisons, containing:

**Attribute Columns:**
* Multiple protected attribute columns (e.g., `Attr1_X`, `Attr2_X`, `Attr1_T`, `Attr2_T`, etc.): Each column corresponds to a protected attribute, with specific values for the subgroup being analyzed or `<NA>` for non-relevant attributes
* The naming convention appears to distinguish between different types of attributes (with `_X` and `_T` suffixes)

**Subgroup Identification:**
* `subgroup_key`: Pipe-separated string identifier showing the specific values for each attribute position (e.g., `*|*|*|*|*|*|*|*|*|*|2|*|*|*|*|*|*|*|*|*` where `2` indicates the value for that attribute and `*` indicates non-relevant attributes)
* `group_key`: Combined identifier for the pairwise comparison, showing both subgroups being compared (e.g., `*|*|...|2|*|...-*|*|...|0|*|...`)

**Discrimination Measures:**
* `diff_outcome`: Numerical difference in outcome rates between the compared groups (can be positive or negative, e.g., `0.22830024346212374`, `-0.10470630905662404`)

**Performance Metrics (repeated for each row):**
* `TSN`: Total Sample Number across all analyses (e.g., `7378`)
* `DSN`: Discriminatory Sample Number - count of unfair results found (e.g., `2`)
* `SUR`: Success/Unfairness Rate calculated as DSN/TSN (e.g., `0.0002710761724044456`)
* `DSS`: Discriminatory Sample Search time - time per discrimination found in seconds (e.g., `154.72836637496948`)
* `total_time`: Total execution time in seconds (e.g., `309.45673274993896`)
* `nodes_visited`: Total samples processed, matching TSN (e.g., `7378`)

**Data Structure Characteristics:**
* Each pairwise comparison generates **two rows** - one for each subgroup in the comparison
* The same performance metrics are replicated across all rows from a single analysis run
* Most attribute columns contain `<NA>` values, with only the relevant attribute for each comparison containing actual values
* The `subgroup_key` uses a positional encoding where each position corresponds to a specific attribute

### **Performance: How has this method's performance been evaluated and what was the result?**

The implementation includes several performance optimizations and evaluation mechanisms:

1. **Scalability Features:**
   * **Timeout Management**: Global `max_runtime_seconds` parameter prevents excessive runtime
   * **Iterative Sampling**: Configurable `n_samples` and `n_max` parameters control memory usage and convergence
   * **Progress Logging**: Regular progress updates every `log_iters` iterations

2. **Efficiency Optimizations:**
   * **Mutual Information Prioritization**: Analyzes attributes most likely to show discrimination first
   * **Early Stopping**: Can halt analysis when time limits are reached
   * **Sampling Strategy**: Uses sampling with replacement to handle varying group sizes efficiently

3. **Robustness Measures:**
   * **Statistical Confidence**: Extremely low default error probability (δ = 0.5 × 10⁻¹⁰)
   * **Empty Group Handling**: Returns neutral outcomes for empty demographic groups
   * **Convergence Detection**: Identifies when analysis fails to converge

4. **Comprehensive Metrics:**
   * **Coverage Metrics**: TSN and DSN provide quantitative measures of analysis breadth
   * **Efficiency Metrics**: SUR and DSS measure discrimination detection efficiency
   * **Time Tracking**: Detailed timing information for performance assessment

The method demonstrates strong performance characteristics with configurable trade-offs between accuracy, runtime, and statistical confidence, making it suitable for both quick assessments and thorough fairness audits.