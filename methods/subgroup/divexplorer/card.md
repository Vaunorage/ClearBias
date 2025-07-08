### **Title**
Identifying Significant Predictive Bias in Classifiers

### **Metric**
The method relies on a statistical measure of predictive bias called `score_bias`. This score is a **log-likelihood ratio** that quantifies the discrepancy between a model's predicted odds and the empirically observed odds for a given subgroup.

-   **Null Hypothesis (H₀):** The classifier is unbiased for the subgroup. The predicted odds `p / (1-p)` are correct.
-   **Alternative Hypothesis (H₁):** The classifier has a consistent multiplicative bias `q` for the subgroup. The true odds are `q * [p / (1-p)]`.

The method finds the subgroup `S` and the bias factor `q` that maximize this log-likelihood ratio score, thereby identifying the "most anomalous" or most biased subgroup.

### **Discrimination Granularity**
The method is designed to detect **intersectional subgroup discrimination**.

-   **Granularity:** It moves beyond pre-defined, single-dimension groups (e.g., only `race` or only `gender`) to find bias in highly specific, multi-dimensional subgroups. As the paper states, it allows "consideration of not just sub-groups of a priori interest or small dimensions, but the space of all possible subgroups of features."
-   **Intersectionality:** It defines a subgroup as a "Cartesian set product" of feature values (e.g., `age < 25` AND `gender = male` AND `crime_type = misdemeanor`). This allows it to identify complex, intersectional biases that might be missed by analyzing features independently. For instance, in the COMPAS analysis, it finds bias in the subgroup of "females, whose initial crimes were misdemeanors, and have COMPAS decile scores ∈ {2, 3, 6, 9, 10}."

### **Location of Discrimination**
The method finds discrimination in the **probabilistic model's predictions**. It is a form of model checking or goodness-of-fit test that analyzes the residuals—the difference between the classifier's predicted outcomes and the actual observed outcomes for a subgroup. The paper notes, "We focus on the source of bias from classification techniques that could be insufficiently flexible to predict well for subgroups in the data."

### **What the Method Finds**
The method identifies **statistically significant subgroups for whom the model systematically over- or under-predicts risk**. It does not compare individuals to each other, but rather compares a model's predictive performance on a specific subgroup to the actual outcomes for that same subgroup. The goal is to answer the question: "can we identify a subgroup(s) that has significantly more predictive bias than would be expected from an unbiased classifier?"

### **Data Structure of Return**
The method returns:
1.  **A description of the most biased subgroup:** This is defined by a set of rules on the input features (e.g., `priors > 5` or `age < 25 AND gender = male`).
2.  **A bias score:** The maximized log-likelihood ratio score for that subgroup.
3.  **A measure of statistical significance:** A p-value, estimated using parametric bootstrapping, which indicates the probability of finding a subgroup this biased by chance alone.

### **Performance Evaluation**
The method's performance was evaluated using both synthetic data and real-world case studies.

1.  **Method of Evaluation:**
    *   **Synthetic Experiments:** The authors generated data with 4 categorical features and injected a known log-odds bias into specific, multi-dimensional subgroups. They then compared the performance of their **Bias Scan** method against a **Lasso regression on residuals**.
    *   **Case Studies:** They applied the method to the real-world COMPAS crime recidivism dataset and a credit delinquency dataset.

2.  **Results:**
    *   On synthetic data, the Bias Scan method significantly outperformed Lasso regression, especially when the bias was spread across multiple, related interactions (i.e., "grouping weak, related signals together").
    *   For example, when bias was injected into eight 3-way interactions, the **Bias Scan achieved 75%/80% recall/precision**, while the Lasso method only achieved 35%/45%.
    *   In the COMPAS case study, the method identified novel, statistically significant biased subgroups not previously reported, such as:
        *   **Under-estimation:** "Young (< 25 years) males are under-estimated (p < 0.005)".
        *   **Over-estimation:** "females, whose initial crimes were misdemeanors, and have COMPAS decile scores ∈ {2,3,6,9,10} are over-estimated (p = 0.035)".

# DivExplorer Algorithm Implementation

## Input Parameters

The DivExplorer algorithm is implemented in the `run_divexploer` function in `main.py`. Here are the key input parameters:

**data_obj**: A `DiscriminationData` object that contains:
- `training_dataframe_with_ypred`: DataFrame containing the dataset with both true labels and predicted labels
- `outcome_column`: Name of the column containing the true class labels
- `y_pred_col`: Name of the column containing the predicted class labels
- `attributes`: List of attribute names in the dataset

**K**: Integer (default=5) - The number of top divergent patterns to return for each metric (FPR and FNR)

### Internal Parameters (used by the underlying algorithm)

When the function calls `FP_DivergenceExplorer`, it uses these parameters:

- **min_support**: Float (default=0.05) - The minimum support threshold for frequent pattern mining. Only patterns that appear in at least this fraction of the dataset will be considered.
- **th_redundancy**: Integer (default=0) - Threshold for redundancy when selecting top-K patterns. Used to filter out similar patterns.

## How the Algorithm Works

1. The algorithm first initializes the `FP_DivergenceExplorer` with the dataset, true class name, and predicted class name.

2. It then uses frequent pattern mining to identify patterns (subgroups) in the data with the `getFrequentPatternDivergence` method.

3. For each pattern, it calculates divergence metrics:
   - **d_fpr**: Divergence in False Positive Rate - how much the FPR in this subgroup differs from the global FPR
   - **d_fnr**: Divergence in False Negative Rate - how much the FNR in this subgroup differs from the global FNR

4. It selects the top-K patterns with the highest absolute divergence for each metric.

5. Finally, it processes the results into a clean DataFrame format, with each pattern's attributes expanded into columns.

## Output

The output is a pandas DataFrame containing the top-K divergent patterns for both FPR and FNR metrics. Each row represents a pattern (subgroup) with the following columns:

- **support**: The fraction of instances in the dataset that match this pattern
- **length**: Number of attributes in the pattern
- **value**: The divergence value (how much this subgroup's error rate differs from the global error rate)
- **metric**: Either 'fpr' or 'fnr' indicating which metric this pattern was selected for
- **Additional columns**: One column for each attribute in the dataset, with values indicating the specific attribute value for this pattern, or None if the attribute is not part of the pattern

## Example

Let's say we have a dataset about loan applications with attributes like 'age', 'income', 'education', and a binary outcome 'loan_approved':

### Input Example:

```python
from data_generator.main import get_real_data, DiscriminationData
from methods.subgroup.divexplorer.main import run_divexploer

# Get a dataset (e.g., the adult dataset)
data_obj, schema = get_real_data('adult', use_cache=False)

# Run DivExplorer with K=3 (top-3 patterns for each metric)
results = run_divexploer(data_obj, K=3)
```

### Output Example:

```
   support  length     value metric    age    sex education  income marital_status occupation
0     0.15       2    0.231    fpr   >45   Male     None     >50K        Married      None
1     0.08       1    0.198    fpr  None  Female     None     None          None      None
2     0.12       2    0.187    fpr  None   None  Bachelor     >50K          None      None
3     0.10       2   -0.245    fnr   <30   None     None     <30K       Divorced      None
4     0.07       1   -0.212    fnr  None   None     None     None          None    Service
5     0.14       2   -0.195    fnr  None   Male     None     <30K          None      None
```

In this example:

- **Rows 0-2** show the top-3 patterns with the highest FPR divergence
- **Rows 3-5** show the top-3 patterns with the highest FNR divergence
- **Positive values for FPR** indicate subgroups where the model makes more false positive errors than average
- **Negative values for FNR** indicate subgroups where the model makes more false negative errors than average
- Each row shows which attribute values define the subgroup (None means that attribute is not part of the pattern)

For instance, row 0 indicates that for people who are male, over 45 years old, and have an income over 50K, the false positive rate is 0.231 higher than the global false positive rate, meaning the model is more likely to incorrectly predict positive outcomes for this group.

This output helps identify specific subgroups where the model performs differently than on the overall population, which is crucial for detecting algorithmic bias.

