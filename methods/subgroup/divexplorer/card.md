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