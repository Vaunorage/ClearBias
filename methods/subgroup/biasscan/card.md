### **Title**
Identifying Significant Predictive Bias in Classifiers

### **Metric**
The method relies on a statistical measure of **predictive bias**, which quantifies the discrepancy between a classifier's predicted risk and the observed outcomes for a subgroup.

The core metric is a subgroup scoring statistic called `scorebias`, which is a likelihood ratio score. It compares two hypotheses:
*   **Null Hypothesis (H₀):** The classifier is unbiased. The odds of an event for any individual (`odds(yi)`) are equal to the odds predicted by the model (`pi / (1-pi)`).
*   **Alternative Hypothesis (H₁):** The classifier has a constant multiplicative bias `q` for a specific subgroup `S`. For individuals within that subgroup, `odds(yi) = q * (pi / (1-pi))`.

The method seeks to find the subgroup `S` and the bias factor `q` that maximize this likelihood ratio score, thus identifying the most statistically significant biased subgroup.

### **Discrimination Granularity**
*   **Type:** The method is designed to find **subgroup discrimination**.
*   **Granularity & Intersectionality:** The primary strength of this method is its ability to move beyond simple, pre-defined groups (like race or gender) and analyze a vast, exponential number of potential subgroups. A subgroup is defined as any multi-dimensional, intersectional combination of feature values (an "M-dimension Cartesian set product"). For example, it can identify a subgroup like "females who initially committed misdemeanors with COMPAS risk scores of 2, 3, 6, 9, or 10." This allows for the discovery of complex, subtle, and previously unconsidered biased subgroups.

### **Location of Discrimination**
The method locates discrimination within the **model's predictions** (the classifier). It is a form of model checking or a goodness-of-fit test. It aims to find "regions of poor classifier fit" by analyzing the residuals (the difference between predicted probabilities and actual outcomes) across all possible subgroups, identifying where the classifier is systematically over- or under-predicting risk.

### **What They Find**
The method identifies one or more **subgroups for which a classifier is statistically biased**. Specifically, it finds subgroups where the model's predicted risk is significantly different from the actual, observed risk. The output highlights whether the subgroup is being **over-estimated** (predicted risk is higher than actual) or **under-estimated** (predicted risk is lower than actual).

It does *not* compare different groups to each other (like disparate impact) but rather compares a single group's predictions to its own ground-truth outcomes.

### **Output Data Structure**
The method returns the **most anomalous subgroup (S*)**, which is described by a set of feature-value pairs that define its members. It also provides:
1.  A **bias score (`scorebias`)** indicating the magnitude of the detected bias.
2.  A **statistical significance value (p-value)**, calculated using a parametric bootstrap, to determine if the detected bias is greater than what would be expected by chance.
3.  The direction of the bias (over- or under-estimation).

### **Performance**
The method's performance was evaluated using both synthetic data and real-world case studies.

*   **Synthetic Experiments:**
    *   **Method:** The authors injected a known bias into synthetic datasets and compared the "bias scan" method against a lasso regression analysis of residuals.
    *   **Result:** The bias scan method demonstrated superior performance, particularly in scenarios where the bias was spread across multiple related interactions (i.e., "grouping weak, related signals"). For example, when bias was spread across eight 3-way interactions, the bias scan achieved ~75% recall and ~80% precision, compared to ~35% recall and ~45% precision for the lasso method.

*   **Real-World Case Studies (COMPAS Recidivism Data):**
    *   **Result:** The method identified significant, multi-dimensional subgroups that were not the primary focus of previous analyses.
        *   **Under-estimated:** Young males (< 25 years) had an observed recidivism rate of 0.60 vs. a predicted rate of 0.50.
        *   **Over-estimated:** Females whose initial crimes were misdemeanors and had specific COMPAS decile scores were significantly over-estimated (observed rate of 0.21 vs. predicted rate of 0.38).