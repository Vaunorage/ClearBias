### Title : Justicia

### Metric: What metric does the method rely on to find discrimination?

Justicia is a formal framework designed to verify fairness metrics that fall into two main categories: **Independence** and **Separation**. It specifically focuses on group-level fairness and excels at handling **subgroup/intersectional discrimination** by analyzing "compound sensitive attributes" (e.g., combinations of race, sex, and age).

The specific metrics it verifies include:
*   **Independence Metrics:** These metrics require the model's prediction to be independent of the protected group.
    *   **Disparate Impact (DI):** Measures the ratio of positive outcomes between the most and least favored groups.
    *   **Statistical Parity (SP):** Measures the difference in the rate of positive outcomes between groups.
*   **Separation Metrics:** These metrics require the model's prediction to be independent of the protected group, conditional on the true outcome.
    *   **Equalized Odds (EO):** Measures the difference in true positive rates (TPR) and false positive rates (FPR) across groups.

### Location: Where does this method try to find discrimination?

Justicia is a **distribution-based verifier**. It finds discrimination by analyzing the **ML model (classifier)** with respect to the **underlying data distribution**.

*   It does not operate on a specific, fixed test sample (like AIF360).
*   Instead, it takes the classifier (e.g., a decision tree or linear model translated to a logical formula) and the probability distribution of the input attributes as its input.
*   Its goal is to verify fairness as a formal property of the model over the entire distribution from which the data is drawn, making it more robust to the specific sample of data used for testing.

### What they find: What exactly does this method try to find?

Justicia's primary goal is to **formally verify fairness metrics**. To do this, it finds the protected groups that are most and least advantaged by the model.

1.  **Core Calculation:** The fundamental unit of computation is the **Positive Predictive Value (PPV)**, i.e., the probability of a positive outcome for a given protected group: `Pr(prediction = 1 | group = a)`.
2.  **Finding Discriminated Groups:** Justicia has two approaches to identify the extent of discrimination:
    *   **Enumeration (`Justicia_enum`):** It iterates through all possible compound protected groups, calculates the PPV for each, and finds the maximum and minimum values.
    *   **Learning (`Justicia_learn`):** A more scalable approach that uses SSAT solvers to directly **learn the most favored group** (the one with the highest PPV) and the **least favored group** (the one with the lowest PPV) without having to enumerate all possibilities.

Ultimately, it finds the maximum disparity between any two subgroups, which is then used to calculate the final fairness score (e.g., the Disparate Impact ratio).

### What does the method return in terms of data structure?

The method returns a **numerical fairness score** for the specified metric.

*   The core SSAT solver computes a **probability** (the PPV for a specific group).
*   The `Justicia_learn` or `Justicia_enum` function uses these probabilities to find the maximum and minimum PPVs across all groups.
*   The final output is a single floating-point number representing the metric, for instance:
    *   The **Disparate Impact ratio** (e.g., 0.25).
    *   The **Statistical Parity difference** (e.g., 0.54).
    *   The **Equalized Odds difference**.

### Performance: How has this method's performance been evaluated and what were the results?

Justicia's performance was evaluated on accuracy, scalability, and robustness against other state-of-the-art verifiers (FairSquare, VeriFair) and sample-based tools (AIF360).

*   **Accuracy:** On a synthetic benchmark with a known ground truth, Justicia was highly accurate (less than 1% error), whereas FairSquare and VeriFair failed to detect the fairness violation.
*   **Scalability:** Justicia was **1 to 3 orders of magnitude faster** than FairSquare and VeriFair. It successfully ran on benchmarks where FairSquare timed out. The `Justicia_learn` approach proved to be exponentially more efficient than the enumeration approach for handling many compound groups.
*   **Robustness:** Compared to the sample-based AIF360, Justicia showed significantly more stable results (lower standard deviation) when the size of the test sample was varied. This is because it operates on the data distribution rather than a single sample.
*   **Capability:** It successfully verified fairness for different classifiers (decision trees, logistic regression) and demonstrated its unique ability to **detect compounded discrimination** in groups with multiple protected attributes (e.g., verifying fairness across 40 different combinations of race, sex, and age on the Adult dataset).