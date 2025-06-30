### Title : Learning Fair Naive Bayes Classifiers by Discovering and Eliminating Discrimination Patterns

### Metric: How discrimination is measured

The paper introduces a new metric to find discrimination at various levels of granularity, from groups to specific subgroups.

*   **Core Metric (Degree of Discrimination):** The fundamental metric is the **degree of discrimination**, defined for a specific context. It measures how the probability of a positive decision `d` changes for an individual when their sensitive attributes `x` are observed, compared to when they are not (i.e., when the individual is only identified by their non-sensitive attributes `y`).
    *   Formula: `Δ(x, y) = P(d|xy) – P(d|y)`
    *   A **discrimination pattern** exists if the absolute value of this degree, `|Δ(x, y)|`, exceeds a user-defined threshold `δ`.

*   **Granularity (Individual, Group, Subgroup):** This single metric can capture different fairness notions depending on the context `y`:
    *   **Group Discrimination (Statistical Parity):** If the set of non-sensitive attributes `y` is empty, the metric approximates statistical parity (`P(d|x) ≈ P(d)`).
    *   **Individual Discrimination:** If `y` includes all non-sensitive attributes, the metric captures a form of individual fairness where individuals with identical non-sensitive features should have similar outcomes regardless of their sensitive features.
    *   **Subgroup Discrimination:** By allowing `y` to be any subset of non-sensitive attributes, the method can detect discrimination within arbitrary subgroups (e.g., for a specific combination of occupation and education level).

*   **Ranking Metric (Divergence Score):** To rank the "most important" patterns, the paper also proposes a **divergence score**. This score combines the degree of discrimination with the probability of the pattern occurring, prioritizing patterns that are both highly discriminatory and affect a larger portion of the population.

### Location: Where discrimination is found

The method finds discrimination directly within the **probabilistic model** itself, not in the training data.

*   The approach analyzes a trained **Naive Bayes classifier**, which is a probabilistic model representing a joint distribution over all features.
*   It searches for discrimination patterns by performing probabilistic inference on this model (`P(d|xy)` and `P(d|y)` are computed from the model's parameters).
*   This is a model-centric approach, distinct from data-centric methods that "repair" the training data before learning. The paper shows that models trained on "fair" data can still contain discrimination patterns (Table 3).

### What they find: The output of the method

The method is designed to discover and return specific, interpretable "discrimination patterns."

*   **What it finds:** The algorithm finds situations where an individual receives a different classification outcome *solely because* their sensitive attribute was observed.
*   **Data Structure Returned:** The output is a **list of discrimination patterns**. Each pattern is a tuple `(x, y)` representing:
    *   `x`: A specific assignment to one or more sensitive attributes (e.g., `{gender=Female, race=White}`).
    *   `y`: A specific assignment to a subset of non-sensitive attributes that forms the context for discrimination (e.g., `{occupation=Sales, marital_status=Married}`).

This provides a precise diagnosis of *when* and *for whom* the model is unfair, going beyond simple group-level statistics.

### Performance: Evaluation and Results

The paper evaluates both the efficiency of the discovery algorithm and the quality of the final fair classifier.

*   **Discovery Performance (Efficiency):**
    *   The paper's branch-and-bound search algorithm is highly efficient. Table 1 shows that it can find the most discriminatory patterns by exploring only a **tiny fraction of the total search space** (e.g., exploring 1 in several million possible patterns on the German dataset).

*   **Fair Learning Performance (Effectiveness):**
    *   **Convergence:** The iterative learning algorithm (a cutting-plane method) converges to a fully fair model in a **very small number of iterations** (e.g., 3-7 iterations on the COMPAS dataset, as shown in Figure 3).
    *   **Model Quality:** The resulting fair models retain high quality. Table 2 shows their log-likelihood is much closer to the unconstrained (best possible) model than to simpler fairness approaches, indicating a good fairness-utility trade-off.
    *   **Accuracy:** The learned fair models are highly accurate. Table 4 shows they **outperform other fairness methods** (like two-naive-Bayes or training on repaired data) in terms of classification accuracy. For two datasets (Adult and German), the fair model was even slightly more accurate than the original, unconstrained model.