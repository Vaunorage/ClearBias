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

# Implementation

## Inputs

The Fair Naive Bayes algorithm takes a `DiscriminationData` object as its primary input. This object contains:

1. **Dataframe (`data.dataframe`)**: A pandas DataFrame containing the dataset with:
   - Feature columns (attributes)
   - Target column (outcome)
   - Protected attribute columns

2. **Metadata**:
   - `data.attr_columns`: List of feature column names
   - `data.protected_attributes`: List of sensitive/protected attribute names
   - `data.sensitive_indices`: Indices of the sensitive attributes

## Processing Steps

1. **Data Binarization**:
   - Converts all attributes and the target column to binary values (0 or 1)
   - For attributes with more than 2 unique values, the median is used as the threshold

2. **Parameter Learning**:
   - Calculates Naive Bayes parameters from the binarized data
   - Creates probability dictionaries for the model

3. **Pattern Finding**:
   - Uses `PatternFinder` to identify discriminating patterns
   - Parameters include:
     - `delta`: Discrimination threshold (default: 0.01)
     - `k`: Number of top discriminating patterns to find (default: 5)

## Outputs

The algorithm returns two main outputs:

1. **Discriminating Patterns DataFrame (`res_df`)**: Contains details about identified patterns:
   - `case_id`: Identifier for each pattern
   - `discrimination_score`: Measure of discrimination for the pattern
   - `p_unfavorable_sensitive`: Probability of unfavorable outcome for sensitive group
   - `p_unfavorable_others`: Probability of unfavorable outcome for non-sensitive group
   - Feature values for both base and sensitive patterns
   - `nature`: Indicates if the pattern is 'base' or 'sensitive'
   - Performance metrics (TSN, DSN, SUR, DSS, total_time, nodes_visited)

2. **Metrics Dictionary (`metrics`)**:
   - `TSN`: Total Searched Nodes (proxy for total sample number)
   - `DSN`: Discriminatory Sample Number (number of discriminating patterns found)
   - `SUR`: Success Rate (DSN/TSN)
   - `DSS`: Discriminatory Sample Search time (time per discriminatory pattern)
   - `total_time`: Total execution time
   - `nodes_visited`: Number of nodes visited during pattern search

## Example

For example, if we have a dataset with attributes like age, income, education, and gender (where gender is a protected attribute), the algorithm would:

1. Binarize all attributes (e.g., age > median becomes 1, else 0)
2. Calculate Naive Bayes parameters
3. Find patterns where outcomes differ significantly between protected groups

### Sample Output

```
--- Discrimination Results ---
   case_id  discrimination_score  p_unfavorable_sensitive  p_unfavorable_others  age  income  education  gender  nature
1        1                 0.152                    0.723                 0.571    1       0          1    None   base
2        1                 0.152                    0.723                 0.571  None    None       None       1   sensitive
3        2                 0.143                    0.698                 0.555    0       1          1    None   base
4        2                 0.143                    0.698                 0.555  None    None       None       1   sensitive

--- Summary Metrics ---
TSN: 128
DSN: 2
SUR: 0.0156
DSS: 0.8745
total_time: 1.75
nodes_visited: 128
```

In this example:
- Two discriminating patterns were found (case_id 1 and 2)
- For case_id 1, individuals with high age (1), low income (0), and high education (1) who belong to the sensitive group (gender=1) have a 72.3% chance of an unfavorable outcome, compared to 57.1% for others with the same attributes
- The algorithm searched 128 nodes and found 2 discriminating patterns, with a success rate of 0.0156

The algorithm helps identify specific attribute combinations where discrimination may be occurring, allowing for targeted bias mitigation.