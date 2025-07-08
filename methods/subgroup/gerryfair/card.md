### **Title**
Preventing Fairness Gerrymandering: Auditing and Learning for Subgroup Fairness

### **Metric**
The method relies on statistical fairness metrics, primarily focusing on two types:

1.  **Statistical Parity (SP) Subgroup Fairness**: This metric aims for the rate of positive classification to be approximately equal across all considered subgroups and equal to the overall positive classification rate.
2.  **False Positive (FP) Subgroup Fairness / Equal Opportunity**: This metric aims for the false positive rate to be approximately equal across all considered subgroups (among negatively-labeled individuals) and equal to the overall false positive rate.

The method seeks to find the subgroup for which the disparity (the difference between the subgroup's rate and the overall rate) is the largest.

### **Individual discrimination, group, subgroup discrimination (granularity and intersectionality)**
The paper's core contribution is moving beyond simple group fairness to address **subgroup fairness**.

*   **Granularity**: The method is designed to check for discrimination not just in a small number of pre-defined, high-level groups (e.g., race, gender) but across an exponentially large collection of structured subgroups. These subgroups are defined by a class of functions `G` over the protected attributes (e.g., all conjunctions of attributes).
*   **Intersectionality**: By considering subgroups formed by combinations of protected attributes (e.g., "black women" as distinct from "black people" or "women"), the method explicitly addresses and prevents *fairness gerrymandering*, where a model appears fair on coarse groups but is highly unfair to specific, intersectional subgroups.

### **Location**
The method tries to find and correct discrimination within **machine learning models (classifiers)**. It presents two distinct but related functionalities:

1.  **Auditing**: Given a pre-existing classifier, the auditing algorithm checks if it violates subgroup fairness for any subgroup within the specified class `G`.
2.  **Learning**: The learning algorithm trains a new classifier that is guaranteed to be fair with respect to all subgroups in `G`, while simultaneously minimizing classification error.

### **What they find**
The method is designed to find the **most-discriminated subgroup** according to the chosen fairness metric.

*   In the **auditing** phase, if a model is found to be unfair, the algorithm returns a **"certificate of unfairness"**. This certificate is the specific subgroup `g` (e.g., a function defining the subgroup) that exhibits the largest fairness violation.
*   In the **learning** phase, this process is used iteratively within a two-player game framework where an "Auditor" player repeatedly finds the most-discriminated subgroup for the "Learner" player's current classifier, forcing the Learner to adjust and improve its fairness on that subgroup.

### **What does the method return in terms of data structure?**
The output depends on whether the goal is auditing or learning:

*   **Auditing**: The method returns a **function `g`** from the class `G` that defines the subgroup with the maximum fairness violation.
*   **Learning**: The method returns a **randomized classifier**, which is a probability distribution over a set of classifiers from a hypothesis class `H`. This can be thought of as an ensemble of models where, for any new data point, a classifier is drawn from this distribution to make a prediction.

### **Performance**
The performance was evaluated both theoretically and empirically.

*   **Theoretical Performance**:
    *   The paper proves that auditing for subgroup fairness is computationally equivalent to the problem of (weak) agnostic learning. This implies the problem is **computationally hard** in the worst case.
    *   Despite the hardness, they provide a learning algorithm (`FairNR`) based on no-regret dynamics that is **provably convergent** to an approximately optimal and fair classifier in a polynomial number of steps, assuming access to a standard learning oracle (a cost-sensitive classification oracle).

*   **Empirical Performance**:
    *   The authors implemented a more practical version of their algorithm (`FairFictPlay`) and tested it on the "Communities and Crime" dataset, using linear classifiers for both the Learner and the Auditor.
    *   **Convergence**: The algorithm was shown to converge effectively in practice, with error and unfairness metrics stabilizing after several thousand iterations.
    *   **Results**: The experiments successfully demonstrated the algorithm's ability to navigate the trade-off between accuracy and fairness. By varying a fairness parameter `γ`, they were able to generate a **Pareto frontier** of models, showing a clear menu of choices from higher-error/higher-fairness models to lower-error/lower-fairness models. This allows a practitioner to select a model that meets their specific needs for balancing accuracy and subgroup fairness.

# GerryFair Algorithm Implementation

## Input Parameters for GerryFair Algorithm

The GerryFair algorithm is implemented in the `run_gerryfair` function in `main.py`. It takes the following input parameters:

### Main Parameters

- **ge**: `DiscriminationData` - The main data object containing:
  - `training_dataframe`: The dataset containing features and labels
  - `protected_attributes`: List of column names that are protected attributes (e.g., race, gender)
  - `outcome_column`: Name of the column containing the outcome/label
  - `non_protected_attributes`: List of column names that are not protected

### Algorithm Parameters

- **C=10** (default) - The regularization parameter that controls the trade-off between model accuracy and fairness. Higher values of C place more emphasis on enforcing fairness constraints.
- **gamma=0.01** (default) - The fairness violation threshold. This parameter determines how much disparity is allowed between groups. Lower values enforce stricter fairness.
- **max_iters=3** (default) - Maximum number of iterations for the fictitious play algorithm. Each iteration involves training a classifier and finding subgroups with fairness violations.

## How GerryFair Works

GerryFair uses a game-theoretic approach called "fictitious play" to iteratively:

1. Train a classifier that minimizes error given the current cost structure
2. Find subgroups where fairness is violated
3. Update costs to penalize unfairness in those subgroups
4. Repeat until convergence or max iterations reached

The algorithm focuses on two fairness definitions:

- **FP (False Positive)**: Equal false positive rates across subgroups
- **FN (False Negative)**: Equal false negative rates across subgroups

## Data Preparation

Before running the algorithm, data is prepared through the `prepare_data_for_gerryfair` function which:

- Converts multi-class outcomes to binary if needed
- Identifies protected attributes
- Creates temporary files for processing
- Cleans the dataset using GerryFair's cleaning function

## Output

The output of the GerryFair algorithm is a pandas DataFrame containing information about the subgroups identified during training that exhibit fairness violations. Each row represents a subgroup found during an iteration, with the following columns:

- **Protected attribute coefficients** - For each protected attribute, a coefficient indicating its importance in defining the subgroup
- **group_size** - The proportion of the dataset that belongs to this subgroup
- **weighted_disparity** - The fairness violation weighted by the group size
- **disparity** - The raw fairness violation (difference in error rates)
- **disparity_direction** - Direction of the disparity (1 or -1)
- **group_rate** - The error rate within the group

## Example Output DataFrame

```
   gender    age  race  education  income  group_size  weighted_disparity  disparity  disparity_direction  group_rate
0    0.75   0.25  None      None    None        0.32              0.042      0.131                    1        0.27
1   -0.65   0.35  None      None    None        0.28              0.037      0.132                   -1        0.18
2    0.15   0.85  None      None    None        0.22              0.031      0.141                    1        0.31
3   -0.45  -0.55  None      None    None        0.18              0.024      0.133                   -1        0.16
```

### Interpretation

In this example:

- The first row shows a subgroup defined primarily by gender (coefficient 0.75) and secondarily by age (0.25)
- This subgroup contains 32% of the dataset (group_size = 0.32)
- It has a weighted fairness violation of 0.042
- The raw disparity is 0.131 (13.1% difference in error rates)
- The disparity_direction is 1 (positive)
- The error rate within this group is 0.27 (27%)

## Usage Example

```python
from data_generator.main import generate_data, DiscriminationData
from methods.subgroup.gerryfair.main import run_gerryfair

# Generate synthetic data with bias
ge = generate_data(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    protected_attributes=['gender', 'age'],
    bias_strength=0.3
)

# Run GerryFair algorithm
results = run_gerryfair(
    ge,
    C=10,         # Regularization parameter
    gamma=0.01,   # Fairness violation threshold
    max_iters=5   # Maximum iterations
)

# Display results
print(results)
```

## Summary

The GerryFair algorithm helps identify and mitigate bias in machine learning models by finding subgroups where fairness is violated and adjusting the model to reduce these violations while maintaining reasonable accuracy.