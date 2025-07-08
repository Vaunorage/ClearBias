### **Title**
FlipTest: Fairness Testing via Optimal Transport

### **Metric**
The method relies on **Optimal Transport** to find a mapping between two protected groups (e.g., men and women). This mapping pairs each individual in the source group with a "similar" counterpart in the target group, minimizing the overall "distance" or cost between all pairs.

The core metric is the **flipset**, defined as "the set of individuals whose classifier output changes post-translation" (Abstract). This set is further divided into:
*   **Positive flipset (F+):** Individuals who are advantaged by their group membership (e.g., a woman is hired, but her male counterpart is not).
*   **Negative flipset (F-):** Individuals who are disadvantaged by their group membership (e.g., a woman is rejected, but her male counterpart is hired).

Discrimination is assessed by analyzing the size, balance, and composition of these flipsets.

### **Individual discrimination, group, subgroup discrimination (granularity and intersectionality)**
FlipTest operates on multiple levels of granularity:
*   **Individual Discrimination:** The method creates pairs of "similar" individuals from different groups. A change in the model's outcome for a specific pair is considered evidence of potential individual-level discrimination.
*   **Group Discrimination:** By comparing the relative sizes of the positive and negative flipsets, the method can test for group-level fairness criteria like demographic parity. For example, a much larger negative flipset than a positive one suggests a group-level bias.
*   **Subgroup Discrimination:** The method can uncover discrimination even when group-level metrics are satisfied. It does this by analyzing the feature distributions of the individuals *within* the flipset and comparing them to the overall population. The paper states, "By comparing the distribution of the flipsets to the distribution of the overall population, it is often possible to identify specific subgroups that the model discriminates against" (Section 2).

### **Location**
The method finds discrimination in the **model**. It is a black-box testing technique that queries a trained classifier to observe its behavior on specifically crafted inputs. The goal is to "uncover[] discrimination in classifiers" (Abstract) by analyzing the model's output on real (in-distribution) samples and their generated counterparts.

### **What they find**
FlipTest aims to find **salient patterns of discriminatory behavior** in a model. It does not claim to prove a causal link between a protected attribute and the outcome. Instead, it identifies:
1.  **Potentially Discriminated Individuals:** The members of the flipset are concrete examples of individuals who may be harmed or advantaged by the model due to their group membership.
2.  **Discriminated Subgroups:** It identifies which subgroups are most affected by analyzing the characteristics of the individuals in the flipset (e.g., finding that the model harms "shorter-haired women" as in Section 2).
3.  **Associated Features:** It identifies which features are most associated with the discriminatory behavior, providing insight into *how* the model might be discriminating.

### **What does the method return in terms of data structure?**
The method returns two main outputs:
1.  **The Flipset:** A set of individuals from the source population whose model-predicted label changes when they are mapped to their counterparts in the target population. This is partitioned into a positive flipset and a negative flipset.
2.  **A Transparency Report:** A ranked list of features that are most associated with the model's differing behavior on the flipset. This report shows (1) the average change for each feature between an individual and their counterpart and (2) how consistently that feature changes in a specific direction (e.g., always increasing). This helps auditors understand the potential mechanism of discrimination.

### **Performance**
The performance of FlipTest was evaluated empirically across four datasets, including real-world case studies (predictive policing and hiring) and comparisons with other fairness auditing methods.

*   **Case Studies:**
    *   On a predictive policing dataset (SSL), FlipTest identified a model's bias against younger black individuals with more narcotics arrests, and the transparency report correctly highlighted "narcotics arrests" as the key feature driving the bias (Section 5.2).
    *   On a synthetic hiring dataset, it successfully detected subgroup discrimination (harming short-haired women) in a model that was designed to be fair at the group level (Section 5.3).
*   **Comparison to Other Methods:**
    *   **vs. Counterfactual Fairness:** On a law school dataset, FlipTest produced "nearly identical results" to the counterfactual fairness method without requiring access to a causal model, which is a major practical advantage (Section 5.4).
    *   **vs. FairTest:** On a synthetic dataset, FlipTest was shown to identify discrimination based on features that are themselves biased (i.e., have different distributions across groups), a scenario that FairTest is not well-suited for (Section 5.5).

# FlipTest Algorithm Implementation

## Overview

The FlipTest algorithm is implemented in the `main.py` file and provides two main functions for analyzing discrimination in datasets:

- `run_fliptest_on_dataset` - Runs FlipTest on a single protected attribute
- `run_fliptest` - Runs FlipTest on multiple protected attributes in a dataset

## Functions

### `run_fliptest_on_dataset`

This function analyzes discrimination for a specific protected attribute.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `discrimination_data` | DiscriminationData | Required | Dataset object containing the dataframe with features, outcomes, and protected attributes |
| `protected_attribute` | str | Required | Name of the protected attribute column to analyze (e.g., "race", "gender") |
| `group1_val` | Any | 0 | Value representing the first group in the protected attribute column |
| `group2_val` | Any | 1 | Value representing the second group in the protected attribute column |

### `run_fliptest`

This function runs FlipTest on multiple protected attributes.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_obj` | DiscriminationData | Required | Dataset object containing the dataframe with features, outcomes, and protected attributes |
| `max_runs` | int | None | Maximum number of protected attributes to check. If None, all protected attributes will be checked |

## Algorithm Process

FlipTest works by:

1. Splitting the dataset into two groups based on the protected attribute
2. Scaling the features to normalize them
3. Computing a distance matrix between individuals in different groups
4. Using optimal transport to find the closest matching pairs between groups
5. Identifying pairs with different outcomes as potentially discriminatory

## Output

The algorithm returns a tuple containing two elements:

### 1. Results DataFrame (`results_df`)

A pandas DataFrame containing all identified discriminatory pairs with the following columns:

- **All original feature columns** from the dataset
- **`indv_key`**: A unique identifier for each individual instance
- **`outcome`**: The predicted outcome for the instance
- **`couple_key`**: A key linking two instances that form a discriminatory pair
- **`diff_outcome`**: The absolute difference in outcomes between the pair
- **`case_id`**: A unique identifier for each discriminatory case
- **`TSN`**: Total Sample Number - total number of input pairs tested
- **`DSN`**: Discriminatory Sample Number - number of discriminatory pairs found
- **`SUR`**: Success Rate - ratio of DSN to TSN
- **`DSS`**: Discriminatory Sample Search time - average time to find a discriminatory sample
- **`protected_attribute`**: The protected attribute used for this pair

### 2. Metrics Dictionary (`metrics_dict`)

A dictionary containing summary statistics:

- **`total_runs`**: Total number of protected attributes tested
- **`successful_runs`**: Number of protected attributes that completed successfully
- **`total_time`**: Total execution time
- **`attribute_metrics`**: Detailed metrics for each protected attribute, including:
  - `TSN`: Total Sample Number
  - `DSN`: Discriminatory Sample Number
  - `SUR`: Success Rate
  - `DSS`: Discriminatory Sample Search time
  - `total_time`: Execution time for this attribute
  - `mean_distance`: Mean L1 distance between matched pairs
  - `protected_attribute`: Name of the protected attribute
  - `group1_val`: Value for group 1
  - `group2_val`: Value for group 2
  - `raw_results`: Raw data used in the analysis

## Example Usage

### Input Example

```python
from data_generator.main import DiscriminationData, generate_optimal_discrimination_data

# Generate synthetic data with controlled bias
data_obj = generate_optimal_discrimination_data(
    nb_groups=100,
    nb_attributes=15,
    prop_protected_attr=0.3,
    nb_categories_outcome=1,
    use_cache=True
)

# Run FlipTest on the top 2 most balanced protected attributes
results_df, metrics = run_fliptest(data_obj, max_runs=2)
```

### Output Example

#### Results DataFrame Example

```
   attr1  attr2  attr3  gender  income  ...  couple_key  diff_outcome  case_id  TSN  DSN   SUR    DSS  protected_attribute
0    0.3    0.7    1.2       0       0  ...  0.3|0.7|...-0.4|0.6|...          1    1     87   43  0.494  0.023  gender
1    0.4    0.6    1.1       1       1  ...  0.3|0.7|...-0.4|0.6|...          1    1     87   43  0.494  0.023  gender
2    0.5    0.2    0.9       0       0  ...  0.5|0.2|...-0.6|0.3|...          2    2     87   43  0.494  0.023  gender
3    0.6    0.3    0.8       1       1  ...  0.5|0.2|...-0.6|0.3|...          2    2     87   43  0.494  0.023  gender
4    0.1    0.8    1.5       0       0  ...  0.1|0.8|...-0.2|0.9|...          3    3     87   43  0.494  0.023  gender
```

#### Metrics Dictionary Example

```python
{
    "total_runs": 2,
    "successful_runs": 2,
    "total_time": 5.23,
    "attribute_metrics": {
        "gender": {
            "TSN": 87,
            "DSN": 43,
            "SUR": 0.494,
            "DSS": 0.023,
            "total_time": 2.12,
            "mean_distance": 0.876,
            "protected_attribute": "gender",
            "group1_val": 0,
            "group2_val": 1,
            "raw_results": {
                # Raw data used in the analysis
            }
        },
        "race": {
            "TSN": 92,
            "DSN": 38,
            "SUR": 0.413,
            "DSS": 0.031,
            "total_time": 3.11,
            "mean_distance": 0.912,
            "protected_attribute": "race",
            "group1_val": 0,
            "group2_val": 1,
            "raw_results": {
                # Raw data used in the analysis
            }
        }
    }
}
```

## Use Case

The FlipTest algorithm is particularly useful for identifying individual fairness violations by finding similar individuals from different protected groups who received different outcomes, suggesting potential discrimination in the decision-making process.