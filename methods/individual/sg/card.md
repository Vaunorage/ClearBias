### **Title**
Black Box Fairness Testing of Machine Learning Models

### **Metric**
The method relies on a simplified, non-probabilistic form of **counterfactual fairness** to find discrimination.

A bias is detected if two individuals, `x` and `x'`, who differ *only* in their protected attributes (e.g., race, gender) but are identical in all other non-protected attributes, receive a different classification outcome from the machine learning model.

Formally, an individual bias instance is a pair of inputs `(x, x')` such that `M(x) ≠ M(x')`, where all non-protected attributes of `x` and `x'` are identical.

### **Individual discrimination, group, subgroup discrimination (granularity and intersectionality)**
*   **Granularity:** The method focuses exclusively on **individual discrimination**. It is designed to find specific pairs of individuals who are treated unfairly, rather than measuring statistical disparities between large demographic groups (group discrimination).
*   **Intersectionality:** The experiments in the paper were conducted by testing for one protected attribute at a time. The authors note that the approach could be extended to handle multiple protected attributes simultaneously (e.g., checking for bias against "black females"), but this would increase the computational cost as it would need to test more combinations.

### **Location**
The method tries to find discrimination in the **behavior of a trained ML model**. It operates in a **black-box** setting, meaning it does not require access to the model's internal structure, code, or original training data. It only needs API access to provide an input and receive an output. While it can leverage seed data (like training or test data) to guide its search more effectively, its goal is to audit the model, not the data itself.

### **What they find**
The method's goal is to find and report **concrete examples of individual discrimination**. It systematically generates test inputs to discover pairs of individuals (`x`, `x'`) that are identical except for a protected attribute but are assigned different labels by the model. The primary objective is to maximize the number of these discriminatory pairs found within a given testing budget (time or number of generated tests).

### **What does the method return in terms of data structure?**
The method returns a **test suite**, which is a collection of generated test inputs. This collection is implicitly divided into two parts:
1.  **Successful Test Cases (`Succ`):** A set of input pairs `(x, x')` that were found to be discriminatory. These are the concrete evidence of bias.
2.  **Generated Test Cases (`Gen`):** The total set of all unique test cases (defined by their non-protected attributes) that the algorithm generated and tested.

### **Performance**
The method's performance was evaluated by comparing it against two other fairness testing tools, **THEMIS** and **AEQUITAS**, on several benchmark datasets.

*   **Primary Evaluation Metric:** The main metric was the **success score (`#Succ / #Gen`)**, which measures the percentage of generated test cases that successfully uncovered discrimination.
*   **Results vs. THEMIS:** The proposed method (referred to as `SG`) demonstrated significantly higher performance. Across 12 benchmarks, SG achieved an average success score of **34.8%**, while THEMIS achieved **6.4%**. SG generated approximately 6 times more successful discriminatory test cases.
*   **Results vs. AEQUITAS:** SG's global and local search strategies were shown to be more effective at finding discriminatory inputs than the random sampling and perturbation methods used by AEQUITAS.
*   **Path Coverage:** The method was also evaluated on its ability to explore different decision paths within the model. It achieved **2.66 times more path coverage** than a random data-based search, indicating it more thoroughly audits the model's logic.
*   **Conclusion:** The experiments empirically demonstrate that the proposed systematic approach, combining local explainability (LIME) with symbolic execution, is significantly more effective and efficient at discovering individual discrimination in black-box models than existing random or perturbation-based methods.

# SG Algorithm Implementation

## Overview

The SG (Symbolic Generation) algorithm is a bias detection technique that uses symbolic execution and local interpretability to find discriminatory instances in machine learning models. Let me detail its input parameters and output.

## Input Parameters

The main function `run_sg()` takes the following parameters:

### 1. **data** (`DiscriminationData`)
- Required parameter
- A data object containing the dataset and metadata about protected attributes
- Contains training data, feature information, and sensitive attribute definitions

### 2. **model_type** (`str`, default='lr')
- The type of model to train
- Default is 'lr' (logistic regression)
- Other options could include 'rf' (random forest), 'dt' (decision tree), etc.

### 3. **cluster_num** (`int`, default=None)
- Number of clusters to use for K-means clustering of seed inputs
- If None, defaults to the number of unique classes in the dataset

### 4. **max_tsn** (`int`, default=100)
- Maximum number of test inputs to generate
- Algorithm terminates when this threshold is reached
- TSN stands for "Total Sample Number"

### 5. **random_seed** (`int`, default=42)
- Seed for random number generation to ensure reproducibility
- Controls randomness in clustering, model training, and LIME explanations

### 6. **max_runtime_seconds** (`int`, default=3900)
- Maximum runtime in seconds (65 minutes by default)
- Algorithm terminates when this time limit is reached

### 7. **one_attr_at_a_time** (`bool`, default=True)
- If True, checks for discrimination one protected attribute at a time
- If False, checks all protected attributes simultaneously

### 8. **db_path** (`str`, default=None)
- Optional path to a database for storing results
- If None, results are not stored in a database

### 9. **analysis_id** (`str`, default=None)
- Optional identifier for the analysis in the database
- Used when storing results in a database

### 10. **use_cache** (`bool`, default=True)
- Whether to use cached models if available
- Speeds up execution by reusing previously trained models

## Algorithm Process

The SG algorithm works through these key steps:

1. Initializes data structures to track discriminatory inputs
2. Seeds test inputs using K-means clustering on the dataset
3. Trains a machine learning model on the data
4. For each seed input:
   - Checks if it leads to discrimination
   - Uses LIME to extract decision rules from the model
   - Performs local search by flipping decision rules
   - Performs global search by systematically exploring the decision space
5. Continues until termination criteria are met (max inputs or runtime)

## Output Structure

The SG algorithm returns two main outputs:

### 1. **res_df** (`pandas.DataFrame`)

A dataframe containing pairs of discriminatory instances with the following structure:

- **Feature columns**: All columns from the original dataset (`discrimination_data.attr_columns`)
- **indv_key**: A string representation of the individual instance (concatenated feature values)
- **outcome**: The model's prediction for this instance
- **couple_key**: A key linking two instances that form a discriminatory pair
- **diff_outcome**: The absolute difference in outcomes between the pair (typically 1 for binary classification)
- **case_id**: A unique identifier for each discriminatory pair
- **Metrics columns**:
  - **TSN**: Total Sample Number (total instances tested)
  - **DSN**: Discriminatory Sample Number (total discriminatory pairs found)
  - **SUR**: Success Rate (DSN/TSN) - the proportion of tested instances that led to discrimination
  - **DSS**: Discriminatory Sample Search time (average time to find each discriminatory instance)

The dataframe contains pairs of rows with the same `case_id` and `couple_key`. Each pair represents two instances that differ only in protected attributes but receive different predictions from the model.

### 2. **metrics** (`dict`)

A dictionary containing summary statistics:

- `TSN`: Total Sample Number
- `DSN`: Discriminatory Sample Number
- `SUR`: Success Rate (DSN/TSN)
- `DSS`: Discriminatory Sample Search time
- `total_time`: Total runtime in seconds
- `dsn_by_attr_value`: Breakdown of discrimination statistics by protected attribute values

## Example Output Dataframe

Here's a more accurate example of what the output dataframe (`res_df`) would look like:

```
   sex  race  age  income  education  ... indv_key  outcome  couple_key  diff_outcome  case_id  TSN  DSN   SUR    DSS
0    1     0   45   50000          3  ... 1|0|45..      1    1|0|45..-    1            0      1000  245  0.245  12.5
1    0     0   45   50000          3  ... 0|0|45..      0    1|0|45..-    1            0      1000  245  0.245  12.5
2    1     1   32   35000          2  ... 1|1|32..      1    1|1|32..-    1            1      1000  245  0.245  12.5
3    1     0   32   35000          2  ... 1|0|32..      0    1|1|32..-    1            1      1000  245  0.245  12.5
...
```

In this example:

- Rows 0 and 1 form a discriminatory pair (case_id=0) where changing the `sex` attribute from 1 to 0 changes the prediction
- Rows 2 and 3 form another discriminatory pair (case_id=1) where changing the `race` attribute from 1 to 0 changes the prediction
- All rows contain the same metrics (TSN, DSN, SUR, DSS) which are the overall statistics for the entire analysis

The key insight is that the output dataframe contains pairs of instances that demonstrate discrimination, where each pair differs only in protected attributes but receives different predictions from the model.