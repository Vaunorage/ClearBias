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

# Implementation of BiasScan Algorithm

After analyzing the code, I can provide a detailed explanation of the BiasScan algorithm's input parameters and output.

## Input Parameters

The BiasScan algorithm is implemented in the `run_bias_scan` function, which takes the following parameters:

### ge
A data generator object that contains:
- **dataframe**: The dataset to analyze
- **attributes**: Dictionary of attributes with boolean values indicating if they're protected
- **outcome_column**: Name of the target variable column

### test_size (default=0.2)
Proportion of the dataset to use for testing
- Example: 0.3 means 30% of data will be used for testing, 70% for training

### random_state (default=42)
Seed for reproducibility of random operations
- Controls the randomness in train/test split and model training

### n_estimators (default=100)
Number of trees in the RandomForest classifier
- Higher values typically improve model performance but increase computation time

### bias_scan_num_iters (default=50)
Number of iterations for the bias scan algorithm
- More iterations may find better subgroups but increase computation time

### bias_scan_scoring (default='Poisson')
Scoring function used to evaluate bias
- Options: 'Bernoulli', 'BerkJones', 'Gaussian', 'Poisson'
- Each scoring function measures discrepancies between observations and expectations differently

### bias_scan_favorable_value (default='high')
Defines which outcome values are considered favorable
- Options: 'high' (maximum value is favorable), 'low' (minimum value is favorable), or specific value

### bias_scan_mode (default='ordinal')
Type of data being analyzed
- Options: 'binary', 'continuous', 'nominal', 'ordinal'
- Determines how the algorithm treats the outcome variable

## How the Algorithm Works

1. The algorithm splits the data into training and testing sets
2. Trains a RandomForest classifier on the training data
3. Makes predictions on the entire dataset
4. Performs two bias scans:
   - One to find subgroups where the model overpredicts outcomes
   - One to find subgroups where the model underpredicts outcomes
5. Formats the results into a dataframe showing the identified biased subgroups

## Output

The algorithm returns a tuple containing:

### result_df
A DataFrame containing the identified biased subgroups with the following columns:
- **group_id**: Unique identifier for each subgroup
- **Attribute columns**: Values of attributes that define the subgroup
- **outcome**: The outcome value for each instance
- **diff_outcome**: The difference in outcome values within a group
- **indv_key**: String representation of individual attribute values
- **couple_key**: String representation of paired instances within a group

### report
A dictionary containing:
- **accuracy**: The accuracy of the trained model on the test set
- **report**: A classification report with precision, recall, and F1-score metrics

## Example Output

Here's an example of what the output dataframe might look like:

```
   group_id  attr1  attr2  attr3  outcome  diff_outcome          indv_key                   couple_key
0         0    1.0    0.0    1.0        0            1          1|0|1|0|0           1|0|1|0|0-1|0|1|1|0
1         0    1.0    0.0    1.0        1            1          1|0|1|1|0           1|0|1|0|0-1|0|1|1|0
2         1    0.0    1.0    0.0        0            1          0|1|0|0|0           0|1|0|0|0-0|1|0|1|0
3         1    0.0    1.0    0.0        1            1          0|1|0|1|0           0|1|0|0|0-0|1|0|1|0
4         2    1.0    1.0    0.0        0            1          1|1|0|0|0           1|1|0|0|0-1|1|0|1|0
5         2    1.0    1.0    0.0        1            1          1|1|0|1|0           1|1|0|0|0-1|1|0|1|0
```

In this example:
- Each pair of rows represents a subgroup (identified by group_id)
- The algorithm found instances with similar attribute values but different outcomes
- diff_outcome shows the magnitude of the difference in outcomes
- Higher values of diff_outcome indicate stronger bias

The classification report would look something like:

```
              precision    recall  f1-score   support
           0       0.85      0.82      0.83       120
           1       0.79      0.83      0.81       105
    accuracy                           0.82       225
   macro avg       0.82      0.82      0.82       225
weighted avg       0.82      0.82      0.82       225
```

This shows the model's performance metrics on the test set, which helps contextualize the bias findings.

## Usage Example

As seen in the test.py file, the algorithm is typically used as follows:

```python
from data_generator.main import generate_data
from methods.subgroup.biasscan.algo import run_bias_scan

# Generate synthetic data with controlled bias
ge = generate_data(
    nb_attributes=6,
    min_number_of_classes=2,
    max_number_of_classes=4,
    prop_protected_attr=0.1,
    nb_groups=100,
    max_group_size=100,
    categorical_outcome=True,
    nb_categories_outcome=4,
    use_cache=True
)

# Run the BiasScan algorithm
result_df, report = run_bias_scan(ge, test_size=0.3, random_state=42, n_estimators=200, bias_scan_num_iters=100,
                                  bias_scan_scoring='Poisson', bias_scan_favorable_value='high',
                                  bias_scan_mode='ordinal')

# Print results
print(result_df)
print(f"Classification Report:\n{report}")
```

The BiasScan algorithm is particularly useful for identifying specific subgroups in the data where a model exhibits biased behavior, allowing for targeted bias mitigation strategies.