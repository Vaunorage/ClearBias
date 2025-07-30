### Title : Probabilistic Verification of Fairness Properties via Concentration

### **Metric: What metric does the method rely on to find discrimination?**

The method is not tied to a single metric. It uses a flexible **Specification Language** that can express a wide range of fairness definitions as arithmetic and logical constraints over the expected outcomes for different groups. The paper provides several examples of fairness metrics that can be verified:

*   **Demographic Parity:** Verifies that minority members receive a favorable outcome at a rate that is at least some fraction (e.g., 80%) of the rate for majority members.
*   **Equal Opportunity:** A refinement of demographic parity, this metric requires that *qualified* members of the minority group receive favorable outcomes at a similar rate to *qualified* members of the majority group.
*   **Path-Specific Causal Fairness:** A causal notion of fairness that checks if a sensitive attribute (e.g., gender) influences the outcome directly, while allowing for indirect influence through non-discriminatory intermediate paths (e.g., years of experience).
*   **Individual Fairness:** The paper also discusses an extension to verify that individuals with similar features are treated similarly, with high probability.

### **Location: Where does this method try to find discrimination?**

The method finds discrimination in the **behavior of the machine learning model** when applied to a given population. The algorithm's inputs are:
1.  A **Classification Program (`f`)**: The machine learning model (e.g., a neural network, SVM) to be verified, which is treated as a black box.
2.  A **Population Model (`P_V`)**: A probabilistic program that generates random members of the population, representing the distribution of data the model will see in the real world.

The method assesses the fairness of the model's decisions (`f`) with respect to the specified data distribution (`P_V`), rather than analyzing a static dataset for bias.

### **What they find: What exactly does this method try to find?**

The method does not "discover" which groups are discriminated against from scratch. Instead, it **verifies a user-defined fairness specification**. The user must first define:
1.  The groups of interest (e.g., majority and minority subpopulations).
2.  The fairness criterion to be checked (e.g., demographic parity).

The algorithm then determines if the model's behavior violates this specific, pre-defined fairness rule for the pre-defined groups.

### **What does the method return in terms of data structure?**

The method returns a **single boolean value (`true` or `false`)** along with a strong probabilistic guarantee.

*   `true`: The fairness specification holds with high probability.
*   `false`: The fairness specification is violated with high probability.

The key feature is the guarantee: the algorithm's response `Ŷ` is correct with a probability of at least `1 - Δ`, where `Δ` is a user-chosen error tolerance. The paper demonstrates that `Δ` can be set to an extremely small value (e.g., 10⁻¹⁰), making the chance of an incorrect answer negligible.

### **Performance: How has this method's performance been evaluated and what was the result?**

The method was implemented in a tool called **VERIFAIR** and evaluated on two benchmarks with the following results:

1.  **FairSquare Benchmark:**
    *   **Comparison:** Compared against `FAIRSQUARE`, a state-of-the-art symbolic verification tool.
    *   **Result:** `VERIFAIR` was significantly faster and more scalable, outperforming `FAIRSQUARE` on all large problem instances. Where `FAIRSQUARE` timed out, `VERIFAIR` completed quickly.

2.  **Quick Draw Benchmark:**
    *   **Setup:** A much more challenging task involving a deep recurrent neural network (RNN) with over 16 million parameters, a scale that `FAIRSQUARE` cannot handle.
    *   **Result:** `VERIFAIR` successfully verified the fairness of this massive model, demonstrating its scalability. For instance, it terminated in about 10 minutes even when the probability of error (`Δ`) was set to 1 in 10 billion. The performance was shown to scale well, with the runtime increasing only linearly as the error tolerance `Δ` was decreased exponentially.

# Implementation of VeriFair Algorithm

## Input

The VeriFair algorithm, as implemented in `run_verifair.py`, takes the following inputs:

### Dataset Parameters

* `dataset_name`: Name of the dataset to analyze (e.g., 'adult')
* `analysis_attribute`: The protected attribute to analyze for discrimination (e.g., 'race', 'gender')
* `group_a_value` and `group_b_value`: The specific values of the protected attribute to compare (e.g., 'Male' vs 'Female')
* `analyze_all_combinations`: Boolean flag to analyze all possible combinations of protected attributes

### Algorithm Parameters

* `c`: The fairness threshold (default: 0.15) - defines the acceptable difference in outcome probabilities between groups
* `Delta`: The indifference parameter (default: 0.0) - allows for some tolerance in the fairness assessment
* `delta`: The confidence parameter (default: 0.5 * 1e-10) - controls the statistical confidence of the results
* `n_samples`: Number of samples to draw in each iteration (default: 1)
* `n_max`: Maximum number of samples to draw (default: 100000)
* `is_causal`: Boolean flag indicating whether to perform causal analysis (default: False)
* `log_iters`: How often to log progress (default: 1000 iterations)

## Process

1. The algorithm loads the specified dataset using the `get_real_data` function
2. It trains a Random Forest classifier on the dataset
3. It then creates samplers for two demographic groups (A and B) based on the specified protected attribute
4. The core verification is performed by the `verify` function, which:
   * Samples predictions from both groups
   * Computes the ratio of positive outcomes between the groups
   * Determines if the model is fair based on the specified fairness threshold `c`

## Output

The algorithm outputs a pandas DataFrame with the following columns:

1. `attribute`: The protected attribute analyzed (e.g., 'gender')
2. `group_a` and `group_b`: The values of the protected attribute compared (e.g., 'Male', 'Female')
3. `is_fair`: Boolean indicating whether the model is fair according to the specified threshold
4. `is_ambiguous`: Boolean indicating whether the result is statistically inconclusive
5. `estimated_ratio`: The estimated ratio of positive outcomes between groups
6. `p_value`: The statistical confidence (2.0 * delta)
7. `successful_samples`: Number of successful sampling iterations
8. `total_samples`: Total number of sampling attempts

## Example

Let's consider an example using the Adult dataset to check for gender discrimination:

```python
# Example function call
results = find_discrimination_with_verifair(
    dataset_name='adult',
    analysis_attribute='sex',
    group_a_value='Male',
    group_b_value='Female',
    c=0.15,  # 15% threshold for fairness
    delta=0.5 * 1e-10,  # High confidence
    n_max=10000  # Maximum number of samples
)
```

### Example Output

```
--- Analysis Summary ---
  attribute group_a group_b  is_fair  is_ambiguous  estimated_ratio    p_value  successful_samples  total_samples
0       sex    Male  Female    False         False           0.372  1.00e-10               8742          10000
```

In this example output:

* The algorithm analyzed the 'sex' attribute, comparing 'Male' vs 'Female'
* It determined the model is NOT fair (`is_fair=False`)
* The result is statistically conclusive (`is_ambiguous=False`)
* The estimated ratio of positive outcomes is 0.372, meaning females receive positive outcomes at approximately 37.2% the rate of males
* The p-value is very low (1.00e-10), indicating high statistical confidence
* The algorithm used 8,742 successful samples out of 10,000 total samples

This indicates a significant gender bias in the model's predictions on the Adult dataset, with males receiving favorable outcomes at a much higher rate than females.

