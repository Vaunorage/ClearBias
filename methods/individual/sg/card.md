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