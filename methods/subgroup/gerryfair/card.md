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