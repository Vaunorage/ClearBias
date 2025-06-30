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