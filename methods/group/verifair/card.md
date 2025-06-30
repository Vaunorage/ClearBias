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