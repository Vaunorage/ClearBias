### **Title**
Explanation-Guided Fairness Testing through Genetic Algorithm

### **Metric**
The method focuses on **individual discrimination**. The paper defines this as a scenario where two individuals, differing only in a protected attribute (e.g., gender, race), receive different decisions from the AI model. The goal is to find pairs of samples (x, x') where x and x' are identical except for the protected attribute, but the model's prediction `f(x)` is not equal to `f(x')`. The paper explicitly states, "Most research on software fairness, including this work, has focused on individual fairness" and does not address group or subgroup fairness.

### **Location**
The method finds discrimination within the **AI model**. It is a model-agnostic, black-box testing approach, meaning it does not need access to the model's internal structure or gradients. It operates by generating new input samples and querying the model for its predictions to reveal biased behaviors. The problem is defined as: "given a black-box model D... can we effectively and efficiently detect individual discriminatory samples for D?".

### **What they find**
The method, named **ExpGA**, is designed to find and generate **individual discriminatory samples**. These are inputs that expose the model's fairness violations. The process involves:
1.  Using interpretable methods (like LIME) to find "seed samples" that are likely to be discriminatory.
2.  Employing a Genetic Algorithm (GA) to mutate these seeds and efficiently search for new inputs that cause the model to change its prediction when a protected attribute is altered.

The final output of the method is a **set of discriminatory samples** that can be used to evaluate and subsequently improve the model's fairness.

### **Performance**
The performance of ExpGA was evaluated based on its **efficiency** and **effectiveness** in finding discriminatory samples, and its ability to improve model fairness through retraining.

*   **Evaluation Metrics:**
    *   **Efficiency (DSS):** Average time to find one discriminatory sample (lower is better).
    *   **Effectiveness (SUR):** Success rate of generating discriminatory samples (higher is better).

*   **Comparison:**
    *   ExpGA was compared against state-of-the-art methods: AEQUITAS, SG, and ADF for tabular data, and MT-NLP for text data.

*   **Results on Tabular Datasets (e.g., Census, Credit):**
    *   ExpGA demonstrated superior performance, requiring on average **less than 0.2 seconds (DSS)** to find a discriminatory sample with a **success rate (SUR) of about 49%**.
    *   It was significantly more efficient than baselines. For example, on one dataset, ExpGA's DSS was 0.03s, compared to 2.00s for AEQUITAS and 4.13s for SG.
    *   The performance was also more stable across different model types (MLP, RF, SVM).

*   **Results on Text Datasets (e.g., IMDB, SST):**
    *   ExpGA outperformed the MT-NLP baseline, being at least **five times more efficient (lower DSS)** and **twice as effective (higher SUR)**. For instance, on the IMDB dataset, ExpGA's DSS was 16.42s vs. MT-NLP's 90.73s.

*   **Fairness Improvement through Retraining:**
    *   By augmenting the training data with the discriminatory samples found by ExpGA and retraining the model, the model's fairness was considerably improved.
    *   After retraining, **over 97% of the original discriminatory samples were no longer misclassified**, while the model's accuracy on normal, non-discriminatory samples remained nearly unchanged.