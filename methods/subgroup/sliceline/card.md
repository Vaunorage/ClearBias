### **Title**
SliceLine: Fast, Linear-Algebra-based Slice Finding for ML Model Debugging

### **Metric**
The method relies on a custom, flexible **Scoring Function** to find problematic data slices. This score quantifies how much worse a model performs on a specific slice compared to its average performance, while also considering the size of the slice.

The score `sc` for a slice `S` is defined as:
`sc = α * ( (se / |S|) / ē - 1 ) - (1 - α) * ( n / |S| - 1 )`

Where:
*   `se / |S|` is the average error on the slice.
*   `ē` is the average error on the entire dataset.
*   `|S|` is the size (number of rows) of the slice.
*   `n` is the size of the entire dataset.
*   `α` is a user-defined weight parameter (between 0 and 1) that balances the importance of the slice's error versus its size. A higher `α` prioritizes slices with high error, even if they are small.

A score `sc > 0` indicates that the model performs worse on the slice than on the overall dataset. The goal is to find slices that maximize this score.

### **Individual discrimination, group, subgroup discrimination**
SliceLine is designed to find **subgroup discrimination** with a high degree of **granularity and intersectionality**.

*   It does not focus on individual discrimination (i.e., comparing one individual to another).
*   It identifies **groups** (or "slices") defined by the conjunction of multiple feature predicates. For example, it can find that a model underperforms for the subgroup where `gender = female` AND `degree = PhD`.
*   By searching through combinations of features, it inherently uncovers intersectional biases that might be missed when looking at single features in isolation.

### **Location**
The method tries to find discrimination by analyzing a **trained model's performance on the data**. It does not modify the model or the training process itself. It operates as a post-hoc debugging tool that takes a model's predictions (and resulting errors) on a dataset and searches for problematic subsets within that data.

### **What they find**
The method finds the **top-K problematic data slices**. A "slice" is a subset of the data defined by a conjunction of predicates on its features. A "problematic" slice is one where a trained ML model performs significantly worse (i.e., has a higher error rate) than its average performance across the entire dataset, according to the scoring function.

The method aims to find discriminated groups (subgroups) relative to the average performance, not to find discriminated individuals.

### **What does the method return in terms of data structure?**
The algorithm returns two main data structures:
1.  **`TS`**: A `K × m` integer-encoded matrix representing the top-K slices found. Each row corresponds to a slice, and the values indicate the feature predicates that define it (with zeros representing "don't care" features).
2.  **`TR`**: A matrix containing the corresponding statistics for each of the top-K slices. This includes the calculated score, total error, average error, and size of each slice.

### **Performance**
The performance was evaluated on its effectiveness (pruning), efficiency (runtime), and scalability using a variety of real-world datasets (Adult, Covtype, KDD98, US Census, Criteo).

*   **Evaluation Environment**: Experiments were run on a powerful scale-up server (112 virtual cores, 768 GB RAM) and a scale-out cluster. The method was implemented in Apache SystemDS, which leverages efficient sparse linear algebra.
*   **Pruning Effectiveness**: The paper shows that the combination of size pruning, score pruning, and handling of parent-child relationships in the search lattice is crucial. Without these techniques, the enumeration of slices becomes computationally infeasible even on small datasets.
*   **End-to-end Runtime & Scalability**:
    *   **Local Performance**: The method is very fast. On the Adult dataset, it completed in **5.6 seconds**, which is significantly faster than the >100s reported for the original SliceFinder work on the same dataset.
    *   **Scalability with Rows**: The method showed excellent scalability with the number of rows, demonstrating near-linear performance degradation as the USCensus dataset was replicated up to 10 times (to ~25M rows).
    *   **Scalability with Columns**: On the massive Criteo dataset (192M rows, 76M one-hot encoded columns), SliceLine was able to effectively enumerate slices and completed in under 45 minutes in a distributed environment, demonstrating its ability to handle extremely high-dimensional, sparse data.