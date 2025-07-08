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

# ExpGA Algorithm Implementation

## Input Parameters of ExpGA Algorithm

The ExpGA (Explainable Genetic Algorithm) is a fairness testing algorithm that identifies discriminatory instances in machine learning models by examining how protected attributes influence model predictions. Here's a detailed breakdown of its input parameters:

## Required Parameters

### `data: DiscriminationData`
A data object containing the dataset with features, labels, and metadata about protected attributes
- Contains information about feature names, protected attributes, input bounds, and sensitive indices

### `threshold_rank: float`
Threshold value that determines when a protected attribute is considered influential
- Used during seed selection to identify promising instances for discrimination testing
- **Example**: `threshold_rank=0.5` means attributes with influence above 50% are considered significant

### `max_global: int`
Maximum number of global samples to generate during the initial discovery phase
- Controls how many samples are created for each protected attribute
- **Example**: `max_global=20000` will generate up to 20,000 samples distributed across protected attributes

### `max_local: int`
Maximum number of iterations for the genetic algorithm's local search phase
- Controls how many generations the GA will evolve to find discriminatory instances
- **Example**: `max_local=100` will run up to 100 generations of evolution

## Optional Parameters

### `model_type: str = 'rf'`
Type of machine learning model to train and test (default: Random Forest)

### `cross_rate: float = 0.9`
Crossover rate for the genetic algorithm (range: 0.0 to 1.0)

### `mutation: float = 0.1`
Mutation rate for the genetic algorithm (range: 0.0 to 1.0)

### `max_runtime_seconds: float = None`
Maximum runtime in seconds before early termination

### `max_tsn: int = None`
Maximum number of test samples (TSN) before early termination

### `random_seed: int = 100`
Seed for random number generation to ensure reproducibility

### `one_attr_at_a_time: bool = False`
- If `True`, tests discrimination by varying one protected attribute at a time
- If `False`, tests all combinations of protected attributes

### `db_path: str = None`
Path to SQLite database for storing results

### `analysis_id: str = None`
Identifier for the analysis run in the database

### `use_gpu: bool = False`
Whether to use GPU acceleration for model training and inference

### `**model_kwargs`
Additional keyword arguments passed to the model training function

## Output

The algorithm returns a tuple with two elements:

### `res_df: pd.DataFrame`
A DataFrame containing the discriminatory instances found with the following structure:

- **attr_columns**: All feature columns from the original dataset
- **indv_key**: A string key identifying each individual instance (pipe-separated feature values)
- **outcome**: The model's prediction for this instance
- **couple_key**: A key linking pairs of instances that demonstrate discrimination (format: "indv_key1-indv_key2")
- **diff_outcome**: The absolute difference in outcomes between the paired instances
- **case_id**: A unique identifier for each discriminatory case
- **TSN**: Total Sample Number (total instances tested)
- **DSN**: Discriminatory Sample Number (instances showing discrimination)
- **SUR**: Success Rate (DSN/TSN)
- **DSS**: Discriminatory Sample Search time (time per discriminatory sample)

### `metrics: Metrics`
A dictionary with the following fairness metrics:

- **TSN**: Total number of test samples evaluated
- **DSN**: Number of discriminatory instances found
- **SUR**: Success Rate (DSN/TSN)
- **DSS**: Average search time per discriminatory sample (seconds)
- **total_time**: Total execution time (seconds)
- **dsn_by_attr_value**: Detailed breakdown of discrimination by protected attribute

## Example Output DataFrame

```python
# Example output DataFrame (res_df)
example_df = pd.DataFrame({
    # Original instance features
    'age': [35, 35, 42, 42],
    'gender': [0, 1, 1, 1],
    'race': [1, 1, 0, 1],
    'income': [50000, 50000, 65000, 65000],
    
    # Additional columns
    'indv_key': ['35|0|1|50000', '35|1|1|50000', '42|1|0|65000', '42|1|1|65000'],
    'outcome': [0, 1, 1, 0],
    'couple_key': ['35|0|1|50000-35|1|1|50000', '35|0|1|50000-35|1|1|50000', 
                  '42|1|0|65000-42|1|1|65000', '42|1|0|65000-42|1|1|65000'],
    'diff_outcome': [1, 1, 1, 1],
    'case_id': [0, 0, 1, 1],
    
    # Metrics added to all rows
    'TSN': [5000, 5000, 5000, 5000],
    'DSN': [2, 2, 2, 2],
    'SUR': [0.0004, 0.0004, 0.0004, 0.0004],
    'DSS': [12.5, 12.5, 12.5, 12.5]
})
```

In this example, each discriminatory case includes two rows: the original instance and the modified instance with different protected attribute values. The pairs show that changing protected attributes (gender in case 0, race in case 1) led to different model outcomes, indicating potential discrimination.

## Example Metrics Dictionary

```python
example_metrics = {
    'TSN': 5000,
    'DSN': 2,
    'SUR': 0.0004,
    'DSS': 12.5,
    'total_time': 25.0,
    'dsn_by_attr_value': {
        'gender': {'TSN': 2500, 'DSN': 1, 'SUR': 0.0004, 'DSS': 12.5},
        'race': {'TSN': 2500, 'DSN': 1, 'SUR': 0.0004, 'DSS': 12.5},
        'total': 2
    }
}
```

The metrics provide an overall summary of the testing process and detailed information about discrimination by each protected attribute.