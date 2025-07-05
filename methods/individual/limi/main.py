import pandas as pd
import numpy as np
import time
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
import warnings

from data_generator.main import get_real_data, DiscriminationData
from methods.utils import train_sklearn_model

warnings.filterwarnings('ignore')


class LatentImitator:
    """
    Implementation of the Latent Imitator (LIMI) framework for fairness testing.
    This class uses PCA to simulate a GAN's latent space and generator.
    """

    def __init__(self, black_box_model, generator, protected_attribute_indices, lambda_val=0.3):
        """
        Initializes the LIMI tester.

        Args:
            black_box_model: The pre-trained classification model to be tested. Must have a `predict_proba` method.
            generator (PCA): A fitted PCA object that acts as our generator.
            protected_attribute_indices (tuple): A tuple of indices for the one-hot encoded protected attribute.
            lambda_val (float): The step size for probing around the surrogate boundary.
        """
        self.black_box_model = black_box_model
        self.generator = generator
        self.protected_attribute_indices = protected_attribute_indices
        self.lambda_val = lambda_val
        self.latent_dim = generator.n_components_
        self.surrogate_model = None

    def _approximate_boundary(self, n_samples=100_000, confidence_threshold=0.7):
        """
        STEP 1: Latent Boundary Approximation (with Refinement)
        Approximates the model's decision boundary in the latent space by training a linear SVM
        on a refined, high-confidence, and balanced dataset, as described in the paper.
        """
        print(f"Step 1: Approximating decision boundary with {n_samples} synthetic samples...")

        # 1. Generate random latent vectors
        z_init = np.random.randn(n_samples, self.latent_dim)

        # 2. Use the generator to create synthetic data samples
        x_synthetic = self.generator.inverse_transform(z_init)

        # --- REFINEMENT STAGE (As per paper) ---

        # 3a. Get prediction probabilities from the black-box model
        y_probs = self.black_box_model.predict_proba(x_synthetic)
        y_pred = np.argmax(y_probs, axis=1)  # Get the hard labels

        # 3b. Filter for high-confidence samples
        print(f"Filtering for samples with confidence > {confidence_threshold}...")
        confidence_scores = np.max(y_probs, axis=1)
        high_confidence_mask = confidence_scores >= confidence_threshold

        z_high_confidence = z_init[high_confidence_mask]
        y_high_confidence = y_pred[high_confidence_mask]

        if len(z_high_confidence) == 0:
            raise ValueError(
                f"No samples met the confidence threshold of {confidence_threshold}. "
                "Try lowering the threshold or increasing n_samples."
            )

        print(f"Retained {len(z_high_confidence)} high-confidence samples out of {n_samples}.")

        # 3c. Balance the dataset using random over-sampling
        print("Balancing the high-confidence dataset using RandomOverSampler...")
        ros = RandomOverSampler(random_state=42)
        z_balanced, y_balanced = ros.fit_resample(z_high_confidence, y_high_confidence)

        print(f"Dataset size after balancing: {len(z_balanced)} samples.")

        # 4. Train a simple, linear surrogate model (SVM) on the REFINED latent vectors and predictions
        print("Training surrogate linear SVM on the refined latent space...")
        surrogate_svm = SVC(kernel='linear', C=1.0, random_state=42)
        surrogate_svm.fit(z_balanced, y_balanced)  # Fit on balanced, high-confidence data

        self.surrogate_model = surrogate_svm
        print("Surrogate boundary learned.")

    def _probe_candidates(self, z_initial_set):
        """
        STEP 2: Latent Candidates Probing
        For a given set of latent vectors, find candidate points near the surrogate boundary.
        """
        print(f"Step 2: Probing for {len(z_initial_set)} candidate triplets...")
        if self.surrogate_model is None:
            raise RuntimeError("You must run _approximate_boundary first.")

        w = self.surrogate_model.coef_[0]
        b = self.surrogate_model.intercept_[0]

        w_norm = np.linalg.norm(w)
        if w_norm == 0:
            raise ValueError("Surrogate model weight vector is zero. Cannot proceed.")
        w_u = w / w_norm

        candidate_triplets = []
        for z in tqdm(z_initial_set, desc="Probing Candidates"):
            dist = (np.dot(z, w) + b) / (w_norm ** 2)
            z0 = z - dist * w
            z_plus = z0 + self.lambda_val * w_u
            z_minus = z0 - self.lambda_val * w_u
            candidate_triplets.append((z0, z_plus, z_minus))

        return candidate_triplets

    def _verify_and_generate(self, candidate_triplets):
        """
        STEP 3: Generation and Verification
        Generate data from latent candidates and check for discrimination.
        """
        print("Step 3: Verifying candidates and generating discriminatory instances...")
        discriminatory_instances = []
        idx1, idx2 = self.protected_attribute_indices

        for triplet in tqdm(candidate_triplets, desc="Verifying Triplets"):
            for z_candidate in triplet:
                x_orig_vector = self.generator.inverse_transform(z_candidate.reshape(1, -1))
                pred_orig = self.black_box_model.predict(x_orig_vector)[0]

                x_mod_vector = x_orig_vector.copy()
                val1 = round(x_mod_vector[0, idx1])
                val2 = round(x_mod_vector[0, idx2])
                x_mod_vector[0, idx1] = val2
                x_mod_vector[0, idx2] = val1
                pred_mod = self.black_box_model.predict(x_mod_vector)[0]

                if pred_orig != pred_mod:
                    discriminatory_instances.append({
                        'original_input': x_orig_vector[0],
                        'modified_input': x_mod_vector[0],
                        'original_prediction': pred_orig,
                        'modified_prediction': pred_mod
                    })
                    break

        return discriminatory_instances

    def test(self, n_test_samples=10_000, n_approx_samples=50_000):
        """
        Runs the full LIMI testing pipeline.
        """
        # Step 1: Learn the surrogate boundary using more samples for better approximation
        self._approximate_boundary(n_samples=n_approx_samples, confidence_threshold=0.7)

        # Generate a new set of random latent vectors to test
        z_test_set = np.random.randn(n_test_samples, self.latent_dim)

        # Step 2: Find candidate points near the boundary
        candidate_triplets = self._probe_candidates(z_test_set)

        # Step 3: Verify candidates and find discriminatory instances
        found_instances = self._verify_and_generate(candidate_triplets)

        return found_instances


def run_limi(discrimination_data: DiscriminationData, lambda_val=0.3, n_test_samples=2000, n_approx_samples=50000):
    start_time = time.time()
    print("\nTraining the black-box model (RandomForest)...")
    # RandomForestClassifier has `predict_proba`, so it's a suitable model for this corrected code.
    model, X_train, X_test, y_train, y_test, feature_names, metrics = train_sklearn_model(
        data=discrimination_data.training_dataframe,
        target_col=discrimination_data.outcome_column,
        sensitive_attrs=discrimination_data.protected_attributes,
        model_type='rf')

    print(f"Black-box model accuracy: {metrics['accuracy']:.2%}")

    # 3. Create our "Generator" (PCA)
    print("\nTraining the 'Generator' (PCA model) to learn the data distribution...")
    n_features = X_train.shape[1]
    LATENT_DIM = min(10, n_features - 1) if n_features > 1 else 1
    generator_pca = PCA(n_components=LATENT_DIM, random_state=42)
    generator_pca.fit(X_train)
    print(f"Generator created with latent space dimension: {LATENT_DIM}")

    # 5. Initialize and run LIMI
    print("\n--- Initializing Latent Imitator (LIMI) ---")
    limi_tester = LatentImitator(
        black_box_model=model,
        generator=generator_pca,
        protected_attribute_indices=tuple(discrimination_data.sensitive_indices),
        lambda_val=lambda_val
    )

    discriminatory_instances = limi_tester.test(n_test_samples=n_test_samples, n_approx_samples=n_approx_samples)

    # 6. Analyze Results
    print("\n--- LIMI Test Results ---")
    print(f"Total discriminatory instances found: {len(discriminatory_instances)}")
    
    # Track metrics for compatibility with other methods
    tot_inputs = set()  # Total inputs tested
    all_discriminations = []  # Discriminatory pairs found
    dsn_by_attr_value = {'total': {'TSN': 0, 'DSN': 0}}
    
    # Initialize counters for protected attributes
    for attr in discrimination_data.protected_attributes:
        for val in discrimination_data.training_dataframe[attr].unique():
            key = f"{attr}={val}"
            dsn_by_attr_value[key] = {'TSN': 0, 'DSN': 0}
    
    if discriminatory_instances:
        for instance in discriminatory_instances:
            # Extract original and modified inputs and their predictions
            org_input = instance['original_input']
            mod_input = instance['modified_input']
            org_pred = instance['original_prediction']
            mod_pred = instance['modified_prediction']
            
            # Add to total inputs
            tot_inputs.add(tuple(org_input))
            tot_inputs.add(tuple(mod_input))
            
            # Add to discriminatory pairs
            all_discriminations.append((org_input, org_pred, mod_input, mod_pred))
            
            # Update counts for protected attributes
            for i, attr in enumerate(discrimination_data.protected_attributes):
                attr_idx = discrimination_data.feature_names.index(attr)
                org_val = org_input[attr_idx]
                mod_val = mod_input[attr_idx]
                
                # Update counts for this attribute value
                key_org = f"{attr}={org_val}"
                key_mod = f"{attr}={mod_val}"
                
                if key_org in dsn_by_attr_value:
                    dsn_by_attr_value[key_org]['TSN'] += 1
                    dsn_by_attr_value[key_org]['DSN'] += 1
                
                if key_mod in dsn_by_attr_value:
                    dsn_by_attr_value[key_mod]['TSN'] += 1
                
                # Update total counts
                dsn_by_attr_value['total']['TSN'] += 2  # Both original and modified
                dsn_by_attr_value['total']['DSN'] += 1  # One discriminatory pair
    
    # Calculate metrics and create result dataframe in the same format as exp_ga
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    tsn = len(tot_inputs)  # Total Sample Number
    dsn = len(all_discriminations)  # Discriminatory Sample Number
    sur = dsn / tsn if tsn > 0 else 0  # Success Rate
    dss = total_time / dsn if dsn > 0 else float('inf')  # Discriminatory Sample Search time
    
    # Update SUR and DSS for each attribute value
    for k, v in dsn_by_attr_value.items():
        if k != 'total':
            dsn_by_attr_value[k]['SUR'] = dsn_by_attr_value[k]['DSN'] / dsn_by_attr_value[k]['TSN'] if \
                dsn_by_attr_value[k]['TSN'] != 0 else 0
            dsn_by_attr_value[k]['DSS'] = dss
    
    # Create metrics dict
    metrics = {
        'TSN': tsn,
        'DSN': dsn,
        'SUR': sur,
        'DSS': dss,
        'total_time': total_time,
        'dsn_by_attr_value': dsn_by_attr_value
    }
    
    # Log results
    print("\nFinal Results:")
    print(f"Total inputs tested: {tsn}")
    print(f"Total discriminatory pairs: {dsn}")
    print(f"Success rate (SUR): {sur:.4f}")
    print(f"Avg. search time per discriminatory sample (DSS): {dss:.4f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    
    # Generate result dataframe in the same format as exp_ga
    res_df = []
    case_id = 0
    
    for org, org_res, counter_org, counter_org_res in all_discriminations:
        # Create dataframes for original and modified inputs
        indv1 = pd.DataFrame([list(org)], columns=discrimination_data.feature_names)
        indv2 = pd.DataFrame([list(counter_org)], columns=discrimination_data.feature_names)
        
        # Create individual keys
        indv_key1 = "|".join(str(x) for x in indv1[discrimination_data.feature_names].iloc[0])
        indv_key2 = "|".join(str(x) for x in indv2[discrimination_data.feature_names].iloc[0])
        
        # Add additional columns
        indv1['indv_key'] = indv_key1
        indv1['outcome'] = org_res
        indv2['indv_key'] = indv_key2
        indv2['outcome'] = counter_org_res
        
        # Create couple_key and diff_outcome
        couple_key = f"{indv_key1}-{indv_key2}"
        diff_outcome = abs(org_res - counter_org_res)
        
        # Combine into a single dataframe
        df_res = pd.concat([indv1, indv2])
        df_res['couple_key'] = couple_key
        df_res['diff_outcome'] = diff_outcome
        df_res['case_id'] = case_id
        res_df.append(df_res)
        case_id += 1
    
    if len(res_df) != 0:
        res_df = pd.concat(res_df)
        # Add metrics to result dataframe
        res_df['TSN'] = tsn
        res_df['DSN'] = dsn
        res_df['SUR'] = sur
        res_df['DSS'] = dss
    else:
        # Create empty dataframe with correct columns if no discriminatory instances found
        res_df = pd.DataFrame(columns=discrimination_data.feature_names + 
                             ['indv_key', 'outcome', 'couple_key', 'diff_outcome', 'case_id',
                              'TSN', 'DSN', 'SUR', 'DSS'])
    
    return res_df, metrics


if __name__ == '__main__':
    discrimination_data, data_schema = get_real_data('adult', use_cache=True)

    results_df, metrics = run_limi(discrimination_data)
    print(results_df)
    print(f"\nTesting Metrics: {metrics}")
