import random

import numpy as np
import collections
import pandas as pd
import time
import logging
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from tqdm import tqdm
from data_generator.main import generate_optimal_discrimination_data, DataSchema, get_real_data
from data_generator.main import DiscriminationData
from methods.utils import train_sklearn_model, make_final_metrics_and_dataframe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('KOSEI')


class KOSEI:
    """
    An implementation of the KOSEI algorithm for individual fairness testing.

    This implementation is designed to work with the DiscriminationData class from the data_generator module.
    """

    def __init__(self,
                 model: Optional[Union[Callable[[np.ndarray], Any], str]] = None,
                 discrimination_data: Optional[DiscriminationData] = None,
                 protected_attribute_indices: Optional[List[int]] = None,
                 attribute_domains: Optional[Dict[int, tuple]] = None,
                 total_attributes: Optional[int] = None,
                 gamma: float = 0.0,
                 model_type: str = 'rf',
                 model_params: Optional[Dict] = None):
        """
        Initialize the KOSEI algorithm.

        Args:
            model: A callable model that takes a numpy array as input and returns a prediction
            discrimination_data: Optional DiscriminationData object containing dataset information
            protected_attribute_indices: List of indices for protected attributes (used if discrimination_data is None)
            attribute_domains: Dictionary mapping attribute indices to their domains (used if discrimination_data is None)
            total_attributes: Total number of attributes (used if discrimination_data is None)
            gamma: Threshold for determining discrimination (default: 0.0)
            model_type: Type of model to train if no model is provided (default: 'rf')
            model_params: Parameters for the model to train if no model is provided (default: None)
            target_col: Target column for the model to train if no model is provided (default: 'class')
        """
        self.model = model
        self.gamma = gamma

        if discrimination_data is not None:
            # Extract information from DiscriminationData object
            self.discrimination_data = discrimination_data
            self.attr_names = discrimination_data.attr_columns
            self.protected_attribute_indices = discrimination_data.sensitive_indices
            self.non_protected_attribute_indices = [
                i for i in range(len(discrimination_data.attr_columns))
                if i not in discrimination_data.sensitive_indices
            ]

            # Convert attribute domains to integer ranges
            self.attribute_domains = {}
            for i, (min_val, max_val) in enumerate(discrimination_data.input_bounds):
                self.attribute_domains[i] = (int(min_val), int(max_val))

            self.total_attributes = len(discrimination_data.attr_columns)

            # Train a model if none is provided or if a model type string is provided
            if model is None or isinstance(model, str):
                print(f"\n--- Training a {'custom' if isinstance(model, str) else 'default'} {model_type} model ---")
                model_type_to_use = model if isinstance(model, str) else model_type

                # Train the model
                trained_model, X_train, X_test, y_train, y_test, feature_names, metrics = train_sklearn_model(
                    data=self.discrimination_data.training_dataframe,
                    model_type=model_type_to_use,
                    model_params=model_params,
                    target_col=self.discrimination_data.outcome_column,
                    sensitive_attrs=self.discrimination_data.sensitive_indices_dict.keys(),
                )

                self.model = trained_model

                print(f"Model trained successfully.")
            else:
                self.model = model
        else:
            # Initialize from direct parameters
            if protected_attribute_indices is None or attribute_domains is None or total_attributes is None:
                raise ValueError("If discrimination_data is not provided, you must specify "
                                 "protected_attribute_indices, attribute_domains, and total_attributes.")
            self.discrimination_data = None
            self.attr_names = [f"attr_{i}" for i in range(total_attributes)]
            self.protected_attribute_indices = protected_attribute_indices
            self.non_protected_attribute_indices = [
                i for i in range(total_attributes)
                if i not in protected_attribute_indices
            ]
            self.attribute_domains = attribute_domains
            self.total_attributes = total_attributes

            # Store the model
            if model is None:
                raise ValueError("If discrimination_data is not provided, you must provide a model.")
            self.model = model

        print(f"KOSEI Initialized.")
        print(f"  - Protected Attributes (indices): {self.protected_attribute_indices}")
        print(f"  - Non-Protected Attributes (indices): {self.non_protected_attribute_indices}")

    def _eval_disc(self, data_item: np.ndarray) -> bool:
        """
        Evaluates if a single data item is discriminatory.

        Args:
            data_item: A numpy array representing a single data point

        Returns:
            bool: True if the item is discriminatory, False otherwise
        """
        original_prediction = self.model.predict(data_item.reshape(1, -1))[0]

        for p_idx in self.protected_attribute_indices:
            min_val, max_val = self.attribute_domains.get(p_idx, (None, None))

            if min_val is None:
                continue

            for value in range(int(min_val), int(max_val) + 1):
                if data_item[p_idx] == value:
                    continue

                counterpart_item = data_item.copy()
                counterpart_item[p_idx] = value

                try:
                    counterpart_prediction = self.model.predict(counterpart_item.reshape(1, -1))[0]
                except (ValueError, TypeError):
                    counterpart_prediction = self.model.predict(counterpart_item)

                if abs(original_prediction - counterpart_prediction) > self.gamma:
                    return True

        return False

    def global_search(self, num_samples: int) -> List[np.ndarray]:
        """
        Performs a simple random global search to find initial seeds.

        Args:
            num_samples: The number of random samples to generate and test.

        Returns:
            A list of initial discriminatory seeds (D_global).
        """
        print(f"\n--- Starting Global Search ({num_samples} samples) ---")
        d_global = []
        found_discriminatory_set = set()

        for _ in tqdm(range(num_samples), desc="Global Search", unit="samples"):
            sample = []
            for i in range(self.total_attributes):
                min_val, max_val = self.attribute_domains[i]
                sample.append(np.random.randint(min_val, max_val + 1))

            sample = np.array(sample)
            sample_tuple = tuple(sample)

            # Evaluate the sample only if it's new
            if sample_tuple not in found_discriminatory_set:
                if self._eval_disc(sample):
                    d_global.append(sample)
                    found_discriminatory_set.add(sample_tuple)

        print(f"Global Search found {len(d_global)} initial discriminatory seeds.")
        return d_global

    def local_search(self, D_global: List[np.ndarray], limit: int) -> List[np.ndarray]:
        """
        Performs the KOSEI local search (implements Algorithm 3).
        
        Args:
            D_global: List of initial discriminatory seeds from global search
            limit: Maximum number of iterations for the local search
            
        Returns:
            List of discriminatory items found during the search
        """
        if not D_global:
            print("\n--- Starting Local Search ---")
            print("No seeds provided by global search. Terminating.")
            return []

        # Use a deque for efficient appends and poplefts (FIFO queue)
        seed_queue = collections.deque(tuple(item) for item in D_global)

        # Use sets for O(1) average time complexity checks for duplicates
        found_discriminatory_set = {tuple(item) for item in D_global}
        evaluated_items_set = {tuple(item) for item in D_global}

        print(f"\n--- Starting Local Search ---")
        print(f"Beginning with {len(seed_queue)} initial seeds and a limit of {limit} iterations...")

        for i in range(limit):
            if not seed_queue:
                print(f"Search terminated early at iteration {i + 1} because the seed queue is empty.")
                break

            current_item_tuple = seed_queue.popleft()
            current_item = np.array(current_item_tuple)

            for p_idx in self.non_protected_attribute_indices:
                for delta in [-1, 1]:
                    perturbed_item = current_item.copy()
                    perturbed_item[p_idx] += delta

                    min_val, max_val = self.attribute_domains.get(p_idx, (None, None))
                    if min_val is not None and not (min_val <= perturbed_item[p_idx] <= max_val):
                        continue

                    perturbed_item_tuple = tuple(perturbed_item)
                    if perturbed_item_tuple in evaluated_items_set:
                        continue

                    evaluated_items_set.add(perturbed_item_tuple)

                    if self._eval_disc(perturbed_item):
                        found_discriminatory_set.add((tuple(current_item), perturbed_item_tuple))
                        seed_queue.append(perturbed_item_tuple)

        print(f"Local search finished.")
        print(f"  - Total unique items evaluated: {len(evaluated_items_set)}")
        print(f"  - Total unique discriminatory items found: {len(found_discriminatory_set)}")

        return found_discriminatory_set

    def run(self, num_samples: int = 1000, local_search_limit: int = 500, random_seed: int = 42, max_runtime_seconds: int = 3600) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the full KOSEI pipeline: global search followed by local search.
        
        Args:
            num_samples: Number of random samples to generate in global search
            local_search_limit: Maximum number of iterations for local search
            
        Returns:
            Tuple containing:
                - DataFrame with discriminatory pairs and metrics
                - Dictionary with metrics summary
        """
        logger.info("\n=== Running Full KOSEI Pipeline ===")
        logger.info(f"Parameters: num_samples={num_samples}, local_search_limit={local_search_limit}, random_seed={random_seed}, max_runtime_seconds={max_runtime_seconds}")
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        start_time = time.time()

        # Initialize tracking variables for metrics
        tot_inputs = set()  # Track all inputs tested
        all_discriminations = []  # Track all discriminatory pairs found
        dsn_by_attr_value = {'total': {'TSN': 0, 'DSN': 0}}  # Track discrimination by attribute value

        # Define a function to check if we should terminate early due to time limit
        def should_terminate():
            elapsed_time = time.time() - start_time
            if elapsed_time > max_runtime_seconds:
                logger.info(f"Early termination triggered after {elapsed_time:.2f} seconds (limit: {max_runtime_seconds} seconds)")
                return True
            return False
            
        # Phase 1: Global Search
        d_global = self.global_search(num_samples=num_samples)
        
        # Check if we should terminate early
        if should_terminate():
            logger.info("Early termination during global search phase")
            # Create minimal results and return
            res_df, metrics = make_final_metrics_and_dataframe(
                discrimination_data=self.discrimination_data,
                tot_inputs=tot_inputs,
                all_discriminations=all_discriminations,
                dsn_by_attr_value=dsn_by_attr_value,
                start_time=start_time,
                logger=logger
            )
            return res_df, metrics

        # Update metrics from global search
        for item in d_global:
            tot_inputs.add(tuple(item))

        # Phase 2: Local Search
        discriminatory_pairs = self.local_search(D_global=d_global, limit=local_search_limit)
        
        # Check if we should terminate early
        if should_terminate():
            logger.info("Early termination during local search phase")
            # Process any results we have so far and return
            # Continue with processing the pairs we have collected

        # Process discriminatory pairs to update metrics
        for pair in discriminatory_pairs:
            if isinstance(pair, tuple) and len(pair) == 2:
                # This is a pair (current_item, perturbed_item)
                current_item, perturbed_item = pair
                current_item_array = np.array(current_item)
                perturbed_item_array = np.array(perturbed_item)

                # Get model predictions for both items
                current_pred = self.model.predict(current_item_array.reshape(1, -1))[0]
                perturbed_pred = self.model.predict(perturbed_item_array.reshape(1, -1))[0]

                # Add to all_discriminations in the format expected by make_final_metrics_and_dataframe
                all_discriminations.append((current_item_array, current_pred, perturbed_item_array, perturbed_pred))

                # Update tot_inputs
                tot_inputs.add(current_item)
                tot_inputs.add(perturbed_item)

                # Update dsn_by_attr_value
                dsn_by_attr_value['total']['TSN'] += 2
                dsn_by_attr_value['total']['DSN'] += 1

                # Track discrimination by protected attribute values
                for idx in self.protected_attribute_indices:
                    attr_val = int(current_item_array[idx])
                    attr_key = f"attr_{idx}_{attr_val}"

                    if attr_key not in dsn_by_attr_value:
                        dsn_by_attr_value[attr_key] = {'TSN': 0, 'DSN': 0}

                    dsn_by_attr_value[attr_key]['TSN'] += 1
                    dsn_by_attr_value[attr_key]['DSN'] += 1

        # Generate final metrics and dataframe
        res_df, metrics = make_final_metrics_and_dataframe(
            discrimination_data=self.discrimination_data,
            tot_inputs=tot_inputs,
            all_discriminations=all_discriminations,
            dsn_by_attr_value=dsn_by_attr_value,
            start_time=start_time,
            logger=logger
        )

        # Summary
        logger.info("\n=== KOSEI Pipeline Complete ===")
        logger.info(f"Total unique discriminatory pairs found: {len(all_discriminations)}")

        return res_df, metrics


def run_kosei(model: Optional[Union[Callable[[np.ndarray], Any], str]] = None,
              data: Optional[DiscriminationData] = None, protected_attribute_indices: Optional[List[int]] = None,
              attribute_domains: Optional[Dict[int, tuple]] = None, total_attributes: Optional[int] = None,
              gamma: float = 0.0, model_type: str = 'rf', model_params: Optional[Dict] = None, num_samples: int = 1000,
              local_search_limit: int = 500, random_seed: int = 42, max_runtime_seconds: int = 3600):
    results_df, metrics = KOSEI(
        model=model,
        discrimination_data=data,
        protected_attribute_indices=protected_attribute_indices,
        attribute_domains=attribute_domains,
        total_attributes=total_attributes,
        gamma=gamma,
        model_type=model_type,
        model_params=model_params
    ).run(
        num_samples=num_samples,
        local_search_limit=local_search_limit,
        random_seed=random_seed,
        max_runtime_seconds=max_runtime_seconds
    )
    return results_df, metrics


# Example usage
if __name__ == "__main__":
    # Create a simple synthetic dataset with discrimination
    discrimination_data, data_schema = get_real_data('adult', use_cache=True)

    results_df, metrics = run_kosei(data=discrimination_data, num_samples=100, local_search_limit=50)

    print(f"\nTesting Metrics: {metrics}")
