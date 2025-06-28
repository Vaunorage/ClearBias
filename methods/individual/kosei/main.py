import numpy as np
import collections
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from tqdm import tqdm
from data_generator.main import get_real_data
from data_generator.main import DiscriminationData
from methods.utils import train_sklearn_model


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
        """
        self.model = model
        self.gamma = gamma

        if discrimination_data is not None:
            self.discrimination_data = discrimination_data
            self.attr_names = discrimination_data.attr_columns
            self.protected_attribute_indices = discrimination_data.sensitive_indices
            self.non_protected_attribute_indices = [
                i for i in range(len(discrimination_data.attr_columns))
                if i not in discrimination_data.sensitive_indices
            ]
            self.attribute_domains = {}
            for i, (min_val, max_val) in enumerate(discrimination_data.input_bounds):
                self.attribute_domains[i] = (int(min_val), int(max_val))
            self.total_attributes = len(discrimination_data.attr_columns)

            if model is None or isinstance(model, str):
                print(f"\n--- Training a {'custom' if isinstance(model, str) else 'default'} {model_type} model ---")
                model_type_to_use = model if isinstance(model, str) else model_type
                trained_model, _, _, _, _, _ = train_sklearn_model(
                    data=self.discrimination_data.training_dataframe,
                    model_type=model_type_to_use,
                    model_params=model_params,
                    target_col=self.discrimination_data.outcome_column,
                    sensitive_attrs=list(self.discrimination_data.sensitive_indices_dict.keys()),
                )
                self.model = trained_model
                print(f"Model trained successfully.")
            else:
                self.model = model
        else:
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

            if model is None:
                raise ValueError("If discrimination_data is not provided, you must provide a model.")
            self.model = model

        print(f"KOSEI Initialized.")
        print(f"  - Protected Attributes (indices): {self.protected_attribute_indices}")
        print(f"  - Non-Protected Attributes (indices): {self.non_protected_attribute_indices}")

    # --- CHANGED: Now returns the counterpart array or None ---
    def _eval_disc(self, data_item: np.ndarray) -> Optional[np.ndarray]:
        """
        Evaluates if a single data item is discriminatory.

        Args:
            data_item: A numpy array representing a single data point.

        Returns:
            The counterpart numpy array that proves discrimination, or None if not discriminatory.
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

                counterpart_prediction = self.model.predict(counterpart_item.reshape(1, -1))[0]

                if abs(original_prediction - counterpart_prediction) > self.gamma:
                    # Return the counterpart that triggered the discrimination
                    return counterpart_item

        return None  # No discrimination found

    # --- CHANGED: Now populates a dictionary of pairs and returns initial seeds ---
    def global_search(self, num_samples: int, found_pairs: Dict[Tuple, np.ndarray]) -> List[np.ndarray]:
        """
        Performs a simple random global search to find initial seeds.

        Args:
            num_samples: The number of random samples to generate and test.
            found_pairs: A dictionary to populate with (original_item_tuple: counterpart_item) pairs.

        Returns:
            A list of initial discriminatory seeds (D_global).
        """
        print(f"\n--- Starting Global Search ({num_samples} samples) ---")
        d_global = []

        # Use tqdm to track progress
        progress_bar = tqdm(total=num_samples, desc="Global Search", unit="sample")

        for _ in range(num_samples):
            sample = []
            for i in range(self.total_attributes):
                min_val, max_val = self.attribute_domains[i]
                # Generate a random integer in the attribute's domain
                sample.append(np.random.randint(min_val, max_val + 1))

            sample = np.array(sample)
            sample_tuple = tuple(sample)

            # Evaluate the sample
            counterpart = self._eval_disc(sample)
            if counterpart is not None:
                d_global.append(sample)
                found_pairs[sample_tuple] = counterpart
                # Update progress bar with additional information
                progress_bar.set_postfix(found=len(d_global), refresh=True)

            # Update progress bar
            progress_bar.update(1)

        progress_bar.close()
        print(f"Global Search found {len(d_global)} initial discriminatory seeds.")
        return d_global

    # --- CHANGED: Now populates a dictionary of pairs ---
    def local_search(self, D_global: List[np.ndarray], limit: int, found_pairs: Dict[Tuple, np.ndarray]) -> None:
        """
        Performs the KOSEI local search (implements Algorithm 3).

        Args:
            D_global: List of initial discriminatory seeds from global search.
            limit: Maximum number of iterations for the local search.
            found_pairs: A dictionary to populate with (original_item_tuple: counterpart_item) pairs.
        """
        if not D_global:
            print("\n--- Starting Local Search ---")
            print("No seeds provided by global search. Terminating.")
            return

        print(f"\n--- Starting Local Search (max {limit} iterations) ---")

        # Use a deque for efficient appends and poplefts (FIFO queue)
        seed_queue = collections.deque(tuple(item) for item in D_global)
        evaluated_items_set = set(seed_queue)  # Keep track of evaluated items
        iterations_done = 0

        # Initialize progress bar for local search
        progress_bar = tqdm(total=limit, desc="Local Search", unit="iter")

        # Initialize counters for progress tracking
        total_discriminatory = len(found_pairs)
        total_evaluated = len(evaluated_items_set)

        while seed_queue and iterations_done < limit:
            iterations_done += 1

            # Get the next seed from the queue
            current_seed = seed_queue.popleft()
            current_seed_array = np.array(current_seed)

            # Perturb each attribute one by one
            for attr_idx in range(self.total_attributes):
                min_val, max_val = self.attribute_domains[attr_idx]
                current_val = current_seed[attr_idx]

                # Try all possible values for this attribute
                for new_val in range(min_val, max_val + 1):
                    if new_val == current_val:
                        continue  # Skip the current value

                    # Create a perturbed version of the current seed
                    perturbed_item = list(current_seed)
                    perturbed_item[attr_idx] = new_val
                    perturbed_item = np.array(perturbed_item)
                    perturbed_item_tuple = tuple(perturbed_item)

                    # Skip if we've already evaluated this item
                    if perturbed_item_tuple in evaluated_items_set:
                        continue

                    # Add to evaluated set
                    evaluated_items_set.add(perturbed_item_tuple)
                    total_evaluated += 1

                    # Evaluate the perturbed item
                    counterpart = self._eval_disc(perturbed_item)
                    if counterpart is not None:
                        # We only add to the queue if it's a new discriminatory item
                        if perturbed_item_tuple not in found_pairs:
                            seed_queue.append(perturbed_item_tuple)
                        found_pairs[perturbed_item_tuple] = counterpart
                        total_discriminatory += 1

            # Update progress bar with additional information
            progress_bar.set_postfix({
                'found': total_discriminatory,
                'queue': len(seed_queue),
                'evaluated': total_evaluated
            }, refresh=True)
            progress_bar.update(1)

        if not seed_queue and iterations_done < limit:
            print(f"Search terminated early at iteration {iterations_done} because the seed queue is empty.")

        progress_bar.close()
        print(f"Local search finished.")
        print(f"  - Total unique items evaluated: {len(evaluated_items_set)}")

    # --- CHANGED: Orchestrates the new flow and returns a list of pairs ---
    def run(self, num_samples: int = 1000, local_search_limit: int = 500) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Run the full KOSEI pipeline: global search followed by local search.

        Args:
            num_samples: Number of random samples to generate in global search.
            local_search_limit: Maximum number of iterations for local search.

        Returns:
            A list of tuples, where each tuple contains two arrays:
            the discriminatory item and its corresponding counterpart.
        """
        print("\n=== Running Full KOSEI Pipeline ===")
        # This dictionary will store original_item_tuple -> counterpart_item
        found_discriminatory_pairs = {}

        # Phase 1: Global Search
        d_global = self.global_search(
            num_samples=num_samples,
            found_pairs=found_discriminatory_pairs
        )

        # Phase 2: Local Search
        self.local_search(
            D_global=d_global,
            limit=local_search_limit,
            found_pairs=found_discriminatory_pairs
        )

        # Summary
        print("\n=== KOSEI Pipeline Complete ===")
        print(f"Total unique discriminatory pairs found: {len(found_discriminatory_pairs)}")

        # Convert the dictionary to the desired list of tuples format
        return [(np.array(original), counterpart) for original, counterpart in found_discriminatory_pairs.items()]


if __name__ == '__main__':
    discrimination_data, data_schema = get_real_data('adult', use_cache=True)

    kosei_tester_with_data = KOSEI(discrimination_data=discrimination_data, model_type='rf')

    all_discrimination_pairs = kosei_tester_with_data.run(
        num_samples=1000, local_search_limit=20
    )
