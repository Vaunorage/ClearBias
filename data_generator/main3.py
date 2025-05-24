import itertools
import math
import copy
import contextlib
import io
import sys
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sdv.sampling import Condition
from tqdm import tqdm

from data_generator.main import estimate_min_attributes_and_classes, generate_data_schema, bin_array_values, \
    DiscriminationData, calculate_actual_metrics


@dataclass
class SubgroupDefinition:
    """Defines a single subgroup with specific attribute values."""
    subgroup_id: str
    attribute_values: Dict[str, Any]  # attribute_name -> value
    size: int
    bias: float = 0.0
    uncertainty_params: Dict[str, float] = None
    frequency: float = 1.0  # For compatibility with existing code

    def __post_init__(self):
        if self.uncertainty_params is None:
            self.uncertainty_params = {
                'epistemic': np.random.uniform(0.05, 0.3),
                'aleatoric': np.random.uniform(0.05, 0.3)
            }


@dataclass
class GroupDefinition:
    """Defines a group as a combination of two subgroups."""
    group_id: str
    subgroup1_id: str
    subgroup2_id: str
    similarity: float
    expected_outcome_difference: float
    granularity: int = 0
    intersectionality: int = 0


@contextlib.contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


class OptimalDataGenerator:
    """
    Improved data generation that first calculates optimal subgroups,
    then generates all possible group combinations.
    """

    def __init__(self, data_schema: 'DataSchema'):
        self.data_schema = data_schema
        self.subgroups: Dict[str, SubgroupDefinition] = {}
        self.groups: Dict[str, GroupDefinition] = {}
        self.generated_subgroup_data: Dict[str, pd.DataFrame] = {}

    def calculate_optimal_subgroups(self,
                                    target_nb_groups: int,
                                    min_subgroups: int = None,
                                    max_subgroups: int = None,
                                    min_group_size: int = 10,
                                    max_group_size: int = 100) -> List[SubgroupDefinition]:
        """
        Calculate the optimal number and configuration of subgroups needed
        to generate the target number of groups.

        Args:
            target_nb_groups: Desired number of groups
            min_subgroups: Minimum number of subgroups to consider
            max_subgroups: Maximum number of subgroups to consider
            min_group_size: Minimum size for each group
            max_group_size: Maximum size for each group

        Returns:
            List of optimal subgroup definitions
        """
        # Calculate minimum number of subgroups needed
        # For n subgroups, we can create C(n,2) = n*(n-1)/2 groups
        min_needed = math.ceil((1 + math.sqrt(1 + 8 * target_nb_groups)) / 2)

        if min_subgroups is None:
            min_subgroups = min_needed
        if max_subgroups is None:
            max_subgroups = min_needed + 10  # Some buffer for optimization

        print(f"Target groups: {target_nb_groups}")
        print(f"Minimum subgroups needed: {min_needed}")
        print(f"Searching in range: {min_subgroups} to {max_subgroups}")

        # Find optimal number of subgroups
        best_config = None
        best_score = float('inf')

        for n_subgroups in range(min_subgroups, max_subgroups + 1):
            max_possible_groups = n_subgroups * (n_subgroups - 1) // 2

            if max_possible_groups >= target_nb_groups:
                # Score based on efficiency (how close to target without waste)
                efficiency = target_nb_groups / max_possible_groups
                waste_penalty = max_possible_groups - target_nb_groups
                score = waste_penalty * (1 - efficiency)

                if score < best_score:
                    best_score = score
                    best_config = {
                        'n_subgroups': n_subgroups,
                        'max_groups': max_possible_groups,
                        'efficiency': efficiency
                    }

        if best_config is None:
            raise ValueError(f"Cannot generate {target_nb_groups} groups with max {max_subgroups} subgroups")

        print(f"Optimal configuration: {best_config['n_subgroups']} subgroups "
              f"-> {best_config['max_groups']} possible groups "
              f"(efficiency: {best_config['efficiency']:.2%})")

        return self._generate_diverse_subgroups(best_config['n_subgroups'],
                                                min_group_size, max_group_size)

    def _generate_diverse_subgroups(self, n_subgroups: int,
                                    min_group_size: int, max_group_size: int) -> List[SubgroupDefinition]:
        """Generate diverse subgroups that maximize coverage of attribute space."""
        subgroups = []

        # Get all possible attribute combinations
        protected_attrs = [attr for attr, is_protected in zip(self.data_schema.attr_names,
                                                              self.data_schema.protected_attr) if is_protected]
        non_protected_attrs = [attr for attr, is_protected in zip(self.data_schema.attr_names,
                                                                  self.data_schema.protected_attr) if not is_protected]

        print(f"Protected attributes: {protected_attrs}")
        print(f"Non-protected attributes: {non_protected_attrs}")

        # Strategy: Create subgroups with different combinations of attribute constraints
        # to ensure good coverage of the attribute space

        used_combinations = set()
        attempt = 0
        max_attempts = n_subgroups * 20

        while len(subgroups) < n_subgroups and attempt < max_attempts:
            attempt += 1

            # Randomly select attributes to constrain for this subgroup
            n_protected = np.random.randint(1, min(len(protected_attrs) + 1, 4)) if protected_attrs else 0
            n_non_protected = np.random.randint(0, min(len(non_protected_attrs) + 1, 4)) if non_protected_attrs else 0

            # Ensure at least one attribute is selected
            if n_protected == 0 and n_non_protected == 0:
                if protected_attrs:
                    n_protected = 1
                elif non_protected_attrs:
                    n_non_protected = 1
                else:
                    continue

            selected_protected = []
            selected_non_protected = []

            if n_protected > 0 and protected_attrs:
                selected_protected = np.random.choice(protected_attrs,
                                                      min(n_protected, len(protected_attrs)),
                                                      replace=False)

            if n_non_protected > 0 and non_protected_attrs:
                selected_non_protected = np.random.choice(non_protected_attrs,
                                                          min(n_non_protected, len(non_protected_attrs)),
                                                          replace=False)

            # Generate attribute values for selected attributes
            attribute_values = {}

            for attr in selected_protected:
                attr_idx = self.data_schema.attr_names.index(attr)
                valid_values = [v for v in self.data_schema.attr_categories[attr_idx] if v != -1]
                if valid_values:
                    attribute_values[attr] = np.random.choice(valid_values)

            for attr in selected_non_protected:
                attr_idx = self.data_schema.attr_names.index(attr)
                valid_values = [v for v in self.data_schema.attr_categories[attr_idx] if v != -1]
                if valid_values:
                    attribute_values[attr] = np.random.choice(valid_values)

            # Skip if no valid attribute values were found
            if not attribute_values:
                continue

            # Create a signature for this combination
            signature = tuple(sorted(attribute_values.items()))

            if signature not in used_combinations:
                used_combinations.add(signature)

                # Calculate subgroup size (will be adjusted during group formation)
                base_size = np.random.randint(min_group_size // 2, max_group_size // 2)

                subgroup = SubgroupDefinition(
                    subgroup_id=f"subgroup_{len(subgroups):03d}",
                    attribute_values=attribute_values,
                    size=base_size,
                    bias=np.random.uniform(-0.5, 0.5),
                    uncertainty_params={
                        'epistemic': np.random.uniform(0.05, 0.3),
                        'aleatoric': np.random.uniform(0.05, 0.3)
                    }
                )

                subgroups.append(subgroup)
                self.subgroups[subgroup.subgroup_id] = subgroup

        if len(subgroups) < n_subgroups:
            print(f"Warning: Only generated {len(subgroups)} subgroups out of {n_subgroups} requested")

        return subgroups

    def generate_all_possible_groups(self,
                                     subgroups: List[SubgroupDefinition],
                                     target_nb_groups: int = None) -> List[GroupDefinition]:
        """Generate all possible group combinations from subgroups."""
        all_combinations = list(itertools.combinations(subgroups, 2))

        if target_nb_groups and target_nb_groups < len(all_combinations):
            # Select the most diverse/interesting groups
            selected_combinations = self._select_diverse_groups(all_combinations, target_nb_groups)
        else:
            selected_combinations = all_combinations

        groups = []
        for i, (sg1, sg2) in enumerate(selected_combinations):
            # Calculate similarity between subgroups
            similarity = self._calculate_subgroup_similarity(sg1, sg2)

            # Estimate expected outcome difference
            outcome_diff = abs(sg1.bias - sg2.bias)

            # Calculate granularity and intersectionality
            granularity, intersectionality = self._calculate_granularity_intersectionality(sg1, sg2)

            group = GroupDefinition(
                group_id=f"group_{i:03d}",
                subgroup1_id=sg1.subgroup_id,
                subgroup2_id=sg2.subgroup_id,
                similarity=similarity,
                expected_outcome_difference=outcome_diff,
                granularity=granularity,
                intersectionality=intersectionality
            )

            groups.append(group)
            self.groups[group.group_id] = group

        return groups

    def _calculate_granularity_intersectionality(self, sg1: SubgroupDefinition,
                                                 sg2: SubgroupDefinition) -> Tuple[int, int]:
        """Calculate granularity and intersectionality for a group."""
        all_attrs = set(sg1.attribute_values.keys()) | set(sg2.attribute_values.keys())

        granularity = 0
        intersectionality = 0

        for attr in all_attrs:
            attr_idx = self.data_schema.attr_names.index(attr)
            is_protected = self.data_schema.protected_attr[attr_idx]

            if is_protected:
                intersectionality += 1
            else:
                granularity += 1

        return granularity, intersectionality

    def _calculate_subgroup_similarity(self, sg1: SubgroupDefinition, sg2: SubgroupDefinition) -> float:
        """Calculate similarity between two subgroups based on their attribute values."""
        all_attrs = set(sg1.attribute_values.keys()) | set(sg2.attribute_values.keys())

        if not all_attrs:
            return 1.0  # Both have no constraints

        matching_attrs = 0
        for attr in all_attrs:
            val1 = sg1.attribute_values.get(attr, None)
            val2 = sg2.attribute_values.get(attr, None)

            if val1 == val2:
                matching_attrs += 1
            elif val1 is None or val2 is None:
                # One has constraint, other doesn't - partial similarity
                matching_attrs += 0.5

        return matching_attrs / len(all_attrs)

    def _select_diverse_groups(self, all_combinations: List[Tuple], target_count: int) -> List[Tuple]:
        """Select the most diverse subset of group combinations."""
        if target_count >= len(all_combinations):
            return all_combinations

        # Strategy: Select groups with diverse similarity scores and outcome differences
        scored_combinations = []

        for sg1, sg2 in all_combinations:
            similarity = self._calculate_subgroup_similarity(sg1, sg2)
            outcome_diff = abs(sg1.bias - sg2.bias)

            # Score favors diverse similarities and meaningful outcome differences
            diversity_score = abs(similarity - 0.5)  # Prefer not too similar, not too different
            outcome_score = min(outcome_diff, 1.0)  # Cap at 1.0

            total_score = diversity_score + outcome_score
            scored_combinations.append((total_score, (sg1, sg2)))

        # Sort by score and take top combinations
        scored_combinations.sort(key=lambda x: x[0], reverse=True)
        return [combo for _, combo in scored_combinations[:target_count]]

    def generate_subgroup_data(self, subgroup: SubgroupDefinition, W: np.ndarray) -> pd.DataFrame:
        """Generate data for a single subgroup using SDV synthesizer and uncertainty calculation."""
        # Check if we already generated this subgroup
        if subgroup.subgroup_id in self.generated_subgroup_data:
            return self.generated_subgroup_data[subgroup.subgroup_id].copy()

        # Use SDV synthesizer to generate data with conditions
        with suppress_stdout():
            if subgroup.attribute_values:
                conditions = [Condition(
                    num_rows=subgroup.size,
                    column_values=subgroup.attribute_values
                )]
                try:
                    subgroup_data = self.data_schema.synthesizer.sample_from_conditions(conditions)
                except Exception as e:
                    print(f"Error generating data for subgroup {subgroup.subgroup_id}: {e}")
                    # Fallback: generate unconstrained data and manually set values
                    subgroup_data = self.data_schema.synthesizer.sample(subgroup.size)
                    for attr, value in subgroup.attribute_values.items():
                        if attr in subgroup_data.columns:
                            subgroup_data[attr] = value
            else:
                subgroup_data = self.data_schema.synthesizer.sample(subgroup.size)

        # Fill any NaN values
        subgroup_data = self._fill_nan_values(subgroup_data)

        # Generate outcomes with uncertainty
        subgroup_data = self._generate_outcomes_with_uncertainty(
            subgroup_data, subgroup, W
        )

        # Add subgroup metadata
        subgroup_data['subgroup_id'] = subgroup.subgroup_id
        subgroup_data['subgroup_bias'] = subgroup.bias
        subgroup_data['epistemic_param'] = subgroup.uncertainty_params['epistemic']
        subgroup_data['aleatoric_param'] = subgroup.uncertainty_params['aleatoric']

        # Store for reuse
        self.generated_subgroup_data[subgroup.subgroup_id] = subgroup_data.copy()

        return subgroup_data

    def _fill_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values in a dataframe using smart defaults."""
        filled_df = df.copy()

        for col in filled_df.columns:
            if not filled_df[col].isna().any():
                continue

            if pd.api.types.is_numeric_dtype(filled_df[col]):
                median_val = filled_df[col].median()
                filled_df[col].fillna(0 if pd.isna(median_val) else median_val, inplace=True)
            else:
                if not filled_df[col].mode().empty:
                    mode_val = filled_df[col].mode().iloc[0]
                    filled_df[col].fillna(mode_val, inplace=True)
                else:
                    filled_df[col].fillna("unknown", inplace=True)

        return filled_df

    def _generate_outcomes_with_uncertainty(self, df: pd.DataFrame,
                                            subgroup: SubgroupDefinition,
                                            W: np.ndarray) -> pd.DataFrame:
        """Generate outcomes with epistemic and aleatoric uncertainty."""
        # Extract features and standardize
        feature_columns = self.data_schema.attr_names
        X = df[feature_columns].values
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

        # Use the last layer of W as weights
        weights = W[-1] if W.ndim > 1 else W

        # Get uncertainty parameters
        epistemic_uncertainty = subgroup.uncertainty_params['epistemic']
        aleatoric_uncertainty = subgroup.uncertainty_params['aleatoric']

        # Generate ensemble predictions
        n_estimators = 50
        ensemble_preds = []

        for i in range(n_estimators):
            # Add epistemic uncertainty through weight perturbation
            weight_noise = np.random.normal(0, epistemic_uncertainty, size=weights.shape)
            perturbed_weights = weights * (1 + weight_noise)

            # Compute base prediction
            base_pred = np.dot(X, perturbed_weights) + subgroup.bias

            # Add aleatoric uncertainty
            noisy_pred = base_pred + np.random.normal(0, aleatoric_uncertainty, size=len(df))

            # Convert to probability
            probs = 1 / (1 + np.exp(-noisy_pred))
            ensemble_preds.append(probs)

        # Calculate final predictions and uncertainties
        ensemble_preds = np.array(ensemble_preds).T
        final_predictions = np.mean(ensemble_preds, axis=1)
        calculated_epistemic = np.var(ensemble_preds, axis=1)
        calculated_aleatoric = np.mean(ensemble_preds * (1 - ensemble_preds), axis=1)

        # Add to dataframe
        df['outcome'] = final_predictions
        df['epis_uncertainty'] = calculated_epistemic
        df['alea_uncertainty'] = calculated_aleatoric
        df['calculated_epistemic'] = calculated_epistemic
        df['calculated_aleatoric'] = calculated_aleatoric

        return df

    def generate_dataset(self, groups: List[GroupDefinition], W: np.ndarray,
                         min_group_size: int = 10, max_group_size: int = 100,
                         min_diff_subgroup_size: float = 0.0,
                         max_diff_subgroup_size: float = 0.5) -> pd.DataFrame:
        """Generate the complete dataset with all groups."""
        all_dataframes = []

        print("Generating dataset from groups...")

        for group in tqdm(groups, desc="Processing groups"):
            sg1 = self.subgroups[group.subgroup1_id]
            sg2 = self.subgroups[group.subgroup2_id]

            # Calculate group size and subgroup size difference
            total_group_size = np.random.randint(min_group_size, max_group_size + 1)
            diff_percentage = np.random.uniform(min_diff_subgroup_size, max_diff_subgroup_size)

            # Calculate individual subgroup sizes
            diff_size = int(total_group_size * diff_percentage)
            sg1_size = max(1, (total_group_size + diff_size) // 2)
            sg2_size = max(1, total_group_size - sg1_size)

            # Update subgroup sizes for this group
            sg1_copy = copy.deepcopy(sg1)
            sg2_copy = copy.deepcopy(sg2)
            sg1_copy.size = sg1_size
            sg2_copy.size = sg2_size

            # Generate data for each subgroup
            sg1_data = self.generate_subgroup_data(sg1_copy, W)
            sg2_data = self.generate_subgroup_data(sg2_copy, W)

            # Take only the required number of rows
            sg1_data = sg1_data.head(sg1_size).copy()
            sg2_data = sg2_data.head(sg2_size).copy()

            # Create subgroup keys based on ordered attribute values (like original code)
            sg1_vals = []
            sg2_vals = []

            # Get ordered attribute values for all attributes (using -1 for unconstrained)
            for attr_name in self.data_schema.attr_names:
                sg1_vals.append(sg1.attribute_values.get(attr_name, -1))
                sg2_vals.append(sg2.attribute_values.get(attr_name, -1))

            # Create keys using the same format as original code
            subgroup1_key = '|'.join(list(map(lambda x: '*' if x == -1 else str(x), sg1_vals)))
            subgroup2_key = '|'.join(list(map(lambda x: '*' if x == -1 else str(x), sg2_vals)))
            group_key = subgroup1_key + '-' + subgroup2_key

            # Assign the keys to the dataframes
            sg1_data['subgroup_key'] = subgroup1_key
            sg2_data['subgroup_key'] = subgroup2_key

            sg1_data['group_key'] = group_key
            sg2_data['group_key'] = group_key

            # Add individual keys (same as original)
            sg1_data['indv_key'] = sg1_data[self.data_schema.attr_names].apply(
                lambda x: '|'.join(list(x.astype(int).astype(str))), axis=1)
            sg2_data['indv_key'] = sg2_data[self.data_schema.attr_names].apply(
                lambda x: '|'.join(list(x.astype(int).astype(str))), axis=1)

            # Combine subgroups into group
            group_data = pd.concat([sg1_data, sg2_data], ignore_index=True)

            # Add group-level metadata
            group_data['similarity_param'] = group.similarity
            group_data['expected_outcome_diff'] = group.expected_outcome_difference
            group_data['granularity_param'] = group.granularity
            group_data['intersectionality_param'] = group.intersectionality
            group_data['group_size'] = len(group_data)
            group_data['diff_subgroup_size'] = diff_percentage
            group_data['frequency_param'] = 1.0  # All groups have equal frequency in this approach

            all_dataframes.append(group_data)

        # Combine all groups
        complete_dataset = pd.concat(all_dataframes, ignore_index=True)

        # Sort by group and individual keys
        complete_dataset = complete_dataset.sort_values(['group_key', 'indv_key'])

        return complete_dataset


# Main function to replace the original generate_data function
def generate_optimal_discrimination_data(
        data_schema: Optional['DataSchema'] = None,
        nb_groups: int = 100,
        nb_attributes: int = 20,
        min_number_of_classes: int = None,
        max_number_of_classes: int = None,
        prop_protected_attr: float = 0.2,
        min_group_size: int = 10,
        max_group_size: int = 100,
        min_diff_subgroup_size: float = 0.0,
        max_diff_subgroup_size: float = 0.5,
        W: Optional[np.ndarray] = None,
        categorical_outcome: bool = True,
        nb_categories_outcome: int = 6,
        **kwargs
) -> 'DiscriminationData':
    """
    Generate discrimination data using the improved optimal architecture.

    Args:
        data_schema: Optional DataSchema with fitted synthesizer. If None, will generate one.
        nb_groups: Target number of groups to generate
        nb_attributes: Number of attributes (used only if data_schema is None)
        min_number_of_classes: Min classes per attribute (used only if data_schema is None)
        max_number_of_classes: Max classes per attribute (used only if data_schema is None)
        prop_protected_attr: Proportion of protected attributes (used only if data_schema is None)
        min_group_size: Minimum size for each group
        max_group_size: Maximum size for each group
        min_diff_subgroup_size: Minimum difference between subgroup sizes
        max_diff_subgroup_size: Maximum difference between subgroup sizes
        W: Weight matrix for outcome generation
        categorical_outcome: Whether to use categorical outcomes
        nb_categories_outcome: Number of categories for outcome if categorical
        **kwargs: Additional arguments for compatibility

    Returns:
        DiscriminationData object with generated data
    """

    # Generate DataSchema if not provided
    if data_schema is None:
        print("No DataSchema provided, generating a new one...")

        # Estimate min/max classes if not provided
        if min_number_of_classes is None or max_number_of_classes is None:
            est_min, est_max = estimate_min_attributes_and_classes(nb_groups)
            min_number_of_classes = min_number_of_classes or est_min
            max_number_of_classes = max_number_of_classes or int(est_max * 1.5)

        # Generate the schema with fitted synthesizer
        data_schema = generate_data_schema(
            min_number_of_classes=min_number_of_classes,
            max_number_of_classes=max_number_of_classes,
            nb_attributes=nb_attributes,
            prop_protected_attr=prop_protected_attr,
            fit_synthesizer=True,
            n_samples=10000
        )
        print(f"Generated DataSchema with {len(data_schema.attr_names)} attributes")
        print(f"Protected attributes: {sum(data_schema.protected_attr)}")
        print(f"Non-protected attributes: {len(data_schema.attr_names) - sum(data_schema.protected_attr)}")

    # Validate that the schema has a fitted synthesizer
    if not hasattr(data_schema, 'synthesizer') or data_schema.synthesizer is None:
        raise ValueError(
            "DataSchema must have a fitted synthesizer. Set fit_synthesizer=True when creating the schema.")

    # Generate weights if not provided
    if W is None:
        hiddenlayers_depth = 3
        nb_attributes_actual = len(data_schema.attr_names)
        W = np.random.uniform(low=0.0, high=1.0, size=(hiddenlayers_depth, nb_attributes_actual))

    # Initialize the optimal generator
    generator = OptimalDataGenerator(data_schema)

    # Step 1: Calculate optimal subgroups
    print("Step 1: Calculating optimal subgroups...")
    subgroups = generator.calculate_optimal_subgroups(
        nb_groups, min_group_size=min_group_size, max_group_size=max_group_size
    )

    # Step 2: Generate all possible groups
    print("Step 2: Generating group combinations...")
    groups = generator.generate_all_possible_groups(subgroups, nb_groups)

    # Step 3: Generate the complete dataset
    print("Step 3: Generating complete dataset...")
    dataset = generator.generate_dataset(
        groups, W, min_group_size, max_group_size,
        min_diff_subgroup_size, max_diff_subgroup_size
    )

    # Step 4: Process outcome column
    outcome_column = 'outcome'
    if categorical_outcome:
        dataset[outcome_column] = bin_array_values(dataset[outcome_column], nb_categories_outcome)
    else:
        dataset[outcome_column] = ((dataset[outcome_column] - dataset[outcome_column].min()) /
                                   (dataset[outcome_column].max() - dataset[outcome_column].min()))

    # Step 5: Create DiscriminationData object
    protected_attr = {k: v for k, v in zip(data_schema.attr_names, data_schema.protected_attr)}
    attr_possible_values = {attr_name: values for attr_name, values in
                            zip(data_schema.attr_names, data_schema.attr_categories)}

    # Create the DiscriminationData object
    data = DiscriminationData(
        dataframe=dataset,
        categorical_columns=list(data_schema.attr_names) + [outcome_column],
        attributes=protected_attr,
        collisions=0,  # No collisions with this approach
        nb_groups=len(groups),
        max_group_size=max_group_size,
        hiddenlayers_depth=W.shape[0],
        outcome_column=outcome_column,
        attr_possible_values=attr_possible_values,
        schema=data_schema,
        generation_arguments={
            'nb_groups': nb_groups,
            'nb_subgroups': len(subgroups),
            'nb_attributes': len(data_schema.attr_names),
            'min_number_of_classes': min_number_of_classes,
            'max_number_of_classes': max_number_of_classes,
            'prop_protected_attr': prop_protected_attr,
            'min_group_size': min_group_size,
            'max_group_size': max_group_size,
            'categorical_outcome': categorical_outcome,
            'nb_categories_outcome': nb_categories_outcome,
            'approach': 'optimal_subgroup_generation',
            'schema_generated': data_schema is None  # Track if schema was auto-generated
        }
    )

    # Step 6: Calculate the missing actual metrics (this is the key addition!)
    print("Step 4: Calculating actual metrics...")
    data = calculate_actual_metrics(data)

    print(f"\nGeneration Summary:")
    print(f"- Schema: {'Auto-generated' if 'schema_generated' not in locals() else 'User-provided'}")
    print(f"- Attributes: {len(data_schema.attr_names)} ({sum(data_schema.protected_attr)} protected)")
    print(f"- Generated {len(subgroups)} unique subgroups")
    print(f"- Created {len(groups)} groups from subgroup combinations")
    print(f"- Final dataset contains {len(dataset)} individuals")
    print(f"- Efficiency: {len(groups) / len(subgroups):.1f} groups per subgroup")

    return data


# Convenience function that matches your original generate_data signature
def generate_data_optimal(
        gen_order: List[int] = None,
        correlation_matrix=None,
        W: np.ndarray = None,
        nb_groups=100,
        nb_attributes=20,
        min_number_of_classes=None,
        max_number_of_classes=None,
        prop_protected_attr=0.2,
        min_group_size=10,
        max_group_size=100,
        min_similarity=0.0,
        max_similarity=1.0,
        min_alea_uncertainty=0.0,
        max_alea_uncertainty=1.0,
        min_epis_uncertainty=0.0,
        max_epis_uncertainty=1.0,
        min_frequency=0.0,
        max_frequency=1.0,
        min_diff_subgroup_size=0.0,
        max_diff_subgroup_size=0.5,
        min_granularity=1,
        max_granularity=None,
        min_intersectionality=1,
        max_intersectionality=None,
        categorical_outcome: bool = True,
        nb_categories_outcome: int = 6,
        use_cache: bool = True,
        corr_matrix_randomness=0.5,
        categorical_distribution: Dict[str, List[float]] = None,
        categorical_influence: float = 0.5,
        data_schema: 'DataSchema' = None,
        predefined_groups: List['GroupDefinition'] = None
) -> 'DiscriminationData':
    """
    Drop-in replacement for your original generate_data function using optimal approach.

    This function provides the same interface as your original generate_data but uses
    the new optimal subgroup generation approach internally.

    Args:
        Similar to your original generate_data function arguments

    Returns:
        DiscriminationData object generated using optimal approach
    """
    # Note: Some parameters like min_similarity, max_similarity etc. are not directly
    # used in the optimal approach but are kept for compatibility

    return generate_optimal_discrimination_data(
        data_schema=data_schema,
        nb_groups=nb_groups,
        nb_attributes=nb_attributes,
        min_number_of_classes=min_number_of_classes,
        max_number_of_classes=max_number_of_classes,
        prop_protected_attr=prop_protected_attr,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        min_diff_subgroup_size=min_diff_subgroup_size,
        max_diff_subgroup_size=max_diff_subgroup_size,
        W=W,
        categorical_outcome=categorical_outcome,
        nb_categories_outcome=nb_categories_outcome
    )


# %%
if __name__ == '__main__':
    data = generate_optimal_discrimination_data(
        nb_groups=100,
        nb_attributes=15,
        prop_protected_attr=0.3
    )
