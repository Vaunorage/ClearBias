import copy
import math
import random
import itertools
import sqlite3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataclasses import dataclass


def max_rank(sets):
    total_combinations = 1
    for s in sets:
        total_combinations *= len(s)
    return total_combinations


def get_tuple_from_multi_set_rank(sets, rank):
    total_combinations = max_rank(sets)
    if rank >= total_combinations:
        raise ValueError("Rank is out of the allowable range (0 to total_combinations - 1)")

    indices = []
    for i in range(len(sets) - 1, -1, -1):
        size = len(sets[i])
        index = rank % size
        indices.insert(0, index)
        rank //= size

    result_tuple = tuple(sets[i][indices[i]] for i in range(len(sets)))
    return result_tuple


def rank_from_tuple(sets, tuple_value):
    if len(sets) != len(tuple_value):
        raise ValueError("The tuple must have the same number of elements as there are sets.")

    rank = 0
    product = 1

    for i in reversed(range(len(sets))):
        element = tuple_value[i]
        set_size = len(sets[i])
        index = sets[i].index(element)
        rank += index * product
        product *= set_size

    return rank


def list_possible_ranks_for_subsets(set, subsets):
    import itertools

    # Initialize a list to hold the ranks for each subset
    all_ranks = []

    # Process each subset
    for subset in subsets:
        # Generate all combinations from the current subset
        all_combinations = list(itertools.product(*subset))

        # Initialize the list of ranks for the current subset
        ranks = []

        # Compute the rank for each combination using the full set
        for combination in all_combinations:
            rank = rank_from_tuple(set, combination)
            ranks.append(rank)

        # Append the list of ranks for the current subset to the main list
        all_ranks.append(ranks)

    return all_ranks


def generate_categorical_distribution(categories, skewness):
    n = len(categories)
    raw_probs = [skewness ** (i - 1) for i in range(1, n + 1)]
    normalization_factor = sum(raw_probs)
    normalized_probs = [p / normalization_factor for p in raw_probs]
    return dict(zip(categories, normalized_probs))


def generate_samples(categories, skewness, num_samples):
    distribution = generate_categorical_distribution(categories, skewness)
    categories, probabilities = zip(*distribution.items())
    samples = random.choices(categories, weights=probabilities, k=num_samples)
    return samples


def generate_data_for_subgroup(sets, subgroup, skewness, num_samples_per_set):
    all_samples = []
    for k, s in enumerate(subgroup):
        if s == -1:
            n_set = copy.deepcopy(sets[k])
            n_set.remove(-1)
            samples = generate_samples(n_set, 1 - skewness, num_samples_per_set)
            all_samples.append(samples)
        else:
            all_samples.append([s] * num_samples_per_set)
    all_samples = np.array(all_samples).T
    return all_samples


def generate_outcome(subgroup2_individuals, Wt, Wx, sets_attr, magnitude, epis_uncertainty):
    protected_attr = subgroup2_individuals[:, np.array(sets_attr)]
    unprotected_attr = subgroup2_individuals[:, ~np.array(sets_attr)]

    sigmoid = lambda z: 1 / (1 + np.exp(-z))

    def normalize_data(X):
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        scale = X_max - X_min
        scale[scale == 0] = 1
        X_normalized = (X - X_min) / scale
        return X_normalized

    outcome = (normalize_data(protected_attr).dot(Wt.T) * (magnitude + 1) + \
               normalize_data(unprotected_attr).dot(Wx.T)).sum(axis=1)
    outcome = random.gauss(outcome, outcome * epis_uncertainty)
    outcome = sigmoid(outcome).reshape(-1, 1)
    return outcome


def convert_to_float(df):
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='ignore')
    return df


def coefficient_of_variation(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    if mean == 0:
        return float('inf')  # Return infinity if the mean is 0, as CV is not defined
    cv = (std_dev / mean) * 100
    return cv


def generate_data_schema(min_number_of_classes, max_number_of_classes, nb_attributes, prop_protected_attr):
    sets = []
    attr_names = []

    if nb_attributes < 2:
        raise ValueError("nb_attributes must be at least 2 to ensure both 0 and 1 can be included.")

    protected_attr = [False] * (nb_attributes - 2) + [False, True]

    for i in range(nb_attributes - 2):
        protected_attr[i] = True if random.random() < prop_protected_attr else False

    random.shuffle(protected_attr)

    for i in range(nb_attributes):
        base_name = 'Attr' + str(i + 1)
        num_classes = random.randint(min_number_of_classes, max_number_of_classes)
        attribute_set = [-1] + list(range(0, num_classes))
        sets.append(attribute_set)

        if protected_attr[i]:
            attr_names.append(base_name + '_T')
        else:
            attr_names.append(base_name + '_X')

    if not any(protected_attr):
        # Randomly select an attribute to make protected
        random_index = random.randint(0, nb_attributes - 1)
        protected_attr[random_index] = True
        attr_names[random_index] = attr_names[random_index][:-2] + '_T'  # Change the suffix to '_T'

    return sets, protected_attr, attr_names


def bin_array_values(array, num_bins):
    min_val = np.min(array)
    max_val = np.max(array)

    bins = np.linspace(min_val, max_val, num_bins + 1)

    binned_indices = np.digitize(array, bins) - 1

    return binned_indices


class CBDataframe(pd.DataFrame):

    @property
    def x_cols(self):
        return list(filter(lambda x: 'Attr' in x, self.columns))

    @property
    def y_col(self):
        return ['outcome']

    def random_split(self, p=0.2):
        train, test = train_test_split(self, test_size=p, random_state=42)
        return CBDataframe(train), CBDataframe(test)

    def xdf(self):
        return self[self.x_cols]

    def ydf(self):
        return self[self.y_col]

    def adf(self):
        return self[self.x_cols + self.y_col]


@dataclass
class GeneratedData:
    dataframe: pd.DataFrame
    categorical_columns: list
    protected_attributes: dict
    collisions: int
    nb_groups: int
    max_group_size: int
    hiddenlayers_depth: int
    outcome_column: str


def generate_data(
        min_number_of_classes=2,
        max_number_of_classes=6,
        nb_attributes=6,
        prop_protected_attr=0.1,
        nb_groups=100,
        max_group_size=100,
        hiddenlayers_depth=3,
        min_similarity=0.0,
        max_similarity=1.0,
        min_alea_uncertainty=0.0,
        max_alea_uncertainty=1.0,
        min_epis_uncertainty=0.0,
        max_epis_uncertainty=1.0,
        min_magnitude=0.0,
        max_magnitude=1.0,
        min_frequency=0.0,
        max_frequency=1.0,
        categorical_outcome: bool = False,
        nb_categories_outcome: int = 6
) -> GeneratedData:
    """
    Generates synthetic data with configurable parameters for protected and unprotected attributes,
    group sizes, and different types of uncertainties.

    Returns:
    - GeneratedData: A dataclass containing the generated dataframe, metadata, and additional information.
    """

    # Generate the data schema based on the input parameters
    sets, sets_attr, attr_names = generate_data_schema(
        min_number_of_classes, max_number_of_classes, nb_attributes, prop_protected_attr
    )

    nb_protected = sum(sets_attr)  # Number of protected attributes
    nb_unprotected = len(sets_attr) - nb_protected  # Number of unprotected attributes

    # Separate indexes for protected and unprotected attributes
    protected_indexes = [index for index, value in enumerate(sets_attr) if value]
    unprotected_indexes = [index for index, value in enumerate(sets_attr) if not value]

    # Randomly initialize weights for protected and unprotected attributes
    Wt = np.random.uniform(low=0.0, high=1.0, size=(hiddenlayers_depth, nb_protected))
    Wx = np.random.uniform(low=0.0, high=1.0, size=(hiddenlayers_depth, nb_unprotected))

    history = set()  # Keep track of generated combinations to avoid duplicates
    results = []  # List to store generated data
    collisions = 0  # Counter for tracking overlapping data points

    for group_num in tqdm(range(nb_groups), desc="Generating data"):
        # Randomly generate hyperparameters for each group
        granularity = random.randint(1, nb_unprotected)
        intersectionality = random.randint(1, nb_protected)
        similarity = random.uniform(min_similarity, max_similarity)
        alea_uncertainty = random.uniform(min_alea_uncertainty, max_alea_uncertainty)
        epis_uncertainty = random.uniform(min_epis_uncertainty, max_epis_uncertainty)
        magnitude = random.uniform(min_magnitude, max_magnitude)
        frequency = random.uniform(min_frequency, max_frequency)

        # Generate all possible combinations of unprotected and protected attributes
        possible_gran = list(itertools.combinations(unprotected_indexes, granularity))
        possible_intersec = list(itertools.combinations(protected_indexes, intersectionality))
        possiblities = list(itertools.product(possible_gran, possible_intersec))

        subgroup1_possible_sets = []
        for pos in possiblities:
            pos = tuple(itertools.chain.from_iterable(pos))
            ress_set = []
            for s in range(len(sets)):
                if s in pos:
                    ss = copy.deepcopy(sets[s])
                    ss.remove(-1)
                    ress_set.append(ss)
                else:
                    ress_set.append([-1])
            subgroup1_possible_sets.append(ress_set)

        # Calculate the ranks for the generated subsets
        possible_ranks_subgroup1 = list_possible_ranks_for_subsets(sets, subgroup1_possible_sets)
        possible_ranks_subgroup1 = set(itertools.chain.from_iterable(possible_ranks_subgroup1))

        # Check if the current combination already exists in the history
        if possible_ranks_subgroup1.issubset(history):
            history.difference_update(possible_ranks_subgroup1)
            collisions += 1
            print(f'granularity {granularity}, intersectionality {intersectionality}, similarity {similarity} maxed !')

        possible_ranks_subgroup1.difference_update(history)

        # Choose a subgroup and mark it as used in history
        subgroup_rank1 = random.choice(list(possible_ranks_subgroup1))
        subgroup1 = get_tuple_from_multi_set_rank(sets, subgroup_rank1)
        history.add(subgroup_rank1)

        # Determine attributes to change in the second subgroup
        possible_chang = [k for k, e in enumerate(subgroup1) if e != -1]
        sim_ind_to_chang = random.choice([e for e in possible_chang if sets_attr[e]])
        random.shuffle(possible_chang)
        possible_chang = possible_chang[:math.ceil(len(possible_chang) * (1 - similarity))]
        possible_chang.append(sim_ind_to_chang)
        possible_chang = set(possible_chang)

        # Generate the second subgroup with changes
        subgroup2_possible_sets = []
        for k, e in enumerate(subgroup1):
            if k in possible_chang:
                p_set = copy.deepcopy(sets[k])
                if len(p_set) > 2:
                    p_set.remove(e)
                p_set.remove(-1)
                subgroup2_possible_sets.append(p_set)
            else:
                subgroup2_possible_sets.append([e])

        possible_ranks_subgroup2 = list_possible_ranks_for_subsets(sets, [subgroup2_possible_sets])
        possible_ranks_subgroup2 = set(itertools.chain.from_iterable(possible_ranks_subgroup2))

        # Check if the second combination already exists in the history
        if possible_ranks_subgroup2.issubset(history):
            history.difference_update(possible_ranks_subgroup2)
            collisions += 1
            print(f'granularity {granularity}, intersectionality {intersectionality}, similarity {similarity} maxed !')

        possible_ranks_subgroup2.difference_update(history)

        # Choose the second subgroup and mark it as used in history
        subgroup_rank2 = random.choice(list(possible_ranks_subgroup2))
        subgroup2 = get_tuple_from_multi_set_rank(sets, subgroup_rank2)
        history.add(subgroup_rank2)

        # Calculate the subgroup sizes and generate individual data for each subgroup
        sub_group_size = math.ceil((max_group_size * frequency) / 2)
        subgroup1_individuals = generate_data_for_subgroup(sets, subgroup1, alea_uncertainty, sub_group_size)
        subgroup2_individuals = generate_data_for_subgroup(sets, subgroup2, alea_uncertainty, sub_group_size)
        group_size = subgroup1_individuals.shape[0] + subgroup2_individuals.shape[1]

        # Generate outcomes for both subgroups
        subgroup1_outcomes = generate_outcome(subgroup1_individuals, Wt, Wx, sets_attr, magnitude, epis_uncertainty)
        subgroup2_outcomes = generate_outcome(subgroup2_individuals, Wt, Wx, sets_attr, magnitude, epis_uncertainty)

        # Create keys and metadata for each individual in the subgroups
        gen_attributes = np.array(
            [
                group_size, min_number_of_classes, max_number_of_classes, nb_attributes, prop_protected_attr,
                nb_groups, max_group_size, hiddenlayers_depth, granularity, intersectionality, similarity,
                alea_uncertainty, epis_uncertainty, magnitude, frequency
            ]
        ).reshape(-1, 1).repeat(sub_group_size, axis=1).T

        subgroup1_key = np.array([f"{'|'.join(list(map(str, subgroup1)))}"]).repeat(sub_group_size, axis=0).reshape(-1,
                                                                                                                    1)
        subgroup2_key = np.array([f"{'|'.join(list(map(str, subgroup2)))}"]).repeat(sub_group_size, axis=0).reshape(-1,
                                                                                                                    1)
        group_key = np.array(
            [f"{'|'.join(list(map(str, subgroup1)))}" + "*" + f"{'|'.join(list(map(str, subgroup2)))}"]
        ).repeat(sub_group_size, axis=0).reshape(-1, 1)

        indv1_key = np.apply_along_axis(lambda x: '|'.join(x), 1, subgroup1_individuals.astype(str)).reshape(-1, 1)
        indv2_key = np.apply_along_axis(lambda x: '|'.join(x), 1, subgroup2_individuals.astype(str)).reshape(-1, 1)
        couple_key = np.concatenate(
            [indv1_key.reshape(-1, 1), np.repeat(['*'], indv1_key.shape[0]).reshape(-1, 1), indv2_key.reshape(-1, 1)],
            axis=1
        )
        couple_key = np.apply_along_axis(lambda x: ''.join(x), 1, couple_key.astype(str)).reshape(-1, 1)

        # Combine all generated data into a single result set
        res_subgroup1 = np.concatenate(
            [group_key, subgroup1_key, couple_key, indv1_key, gen_attributes, subgroup1_individuals,
             subgroup1_outcomes],
            axis=1
        )
        res_subgroup2 = np.concatenate(
            [group_key, subgroup2_key, couple_key, indv2_key, gen_attributes, subgroup2_individuals,
             subgroup2_outcomes],
            axis=1
        )

        results.append(np.concatenate([res_subgroup1, res_subgroup2]))

    # Define column names for the results dataframe
    col_names = [
                    'group_key', 'subgroup_key', 'couple_key', 'indv_key', 'group_size', 'min_number_of_classes',
                    'max_number_of_classes', 'nb_attributes', 'prop_protected_attr', 'nb_groups', 'max_group_size',
                    'hiddenlayers_depth', 'granularity', 'intersectionality', 'similarity', 'alea_uncertainty',
                    'epis_uncertainty', 'magnitude', 'frequency'
                ] + attr_names + ['outcome']

    # Combine all generated results into a single dataframe
    results = pd.DataFrame(np.concatenate(results), columns=col_names)
    results['collisions'] = collisions

    results = convert_to_float(results)

    # If categorical outcomes are required, bin the outcome into categories
    if categorical_outcome:
        results['outcome'] = bin_array_values(results['outcome'], nb_categories_outcome)

    results = results.sort_values(['group_key', 'couple_key'])

    # Calculate the difference in outcomes within each group and couple
    results[f'diff_outcome'] = results.groupby(['group_key', 'couple_key'])[f'outcome'].diff().abs().bfill()
    results['diff_variation'] = coefficient_of_variation(results['diff_outcome'])

    # Map the attributes to whether they are protected or not
    protected_attr = {k: e for k, e in zip(attr_names, sets_attr)}

    # Identify categorical columns
    categorical_columns = [col for col in results.columns if results[col].dtype == 'object']

    # Return a GeneratedData instance containing the results and additional metadata
    return GeneratedData(
        dataframe=results,
        categorical_columns=categorical_columns,
        protected_attributes=protected_attr,
        collisions=collisions,
        nb_groups=nb_groups,
        max_group_size=max_group_size,
        hiddenlayers_depth=hiddenlayers_depth,
        outcome_column='outcome'
    )
