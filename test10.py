from typing import Dict, Set, List, Tuple
from itertools import combinations, product, chain
from typing import Dict, Set, List
import math


def get_all_subsets(attributes: Dict[str, Set]) -> List[Dict[str, Set]]:
    """
    Generate all possible subsets of attributes (power set excluding empty set)
    """
    attr_items = list(attributes.items())
    n = len(attr_items)
    # Generate all possible combinations of indices (excluding empty set)
    all_subsets = []
    for r in range(1, n + 1):
        for combo in combinations(attr_items, r):
            all_subsets.append(dict(combo))
    return all_subsets


def get_all_discrimination_possibilities(
        T: Dict[str, Set],  # Protected attributes and their possible values
        X: Dict[str, Set]  # Non-protected attributes and their possible values
) -> List[Tuple[Dict[str, int], Dict[str, int]]]:
    """
    Returns all possible discrimination patterns by:
    1. Generating all possible subsets of T and X
    2. For each combination of subsets, generating all possible value combinations
    """
    # Step 1: Generate all possible subsets of T and X
    T_subsets = get_all_subsets(T)
    X_subsets = get_all_subsets(X)

    all_possibilities = []

    for T_subset in T_subsets:
        for X_subset in X_subsets:
            subs = {**T_subset, **X_subset}
            possible_subgroups = product(*[list(e) for e in subs.values()])
            possible_groups = combinations(possible_subgroups, r=2)
            for group in possible_groups:
                grp1 = {k: v for k, v in zip(subs.keys(), group[0])}
                grp2 = {k: v for k, v in zip(subs.keys(), group[1])}
                all_possibilities.append((grp1, grp2))

    return all_possibilities


def calculate_discrimination_space(
        T: Dict[str, Set],
        X: Dict[str, Set]
) -> int:
    """
    Calculate the size considering pairs of groups
    """
    total = 0

    # For each possible subset selection of attributes
    for i in range(1, len(T) + 1):
        for j in range(1, len(X) + 1):
            # Number of ways to select attributes
            ct = math.comb(len(T), i)
            cx = math.comb(len(X), j)

            # For each selection, calculate total number of possible value combinations
            num_values_t = math.prod([len(val_set) for val_set in list(T.values())[:i]])
            num_values_x = math.prod([len(val_set) for val_set in list(X.values())[:j]])

            # Total number of value combinations for this selection
            total_combinations = num_values_t * num_values_x

            # Number of possible pairs of these combinations
            num_pairs = math.comb(total_combinations, 2)

            # Add to total
            total += ct * cx * num_pairs

    return total

# Example usage:
if __name__ == "__main__":
    # Define the attributes and their possible values
    T = {
        't1': {0, 1},
        't2': {0, 1}
    }

    X = {
        'x1': {0, 1},
        'x2': {0, 1}
    }

    result = calculate_discrimination_space(T, X)
    print(f"Size of discrimination search space: {result}")

    # Get all possibilities
    all_possibilities = get_all_discrimination_possibilities(T, X)
    print(f"\nTotal number of possibilities: {len(all_possibilities)}")

    # Print first few examples
    print("\nFirst 5 discrimination patterns:")
    for i, (t_dict, x_dict) in enumerate(all_possibilities[:5], 1):
        print(f"\nPattern {i}:")
        print(f"Protected attributes subset: {t_dict}")
        print(f"Non-protected attributes subset: {x_dict}")
