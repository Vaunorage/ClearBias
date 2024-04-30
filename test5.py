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


# %%

sets = [
    [0, 1, 2, 3, 4],
    [0, 1, 2],
    ['a', 'b']
]
rank_r = 23
tuple_result = get_tuple_from_multi_set_rank(sets, rank_r)
rank_rr = rank_from_tuple(sets, tuple_result)
print("Tuple for rank 23:", tuple_result, rank_rr)
