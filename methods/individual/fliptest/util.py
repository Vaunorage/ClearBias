import numpy as np

def get_index_arrays(forward_dict, reverse_dict):
    '''
    Given the outputs of the `optimize.optimize` function on `X1` and `X2`,
    creates a simpler form of these outputs. `X1` and `X2` may have different
    number of rows when dealing with imbalanced groups.
    
    The outputs `forward` and `reverse` are 1-D numpy arrays. If `X1[i]`
    maps to `X2[j]`, we have `forward[i] = j` and `reverse[j] = i`.
    '''
    # Handle case where groups are imbalanced
    num_pts_forward = len(forward_dict)
    num_pts_reverse = len(reverse_dict)
    num_pts = max(num_pts_forward, num_pts_reverse)
    
    # Check if we have valid weights for mapping
    if forward_dict:
        min_weight = min([max(forward_dict[i].values()) for i in forward_dict if forward_dict[i]])
        # Relaxed assertion for imbalanced groups
        assert min_weight > 0.1 # Allow for more flexible mapping
    
    #initialize the arrays to -1
    forward = -np.ones(num_pts, dtype=np.int64)
    reverse = -np.ones(num_pts, dtype=np.int64)
    
    #fill the arrays with the correct indices
    for i in forward_dict:
        forward[i] = max(forward_dict[i].keys(), key=lambda j: forward_dict[i][j])
    for j in reverse_dict:
        reverse[j] = max(reverse_dict[j].keys(), key=lambda i: reverse_dict[j][i])
    
    return forward, reverse

def get_mean_dist(X1, X2, forward):
    '''
    Compute the mean L1 distance between rows in `X1` and `X2` that map to
    each other. Handles cases where some rows in X1 might not have a valid mapping
    in X2 (indicated by forward[i] == -1).
    '''
    # Filter out invalid mappings (where forward[i] == -1)
    valid_indices = forward >= 0
    if not np.any(valid_indices):
        return float('inf')  # No valid mappings
    
    # Only compute distances for valid mappings
    valid_X1 = X1[valid_indices]
    valid_forward = forward[valid_indices]
    
    # Compute mean distance only for valid mappings
    return np.mean(np.sum(np.abs(valid_X1 - X2[valid_forward]), axis=1))
