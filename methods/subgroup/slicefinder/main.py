from typing import List, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from methods.subgroup.slicefinder.slice_finder import SliceFinder, Slice
from methods.subgroup.slicefinder.decision_tree import DecisionTree, Node
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from methods.subgroup.slicefinder.slice_finder import Slice


def lattice_slices_to_dataframe(slices: List[Slice], data_obj) -> pd.DataFrame:
    """
    Convert a list of lattice slices to a pandas DataFrame.

    Each slice becomes a row in the DataFrame. All attributes from data_obj are included as columns.
    If a slice doesn't have a filter for a particular attribute, the value is set to None.

    Args:
        slices: List of Slice objects from SliceFinder
        data_obj: Data object containing the dataset information

    Returns:
        pandas DataFrame with one row per slice and columns for all attributes
    """
    # Get all attribute names from data_obj
    all_attributes = list(data_obj.xdf.columns)

    # Create a list to hold the data for each slice
    rows = []

    # Process each slice
    for i, slice_obj in enumerate(slices):
        # Create a dictionary for this slice with all attributes set to None initially
        row_data = {attr: None for attr in all_attributes}

        # Add slice index and size information
        row_data['slice_index'] = i
        row_data['slice_size'] = slice_obj.size

        # Add effect size and metric if available
        if slice_obj.effect_size is not None:
            row_data['effect_size'] = slice_obj.effect_size
        if slice_obj.metric is not None:
            row_data['metric'] = slice_obj.metric

        # Fill in the values for attributes that are in the slice's filters
        for attr, conditions in slice_obj.filters.items():
            # Remove any suffix like '_X' that might be in the attribute name
            base_attr = attr.split('_')[0] if '_' in attr else attr

            # Check if this is a valid attribute (might have been transformed)
            if base_attr in all_attributes:
                # Convert the conditions to a string representation
                row_data[base_attr] = str(conditions[0][0])
            else:
                # If the attribute doesn't match directly, try to find a close match
                for col in all_attributes:
                    if base_attr in col:
                        row_data[col] = str(conditions[0][0])
                        break

        rows.append(row_data)

    # Create DataFrame from the rows
    df = pd.DataFrame(rows)

    return df


def traverse_node_tree(node, case_num, path_data=None):
    """
    Recursively traverse the node tree until reaching leaf nodes.
    Prioritizes nodes with eff_size attribute when choosing which path to follow.

    Args:
        node: Current node being traversed
        case_num: Case number (node item number)
        path_data: List to accumulate path information

    Returns:
        List of dictionaries containing path information
    """
    if path_data is None:
        path_data = []

    # If this is a leaf node (no children), return the accumulated path
    if node.left_child is None and node.right_child is None:
        return path_data

    results = []

    # Traverse left child if it exists
    if node.left_child is not None:
        # Create path entry for left traversal
        left_path_entry = {
            'case': f'case {case_num}',
            'feature': node.desc[0] if node.desc else 'unknown',
            'feature_name': node.desc[0] if node.desc else 'unknown',
            'threshold': node.desc[1] if node.desc and len(node.desc) > 1 else 'unknown',
            'operator': '<',
            'val_diff_outcome': getattr(node.left_child, 'eff_size', 0) if hasattr(node.left_child, 'eff_size') else 0
        }

        # Continue traversing left subtree
        left_results = traverse_node_tree(
            node.left_child,
            case_num,
            path_data + [left_path_entry]
        )
        results.extend(left_results)

    # Traverse right child if it exists
    if node.right_child is not None:
        # Create path entry for right traversal
        right_path_entry = {
            'case': f'case {case_num}',
            'feature': node.desc[0] if node.desc else 'unknown',
            'feature_name': node.desc[0] if node.desc else 'unknown',
            'threshold': node.desc[1] if node.desc and len(node.desc) > 1 else 'unknown',
            'operator': '>=',
            'val_diff_outcome': getattr(node.right_child, 'eff_size', 0) if hasattr(node.right_child, 'eff_size') else 0
        }

        # Continue traversing right subtree
        right_results = traverse_node_tree(
            node.right_child,
            case_num,
            path_data + [right_path_entry]
        )
        results.extend(right_results)

    return results


def create_dataframe_from_nodes_for_tree_method(nodes_list: List[Node]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from a list of Node objects.

    Args:
        nodes_list: List of Node objects

    Returns:
        pandas DataFrame with columns: case, feature, feature_name, threshold, operator, val_diff_outcome, order
    """
    all_paths = []

    for i, node in enumerate(nodes_list, 1):
        # Get all paths from this node tree
        node_paths = traverse_node_tree(node, i)

        # Add order column for each case
        for order, path in enumerate(node_paths, 1):
            path['order'] = order

        all_paths.extend(node_paths)

    # Create DataFrame
    if all_paths:
        df = pd.DataFrame(all_paths)
        # Reorder columns to match your desired format
        column_order = ['case', 'feature', 'feature_name', 'threshold', 'operator', 'val_diff_outcome', 'order']
        df = df[column_order]
    else:
        # Create empty DataFrame with correct columns if no paths found
        df = pd.DataFrame(
            columns=['case', 'feature', 'feature_name', 'threshold', 'operator', 'val_diff_outcome', 'order'])

    return df


def run_slicefinder(
        data_obj,
        approach: str = "both",
        model: Optional[object] = None,
        # Model parameters
        max_depth: int = 5,
        n_estimators: int = 10,
        # Common slice parameters
        k: int = 5,
        epsilon: float = 0.3,
        # Lattice search specific parameters
        degree: int = 2,
        max_workers: int = 4,
        # Decision tree specific parameters
        dt_max_depth: int = 3,
        min_size: int = 100,
        min_effect_size: float = 0.3,
        # Display options
        verbose: bool = True,
        drop_na: bool = True
) -> dict:
    """
    Run SliceFinder analysis using lattice search and/or decision tree approaches.

    Parameters:
    -----------
    data_obj : object
        Data object containing dataframe, xdf, and ydf attributes
    schema : object
        Schema object for the dataset
    approach : str, default="both"
        Which approach to use: "lattice", "decision_tree", or "both"
    model : object, optional
        Pre-trained model to use. If None, will train RandomForestClassifier
    max_depth : int, default=5
        Maximum depth for RandomForestClassifier
    n_estimators : int, default=10
        Number of estimators for RandomForestClassifier
    k : int, default=5
        Number of slices to return
    epsilon : float, default=0.3
        Minimum effect size threshold for lattice search
    degree : int, default=2
        Maximum complexity of slice filters for lattice search
    max_workers : int, default=4
        Number of parallel workers for lattice search
    dt_max_depth : int, default=3
        Maximum depth for decision tree approach
    min_size : int, default=100
        Minimum number of samples required to split a node in decision tree
    min_effect_size : float, default=0.3
        Minimum effect size threshold for decision tree
    verbose : bool, default=True
        Whether to print detailed results
    drop_na : bool, default=True
        Whether to drop missing values from data

    Returns:
    --------
    dict : Dictionary containing results from both approaches
        - 'lattice_slices': List of slices from lattice search (if run)
        - 'dt_slices': List of slices from decision tree (if run)
        - 'model': The trained model used
        - 'data': The processed data (X, y)
        - 'summary': Summary statistics
    """

    # Validate approach parameter
    valid_approaches = ["lattice", "decision_tree", "both"]
    if approach not in valid_approaches:
        raise ValueError(f"approach must be one of {valid_approaches}")

    # ========== DATA PREPARATION ==========
    if verbose:
        print("===== DATA PREPARATION =====")

    # Handle missing values
    if drop_na:
        data = data_obj.dataframe.dropna()
        if verbose:
            print(f"Dropped missing values. Shape: {data.shape}")
    else:
        data = data_obj.dataframe

    # Split features and target
    X = data_obj.xdf
    y = data_obj.ydf

    if verbose:
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")

    # ========== MODEL TRAINING ==========
    if model is None:
        if verbose:
            print(f"\nTraining RandomForestClassifier with max_depth={max_depth}, n_estimators={n_estimators}")
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
        model.fit(X, y)
    elif verbose:
        print("\nUsing provided pre-trained model")

    # Initialize results dictionary
    results = {
        'lattice_slices': None,
        'dt_slices': None,
    }

    # ========== APPROACH 1: LATTICE SEARCH ==========
    if approach in ["lattice", "both"]:
        if verbose:
            print("\n===== USING LATTICE SEARCH APPROACH =====")

        try:
            # Initialize SliceFinder with your model and data
            slice_finder = SliceFinder(model, (X, y))

            # Find interesting slices
            lattice_slices = slice_finder.find_slice(
                k=k,
                epsilon=epsilon,
                degree=degree,
                max_workers=max_workers
            )

            slices_df = lattice_slices_to_dataframe(lattice_slices, data_obj)

            results['lattice_slices'] = slices_df

            if verbose:
                print(f"\nFound {len(lattice_slices)} slices using lattice search:")
                for i, slice_obj in enumerate(lattice_slices):
                    print(f"\n----- Lattice Slice {i + 1} -----")
                    print(f"Slice: {slice_obj}")
                    print(f"Effect size: {slice_obj.effect_size:.4f}")
                    print(f"Size: {slice_obj.size} samples")

        except Exception as e:
            if verbose:
                print(f"Error in lattice search: {e}")
            results['lattice_slices'] = None

    # ========== APPROACH 2: DECISION TREE ==========
    if approach in ["decision_tree", "both"]:
        if verbose:
            print("\n===== USING DECISION TREE APPROACH =====")

        try:
            # Initialize DecisionTree with your data and model
            dt_finder = DecisionTree((X, y), model)

            # Build the decision tree
            dt_finder = dt_finder.fit(max_depth=dt_max_depth, min_size=min_size)

            # Find interesting slices
            dt_slices = dt_finder.recommend_slices(k=k, min_effect_size=min_effect_size)

            dt_slices = create_dataframe_from_nodes_for_tree_method(dt_slices)

            results['dt_slices'] = dt_slices

            if verbose:
                print(f"\nFound {len(dt_slices)} slices using decision tree:")
                for i, node in enumerate(dt_slices):
                    print(f"\n----- Decision Tree Slice {i + 1} -----")

                    # Get the path from root to this node
                    ancestry = node.__ancestry__()
                    if ancestry:
                        print("Path from root:", " → ".join(ancestry))

                    # Display the node description
                    print(f"Node: {node}")

                    # Display effect size and slice size
                    print(f"Effect size: {node.eff_size:.4f}")
                    print(f"Slice size: {node.size} samples")

        except Exception as e:
            if verbose:
                print(f"Error in decision tree approach: {e}")
            results['dt_slices'] = None

    return results


# Example usage:
if __name__ == "__main__":
    # Load your dataset
    from data_generator.main import get_real_data

    data_obj, schema = get_real_data('adult', use_cache=True)

    # Run with both approaches (default)
    results = run_slicefinder(data_obj, approach="lattice", model=None, max_depth=5,
                              n_estimators=1, k=2, epsilon=0.3, degree=2,
                              max_workers=4, dt_max_depth=3, min_size=100,
                              min_effect_size=0.3, verbose=True, drop_na=True)

    print(results)
