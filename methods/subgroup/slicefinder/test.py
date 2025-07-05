import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Union, Optional

from methods.subgroup.slicefinder.slice_finder import SliceFinder, Slice
from methods.subgroup.slicefinder.decision_tree import DecisionTree

def slices_to_dataframe(slices: List[Union[Slice, object]], slice_type: str = "slice") -> pd.DataFrame:
    """
    Convert a list of Slice objects to a pandas DataFrame.

    Parameters:
    -----------
    slices : List[Union[Slice, object]]
        List of Slice objects or decision tree nodes
    slice_type : str, default="slice"
        Type of slices: "slice" for lattice search results, "node" for decision tree results

    Returns:
    --------
    pd.DataFrame : DataFrame with slice information
    """
    if not slices:
        return pd.DataFrame()

    data = []

    for i, slice_obj in enumerate(slices):
        if slice_type == "slice":
            # Handle lattice search Slice objects
            row = {
                'slice_id': i + 1,
                'description': str(slice_obj),
                'filters': slice_obj.filters,
                'size': slice_obj.size,
                'effect_size': slice_obj.effect_size,
                'metric': slice_obj.metric,
                'data_idx': slice_obj.data_idx
            }

            filtrs = {k: v[0][0] for k, v in slice_obj.filters.items()}
            filtrs = {**filtrs, **{e:None for e in data_obj.attr_columns if e not in filtrs}}
            row = {**row, **filtrs}

            # Flatten filters for easier analysis
            for filter_name, filter_values in slice_obj.filters.items():
                row[f'filter_{filter_name}'] = filter_values

        else:  # slice_type == "node" for decision tree results
            # Handle decision tree node objects
            row = {
                'slice_id': i + 1,
                'description': str(slice_obj),
                'size': slice_obj.size,
                'effect_size': slice_obj.eff_size,
                'node_obj': slice_obj
            }

            # Add ancestry path if available
            try:
                ancestry = slice_obj.__ancestry__()
                row['ancestry_path'] = " → ".join(ancestry) if ancestry else ""
            except:
                row['ancestry_path'] = ""

        data.append(row)

    df = pd.DataFrame(data)

    # Sort by effect size (descending)
    if 'effect_size' in df.columns:
        df = df.sort_values('effect_size', ascending=False).reset_index(drop=True)

    return df


def get_slice_summary(slices: List[Union[Slice, object]], slice_type: str = "slice") -> dict:
    """
    Get summary statistics for a list of slices.

    Parameters:
    -----------
    slices : List[Union[Slice, object]]
        List of Slice objects or decision tree nodes
    slice_type : str, default="slice"
        Type of slices: "slice" for lattice search results, "node" for decision tree results

    Returns:
    --------
    dict : Summary statistics
    """
    if not slices:
        return {}

    if slice_type == "slice":
        effect_sizes = [s.effect_size for s in slices if s.effect_size is not None]
        sizes = [s.size for s in slices]
    else:  # decision tree nodes
        effect_sizes = [s.eff_size for s in slices if hasattr(s, 'eff_size')]
        sizes = [s.size for s in slices if hasattr(s, 'size')]

    summary = {
        'total_slices': len(slices),
        'avg_effect_size': np.mean(effect_sizes) if effect_sizes else 0,
        'max_effect_size': np.max(effect_sizes) if effect_sizes else 0,
        'min_effect_size': np.min(effect_sizes) if effect_sizes else 0,
        'avg_size': np.mean(sizes) if sizes else 0,
        'max_size': np.max(sizes) if sizes else 0,
        'min_size': np.min(sizes) if sizes else 0,
    }

    return summary


def run_slicefinder(
        data_obj,
        schema,
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
        'lattice_df': None,
        'dt_df': None,
        'model': model,
        'data': (X, y),
        'summary': {}
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

            results['lattice_slices'] = lattice_slices

            # Convert to DataFrame
            if lattice_slices:
                results['lattice_df'] = slices_to_dataframe(lattice_slices, slice_type="slice")

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
            results['lattice_slices'] = []

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
            results['dt_slices'] = []

    # ========== SUMMARY ==========
    if verbose and approach == "both":
        print("\n===== COMPARISON OF APPROACHES =====")

        lattice_count = len(results['lattice_slices']) if results['lattice_slices'] else 0
        dt_count = len(results['dt_slices']) if results['dt_slices'] else 0

        print(f"Lattice search found {lattice_count} slices")
        print(f"Decision tree found {dt_count} slices")

        # Compare average effect sizes
        if results['lattice_slices']:
            avg_effect_lattice = sum(s.effect_size for s in results['lattice_slices']) / len(results['lattice_slices'])
            print(f"Average effect size (lattice): {avg_effect_lattice:.4f}")
            results['summary']['avg_effect_lattice'] = avg_effect_lattice

        if results['dt_slices']:
            avg_effect_dt = sum(node.eff_size for node in results['dt_slices']) / len(results['dt_slices'])
            print(f"Average effect size (decision tree): {avg_effect_dt:.4f}")
            results['summary']['avg_effect_dt'] = avg_effect_dt

    # Store summary information
    results['summary']['approach'] = approach
    results['summary']['lattice_count'] = len(results['lattice_slices']) if results['lattice_slices'] else 0
    results['summary']['dt_count'] = len(results['dt_slices']) if results['dt_slices'] else 0

    # Convert slices to DataFrames
    if results['lattice_slices']:
        results['lattice_df'] = slices_to_dataframe(results['lattice_slices'], slice_type="slice")
        results['summary']['lattice_summary'] = get_slice_summary(results['lattice_slices'], slice_type="slice")

    if results['dt_slices']:
        results['dt_df'] = slices_to_dataframe(results['dt_slices'], slice_type="node")
        results['summary']['dt_summary'] = get_slice_summary(results['dt_slices'], slice_type="node")

    return results


# Example usage:
if __name__ == "__main__":
    # Load your dataset
    from data_generator.main import get_real_data

    data_obj, schema = get_real_data('adult', use_cache=True)

    # Run with both approaches (default)
    results = run_slicefinder(data_obj, schema)

    # Access DataFrames
    if 'lattice_df' in results:
        print("\nLattice Search Results DataFrame:")
        print(results['lattice_df'][['slice_id', 'description', 'size', 'effect_size']].head())

    if 'dt_df' in results:
        print("\nDecision Tree Results DataFrame:")
        print(results['dt_df'][['slice_id', 'description', 'size', 'effect_size']].head())

    # Run with only lattice search
    # results = run_slicefinder(data_obj, schema, approach="lattice", k=3, epsilon=0.2)

    # Run with only decision tree
    # results = run_slicefinder(data_obj, schema, approach="decision_tree", k=3, dt_max_depth=4)

    # Run with custom model
    # from sklearn.linear_model import LogisticRegression
    # custom_model = LogisticRegression()
    # custom_model.fit(data_obj.xdf, data_obj.ydf)
    # results = run_slicefinder(data_obj, schema, model=custom_model, verbose=False)
