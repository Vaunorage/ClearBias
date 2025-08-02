import pandas as pd
from pathlib import Path
import time
import logging
from data_generator.main import DiscriminationData, get_real_data
from methods.subgroup.gerryfair import clean, model
from methods.utils import make_subgroup_metrics_and_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data_for_gerryfair(data, protected_attrs, outcome_col):
    """
    Prepare data for GerryFair by creating the necessary dataset and attributes files

    Args:
        data: DataFrame containing the dataset
        protected_attrs: List of column names that are protected attributes

    Returns:
        X: DataFrame with all features
        X_prime: DataFrame with protected attributes only
        y: Series with binary labels
    """
    # Make a copy of the data to avoid modifying the original
    df = data.copy()

    # For GerryFair, we need binary labels (0/1)
    # If the outcome is not binary, we'll convert it to binary (predicting the first class vs others)
    if df[outcome_col].nunique() > 2:
        print(f"Converting {df[outcome_col].nunique()}-class outcome to binary (class 0 vs rest)")
        df['outcome'] = (df[outcome_col] != 0).astype(int)

    # Create attributes DataFrame to identify protected attributes
    attr_data = []
    for col in df.columns:
        if col == outcome_col:
            # Label column
            attr_data.append(2)
        elif col in protected_attrs:
            # Protected attribute
            attr_data.append(1)
        else:
            # Unprotected attribute
            attr_data.append(0)

    # Create temporary files for GerryFair
    temp_dir = Path("temp_gerryfair")
    temp_dir.mkdir(exist_ok=True)

    # Save dataset and attributes
    dataset_path = temp_dir / "temp_dataset.csv"
    attributes_path = temp_dir / "temp_attributes.csv"

    df.to_csv(dataset_path, index=False)
    pd.DataFrame([attr_data], columns=df.columns).to_csv(attributes_path, index=False)

    # Clean the dataset using GerryFair's function
    X, X_prime, y = clean.clean_dataset(dataset_path, attributes_path, centered=True)

    return X, X_prime, y


def run_gerryfair(data: DiscriminationData, C=10, gamma=0.01, max_iters=3, heatmap_iter=10):
    start_time = time.time()
    print("Generating synthetic data...")

    # Extract protected attributes directly from the DiscriminationData object
    protected_attrs = data.protected_attributes
    print(f"Protected attributes: {protected_attrs}")

    # Prepare data for GerryFair
    X, X_prime, y = prepare_data_for_gerryfair(data.training_dataframe, data.protected_attributes, data.outcome_column)

    print("\nData preparation complete.")
    print(f"Total samples: {len(X)}")
    print(f"Features shape: {X.shape}")
    print(f"Protected attributes shape: {X_prime.shape}")

    # Split data into train and test sets
    train_size = int(0.7 * len(X))

    X_train = X.iloc[:train_size]
    X_prime_train = X_prime.iloc[:train_size]
    y_train = y.iloc[:train_size]

    # Train a simple model (GerryFair's model)
    print("\nTraining a model...")
    fair_model_fp = model.Model(C=C, printflag=True, gamma=gamma, fairness_def='FP',
                        heatmapflag=True, heatmap_iter=heatmap_iter,  max_iters=max_iters)

    fair_model_fn = model.Model(C=C, printflag=True, gamma=gamma, fairness_def='FP',
                                heatmapflag=True, heatmap_iter=heatmap_iter,  max_iters=max_iters)
    # Train the model
    fp_errors, fp_difference, fp_subgroups_history = fair_model_fp.train(X_train, X_prime_train, y_train)
    fn_errors, fn_difference, fn_subgroups_history = fair_model_fn.train(X_train, X_prime_train, y_train)

    print("\n--- History of Subgroups Found During Training ---")
    all_subgroups = fp_subgroups_history + fn_subgroups_history
    subgroup_data = []
    protected_attr_names = X_prime_train.columns.tolist()
    
    # Get unique values for each protected attribute
    attr_values = {}
    for attr in protected_attr_names:
        attr_values[attr] = X_prime_train[attr].unique()
        print(f"Attribute '{attr}' has values: {attr_values[attr]}")
    
    for i, group_obj in enumerate(all_subgroups):
        description = []
        # Handle coefficients - check if it's a 1D or 2D array
        if hasattr(group_obj.func.b1, 'coef_'):
            if group_obj.func.b1.coef_.ndim > 1:
                coefficients = group_obj.func.b1.coef_[0]
            else:
                coefficients = group_obj.func.b1.coef_
        else:
            # If no coefficients attribute, use empty array
            coefficients = []
            
        # Handle intercept - check if it's a scalar or array
        if hasattr(group_obj.func.b1, 'intercept_'):
            if hasattr(group_obj.func.b1.intercept_, '__len__') and not isinstance(group_obj.func.b1.intercept_, (str, bytes)):
                # It's an array-like object
                intercept = group_obj.func.b1.intercept_[0]
            else:
                # It's a scalar
                intercept = group_obj.func.b1.intercept_
        else:
            # If no intercept attribute, use 0
            intercept = 0
        
        # Mathematical description (original)
        math_description = []
        for attr_name, coef in zip(protected_attr_names, coefficients):
            if abs(coef) > 1e-6:  # Only include non-trivial coefficients
                math_description.append(f"{coef:.2f} * {attr_name}")
        
        math_description_str = " + ".join(math_description) + f" > {-intercept:.2f}"
        
        # Identify the discriminated groups based on attribute values
        discriminated_groups = []
        
        try:
            # For each sample in the training data, check if it belongs to this subgroup
            subgroup_mask = group_obj.predict(X_prime_train) == 1
            subgroup_size = subgroup_mask.sum()
            
            if subgroup_size > 0:
                # Get samples that belong to this subgroup
                subgroup_samples = X_prime_train[subgroup_mask]
                print(f"\nSubgroup {i} ({subgroup_size} samples):")
                
                # For each protected attribute, find the most common values in the subgroup
                group_description = []
                for attr in protected_attr_names:
                    if attr in subgroup_samples.columns:
                        # Print the distribution of values for this attribute in the subgroup
                        value_counts = subgroup_samples[attr].value_counts(normalize=True)
                        print(f"  {attr} distribution:")
                        for val, freq in value_counts.items():
                            print(f"    {val}: {freq:.1%}")
                        
                        # Find significant values (those that appear more frequently than in the overall population)
                        overall_counts = X_prime_train[attr].value_counts(normalize=True)
                        
                        # Compare subgroup distribution to overall distribution
                        for val, freq in value_counts.items():
                            overall_freq = overall_counts.get(val, 0)
                            if freq > overall_freq * 1.5 and freq > 0.2:  # 50% more frequent and at least 20% present
                                group_description.append(f"{attr} = {val} ({freq:.1%}, overall: {overall_freq:.1%})")
                
                if group_description:
                    discriminated_groups = group_description
                else:
                    # If no clear pattern, show the top values for each attribute
                    for attr in protected_attr_names:
                        if attr in subgroup_samples.columns:
                            top_val = subgroup_samples[attr].value_counts(normalize=True).idxmax()
                            top_freq = subgroup_samples[attr].value_counts(normalize=True).max()
                            discriminated_groups.append(f"{attr} most common: {top_val} ({top_freq:.1%})")
            else:
                discriminated_groups = ["No samples found in this subgroup"]
        except Exception as e:
            print(f"Error analyzing subgroup {i}: {str(e)}")
            discriminated_groups = [f"Error analyzing subgroup: {str(e)}"]

        subgroup_info = {
            'subgroup_id': i,
            'type': 'FP' if i < len(fp_subgroups_history) else 'FN',
            'mathematical_description': math_description_str,
            'attribute_values': discriminated_groups,
            'sample_count': subgroup_mask.sum(),
            'coefficients': coefficients,
            'intercept': intercept
        }
        subgroup_data.append(subgroup_info)

    res_df = pd.DataFrame(subgroup_data)

    tsn = len(X_train)
    dsn = len(res_df)

    res_df, metrics = make_subgroup_metrics_and_dataframe(
        res_df,
        tsn,
        dsn,
        start_time,
    )

    print("\nAudit complete!")
    return res_df, metrics


if __name__ == "__main__":
    ge, schema = get_real_data('adult', use_cache=False)
    res_df, metrics = run_gerryfair(ge, C=10, gamma=0.001, max_iters=2)
    print("\n--- Results ---")
    print(res_df)
    print(f"\n--- Metrics ---")
    print(metrics)
