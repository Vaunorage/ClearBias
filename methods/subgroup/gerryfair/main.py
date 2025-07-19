import pandas as pd
from pathlib import Path
from data_generator.main import DiscriminationData, get_real_data
from methods.subgroup.gerryfair import clean, model


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


def run_gerryfair(ge: DiscriminationData, C=10, gamma=0.01, max_iters=3):
    print("Generating synthetic data...")

    # Extract protected attributes directly from the DiscriminationData object
    protected_attrs = ge.protected_attributes
    print(f"Protected attributes: {protected_attrs}")

    # Prepare data for GerryFair
    X, X_prime, y = prepare_data_for_gerryfair(ge.training_dataframe, ge.protected_attributes, ge.outcome_column)

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
    fair_model_fp = model.Model(C=C, printflag=True, gamma=gamma, fairness_def='FP')
    fair_model_fp.set_options(max_iters=max_iters)

    fair_model_fn = model.Model(C=C, printflag=True, gamma=gamma, fairness_def='FP')
    fair_model_fn.set_options(max_iters=max_iters)

    # Train the model
    fp_errors, fp_difference, fp_subgroups_history = fair_model_fp.train(X_train, X_prime_train, y_train)
    fn_errors, fn_difference, fn_subgroups_history = fair_model_fn.train(X_train, X_prime_train, y_train)

    print("\n--- History of Subgroups Found During Training ---")
    final_vals = []
    for i, group_obj in enumerate(fp_subgroups_history+fn_subgroups_history):
        print(f"\nIteration {i + 1}:")
        print(f"  Fairness Violation: {group_obj.weighted_disparity}")
        group_coeffs = {k: v for k, v in zip(ge.protected_attributes, group_obj.func.b1.coef_)}
        print(f"  Group Characteristics (coefficients for protected attributes): {group_coeffs}")
        final_val = {**group_coeffs, **{e: None for e in ge.non_protected_attributes}, **group_obj.to_dict()}
        final_vals.append(final_val)
    final_vals = pd.DataFrame(final_vals)

    print("\nAudit complete!")
    return final_vals


if __name__ == "__main__":
    ge, schema = get_real_data('adult', use_cache=False)
    res = run_gerryfair(ge, C=10, gamma=0.001, max_iters=4)
    print(res)
