import pandas as pd
import numpy as np
from pathlib import Path
from data_generator.main import generate_data, DiscriminationData
from methods.subgroup.gerryfair import clean, auditor, model


def prepare_data_for_gerryfair(data, protected_attrs):
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
    if df['outcome'].nunique() > 2:
        print(f"Converting {df['outcome'].nunique()}-class outcome to binary (class 0 vs rest)")
        df['outcome'] = (df['outcome'] != 0).astype(int)

    # Create attributes DataFrame to identify protected attributes
    attr_data = []
    for col in df.columns:
        if col == 'outcome':
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


def run_gerryfair(ge: DiscriminationData, C = 10, gamma = 0.01, max_iters=25):
    print("Generating synthetic data...")

    # Extract protected attributes directly from the DiscriminationData object
    protected_attrs = ge.protected_attributes
    print(f"Protected attributes: {protected_attrs}")

    # Prepare data for GerryFair
    X, X_prime, y = prepare_data_for_gerryfair(data, protected_attrs)

    print("\nData preparation complete.")
    print(f"Total samples: {len(X)}")
    print(f"Features shape: {X.shape}")
    print(f"Protected attributes shape: {X_prime.shape}")

    # Split data into train and test sets
    train_size = int(0.7 * len(X))

    X_train = X.iloc[:train_size]
    X_prime_train = X_prime.iloc[:train_size]
    y_train = y.iloc[:train_size]

    X_test = X.iloc[train_size:].reset_index(drop=True)
    X_prime_test = X_prime.iloc[train_size:].reset_index(drop=True)
    y_test = y.iloc[train_size:].reset_index(drop=True)

    # Train a simple model (GerryFair's model)
    print("\nTraining a model...")
    fair_model = model.Model(C=C, printflag=True, gamma=gamma, fairness_def='FP')
    fair_model.set_options(max_iters=max_iters)

    # Train the model
    errors, fp_difference = fair_model.train(X_train, X_prime_train, y_train)

    print("\nGenerating predictions...")
    train_predictions = fair_model.predict(X_train)
    test_predictions = fair_model.predict(X_test)

    # Audit predictions for both fairness definitions
    print("\nAuditing predictions for False Positive (FP) fairness:")
    fp_auditor = auditor.Auditor(X_prime_train, y_train, 'FP')
    fp_group, fp_violation = fp_auditor.audit(train_predictions)
    print(f"FP fairness violation: {fp_violation}")

    print("\nAuditing predictions for False Negative (FN) fairness:")
    fn_auditor = auditor.Auditor(X_prime_train, y_train, 'FN')
    fn_group, fn_violation = fn_auditor.audit(train_predictions)
    print(f"FN fairness violation: {fn_violation}")

    # Evaluate model performance
    train_accuracy = np.mean([(1 if p >= 0.5 else 0) == y for p, y in zip(train_predictions, y_train)])
    test_accuracy = np.mean([(1 if p >= 0.5 else 0) == y for p, y in zip(test_predictions, y_test)])

    print("\nModel Performance:")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Generate Pareto frontier to see accuracy-fairness tradeoff
    print("\nGenerating Pareto frontier for different fairness constraints...")
    gamma_list = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    errors, fp_violations, fn_violations = fair_model.pareto(X_train, X_prime_train, y_train, gamma_list)

    print("\nPareto frontier results:")
    print("Gamma\tError\tFP Violation\tFN Violation")
    for i, gamma in enumerate(gamma_list):
        print(f"{gamma:.3f}\t{errors[i]:.4f}\t{fp_violations[i]:.4f}\t{fn_violations[i]:.4f}")

    print("\nAudit complete!")


if __name__ == "__main__":
    ge = generate_data(
        nb_attributes=6,
        min_number_of_classes=2,
        max_number_of_classes=4,
        prop_protected_attr=0.1,
        nb_groups=100,
        max_group_size=100,
        categorical_outcome=True,
        nb_categories_outcome=4,
        use_cache=True
    )

    # Get the data and access attributes directly from the DiscriminationData object
    data = ge.dataframe
    run_gerryfair()
