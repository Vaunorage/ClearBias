import sqlite3
import pandas as pd
import numpy as np
from path import HERE

DB_PATH = HERE.joinpath("experiments/reproduce_paper/exp.db")


def ref_data():
    reference_data = []

    # Table 3: MLP model
    ## ExpGA
    reference_data.append(["ExpGA", "Census", "gender", "MLP", 491323, 141049, 0.03, 28.70])
    reference_data.append(["ExpGA", "Census", "age", "MLP", 43465, 29467, 0.12, 67.79])
    reference_data.append(["ExpGA", "Census", "race", "MLP", 90303, 30171, 0.12, 33.41])
    reference_data.append(["ExpGA", "Credit", "gender", "MLP", 281130, 65009, 0.06, 23.12])
    reference_data.append(["ExpGA", "Credit", "age", "MLP", 46357, 34048, 0.11, 73.44])
    reference_data.append(["ExpGA", "Bank", "age", "MLP", 46802, 27099, 0.13, 57.90])

    ## AEQUITAS
    reference_data.append(["Aequitas", "Census", "gender", "MLP", 85629, 1797, 2.00, 2.10])
    reference_data.append(["Aequitas", "Census", "age", "MLP", 12243, 1107, 3.25, 9.04])
    reference_data.append(["Aequitas", "Census", "race", "MLP", 22053, 551, 6.53, 2.50])
    reference_data.append(["Aequitas", "Credit", "gender", "MLP", 38096, 1596, 2.26, 4.19])
    reference_data.append(["Aequitas", "Credit", "age", "MLP", 11756, 5146, 0.70, 43.77])
    reference_data.append(["Aequitas", "Bank", "age", "MLP", 14808, 2915, 1.23, 19.68])

    ## SG
    reference_data.append(["SG", "Census", "gender", "MLP", 7070, 871, 4.13, 12.31])
    reference_data.append(["SG", "Census", "age", "MLP", 4190, 1530, 2.35, 36.51])
    reference_data.append(["SG", "Census", "race", "MLP", 4355, 588, 6.12, 13.50])
    reference_data.append(["SG", "Credit", "gender", "MLP", 7569, 1321, 2.73, 17.45])
    reference_data.append(["SG", "Credit", "age", "MLP", 7699, 4659, 0.77, 60.5])
    reference_data.append(["SG", "Bank", "age", "MLP", 3776, 2592, 1.39, 68.63])

    ## ADF
    reference_data.append(["ADF", "Census", "gender", "MLP", 116119, 24974, 0.14, 21.51])
    reference_data.append(["ADF", "Census", "age", "MLP", 40135, 21795, 0.17, 54.30])
    reference_data.append(["ADF", "Census", "race", "MLP", 47296, 8097, 0.44, 17.12])
    reference_data.append(["ADF", "Credit", "gender", "MLP", 62242, 8052, 0.45, 12.93])
    reference_data.append(["ADF", "Credit", "age", "MLP", 21649, 5957, 0.60, 27.51])
    reference_data.append(["ADF", "Bank", "age", "MLP", 27325, 11852, 0.30, 43.37])

    # Table 4: RF model
    ## ExpGA
    reference_data.append(["ExpGA", "Census", "gender", "RF", 274521, 156259, 0.02, 56.92])
    reference_data.append(["ExpGA", "Census", "age", "RF", 51192, 46035, 0.08, 89.93])
    reference_data.append(["ExpGA", "Census", "race", "RF", 78349, 47810, 0.08, 61.02])
    reference_data.append(["ExpGA", "Credit", "gender", "RF", 20295, 4247, 0.85, 20.93])
    reference_data.append(["ExpGA", "Credit", "age", "RF", 10183, 3673, 0.98, 36.07])
    reference_data.append(["ExpGA", "Bank", "age", "RF", 13284, 8885, 0.41, 66.99])

    ## AEQUITAS
    reference_data.append(["Aequitas", "Census", "gender", "RF", 56776, 1210, 2.98, 2.13])
    reference_data.append(["Aequitas", "Census", "age", "RF", 21330, 13472, 0.27, 63.16])
    reference_data.append(["Aequitas", "Census", "race", "RF", 29309, 362, 9.94, 1.24])
    reference_data.append(["Aequitas", "Credit", "gender", "RF", 3245, 148, 24.32, 4.56])
    reference_data.append(["Aequitas", "Credit", "age", "RF", 869, 153, 23.53, 17.60])
    reference_data.append(["Aequitas", "Bank", "age", "RF", 3952, 941, 3.83, 23.81])

    ## SG
    reference_data.append(["SG", "Census", "gender", "RF", 10906, 595, 6.05, 5.46])
    reference_data.append(["SG", "Census", "age", "RF", 17490, 12441, 0.29, 71.50])
    reference_data.append(["SG", "Census", "race", "RF", 12572, 894, 4.03, 7.11])
    reference_data.append(["SG", "Credit", "gender", "RF", 7269, 2192, 1.64, 30.15])
    reference_data.append(["SG", "Credit", "age", "RF", 6349, 2654, 1.36, 41.80])
    reference_data.append(["SG", "Bank", "age", "RF", 3871, 2530, 1.42, 65.36])

    # Table 5: SVM model
    ## ExpGA
    reference_data.append(["ExpGA", "Census", "gender", "SVM", 804328, 145226, 0.02, 18.06])
    reference_data.append(["ExpGA", "Census", "age", "SVM", 124816, 78064, 0.05, 62.54])
    reference_data.append(["ExpGA", "Census", "race", "SVM", 226434, 78627, 0.05, 34.72])
    reference_data.append(["ExpGA", "Credit", "gender", "SVM", 118428, 47570, 0.08, 40.17])
    reference_data.append(["ExpGA", "Credit", "age", "SVM", 21058, 9377, 0.38, 44.53])
    reference_data.append(["ExpGA", "Bank", "age", "SVM", 12746, 29574, 0.12, 69.19])

    ## AEQUITAS
    reference_data.append(["Aequitas", "Census", "gender", "SVM", 109706, 386, 9.33, 0.35])
    reference_data.append(["Aequitas", "Census", "age", "SVM", 81275, 2489, 1.45, 3.06])
    reference_data.append(["Aequitas", "Census", "race", "SVM", 62954, 277, 13.0, 0.44])
    reference_data.append(["Aequitas", "Credit", "gender", "SVM", 152432, 3683, 0.98, 2.42])
    reference_data.append(["Aequitas", "Credit", "age", "SVM", 53344, 8419, 0.43, 15.78])
    reference_data.append(["Aequitas", "Bank", "age", "SVM", 31276, 41, 87.80, 0.13])

    ## SG
    reference_data.append(["SG", "Census", "gender", "SVM", 1119, 0, np.nan, 0])
    reference_data.append(["SG", "Census", "age", "SVM", 2998, 624, 5.77, 20.81])
    reference_data.append(["SG", "Census", "race", "SVM", 1141, 0, np.nan, 0])
    reference_data.append(["SG", "Credit", "gender", "SVM", 9204, 1912, 1.88, 20.77])
    reference_data.append(["SG", "Credit", "age", "SVM", 7952, 1918, 1.88, 24.12])
    reference_data.append(["SG", "Bank", "age", "SVM", 744, 248, 14.52, 33.33])

    # Create DataFrame
    columns = ["algorithm", "dataset", "feature", "model", "TSN", "DSN", "DSS", "SUR"]
    reference_data = pd.DataFrame(reference_data, columns=columns)

    # Convert numeric columns to appropriate types
    numeric_cols = ["TSN", "DSN", "DSS", "SUR"]
    for col in numeric_cols:
        reference_data[col] = pd.to_numeric(reference_data[col])

    reference_data['SUR'] = reference_data['SUR'] / 100
    return reference_data


def get_experiment_results():
    conn = sqlite3.connect(DB_PATH)

    quer1 = """
SELECT
    algorithm AS Algorithm,
    ROUND(AVG(CAST(total_SUR AS REAL)), 4) AS "Average SUR"
FROM
    paper_reproduction_results
GROUP BY
    algorithm;
    """

    quer2 = """
SELECT
    algorithm AS Algorithm,
    model AS Model,
    ROUND(AVG(CAST(SUR AS REAL)), 4) AS "Average SUR"
FROM
    paper_reproduction_results
GROUP BY
    algorithm, model
"""

    res1 = pd.read_sql_query(quer1, conn)
    res2 = pd.read_sql_query(quer2, conn)

    conn.close()
    return res1, res2


def get_dsr_from_experiments():
    """
    Calculate DSR from the database and return results in the same format as main()
    """
    conn = sqlite3.connect(DB_PATH)

    # Query for average DSR by algorithm
    query_dsr_by_algo = """
    SELECT
        algorithm AS Algorithm,
        ROUND(AVG(CAST(total_SUR AS REAL)), 4) AS "Average DSR"
    FROM
        paper_reproduction_results
    GROUP BY
        algorithm
    ORDER BY
        algorithm;
    """

    # Query for average DSR by algorithm and model
    query_dsr_by_algo_model = """
    SELECT
        algorithm AS Algorithm,
        model AS Model,
        ROUND(AVG(CAST(total_SUR AS REAL)), 4) AS "Average DSR"
    FROM
        paper_reproduction_results
    GROUP BY
        algorithm, model
    ORDER BY
        algorithm, model;
    """

    # Execute queries
    dsr_by_algo = pd.read_sql_query(query_dsr_by_algo, conn)
    dsr_by_algo_model = pd.read_sql_query(query_dsr_by_algo_model, conn)

    # Convert numeric values to strings with 4 decimal places (to match main function output)
    dsr_by_algo['Average DSR'] = dsr_by_algo['Average DSR'].apply(lambda x: f"{x:.4f}")
    dsr_by_algo_model['Average DSR'] = dsr_by_algo_model['Average DSR'].apply(lambda x: f"{x:.4f}")

    # Filter out SVM models if needed (matching main function behavior)
    dsr_by_algo_model_filtered = dsr_by_algo_model[dsr_by_algo_model['Model'] != 'SVM'].copy()
    dsr_by_algo_model_filtered = dsr_by_algo_model_filtered.sort_values(by=['Average DSR'])

    conn.close()
    return dsr_by_algo, dsr_by_algo_model, dsr_by_algo_model_filtered


def analyze_and_compare():
    """
    Analyze both reference data and experimental data, and print results for comparison
    """
    print("\n===== REFERENCE DATA ANALYSIS =====\n")
    # Process reference data (similar to main function)
    df = ref_data()
    df['DSR'] = df['DSN'] / df['TSN']

    # Reference data - Average DSR by algorithm
    ref_avg_dsr = df.groupby('algorithm')['DSR'].mean().reset_index()
    ref_avg_dsr['DSR'] = ref_avg_dsr['DSR'].apply(lambda x: f"{x:.4f}")
    ref_avg_dsr = ref_avg_dsr.sort_values('algorithm')
    ref_avg_dsr.columns = ['Algorithm', 'Average DSR']

    # Reference data - Average DSR by algorithm and model
    ref_avg_dsr_by_model = df.groupby(['algorithm', 'model'])['DSR'].mean().reset_index()
    ref_avg_dsr_by_model['DSR'] = ref_avg_dsr_by_model['DSR'].apply(lambda x: f"{x:.4f}")
    ref_avg_dsr_by_model.columns = ['Algorithm', 'Model', 'Average DSR']

    # Filter out SVM models if needed (matching main function behavior)
    ref_avg_dsr_by_model_filtered = ref_avg_dsr_by_model[ref_avg_dsr_by_model['Model'] != 'SVM'].copy()
    ref_avg_dsr_by_model_filtered = ref_avg_dsr_by_model_filtered.sort_values(by=['Average DSR'])

    # Print reference data results
    print("\nReference Data - Average DSR by Algorithm:")
    print("----------------------------------------")
    print(ref_avg_dsr.to_string(index=False))

    print("\nReference Data - Average DSR by Algorithm and Model:")
    print("------------------------------------------------")
    print(ref_avg_dsr_by_model.to_string(index=False))

    print("\nReference Data - Average DSR by Algorithm and Model (excluding SVM, sorted by DSR):")
    print("-------------------------------------------------------------------------------")
    print(ref_avg_dsr_by_model_filtered.to_string(index=False))

    print("\n\n===== EXPERIMENTAL DATA ANALYSIS =====\n")

    # Get experimental data
    exp_dsr_by_algo, exp_dsr_by_algo_model, exp_dsr_by_algo_model_filtered = get_dsr_from_experiments()

    # Print experimental data results
    print("\nExperimental Data - Average DSR by Algorithm:")
    print("------------------------------------------")
    print(exp_dsr_by_algo.to_string(index=False))

    print("\nExperimental Data - Average DSR by Algorithm and Model:")
    print("--------------------------------------------------")
    print(exp_dsr_by_algo_model.to_string(index=False))

    print("\nExperimental Data - Average DSR by Algorithm and Model (excluding SVM, sorted by DSR):")
    print("-----------------------------------------------------------------------------------")
    print(exp_dsr_by_algo_model_filtered.to_string(index=False))

    print("\n\n===== ADDITIONAL EXPERIMENTAL DATA (SUR) =====\n")

    # Get SUR data from experiments
    sur_by_algo, sur_by_algo_model = get_experiment_results()

    # Print SUR results
    print("\nExperimental Data - Average SUR by Algorithm:")
    print("------------------------------------------")
    print(sur_by_algo.to_string(index=False))

    print("\nExperimental Data - Average SUR by Algorithm and Model:")
    print("--------------------------------------------------")
    print(sur_by_algo_model.to_string(index=False))

    # Calculate and display some comparison metrics
    print("\n\n===== COMPARISON: REFERENCE vs EXPERIMENTAL =====\n")

    # Merge reference and experimental data for comparison
    merged_algo = pd.merge(ref_avg_dsr, exp_dsr_by_algo, on='Algorithm', how='outer',
                           suffixes=(' (Ref)', ' (Exp)'))

    merged_algo = pd.merge(ref_avg_dsr, exp_dsr_by_algo, on='Algorithm', how='outer',
                           suffixes=(' (Ref)', ' (Exp)'))

    # Convert DSR values to float for ranking (they might be strings)
    if 'Average DSR (Ref)' in merged_algo.columns:
        merged_algo['Average DSR (Ref)'] = pd.to_numeric(merged_algo['Average DSR (Ref)'], errors='coerce')
    if 'Average DSR (Exp)' in merged_algo.columns:
        merged_algo['Average DSR (Exp)'] = pd.to_numeric(merged_algo['Average DSR (Exp)'], errors='coerce')

    # Add ranking columns (lower DSR is better, so ascending=True)
    merged_algo['Rank (Ref)'] = merged_algo['Average DSR (Ref)'].rank(ascending=True).astype(int)
    merged_algo['Rank (Exp)'] = merged_algo['Average DSR (Exp)'].rank(ascending=True).astype(int)

    # Format DSR values back to strings with 4 decimal places for display
    merged_algo['Average DSR (Ref)'] = merged_algo['Average DSR (Ref)'].apply(
        lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    merged_algo['Average DSR (Exp)'] = merged_algo['Average DSR (Exp)'].apply(
        lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

    # Print the result
    print("Average DSR by Algorithm - Reference vs Experimental (with Rankings):")
    print("------------------------------------------------------------------")
    print(merged_algo.to_string(index=False))


if __name__ == "__main__":
    analyze_and_compare()
