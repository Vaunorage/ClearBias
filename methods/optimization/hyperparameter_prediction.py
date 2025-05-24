import sqlite3
import json
import pandas as pd
from typing import Dict, Any

from path import HERE


def get_best_hyperparameters(method: str, dataset: str,
                             db_path: str = HERE.joinpath("methods/optimization/optimizations.db")) -> Dict[
    str, Any]:
    """
    Retrieve the best hyperparameters for a given discrimination discovery method and dataset.

    Args:
        method: The fairness testing method ('expga', 'sg', 'aequitas', 'adf')
        dataset: The dataset name ('adult', 'credit', 'bank', etc.)
        db_path: Path to the SQLite database containing optimization results

    Returns:
        Dictionary containing the best parameters
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)

    try:
        # Query to find the trial with the highest SUR for the given method and dataset
        query = """
        SELECT parameters, metrics, study_name
        FROM optimization_trials
        WHERE method = ? AND study_name LIKE ?
        ORDER BY json_extract(metrics, '$.SUR') DESC
        LIMIT 1
        """

        cursor = conn.execute(query, (method, f"%{dataset}%{method}%"))
        row = cursor.fetchone()

        if not row:
            print(f"No results found for method '{method}' on dataset '{dataset}'")
            return {}

        parameters_json, metrics_json, study_name = row

        # Parse the JSON strings
        parameters = json.loads(parameters_json)
        metrics = json.loads(metrics_json)

        # Print some information about the best run
        print(f"Best parameters for {method} on {dataset}:")
        print(f"Study: {study_name}")
        print(f"SUR: {metrics.get('SUR', 'N/A')}")
        print(f"DSN: {metrics.get('DSN', 'N/A')}")
        print(f"TSN: {metrics.get('TSN', 'N/A')}")

        # Remove runtime parameters that aren't hyperparameters
        for key in ['max_runtime_seconds', 'max_tsn', 'db_path', 'analysis_id']:
            if key in parameters:
                del parameters[key]

        return parameters

    finally:
        conn.close()


# Example usage:
if __name__ == "__main__":
    db_path = HERE.joinpath("methods/optimization/optimizations.db")
    best_params = get_best_hyperparameters('expga', 'adult', db_path=db_path)
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
