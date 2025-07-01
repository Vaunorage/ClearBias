# wrapper for logistic regression - Updated with Z3 solver
import z3
import math  # Use math.gcd instead of fractions.gcd
from functools import reduce
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sklearn.datasets
import numpy as np
from sklearn.datasets import load_iris
from sklearn import metrics
import numpy as np
import os
import pickle
from datetime import datetime
import random


def predict_lin(X, weights, bias):
    X = np.array(X)
    dot_result = np.dot(X, np.array(weights))
    return (dot_result >= -1 * bias).astype(int)


class linear_classifier_wrap_Wrapper():
    """
    Updated wrapper that uses Z3 solver instead of pysat/pypblib.
    Z3 has excellent Windows support and handles pseudo-boolean constraints natively.
    """

    def _find_gcd(self, list):
        x = reduce(math.gcd, list)
        return x

    def __init__(self, weights, attributes, sensitive_attributes, bias, converted_data,
                 original_model_prediction, convert_to_cnf=True, negate=False, verbose=True):
        self.weights = weights
        self.bias = bias
        self.classifier = None
        self.solver = None
        self.variables = []
        self.num_attributes = len(weights)
        self.sensitive_attributes = sensitive_attributes
        self.attributes = attributes
        self.auxiliary_variables = []
        self._store_benchmark = False

        if (self._store_benchmark):
            os.system("mkdir -p pseudo_Boolean_benchmarks")

        self._convert_weights_to_integers(converted_data, original_model_prediction)

        if (convert_to_cnf):
            self._create_z3_model(negate)

        if (verbose):
            print("Expression: ", " ".join(map(str, self.weights)), "?=", -1 * self.bias)

    def predict(self, X, weights, bias):
        X = np.array(X)
        dot_result = np.dot(X, np.array(weights))
        return (dot_result >= -1 * bias).astype(int)

    def _convert_weights_to_integers(self, converted_data, original_model_prediction):
        # convert w, bias to integer
        _max = abs(np.array(self.weights + [self.bias])).max()
        best_accuracy = -1
        best_weights = None
        best_bias = None
        best_multiplier = None

        # gridsearch to find best multiplier
        for multiplier in range(101):
            _weights = [int(float(weight / _max) * multiplier) for weight in self.weights]
            _bias = int(float(self.bias / _max) * multiplier)
            measured_accuracy = metrics.f1_score(
                self.predict(converted_data, _weights, _bias),
                original_model_prediction
            )
            if (measured_accuracy > best_accuracy):
                best_weights = _weights
                best_bias = _bias
                best_multiplier = multiplier
                best_accuracy = measured_accuracy

        assert best_weights is not None
        assert best_bias is not None
        self.weights = best_weights
        self.bias = best_bias

    def _create_z3_model(self, negate=False):
        """
        Create Z3 model with pseudo-boolean constraints
        """
        # Create Z3 solver
        self.solver = z3.Solver()

        # Create boolean variables
        self.variables = []
        for i in range(len(self.attributes)):
            var_name = f"x_{i}"
            var = z3.Bool(var_name)
            self.variables.append(var)

        # Create additional variables for sensitive attributes
        for group_idx, group in enumerate(self.sensitive_attributes):
            for attr_idx, attr in enumerate(group):
                var_name = f"sensitive_{group_idx}_{attr_idx}"
                var = z3.Bool(var_name)
                self.variables.append(var)

        # Create the pseudo-boolean constraint
        # sum(weights[i] * x[i]) >= -bias (or <= -bias-1 if negated)
        weighted_sum = z3.Sum([
            z3.If(var, weight, 0)
            for var, weight in zip(self.variables[:len(self.weights)], self.weights)
        ])

        bound = -1 * self.bias

        if negate:
            # For negation: sum(weights[i] * x[i]) <= bound - 1
            constraint = weighted_sum <= bound - 1
        else:
            # Normal case: sum(weights[i] * x[i]) >= bound
            constraint = weighted_sum >= bound

        self.solver.add(constraint)

        if self._store_benchmark:
            self._save_z3_benchmark(negate, bound)

        print(f"Z3 model created with constraint: {constraint}")

    def _save_z3_benchmark(self, negate, bound):
        """Save Z3 benchmark files for debugging"""
        benchmark_file = "Z3_Justicia_" + str(len(self.weights)) + datetime.now().strftime('_%Y_%m_%d_%H_%M_%S_') + str(
            random.randint(0, 100000))

        # Save Z3 format
        with open(f"pseudo_Boolean_benchmarks/{benchmark_file}.smt2", "w") as f:
            f.write(self.solver.to_smt2())

        # Save human-readable format
        operator = "<=" if negate else ">="
        s = f"* Z3 Model - #variables= {len(self.weights)} #constraints= 1\n"
        weight_strings = [(f"+{weight}" if weight > 0 else f"{weight}") for weight in self.weights]
        var_strings = [f"x_{i}" for i in range(len(self.weights))]
        s += " ".join([f"{w}*{v}" for w, v in zip(weight_strings, var_strings)]) + f" {operator} {bound}\n"

        with open(f"pseudo_Boolean_benchmarks/{benchmark_file}.txt", "w") as f:
            f.write(s)

    def check_assignment(self):
        """Check whether there is a satisfying assignment to the classifier."""
        if self.solver is None:
            print("No solver created. Call _create_z3_model first.")
            return False

        result = self.solver.check()
        is_sat = result == z3.sat

        print("is SAT? ", is_sat)

        if is_sat:
            model = self.solver.model()
            print("Z3 model found:")
            assignment = []
            for i, var in enumerate(self.variables):
                value = model.eval(var, model_completion=True)
                assignment.append(z3.is_true(value))
                print(f"  {var} = {value}")
            return True, assignment
        else:
            print("No satisfying assignment found")
            if result == z3.unsat:
                print("The constraint is unsatisfiable")
            elif result == z3.unknown:
                print("Z3 could not determine satisfiability")
            return False, None

    def solve_with_assumptions(self, assumptions=None):
        """
        Solve with additional assumptions
        assumptions should be a list of Z3 boolean expressions
        """
        if self.solver is None:
            print("No solver created. Call _create_z3_model first.")
            return False, None

        # Create a new solver context for assumptions
        temp_solver = z3.Solver()
        temp_solver.add(self.solver.assertions())

        if assumptions:
            for assumption in assumptions:
                temp_solver.add(assumption)

        result = temp_solver.check()
        is_sat = result == z3.sat

        if is_sat:
            model = temp_solver.model()
            assignment = []
            for var in self.variables:
                value = model.eval(var, model_completion=True)
                assignment.append(z3.is_true(value))
            return True, assignment
        else:
            return False, None

    def add_constraint(self, constraint):
        """Add an additional Z3 constraint to the solver"""
        if self.solver is None:
            print("No solver created. Call _create_z3_model first.")
            return

        self.solver.add(constraint)
        print(f"Added constraint: {constraint}")

    def get_variable(self, index):
        """Get Z3 variable by index"""
        if 0 <= index < len(self.variables):
            return self.variables[index]
        else:
            raise IndexError(f"Variable index {index} out of range")

    def reset_solver(self):
        """Reset the solver (clear all constraints)"""
        if self.solver:
            self.solver.reset()


class Z3PseudoBooleanSolver():
    """
    Standalone Z3-based pseudo-boolean constraint solver
    """

    def __init__(self, weights, bias, variable_names=None):
        self.weights = weights
        self.bias = bias
        self.solver = z3.Solver()
        self.variables = []

        # Create variables
        if variable_names is None:
            variable_names = [f"x_{i}" for i in range(len(weights))]

        for name in variable_names:
            var = z3.Bool(name)
            self.variables.append(var)

    def add_constraint(self, operator=">=", negate=False):
        """
        Add pseudo-boolean constraint: sum(weights[i] * x[i]) operator bias
        operator: ">=" or "<=" or "="
        """
        weighted_sum = z3.Sum([
            z3.If(var, weight, 0)
            for var, weight in zip(self.variables, self.weights)
        ])

        bound = -1 * self.bias

        if negate:
            bound = bound - 1

        if operator == ">=":
            constraint = weighted_sum >= bound
        elif operator == "<=":
            constraint = weighted_sum <= bound
        elif operator == "=":
            constraint = weighted_sum == bound
        else:
            raise ValueError(f"Unknown operator: {operator}")

        self.solver.add(constraint)
        return constraint

    def solve(self):
        """Solve and return result"""
        result = self.solver.check()

        if result == z3.sat:
            model = self.solver.model()
            assignment = {}
            for var in self.variables:
                value = model.eval(var, model_completion=True)
                assignment[str(var)] = z3.is_true(value)
            return True, assignment
        else:
            return False, None

    def add_custom_constraint(self, constraint):
        """Add any Z3 constraint"""
        self.solver.add(constraint)

    def get_variable(self, name_or_index):
        """Get variable by name or index"""
        if isinstance(name_or_index, int):
            return self.variables[name_or_index]
        else:
            for var in self.variables:
                if str(var) == name_or_index:
                    return var
            raise ValueError(f"Variable {name_or_index} not found")


# Supporting functions remain the same
def init_iris():
    """
    Returns weights, bias, features (including sensitive features), and sensitive features
    """
    target = "target"
    dataset = load_iris()
    dataset[target] = np.where(dataset[target] == 2, 0, dataset[target])

    # Convert to DataFrame
    data_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    data_df[target] = dataset[target]

    # Simple discretization
    for col in data_df.columns[:-1]:  # Exclude target
        data_df[col] = pd.cut(data_df[col], bins=3, labels=[0, 1, 2])

    # get X,y
    X = data_df.drop([target], axis=1)
    y = data_df[target]

    # One-hot encoding
    X = pd.get_dummies(X, columns=X.columns.tolist())

    # split into train_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    known_sensitive_attributes = [['sepal length (cm)_1']]

    attributes = X_train.columns.tolist()
    sensitive_attributes = known_sensitive_attributes
    probs = None

    # For linear classifier, we use Logistic regression model of sklearn
    clf = LogisticRegression(random_state=0)
    clf = clf.fit(X_train, y_train)

    print("\nFeatures: ", X_train.columns.tolist())
    print("\nWeights: ", clf.coef_)
    print("\nBias:", clf.intercept_[0])

    print("Train Accuracy Score: ", clf.score(X_train, y_train), "positive ratio: ", y_train.mean())
    print("Test Accuracy Score: ", clf.score(X_test, y_test), "positive ratio: ", y_test.mean())

    return clf.coef_[0], clf.intercept_[0], attributes, sensitive_attributes, probs


# Example usage and testing
if __name__ == "__main__":
    print("Testing Z3-based Linear Classifier Wrapper")
    print("=" * 50)

    # Test with iris dataset
    weights, bias, attributes, sensitive_attributes, probs = init_iris()

    # Create some dummy data for testing
    converted_data = np.random.randint(0, 2, (100, len(weights)))
    original_model_prediction = np.random.randint(0, 2, 100)

    print("\n1. Testing main wrapper:")
    # Test the updated wrapper
    wrapper = linear_classifier_wrap_Wrapper(
        weights=weights.tolist(),
        attributes=attributes,
        sensitive_attributes=sensitive_attributes,
        bias=bias,
        converted_data=converted_data,
        original_model_prediction=original_model_prediction,
        convert_to_cnf=True,
        verbose=True
    )

    # Test solving
    is_sat, assignment = wrapper.check_assignment()

    print("\n2. Testing standalone Z3 solver:")
    # Test standalone solver
    z3_solver = Z3PseudoBooleanSolver(weights=weights.tolist(), bias=bias)
    constraint = z3_solver.add_constraint(operator=">=")
    print(f"Added constraint: {constraint}")

    is_sat2, assignment2 = z3_solver.solve()
    print(f"Standalone Z3 result: SAT={is_sat2}")
    if assignment2:
        print("Assignment:", assignment2)

    print("\n3. Testing with custom constraints:")
    # Example of adding custom constraints
    if len(wrapper.variables) >= 2:
        # Add constraint that first two variables cannot both be true
        custom_constraint = z3.Not(z3.And(wrapper.variables[0], wrapper.variables[1]))
        wrapper.add_constraint(custom_constraint)

        is_sat3, assignment3 = wrapper.check_assignment()
        print(f"With custom constraint: SAT={is_sat3}")

    print("\nZ3 testing completed!")