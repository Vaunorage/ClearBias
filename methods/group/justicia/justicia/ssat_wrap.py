import os
import subprocess
from z3 import *
import numpy as np
import warnings


class Fairness_verifier():

    def __init__(self, timeout=400):
        self.num_variables = 0
        self.num_clauses = 0
        self.timeout = max(int(timeout), 10)
        self.execution_error = False
        self.solver = Solver()
        self.vars_dict = {}  # Map variable indices to Z3 Bool variables
        pass

    def __str__(self):
        s = "\nFairness verifier ->\n-number of variables: %s\n-number is clauses %s" % (
            self.num_variables, self.num_clauses)
        return s

    def _get_or_create_var(self, var_index):
        """Get or create a Z3 Boolean variable for the given index"""
        abs_index = abs(var_index)
        if abs_index not in self.vars_dict:
            self.vars_dict[abs_index] = Bool(f'x_{abs_index}')
        return self.vars_dict[abs_index]

    def _apply_random_quantification(self, var, prob):
        return "r " + str(prob) + " " + str(var) + " 0\n"

    def _apply_exist_quantification(self, var):
        return "e " + str(abs(var)) + " 0\n"

    def invoke_SSAT_solver(self, filename, find_maximization=True, verbose=True):

        # print("\n\n")
        # print(self.formula)
        # print("\n\n")

        dir_path = os.path.dirname(os.path.realpath(filename))

        # Execute and read output
        cmd = "timeout " + str(self.timeout) + " stdbuf -oL " + " abc -c \"ssat -v " + str(dir_path) + "/" + str(
            filename) + \
              "\" 1>" + str(dir_path) + "/" + str(filename) + "_out.log" + \
              " 2>" + str(dir_path) + "/" + str(filename) + "_err.log"
        os.system(cmd)

        if (verbose):
            f = open(str(dir_path) + "/" + str(filename) + "_err.log", "r")
            lines = f.readlines()
            if (len(lines) > 0):
                print("Error print of SSAT (if any)")
            print(("").join(lines))
            f.close()

        f = open(str(dir_path) + "/" + str(filename) + "_out.log", "r")
        lines = f.readlines()
        f.close()

        # os.system("rm " + str(dir_path) + "/" +str(filename) + "_out")

        # process output
        upper_bound = None
        lower_bound = None
        read_optimal_assignment = False
        self.sol_prob = None
        self.assignment_to_exists_variables = None

        for line in lines:
            if (read_optimal_assignment):
                try:
                    self.assignment_to_exists_variables = list(
                        map(int, line[:-2].strip().split(" ")))
                except:
                    warnings.warn(
                        "Assignment extraction failure: existential variable is probably not in CNF")
                    self.execution_error = True
                if (verbose):
                    print("Learned assignment:",
                          self.assignment_to_exists_variables)
                read_optimal_assignment = False
            if (line.startswith("  > Satisfying probability:")):
                self.sol_prob = float(line.split(" ")[-1])
            if (line.startswith("  > Best upper bound:")):
                upper_bound = float(line.split(" ")[-1])
            if (line.startswith("  > Best lower bound:")):
                lower_bound = float(line.split(" ")[-1])
            if (line.startswith("  > Found an optimizing assignment to exist vars:")):
                read_optimal_assignment = True
                # read optimal assignment in the next iteration

        # When conclusive solution is not found
        if (self.sol_prob is None):
            if (find_maximization):
                try:
                    assert upper_bound is not None
                    self.sol_prob = upper_bound
                except:
                    self.execution_error = True
            else:
                try:
                    assert lower_bound is not None
                    self.sol_prob = lower_bound
                except:
                    self.execution_error = True

        if (not find_maximization):
            self.sol_prob = 1 - self.sol_prob
        if (verbose):
            print("Probability:", self.sol_prob)
            print("\n===================================\n")

        # remove formula file
        # os.system("rm " + str(dir_path) + "/" +str(filename))

        return self.execution_error

    def _construct_clause(self, vars):
        self.num_clauses += 1
        clause = ""
        for var in vars:
            clause += str(var) + " "

        clause += "0\n"
        return clause

    def _construct_header(self):
        return "p cnf " + str(self.num_variables) + " " + str(self.num_clauses) + "\n"

    def _z3_clause_from_literals(self, literals):
        """Convert a list of literals to a Z3 clause (disjunction)"""
        if not literals:
            return False

        clause_terms = []
        for lit in literals:
            var = self._get_or_create_var(abs(lit))
            if lit > 0:
                clause_terms.append(var)
            else:
                clause_terms.append(Not(var))

        if len(clause_terms) == 1:
            return clause_terms[0]
        else:
            return Or(*clause_terms)

    def _z3_cnf_from_clauses(self, clauses):
        """Convert a list of clauses to a Z3 CNF formula"""
        if not clauses:
            return True

        cnf_clauses = []
        for clause in clauses:
            cnf_clauses.append(self._z3_clause_from_literals(clause))

        if len(cnf_clauses) == 1:
            return cnf_clauses[0]
        else:
            return And(*cnf_clauses)

    def _create_exactly_one_constraint(self, variables):
        """Create exactly-one constraint using Z3"""
        z3_vars = [self._get_or_create_var(abs(var)) if var > 0 else Not(self._get_or_create_var(abs(var))) for var in
                   variables]

        # Exactly one: at least one AND at most one
        at_least_one = Or(*z3_vars)

        # At most one: for each pair, not both can be true
        at_most_one_constraints = []
        for i in range(len(z3_vars)):
            for j in range(i + 1, len(z3_vars)):
                at_most_one_constraints.append(Not(And(z3_vars[i], z3_vars[j])))

        if at_most_one_constraints:
            at_most_one = And(*at_most_one_constraints)
            return And(at_least_one, at_most_one)
        else:
            return at_least_one

    def _negate_cnf_formula(self, clauses):
        """Negate a CNF formula using Z3 and convert back to CNF"""
        # Convert clauses to Z3 formula
        formula = self._z3_cnf_from_clauses(clauses)

        # Negate the formula
        negated_formula = Not(formula)

        # Simplify and convert to CNF
        simplified = simplify(negated_formula)

        # Convert back to clause representation
        # This is a simplified conversion - for complex formulas,
        # you might need more sophisticated CNF conversion
        return self._z3_to_clause_list(simplified)

    def _z3_to_clause_list(self, formula):
        """Convert Z3 formula back to clause list format"""
        # This is a simplified implementation
        # For complex formulas, you might need Tseitin transformation

        # Use Z3's tactics to convert to CNF
        tactic = Tactic('tseitin-cnf')
        cnf_goal = tactic(formula)

        clauses = []
        for goal in cnf_goal:
            for clause in goal:
                literals = []
                if is_or(clause):
                    for arg in clause.children():
                        literals.append(self._extract_literal(arg))
                else:
                    literals.append(self._extract_literal(clause))
                clauses.append(literals)

        return clauses

    def _extract_literal(self, expr):
        """Extract literal value from Z3 expression"""
        if is_not(expr):
            var_name = str(expr.children()[0])
            var_index = int(var_name.split('_')[1])
            return -var_index
        else:
            var_name = str(expr)
            var_index = int(var_name.split('_')[1])
            return var_index

    def encoding_Enum_SSAT(self, classifier, attributes, sensitive_attributes, auxiliary_variables, probs, filename,
                           sensitive_attributes_assignment, dependency_constraints=[], verbose=True):
        # classifier is assumed to be a boolean formula. It is a 2D list
        # attributes is a list of variables. The last attribute is the sensitive attribute
        # probs is a dictionary containing the i.i.d. probabilities of attributes
        # sensitive attribute is assumed to have the next index after attributes

        self.instance = "Enum"

        self.num_variables = np.array(
            attributes + [abs(_var) for _group in sensitive_attributes for _var in _group] + auxiliary_variables).max()
        self.formula = ""

        # random quantification over non-sensitive attributes
        for attribute in attributes:
            self.formula += self._apply_random_quantification(
                attribute, probs[attribute])

        # the sensitive attribute is exist quantified
        for group in sensitive_attributes:
            for var in group:
                self.formula += self._apply_exist_quantification(var)

        # for each sensitive-attribute (one hot vector), at least one group must be True
        self._formula_for_equal_one_constraints = ""
        equal_one_clauses = []

        for _group in sensitive_attributes:
            if (len(_group) > 1):  # when categorical attribute is not Boolean
                # Create exactly-one constraint using Z3
                exactly_one_formula = self._create_exactly_one_constraint(_group)

                # Convert to CNF clause format
                cnf_clauses = self._z3_to_clause_list(exactly_one_formula)
                equal_one_clauses.extend(cnf_clauses)

                # Update auxiliary variables and num_variables
                max_var_in_constraint = max([abs(lit) for clause in cnf_clauses for lit in clause])
                if max_var_in_constraint > self.num_variables:
                    auxiliary_variables.extend(range(self.num_variables + 1, max_var_in_constraint + 1))
                    self.num_variables = max_var_in_constraint

        # Convert equal_one_clauses to string format
        for clause in equal_one_clauses:
            self._formula_for_equal_one_constraints += self._construct_clause(clause)

        # other variables (auxiliary) are exist quantified
        for var in auxiliary_variables:
            self.formula += self._apply_exist_quantification(var)

        # clauses for the classifier
        for clause in classifier:
            self.formula += self._construct_clause(clause)

        # clauses for the dependency constraints
        for clause in dependency_constraints:
            self.formula += self._construct_clause(clause)

        # append previous constraints. TODO this is crucial as auxiliary variables are added after deriving the constraint.
        self.formula += self._formula_for_equal_one_constraints

        # specify group to measure fairness
        for _var in sensitive_attributes_assignment:
            self.formula += self._construct_clause([_var])

        # store in a file
        self.formula = self._construct_header() + self.formula[:-1]
        file = open(filename, "w")
        file.write(self.formula)
        file.close()

        # if(verbose):
        #     print("SSAT instance ->")
        #     print(self.formula)

    def encoding_Learn_SSAT(self, classifier, attributes, sensitive_attributes, auxiliary_variables, probs, filename,
                            dependency_constraints=[], ignore_sensitive_attribute=None, verbose=True,
                            find_maximization=True, negate_dependency_CNF=False):
        """
        Maximization-minimization algorithm
        """

        self.instance = "Learn"

        # TODO here we call twice: 1) get the sensitive feature with maximum favor, 2) get the sensitive feature with minimum favor
        # classifier is assumed to be a boolean formula. It is a 2D list
        # attributes, sensitive_attributes and auxiliary_variables is a list of variables
        # probs is the list of i.i.d. probabilities of attributes that are not sensitive

        self.num_variables = np.array(
            attributes + [abs(_var) for _group in sensitive_attributes for _var in _group] + auxiliary_variables).max()
        self.formula = ""

        # the sensitive attribute is exist quantified
        for group in sensitive_attributes:
            if (ignore_sensitive_attribute == None or group != ignore_sensitive_attribute):
                for var in group:
                    self.formula += self._apply_exist_quantification(var)

        # random quantification over non-sensitive attributes
        if (len(attributes) > 0):
            for attribute in attributes:
                self.formula += self._apply_random_quantification(
                    attribute, probs[attribute])
        else:
            # dummy random variables
            self.formula += self._apply_random_quantification(
                self.num_variables, 0.5)
            self.num_variables += 1

        if (ignore_sensitive_attribute != None):
            for var in ignore_sensitive_attribute:
                self.formula += self._apply_exist_quantification(var)

        # Negate the classifier
        if (not find_maximization):
            # Use Z3 to negate the classifier
            negated_clauses = self._negate_cnf_formula(classifier)
            classifier = negated_clauses

            # Update auxiliary variables and num_variables
            max_var_in_negated = max(
                [abs(lit) for clause in classifier for lit in clause]) if classifier else self.num_variables
            if max_var_in_negated > self.num_variables:
                auxiliary_variables.extend(range(self.num_variables + 1, max_var_in_negated + 1))
                self.num_variables = max_var_in_negated

        # for each sensitive-attribute (one hot vector), at least one group must be True
        self._formula_for_equal_one_constraints = ""
        equal_one_clauses = []

        for _group in sensitive_attributes:
            if (len(_group) > 1):  # when categorical attribute is not Boolean
                # Create exactly-one constraint using Z3
                exactly_one_formula = self._create_exactly_one_constraint(_group)

                # Convert to CNF clause format
                cnf_clauses = self._z3_to_clause_list(exactly_one_formula)
                equal_one_clauses.extend(cnf_clauses)

                # Update auxiliary variables and num_variables
                max_var_in_constraint = max([abs(lit) for clause in cnf_clauses for lit in clause])
                if max_var_in_constraint > self.num_variables:
                    auxiliary_variables.extend(range(self.num_variables + 1, max_var_in_constraint + 1))
                    self.num_variables = max_var_in_constraint

        # Convert equal_one_clauses to string format
        for clause in equal_one_clauses:
            self._formula_for_equal_one_constraints += self._construct_clause(clause)

        # other variables (auxiliary) are exist quantified
        for var in auxiliary_variables:
            self.formula += self._apply_exist_quantification(var)

        # clauses for the classifier
        for clause in classifier:
            self.formula += self._construct_clause(clause)

        # clauses for the dependency constraints
        for clause in dependency_constraints:
            self.formula += self._construct_clause(clause)

        self.formula += self._formula_for_equal_one_constraints

        # store in a file
        self.formula = self._construct_header() + self.formula[:-1]
        file = open(filename, "w")
        file.write(self.formula)
        file.close()

        # if(verbose):
        #     print("SSAT instance ->")
        #     print(self.formula)