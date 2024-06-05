import numpy as np
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2


class CustomProblem(Problem):
    def __init__(self, number_of_attributes, protected_indices, highest_value):
        super().__init__(
            n_var=number_of_attributes * 2,  # x1 and x2 concatenated
            n_obj=6,  # Number of objectives
            n_constr=2,  # Number of constraints
            xl=np.full((number_of_attributes * 2,), -1),  # Lower bounds for each variable
            xu=np.full((number_of_attributes * 2,), highest_value),  # Upper bounds for each variable
            type_var=np.int  # Ensure variables are treated as discrete
        )
        self.protected_indices = protected_indices
        self.K = number_of_attributes

    def _evaluate(self, x, out, *args, **kwargs):
        x1 = x[:self.K]
        x2 = x[self.K:]

        # Constraints
        c1 = np.all([not (x1[i] == -1 and x2[i] == -1) for i in self.protected_indices])
        c2 = np.all([(x1[i] == -1) == (x2[i] == -1) for i in range(self.K)])

        feasible = c1 and c2
        out["G"] = [-1 if c1 else 0, -1 if c2 else 0]  # Constraint handling

        if feasible:
            # Objectives (dummy implementations)
            aleatoric = np.random.rand()  # Dummy aleatoric uncertainty function
            epistemic = np.random.rand()  # Dummy epistemic uncertainty function
            group_size = np.sum(x1 != -1) + np.sum(x2 != -1)  # Example group size computation
            magnitude = np.abs(np.sum(x1) - np.sum(x2))  # Example magnitude computation
            granularity = np.var(np.concatenate((x1[x1 != -1], x2[x2 != -1])))  # Example granularity
            intersectionality = np.sum((x1 == x2) & (x1 != -1))  # Example intersectionality

            out["F"] = [-aleatoric, -epistemic, group_size, magnitude, granularity, intersectionality]
        else:
            out["F"] = [np.inf, np.inf, -np.inf, -np.inf, -np.inf, -np.inf]


# Example usage
problem = CustomProblem(number_of_attributes=10, protected_indices=[0, 1, 2], highest_value=10)
algorithm = NSGA2(pop_size=100, sampling=get_sampling("int_random"), crossover=get_crossover("int_sbx"),
                  mutation=get_mutation("int_pm"))
res = minimize(problem, algorithm, ('n_gen', 50), verbose=True)

print("Best solution found:\nX = %s\nF = %s" % (res.X, res.F))
