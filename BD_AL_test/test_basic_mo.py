import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

class SimpleProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=-5, xu=5)
    
    def _evaluate(self, X, out, *args, **kwargs):
        f1 = X[:, 0] ** 2
        f2 = (X[:, 0] - 2) ** 2 + X[:, 1] ** 2
        out["F"] = np.column_stack([f1, f2])

# Create and solve
problem = SimpleProblem()
algorithm = NSGA2(pop_size=20)

result = minimize(
    problem,
    algorithm,
    ('n_gen', 10),
    verbose=False
)

print(f"Pareto solutions found: {len(result.F)}")
print(f"Sample objectives: {result.F[:3]}")
