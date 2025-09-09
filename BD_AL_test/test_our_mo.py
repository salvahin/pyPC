import numpy as np
import ast
from multi_objective_fitness import MOFitnessFactory
from tree_converter import TreeVisitor

# Load a simple test program
path = "test_programs/minimum.py"
with open(path, 'r') as f:
    lines = f.readlines()
    tree = ast.parse(''.join(lines))

visitor = TreeVisitor()
visitor.visit(tree)

# Create problem
problem = MOFitnessFactory.create_dual_objective(
    visitor, 4, objective_type='conflicting'
)

print(f"Problem created: {problem}")
print(f"n_var: {problem.n_var}, n_obj: {problem.n_obj}")

# Test evaluation
X = np.random.random((5, 4)) * 10 - 5  # 5 solutions
out = {}
problem._evaluate(X, out)

print(f"Objectives shape: {out['F'].shape}")
print(f"Sample objectives: {out['F'][:3]}")
