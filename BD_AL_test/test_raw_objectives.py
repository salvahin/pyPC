import numpy as np
import ast
from multi_objective_fitness import MultiObjectiveFitness
from tree_converter import TreeVisitor

# Load test program
path = "test_programs/minimum.py"
with open(path, 'r') as f:
    lines = f.readlines()
    tree = ast.parse(''.join(lines))

visitor = TreeVisitor()
visitor.visit(tree)

# Create fitness without normalization
mo_fitness = MultiObjectiveFitness(visitor, objective_type='conflicting', normalize_objectives=False)

# Test with random solutions
X = np.random.random((5, 4)) * 10 - 5

objectives = mo_fitness.multi_objective_fitness(X)
print(f"Raw objectives shape: {objectives.shape}")
print(f"Raw objectives:\n{objectives}")
print(f"\nCoverage values: {mo_fitness.population_coverage}")
print(f"Complexity values: {mo_fitness.population_complexity}")
