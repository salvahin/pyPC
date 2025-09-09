#!/usr/bin/env python3
import numpy as np
from algorithm_factory import AlgorithmFactory
from config_loader import ConfigLoader
from multi_objective_fitness import MOFitnessFactory
from tree_converter import TreeVisitor
import ast

# Load configs
cl = ConfigLoader()
cl.load_all()
af = AlgorithmFactory(cl)

# Create algorithm
algo = af.create_algorithm('NSGA2', custom_params={'pop_size': 20})
print(f"Algorithm type: {type(algo)}")
print(f"Has pop: {hasattr(algo, 'pop')}")

# Load a test program
path = "test_programs/minimum.py"
with open(path, 'r') as f:
    lines = f.readlines()
    tree = ast.parse(''.join(lines))

visitor = TreeVisitor()
visitor.visit(tree)

# Create problem with conflicting objectives
problem = MOFitnessFactory.create_dual_objective(
    visitor, 4, objective_type='conflicting'
)

print(f"Problem created: {problem}")
print(f"Problem n_obj: {problem.n_obj}")

# Try to run
from pymoo.optimize import minimize

try:
    result = minimize(
        problem,
        algo,
        ('n_gen', 5),
        verbose=False
    )
    print(f"Success! Result: {result.F.shape}")
except Exception as e:
    print(f"Error: {e}")
