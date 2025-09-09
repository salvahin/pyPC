#!/usr/bin/env python3
"""
Simple single test runner for multi-objective optimization
"""

import numpy as np
import ast
import time
from multi_objective_fitness import MOFitnessFactory
from tree_converter import TreeVisitor
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

def run_test(program_path, program_name, n_generations=30, pop_size=30):
    """Run NSGA2 on a single test program"""
    
    print(f"Testing: {program_name}")
    print("="*50)
    
    # Load and parse program
    with open(program_path, 'r') as f:
        tree = ast.parse(f.read())
    
    visitor = TreeVisitor()
    visitor.visit(tree)
    
    # Create problem
    problem = MOFitnessFactory.create_dual_objective(
        visitor, 4, objective_type='conflicting'
    )
    
    # Create algorithm
    algorithm = NSGA2(pop_size=pop_size)
    
    # Track progress
    def callback_func(algorithm):
        if algorithm.n_gen % 10 == 0:
            F = algorithm.pop.get("F")
            if len(F) > 0:
                best_coverage = -np.min(F[:, 0])  # First objective is negated coverage
                avg_complexity = np.mean(F[:, 1])
                print(f"Gen {algorithm.n_gen}: Best coverage: {best_coverage:.2%}, "
                      f"Avg complexity: {avg_complexity:.3f}")
    
    # Run optimization
    start_time = time.time()
    
    result = minimize(
        problem,
        algorithm,
        ('n_gen', n_generations),
        callback=callback_func,
        verbose=False
    )
    
    elapsed = time.time() - start_time
    
    # Analyze results
    if result.F is not None and len(result.F) > 0:
        pareto_front = result.F
        
        # Extract metrics
        coverages = -pareto_front[:, 0]  # Convert back from negative
        complexities = pareto_front[:, 1]
        
        print(f"\nResults:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Pareto solutions: {len(pareto_front)}")
        print(f"  Coverage range: [{min(coverages):.2%}, {max(coverages):.2%}]")
        print(f"  Complexity range: [{min(complexities):.3f}, {max(complexities):.3f}]")
        
        # Show best solutions
        print(f"\nBest solutions:")
        best_cov_idx = np.argmax(coverages)
        print(f"  Best coverage: {coverages[best_cov_idx]:.2%} "
              f"(complexity: {complexities[best_cov_idx]:.3f})")
        
        best_comp_idx = np.argmin(complexities)
        print(f"  Best complexity: {complexities[best_comp_idx]:.3f} "
              f"(coverage: {coverages[best_comp_idx]:.2%})")
        
        # Check for trade-offs
        cov_spread = max(coverages) - min(coverages)
        comp_spread = max(complexities) - min(complexities)
        
        if cov_spread > 0.1 and comp_spread > 0.2:
            print("\n✓ Meaningful trade-offs detected!")
        else:
            print("\n✗ Limited trade-offs")
    else:
        print("No solutions found!")

if __name__ == "__main__":
    # Test programs
    test_programs = [
        ("test_programs/minimum.py", "Minimum (baseline)"),
        ("test_programs/bubble_sort.py", "Bubble Sort"),
        ("test_programs/complex_conditions.py", "Complex Conditions"),
    ]
    
    for path, name in test_programs:
        try:
            run_test(path, name)
            print()
        except Exception as e:
            print(f"Error testing {name}: {e}")
            print()