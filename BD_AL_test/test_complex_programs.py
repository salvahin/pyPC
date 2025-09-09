#!/usr/bin/env python3
"""
Test the new complex programs with multi-objective optimization
"""

import numpy as np
import ast
from multi_objective_fitness import MultiObjectiveFitness
from tree_converter import TreeVisitor
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from multi_objective_fitness import MOFitnessFactory

def test_program(program_path, program_name):
    print(f"\n{'='*60}")
    print(f"Testing: {program_name}")
    print('='*60)
    
    # Load and parse the program
    with open(program_path, 'r') as f:
        lines = f.readlines()
        tree = ast.parse(''.join(lines))
    
    visitor = TreeVisitor()
    visitor.visit(tree)
    
    # Create multi-objective fitness evaluator
    mo_fitness = MultiObjectiveFitness(visitor, objective_type='conflicting')
    
    # Test with random solutions
    n_solutions = 20
    n_vars = 4
    X = np.random.random((n_solutions, n_vars)) * 10 - 5
    
    # Evaluate objectives
    objectives = mo_fitness.multi_objective_fitness(X)
    
    # Analyze results
    coverages = list(mo_fitness.population_coverage.values()) if mo_fitness.population_coverage else [0]
    complexities = list(mo_fitness.population_complexity.values()) if mo_fitness.population_complexity else [0]
    
    print(f"\nProgram Statistics:")
    print(f"  - Total nodes: {len(mo_fitness.walked_tree) if mo_fitness.walked_tree else 0}")
    print(f"  - Unique paths: {len(mo_fitness.unique_paths) if hasattr(mo_fitness, 'unique_paths') else 0}")
    
    print(f"\nCoverage Results:")
    print(f"  - Min coverage: {min(coverages):.2%}")
    print(f"  - Max coverage: {max(coverages):.2%}")
    print(f"  - Mean coverage: {np.mean(coverages):.2%}")
    print(f"  - Std coverage: {np.std(coverages):.2%}")
    
    print(f"\nComplexity Results:")
    print(f"  - Min complexity: {min(complexities):.3f}")
    print(f"  - Max complexity: {max(complexities):.3f}")
    print(f"  - Mean complexity: {np.mean(complexities):.3f}")
    print(f"  - Std complexity: {np.std(complexities):.3f}")
    
    # Check for trade-offs
    high_coverage_solutions = [i for i, c in enumerate(coverages) if c > 0.8]
    if high_coverage_solutions:
        high_cov_complexities = [complexities[i] for i in high_coverage_solutions]
        print(f"\nHigh Coverage Solutions (>80%):")
        print(f"  - Count: {len(high_coverage_solutions)}")
        print(f"  - Complexity range: [{min(high_cov_complexities):.3f}, {max(high_cov_complexities):.3f}]")
    
    # Run NSGA-II to find Pareto front
    print(f"\nRunning NSGA-II optimization...")
    problem = MOFitnessFactory.create_dual_objective(
        visitor, n_vars, objective_type='conflicting'
    )
    
    algorithm = NSGA2(pop_size=50)
    result = minimize(
        problem,
        algorithm,
        ('n_gen', 50),
        verbose=False
    )
    
    if result.F is not None and len(result.F) > 0:
        print(f"\nPareto Front:")
        print(f"  - Solutions found: {len(result.F)}")
        
        # Analyze Pareto front diversity
        pareto_coverages = result.F[:, 0]  # Note: negated for minimization
        pareto_complexities = result.F[:, 1]
        
        print(f"  - Coverage range: [{-max(pareto_coverages):.2%}, {-min(pareto_coverages):.2%}]")
        print(f"  - Complexity range: [{min(pareto_complexities):.3f}, {max(pareto_complexities):.3f}]")
        
        # Check if we have meaningful trade-offs
        coverage_spread = max(pareto_coverages) - min(pareto_coverages)
        complexity_spread = max(pareto_complexities) - min(pareto_complexities)
        
        print(f"\nTrade-off Analysis:")
        print(f"  - Coverage spread: {coverage_spread:.3f}")
        print(f"  - Complexity spread: {complexity_spread:.3f}")
        
        if coverage_spread > 0.1 and complexity_spread > 0.5:
            print("  ✓ Meaningful trade-offs detected!")
        else:
            print("  ✗ Limited trade-offs (program may still be too simple)")
    
    return {
        'program': program_name,
        'total_nodes': len(visitor.walked_tree),
        'max_coverage': max(coverages),
        'coverage_std': np.std(coverages),
        'pareto_size': len(result.F) if result.F is not None else 0,
        'has_tradeoffs': coverage_spread > 0.1 and complexity_spread > 0.5 if result.F is not None and len(result.F) > 0 else False
    }

def main():
    programs = [
        ('test_programs/deep_branching.py', 'Deep Branching'),
        ('test_programs/complex_conditions.py', 'Complex Conditions'),
        ('test_programs/minimum.py', 'Minimum (baseline)'),
        ('test_programs/bubble_sort.py', 'Bubble Sort (baseline)'),
    ]
    
    results = []
    for path, name in programs:
        try:
            result = test_program(path, name)
            results.append(result)
        except Exception as e:
            print(f"Error testing {name}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for r in results:
        print(f"\n{r['program']}:")
        print(f"  Nodes: {r['total_nodes']}, Max Coverage: {r['max_coverage']:.2%}")
        print(f"  Pareto Size: {r['pareto_size']}, Has Trade-offs: {r['has_tradeoffs']}")

if __name__ == '__main__':
    main()