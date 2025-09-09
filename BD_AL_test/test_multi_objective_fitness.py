"""
Test script for Multi-Objective Fitness
Verifies multi-objective problem formulation
"""

import sys
import ast
import numpy as np
from multi_objective_fitness import (
    MultiObjectiveFitness, 
    MultiObjectiveProblem,
    MOFitnessFactory
)
from tree_converter import TreeVisitor


def test_multi_objective_creation():
    """Test creating multi-objective fitness"""
    print("=" * 60)
    print("TEST 1: Multi-Objective Fitness Creation")
    print("=" * 60)
    
    # Load a simple test program
    test_program = "test_programs/minimum.py"
    
    try:
        with open(test_program, 'r') as f:
            lines = f.readlines()
            tree = ast.parse(''.join(lines))
        
        visitor = TreeVisitor()
        visitor.visit(tree)
        
        # Create multi-objective fitness
        mo_fitness = MultiObjectiveFitness(visitor, use_three_objectives=False)
        print("✓ Created dual-objective fitness")
        
        mo_fitness_3obj = MultiObjectiveFitness(visitor, use_three_objectives=True)
        print("✓ Created three-objective fitness")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to create MO fitness: {e}")
        return False


def test_objective_calculation():
    """Test objective calculation"""
    print("\n" + "=" * 60)
    print("TEST 2: Objective Calculation")
    print("=" * 60)
    
    test_program = "test_programs/minimum.py"
    
    try:
        with open(test_program, 'r') as f:
            lines = f.readlines()
            tree = ast.parse(''.join(lines))
        
        visitor = TreeVisitor()
        visitor.visit(tree)
        
        # Create dual-objective fitness
        mo_fitness = MultiObjectiveFitness(visitor, use_three_objectives=False)
        
        # Test with random population
        n_particles = 5
        dimensions = 4
        population = np.random.uniform(-100, 100, (n_particles, dimensions))
        
        # Calculate objectives
        objectives = mo_fitness.multi_objective_fitness(population)
        
        print(f"Population shape: {population.shape}")
        print(f"Objectives shape: {objectives.shape}")
        print(f"✓ Calculated objectives for {n_particles} particles")
        
        # Check objectives
        assert objectives.shape == (n_particles, 2), "Should have 2 objectives"
        
        # Print sample objectives
        print("\nSample objectives (fitness, -coverage):")
        for i in range(min(3, n_particles)):
            fitness = objectives[i, 0]
            neg_coverage = objectives[i, 1]
            coverage = -neg_coverage  # Convert back to positive
            print(f"  Particle {i+1}: fitness={fitness:.4f}, coverage={coverage:.2%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed objective calculation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mo_problem():
    """Test multi-objective problem formulation"""
    print("\n" + "=" * 60)
    print("TEST 3: Multi-Objective Problem")
    print("=" * 60)
    
    test_program = "test_programs/three_number_sort.py"
    
    try:
        with open(test_program, 'r') as f:
            lines = f.readlines()
            tree = ast.parse(''.join(lines))
        
        visitor = TreeVisitor()
        visitor.visit(tree)
        
        # Create MO problem
        mo_fitness = MultiObjectiveFitness(visitor)
        problem = MultiObjectiveProblem(mo_fitness, dimensions=3, n_objectives=2)
        
        print(f"Problem created:")
        print(f"  Variables: {problem.n_var}")
        print(f"  Objectives: {problem.n_obj}")
        print(f"  Lower bound: {problem.xl[0]}")
        print(f"  Upper bound: {problem.xu[0]}")
        
        # Test evaluation
        test_pop = np.random.uniform(-100, 100, (10, 3))
        out = {}
        problem._evaluate(test_pop, out)
        
        assert "F" in out, "Should have objective values"
        assert out["F"].shape == (10, 2), "Should have correct shape"
        
        print(f"✓ Problem evaluation successful")
        print(f"  Output shape: {out['F'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed MO problem test: {e}")
        return False


def test_mo_factory():
    """Test MOFitnessFactory"""
    print("\n" + "=" * 60)
    print("TEST 4: Multi-Objective Factory")
    print("=" * 60)
    
    test_program = "test_programs/bubble_sort.py"
    
    try:
        with open(test_program, 'r') as f:
            lines = f.readlines()
            tree = ast.parse(''.join(lines))
        
        visitor = TreeVisitor()
        visitor.visit(tree)
        
        # Test dual objective creation
        problem_2obj = MOFitnessFactory.create_dual_objective(visitor, dimensions=4)
        print(f"✓ Created dual-objective problem")
        print(f"  Objectives: {problem_2obj.n_obj}")
        
        # Test three objective creation
        problem_3obj = MOFitnessFactory.create_three_objective(visitor, dimensions=4)
        print(f"✓ Created three-objective problem")
        print(f"  Objectives: {problem_3obj.n_obj}")
        
        # Test from config
        config = {
            'dimensions': 4,
            'n_objectives': 2,
            'bounds': (-1000, 1000)
        }
        problem_config = MOFitnessFactory.create_from_config(visitor, config)
        print(f"✓ Created problem from config")
        print(f"  Bounds: [{problem_config.xl[0]}, {problem_config.xu[0]}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed factory test: {e}")
        return False


def test_pareto_metrics():
    """Test Pareto metrics calculation"""
    print("\n" + "=" * 60)
    print("TEST 5: Pareto Metrics")
    print("=" * 60)
    
    test_program = "test_programs/minimum.py"
    
    try:
        with open(test_program, 'r') as f:
            lines = f.readlines()
            tree = ast.parse(''.join(lines))
        
        visitor = TreeVisitor()
        visitor.visit(tree)
        
        # Create MO fitness
        mo_fitness = MultiObjectiveFitness(visitor)
        
        # Calculate objectives for population
        population = np.random.uniform(-50, 50, (20, 4))
        objectives = mo_fitness.multi_objective_fitness(population)
        
        # Get metrics
        metrics = mo_fitness.get_pareto_metrics()
        
        print("Pareto metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        assert 'max_coverage' in metrics
        assert 'min_fitness' in metrics
        assert metrics['max_coverage'] >= 0 and metrics['max_coverage'] <= 1
        
        print("✓ Metrics calculated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed metrics test: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("MULTI-OBJECTIVE FITNESS TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("MO Fitness Creation", test_multi_objective_creation),
        ("Objective Calculation", test_objective_calculation),
        ("MO Problem Formulation", test_mo_problem),
        ("MO Factory", test_mo_factory),
        ("Pareto Metrics", test_pareto_metrics)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Multi-objective fitness is working correctly.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)