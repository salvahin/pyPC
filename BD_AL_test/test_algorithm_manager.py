"""
Test script for AlgorithmManager
Verifies that the algorithm manager works correctly with existing algorithms
"""

import sys
import numpy as np
from algorithm_manager import AlgorithmManager, AlgorithmType
from test_fitness import Fitness
from tree_converter import TreeVisitor
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import ast


class TestFitnessProblem(Problem):
    """Test problem for algorithm validation"""
    
    def __init__(self, fitness=None, dimensions=None):
        xl = -999999
        xu = 999999
        self.fitness = fitness
        self.ndim = dimensions
        super().__init__(n_var=dimensions, n_obj=1, xl=xl, xu=xu)
    
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.fitness.fitness_function(x)


def test_algorithm_creation():
    """Test creating algorithm instances"""
    print("=" * 60)
    print("TEST 1: Algorithm Creation")
    print("=" * 60)
    
    manager = AlgorithmManager()
    
    # Test PSO creation
    print("\n1. Testing PSO creation...")
    try:
        pso = manager.get_algorithm("PSO")
        print("   ✓ PSO created successfully")
        print(f"   Algorithm type: {type(pso)}")
    except Exception as e:
        print(f"   ✗ Failed to create PSO: {e}")
        return False
    
    # Test GA creation
    print("\n2. Testing GA creation...")
    try:
        ga = manager.get_algorithm("GA")
        print("   ✓ GA created successfully")
        print(f"   Algorithm type: {type(ga)}")
    except Exception as e:
        print(f"   ✗ Failed to create GA: {e}")
        return False
    
    # Test DE creation
    print("\n3. Testing DE creation...")
    try:
        de = manager.get_algorithm("DE")
        print("   ✓ DE created successfully")
        print(f"   Algorithm type: {type(de)}")
    except Exception as e:
        print(f"   ✗ Failed to create DE: {e}")
        return False
    
    # Test with custom parameters
    print("\n4. Testing PSO with custom parameters...")
    try:
        custom_pso = manager.get_algorithm("PSO", {"pop_size": 50, "w": 0.9})
        print("   ✓ PSO with custom params created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create PSO with custom params: {e}")
        return False
    
    print("\n✓ All algorithm creation tests passed!")
    return True


def test_algorithm_listing():
    """Test algorithm listing and filtering"""
    print("\n" + "=" * 60)
    print("TEST 2: Algorithm Listing and Filtering")
    print("=" * 60)
    
    manager = AlgorithmManager()
    
    # List all algorithms
    print("\n1. All available algorithms:")
    all_algos = manager.list_algorithms()
    print(f"   Found {len(all_algos)} algorithms: {', '.join(all_algos)}")
    
    # List by type
    print("\n2. Algorithms by type:")
    for algo_type in AlgorithmType:
        algos = manager.list_algorithms(algo_type)
        if algos:
            print(f"   {algo_type.value}: {', '.join(algos)}")
    
    # Test compatibility checking
    print("\n3. Compatible algorithms for 3D unconstrained problem:")
    compatible = manager.get_compatible_algorithms(dimensions=3, has_constraints=False)
    print(f"   Compatible: {', '.join(compatible)}")
    
    print("\n✓ Algorithm listing tests passed!")
    return True


def test_algorithm_with_problem():
    """Test algorithms with actual fitness problem"""
    print("\n" + "=" * 60)
    print("TEST 3: Algorithm Integration with Fitness Problem")
    print("=" * 60)
    
    manager = AlgorithmManager()
    
    # Load a simple test program
    test_program = "test_programs/minimum.py"
    print(f"\n1. Loading test program: {test_program}")
    
    try:
        with open(test_program, 'r') as f:
            lines = f.readlines()
            tree = ast.parse(''.join(lines))
        
        visitor = TreeVisitor()
        visitor.visit(tree)
        fitness = Fitness(visitor)
        dimensions = 4  # minimum.py uses 4 dimensions
        
        print("   ✓ Test program loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load test program: {e}")
        return False
    
    # Test PSO with the problem
    print("\n2. Testing PSO with fitness problem...")
    try:
        pso = manager.get_algorithm("PSO", {"pop_size": 20})
        problem = TestFitnessProblem(fitness, dimensions)
        
        result = minimize(
            problem,
            pso,
            ('n_gen', 10),  # Just 10 generations for testing
            verbose=False
        )
        
        print(f"   ✓ PSO optimization completed")
        print(f"   Best fitness: {result.F[0]:.6f}")
        print(f"   Best solution: {result.X}")
    except Exception as e:
        print(f"   ✗ PSO optimization failed: {e}")
        return False
    
    # Test GA with the problem
    print("\n3. Testing GA with fitness problem...")
    try:
        ga = manager.get_algorithm("GA", {"pop_size": 20})
        problem = TestFitnessProblem(fitness, dimensions)
        
        result = minimize(
            problem,
            ga,
            ('n_gen', 10),
            verbose=False
        )
        
        print(f"   ✓ GA optimization completed")
        print(f"   Best fitness: {result.F[0]:.6f}")
        print(f"   Best solution: {result.X}")
    except Exception as e:
        print(f"   ✗ GA optimization failed: {e}")
        return False
    
    print("\n✓ Algorithm integration tests passed!")
    return True


def test_batch_creation():
    """Test batch algorithm creation"""
    print("\n" + "=" * 60)
    print("TEST 4: Batch Algorithm Creation")
    print("=" * 60)
    
    manager = AlgorithmManager()
    
    print("\n1. Creating batch of algorithms...")
    algorithm_names = ["PSO", "GA", "DE", "NelderMead"]
    
    try:
        batch = manager.create_batch(algorithm_names, {"pop_size": 30})
        print(f"   ✓ Created batch with {len(batch)} algorithms")
        
        for name, algo in batch:
            print(f"   - {name}: {type(algo).__name__}")
    except Exception as e:
        print(f"   ✗ Failed to create batch: {e}")
        return False
    
    print("\n✓ Batch creation test passed!")
    return True


def test_algorithm_info():
    """Test algorithm information retrieval"""
    print("\n" + "=" * 60)
    print("TEST 5: Algorithm Information")
    print("=" * 60)
    
    manager = AlgorithmManager()
    
    print("\n1. Getting algorithm info for PSO...")
    try:
        info = manager.get_algorithm_info("PSO")
        print(f"   Name: {info.name}")
        print(f"   Type: {info.type.value}")
        print(f"   Description: {info.description}")
        print(f"   Reference: {info.reference}")
        print(f"   Default params: {info.params}")
    except Exception as e:
        print(f"   ✗ Failed to get info: {e}")
        return False
    
    print("\n2. Algorithm summary:")
    print(manager.get_algorithm_summary())
    
    print("\n✓ Algorithm info test passed!")
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("ALGORITHM MANAGER TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Algorithm Creation", test_algorithm_creation),
        ("Algorithm Listing", test_algorithm_listing),
        ("Problem Integration", test_algorithm_with_problem),
        ("Batch Creation", test_batch_creation),
        ("Algorithm Info", test_algorithm_info)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with error: {e}")
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
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)