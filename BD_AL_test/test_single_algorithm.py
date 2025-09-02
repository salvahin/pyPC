#!/usr/bin/env python3
"""
Test script for Single Algorithm Runner
Verifies the runner works with a quick test
"""

import sys
from run_single_algorithm import SingleAlgorithmRunner


def test_quick_run():
    """Test a quick run with PSO on subset of programs"""
    print("=" * 80)
    print("TESTING SINGLE ALGORITHM RUNNER")
    print("=" * 80)
    
    # Create runner with minimal settings for testing
    runner = SingleAlgorithmRunner(
        algorithm_name="PSO",
        variant=None,
        custom_params={"pop_size": 20},  # Small population for speed
        num_runs=2,  # Two runs to test statistics
        max_generations=10,  # Few generations for quick test
        verbose=True
    )
    
    # Override test programs to use only a subset for testing
    runner.test_programs = {
        "minimum": runner.test_programs["minimum"],
        "three_number_sort": runner.test_programs["three_number_sort"]
    }
    
    print(f"\nTesting {runner.algorithm_name} with:")
    print(f"  Population size: 20")
    print(f"  Generations: 10")
    print(f"  Runs per program: 2")
    print(f"  Test programs: minimum, three_number_sort")
    
    try:
        # Run tests
        results = runner.run_all_tests()
        
        # Check results
        assert len(results) == 2, "Should have results for 2 programs"
        
        for prog_name, runs in results.items():
            assert len(runs) == 2, f"Should have 2 runs for {prog_name}"
            
            for run in runs:
                assert run.best_fitness < float('inf'), "Should have valid fitness"
                assert run.final_coverage >= 0, "Should have valid coverage"
                assert run.execution_time > 0, "Should have execution time"
                assert len(run.convergence_history) > 0, "Should have convergence history"
        
        # Print summary
        runner.print_summary(results)
        
        # Test metrics statistics
        stats = runner.metrics.get_statistics()
        assert "METRICS SUMMARY" in stats
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_algorithm_variants():
    """Test running with algorithm variants"""
    print("\n" + "=" * 80)
    print("TESTING ALGORITHM VARIANTS")
    print("=" * 80)
    
    # Test PSO aggressive variant
    runner = SingleAlgorithmRunner(
        algorithm_name="PSO",
        variant="aggressive",
        num_runs=1,
        max_generations=5,
        verbose=True
    )
    
    # Use only one test program
    runner.test_programs = {"minimum": runner.test_programs["minimum"]}
    
    print(f"\nTesting PSO aggressive variant on minimum.py")
    
    try:
        results = runner.run_all_tests()
        assert len(results) == 1
        assert results["minimum"][0].algorithm_name == "PSO_aggressive"
        
        print("✓ Variant test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Variant test failed: {e}")
        return False


def test_different_algorithms():
    """Test different algorithms"""
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT ALGORITHMS")
    print("=" * 80)
    
    algorithms = ["GA", "DE", "NelderMead"]
    test_program = {"minimum": None}  # Will be loaded from config
    
    all_passed = True
    
    for algo_name in algorithms:
        print(f"\nTesting {algo_name}...")
        
        try:
            runner = SingleAlgorithmRunner(
                algorithm_name=algo_name,
                custom_params={"pop_size": 10} if algo_name != "NelderMead" else {},
                num_runs=1,
                max_generations=5,
                verbose=False
            )
            
            runner.test_programs = {"minimum": runner.test_programs["minimum"]}
            results = runner.run_all_tests()
            
            assert len(results) == 1
            assert results["minimum"][0].best_fitness < float('inf')
            
            print(f"  ✓ {algo_name} works correctly")
            
        except Exception as e:
            print(f"  ✗ {algo_name} failed: {e}")
            all_passed = False
    
    return all_passed


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("SINGLE ALGORITHM RUNNER TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Quick Run Test", test_quick_run),
        ("Algorithm Variants", test_algorithm_variants),
        ("Different Algorithms", test_different_algorithms)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nRunning: {name}")
        print("-" * 40)
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"Test '{name}' crashed: {e}")
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
        print("\n🎉 All tests passed! The single algorithm runner is working correctly.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)