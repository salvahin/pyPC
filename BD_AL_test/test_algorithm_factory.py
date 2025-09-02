"""
Test script for Algorithm Factory
Verifies algorithm creation from configuration
"""

import sys
from algorithm_factory import AlgorithmFactory
from config_loader import ConfigLoader


def test_single_algorithm_creation():
    """Test creating single algorithms"""
    print("=" * 60)
    print("TEST 1: Single Algorithm Creation")
    print("=" * 60)
    
    factory = AlgorithmFactory()
    
    # Test creating PSO
    print("\n1. Creating PSO from configuration...")
    try:
        pso = factory.create_algorithm("PSO")
        print(f"   ✓ PSO created: {type(pso).__name__}")
    except Exception as e:
        print(f"   ✗ Failed to create PSO: {e}")
        return False
    
    # Test creating GA with variant
    print("\n2. Creating GA with large_pop variant...")
    try:
        ga = factory.create_algorithm("GA", variant="large_pop")
        print(f"   ✓ GA created with variant: {type(ga).__name__}")
    except Exception as e:
        print(f"   ✗ Failed to create GA variant: {e}")
        return False
    
    # Test creating DE with custom params
    print("\n3. Creating DE with custom parameters...")
    try:
        de = factory.create_algorithm("DE", custom_params={"CR": 0.9, "F": 0.8})
        print(f"   ✓ DE created with custom params: {type(de).__name__}")
    except Exception as e:
        print(f"   ✗ Failed to create DE: {e}")
        return False
    
    # Test creating CMAES (requires dimensions)
    print("\n4. Creating CMAES with dimensions...")
    try:
        cmaes = factory.create_algorithm("CMAES", dimensions=5)
        print(f"   ✓ CMAES created: {type(cmaes).__name__}")
    except Exception as e:
        print(f"   ✗ Failed to create CMAES: {e}")
        return False
    
    print("\n✓ Single algorithm creation tests passed!")
    return True


def test_experiment_algorithms():
    """Test creating algorithms from experiment config"""
    print("\n" + "=" * 60)
    print("TEST 2: Experiment Algorithm Creation")
    print("=" * 60)
    
    factory = AlgorithmFactory()
    
    # Test quick_test experiment
    print("\n1. Creating algorithms for 'quick_test' experiment...")
    try:
        algorithms = factory.create_from_experiment("quick_test", dimensions=4)
        print(f"   ✓ Created {len(algorithms)} algorithms")
        for name, algo in algorithms:
            print(f"     - {name}: {type(algo).__name__}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test parameter_sensitivity experiment (has multiple configs)
    print("\n2. Creating algorithms for 'parameter_sensitivity' experiment...")
    try:
        algorithms = factory.create_from_experiment("parameter_sensitivity", dimensions=4)
        print(f"   ✓ Created {len(algorithms)} algorithm configurations")
        for name, algo in algorithms[:5]:  # Show first 5
            print(f"     - {name}: {type(algo).__name__}")
        if len(algorithms) > 5:
            print(f"     ... and {len(algorithms) - 5} more")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n✓ Experiment algorithm creation tests passed!")
    return True


def test_preset_creation():
    """Test creating algorithms from presets"""
    print("\n" + "=" * 60)
    print("TEST 3: Preset Algorithm Creation")
    print("=" * 60)
    
    factory = AlgorithmFactory()
    
    # Test fast preset
    print("\n1. Creating algorithms from 'fast' preset...")
    try:
        algorithms = factory.create_from_preset("fast", dimensions=3)
        print(f"   ✓ Created {len(algorithms)} algorithms from fast preset")
        for name, algo in algorithms:
            print(f"     - {name}: {type(algo).__name__}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test paper_comparison preset
    print("\n2. Creating algorithms from 'paper_comparison' preset...")
    try:
        algorithms = factory.create_from_preset("paper_comparison", dimensions=4)
        print(f"   ✓ Created {len(algorithms)} algorithms from paper_comparison preset")
        for name, algo in algorithms:
            print(f"     - {name}: {type(algo).__name__}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n✓ Preset creation tests passed!")
    return True


def test_variant_creation():
    """Test creating all variants of an algorithm"""
    print("\n" + "=" * 60)
    print("TEST 4: Algorithm Variant Creation")
    print("=" * 60)
    
    factory = AlgorithmFactory()
    
    # Test PSO variants
    print("\n1. Creating all PSO variants...")
    try:
        algorithms = factory.create_with_variants("PSO", dimensions=3)
        print(f"   ✓ Created {len(algorithms)} PSO variants")
        for name, algo in algorithms:
            print(f"     - {name}: {type(algo).__name__}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test DE variants
    print("\n2. Creating all DE variants...")
    try:
        algorithms = factory.create_with_variants("DE", dimensions=4)
        print(f"   ✓ Created {len(algorithms)} DE variants")
        for name, algo in algorithms:
            print(f"     - {name}: {type(algo).__name__}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n✓ Variant creation tests passed!")
    return True


def test_population_scaling():
    """Test creating algorithms with different population sizes"""
    print("\n" + "=" * 60)
    print("TEST 5: Population Size Scaling")
    print("=" * 60)
    
    factory = AlgorithmFactory()
    
    print("\n1. Creating GA with different population sizes...")
    try:
        pop_sizes = [20, 50, 100, 200]
        algorithms = factory.create_population_scaled("GA", pop_sizes, dimensions=3)
        print(f"   ✓ Created {len(algorithms)} GA instances with different pop sizes")
        for name, algo in algorithms:
            print(f"     - {name}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n✓ Population scaling test passed!")
    return True


def test_compatibility_checking():
    """Test algorithm compatibility with test programs"""
    print("\n" + "=" * 60)
    print("TEST 6: Algorithm Compatibility Checking")
    print("=" * 60)
    
    factory = AlgorithmFactory()
    
    # Test compatibility with bubble_sort (4 dimensions)
    print("\n1. Getting compatible algorithms for 'bubble_sort'...")
    try:
        compatible = factory.get_compatible_algorithms("bubble_sort")
        print(f"   ✓ Found {len(compatible)} compatible algorithms")
        print(f"   Compatible: {', '.join(compatible[:5])}...")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test compatibility with minimum (4 dimensions)
    print("\n2. Getting compatible algorithms for 'minimum'...")
    try:
        compatible = factory.get_compatible_algorithms("minimum")
        print(f"   ✓ Found {len(compatible)} compatible algorithms")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n✓ Compatibility checking tests passed!")
    return True


def test_batch_creation():
    """Test batch algorithm creation"""
    print("\n" + "=" * 60)
    print("TEST 7: Batch Algorithm Creation")
    print("=" * 60)
    
    factory = AlgorithmFactory()
    
    print("\n1. Creating batch from configuration list...")
    
    configurations = [
        {"name": "PSO", "label": "PSO_standard"},
        {"name": "PSO", "variant": "aggressive", "label": "PSO_aggressive"},
        {"name": "GA", "params": {"pop_size": 150}, "label": "GA_150"},
        {"name": "DE", "variant": "best1bin", "label": "DE_best"},
        {"name": "CMAES", "dimensions": 5, "label": "CMAES_5D"}
    ]
    
    try:
        algorithms = factory.batch_create(configurations)
        print(f"   ✓ Created {len(algorithms)} algorithms from batch")
        for name, algo in algorithms:
            print(f"     - {name}: {type(algo).__name__}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n✓ Batch creation test passed!")
    return True


def test_validation():
    """Test configuration validation"""
    print("\n" + "=" * 60)
    print("TEST 8: Configuration Validation")
    print("=" * 60)
    
    factory = AlgorithmFactory()
    
    # Test valid configuration
    print("\n1. Validating valid PSO configuration...")
    try:
        valid, errors = factory.validate_algorithm_config(
            "PSO", 
            {"pop_size": 100, "w": 0.7, "c1": 2.0, "c2": 2.0}
        )
        if valid:
            print("   ✓ Configuration is valid")
        else:
            print(f"   ✗ Unexpected validation failure: {errors}")
            return False
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test invalid configuration (wrong type)
    print("\n2. Validating invalid configuration (type mismatch)...")
    try:
        valid, errors = factory.validate_algorithm_config(
            "PSO",
            {"pop_size": "not_a_number", "w": 0.7}
        )
        if not valid:
            print(f"   ✓ Correctly detected invalid config: {errors[0]}")
        else:
            print("   ✗ Should have detected type mismatch")
            return False
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n✓ Validation tests passed!")
    return True


def test_algorithm_info():
    """Test getting algorithm information"""
    print("\n" + "=" * 60)
    print("TEST 9: Algorithm Information")
    print("=" * 60)
    
    factory = AlgorithmFactory()
    
    print("\n1. Getting PSO information...")
    try:
        info = factory.get_algorithm_info("PSO")
        print("   ✓ PSO Information:")
        for line in info.split("\n"):
            print(f"     {line}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n✓ Algorithm info test passed!")
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("ALGORITHM FACTORY TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Single Algorithm Creation", test_single_algorithm_creation),
        ("Experiment Algorithms", test_experiment_algorithms),
        ("Preset Creation", test_preset_creation),
        ("Variant Creation", test_variant_creation),
        ("Population Scaling", test_population_scaling),
        ("Compatibility Checking", test_compatibility_checking),
        ("Batch Creation", test_batch_creation),
        ("Configuration Validation", test_validation),
        ("Algorithm Information", test_algorithm_info)
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