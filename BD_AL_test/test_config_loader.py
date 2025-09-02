"""
Test script for Configuration Loader
Verifies YAML configuration loading and validation
"""

import sys
from config_loader import ConfigLoader


def test_config_loading():
    """Test loading all configuration files"""
    print("=" * 60)
    print("TEST 1: Configuration Loading")
    print("=" * 60)
    
    loader = ConfigLoader()
    
    # Test loading algorithms config
    print("\n1. Loading algorithms configuration...")
    try:
        algo_config = loader.load_algorithms()
        print(f"   ✓ Loaded {len(algo_config['algorithms'])} algorithms")
        print(f"   ✓ Loaded {len(algo_config['presets'])} presets")
    except Exception as e:
        print(f"   ✗ Failed to load algorithms config: {e}")
        return False
    
    # Test loading experiments config
    print("\n2. Loading experiments configuration...")
    try:
        exp_config = loader.load_experiments()
        print(f"   ✓ Loaded {len(exp_config['experiments'])} experiments")
        print(f"   ✓ Found global settings: {list(exp_config['global_settings'].keys())}")
    except Exception as e:
        print(f"   ✗ Failed to load experiments config: {e}")
        return False
    
    # Test loading test programs config
    print("\n3. Loading test programs configuration...")
    try:
        prog_config = loader.load_test_programs()
        print(f"   ✓ Loaded {len(prog_config['test_programs'])} test programs")
        print(f"   ✓ Loaded {len(prog_config['test_suites'])} test suites")
    except Exception as e:
        print(f"   ✗ Failed to load test programs config: {e}")
        return False
    
    print("\n✓ All configuration files loaded successfully!")
    return True


def test_algorithm_retrieval():
    """Test retrieving algorithm configurations"""
    print("\n" + "=" * 60)
    print("TEST 2: Algorithm Configuration Retrieval")
    print("=" * 60)
    
    loader = ConfigLoader()
    
    # Test getting specific algorithm config
    print("\n1. Getting PSO configuration...")
    try:
        pso_config = loader.get_algorithm_config("PSO")
        print(f"   Name: {pso_config['name']}")
        print(f"   Type: {pso_config['type']}")
        print(f"   Enabled: {pso_config.get('enabled', True)}")
        print(f"   Default params: {pso_config['default_params']}")
    except Exception as e:
        print(f"   ✗ Failed to get PSO config: {e}")
        return False
    
    # Test getting algorithm parameters
    print("\n2. Getting GA parameters...")
    try:
        ga_params = loader.get_algorithm_params("GA")
        print(f"   Parameters: {ga_params}")
    except Exception as e:
        print(f"   ✗ Failed to get GA params: {e}")
        return False
    
    # Test getting variant parameters
    print("\n3. Getting PSO variant parameters...")
    try:
        pso_aggressive = loader.get_algorithm_params("PSO", "aggressive")
        print(f"   Aggressive variant: {pso_aggressive}")
    except Exception as e:
        print(f"   ✗ Failed to get PSO variant: {e}")
        return False
    
    # Test getting preset algorithms
    print("\n4. Getting preset algorithms...")
    try:
        fast_algos = loader.get_preset_algorithms("fast")
        print(f"   Fast preset: {fast_algos}")
        
        paper_algos = loader.get_preset_algorithms("paper_comparison")
        print(f"   Paper comparison preset: {paper_algos}")
    except Exception as e:
        print(f"   ✗ Failed to get preset: {e}")
        return False
    
    # Test getting enabled algorithms
    print("\n5. Getting enabled algorithms...")
    try:
        enabled = loader.get_enabled_algorithms()
        print(f"   Enabled algorithms: {len(enabled)} total")
        print(f"   List: {', '.join(enabled[:5])}...")
    except Exception as e:
        print(f"   ✗ Failed to get enabled algorithms: {e}")
        return False
    
    print("\n✓ Algorithm retrieval tests passed!")
    return True


def test_experiment_retrieval():
    """Test retrieving experiment configurations"""
    print("\n" + "=" * 60)
    print("TEST 3: Experiment Configuration Retrieval")
    print("=" * 60)
    
    loader = ConfigLoader()
    
    # Test getting experiment config
    print("\n1. Getting quick_test experiment...")
    try:
        quick_test = loader.get_experiment_config("quick_test")
        print(f"   Name: {quick_test['name']}")
        print(f"   Description: {quick_test['description']}")
        print(f"   Algorithms: {quick_test['algorithms']}")
        print(f"   Test programs: {quick_test['test_programs']}")
        print(f"   Repetitions: {quick_test['repetitions']}")
    except Exception as e:
        print(f"   ✗ Failed to get experiment: {e}")
        return False
    
    # Test getting global settings
    print("\n2. Getting global settings...")
    try:
        global_settings = loader.get_global_settings()
        print(f"   Random seeds: {global_settings['random_seeds']}")
        print(f"   Max generations: {global_settings['max_generations']}")
        print(f"   Population size: {global_settings['population_size']}")
    except Exception as e:
        print(f"   ✗ Failed to get global settings: {e}")
        return False
    
    # Test getting termination criteria
    print("\n3. Getting termination criteria...")
    try:
        termination = loader.get_termination_criteria()
        print(f"   Max generations: {termination['max_generations']}")
        print(f"   Target fitness: {termination['target_fitness']}")
        print(f"   Time limit: {termination['time_limit']}s")
    except Exception as e:
        print(f"   ✗ Failed to get termination criteria: {e}")
        return False
    
    # Test getting output config
    print("\n4. Getting output configuration...")
    try:
        output = loader.get_output_config()
        print(f"   Base directory: {output['base_directory']}")
        print(f"   Structure: {output['structure']}")
    except Exception as e:
        print(f"   ✗ Failed to get output config: {e}")
        return False
    
    print("\n✓ Experiment retrieval tests passed!")
    return True


def test_test_program_retrieval():
    """Test retrieving test program configurations"""
    print("\n" + "=" * 60)
    print("TEST 4: Test Program Configuration Retrieval")
    print("=" * 60)
    
    loader = ConfigLoader()
    
    # Test getting specific test program
    print("\n1. Getting bubble_sort configuration...")
    try:
        bubble_sort = loader.get_test_program_config("bubble_sort")
        print(f"   Path: {bubble_sort['path']}")
        print(f"   Dimensions: {bubble_sort['dimensions']}")
        print(f"   Category: {bubble_sort['category']}")
        print(f"   Complexity: {bubble_sort['complexity']}")
    except Exception as e:
        print(f"   ✗ Failed to get test program: {e}")
        return False
    
    # Test getting test suite programs
    print("\n2. Getting test suite programs...")
    try:
        basic_suite = loader.get_test_suite_programs("basic")
        print(f"   Basic suite: {basic_suite}")
        
        games_suite = loader.get_test_suite_programs("games")
        print(f"   Games suite ({len(games_suite)} programs): {games_suite[:3]}...")
    except Exception as e:
        print(f"   ✗ Failed to get test suite: {e}")
        return False
    
    # Test getting all programs suite
    print("\n3. Getting 'all' test suite...")
    try:
        all_programs = loader.get_test_suite_programs("all")
        print(f"   Total programs in 'all' suite: {len(all_programs)}")
    except Exception as e:
        print(f"   ✗ Failed to get all programs: {e}")
        return False
    
    print("\n✓ Test program retrieval tests passed!")
    return True


def test_config_summary():
    """Test configuration summary generation"""
    print("\n" + "=" * 60)
    print("TEST 5: Configuration Summary")
    print("=" * 60)
    
    loader = ConfigLoader()
    
    print("\n1. Generating configuration summary...")
    try:
        summary = loader.get_config_summary()
        print(summary)
    except Exception as e:
        print(f"   ✗ Failed to generate summary: {e}")
        return False
    
    print("\n✓ Configuration summary test passed!")
    return True


def test_error_handling():
    """Test error handling for invalid configurations"""
    print("\n" + "=" * 60)
    print("TEST 6: Error Handling")
    print("=" * 60)
    
    loader = ConfigLoader()
    
    # Test invalid algorithm name
    print("\n1. Testing invalid algorithm name...")
    try:
        loader.get_algorithm_config("InvalidAlgo")
        print("   ✗ Should have raised error for invalid algorithm")
        return False
    except ValueError as e:
        print(f"   ✓ Correctly raised error: {e}")
    
    # Test invalid variant name
    print("\n2. Testing invalid variant name...")
    try:
        loader.get_algorithm_params("PSO", "invalid_variant")
        print("   ✗ Should have raised error for invalid variant")
        return False
    except ValueError as e:
        print(f"   ✓ Correctly raised error: {e}")
    
    # Test invalid preset name
    print("\n3. Testing invalid preset name...")
    try:
        loader.get_preset_algorithms("invalid_preset")
        print("   ✗ Should have raised error for invalid preset")
        return False
    except ValueError as e:
        print(f"   ✓ Correctly raised error: {e}")
    
    # Test invalid experiment name
    print("\n4. Testing invalid experiment name...")
    try:
        loader.get_experiment_config("invalid_experiment")
        print("   ✗ Should have raised error for invalid experiment")
        return False
    except ValueError as e:
        print(f"   ✓ Correctly raised error: {e}")
    
    # Test invalid test program name
    print("\n5. Testing invalid test program name...")
    try:
        loader.get_test_program_config("invalid_program")
        print("   ✗ Should have raised error for invalid program")
        return False
    except ValueError as e:
        print(f"   ✓ Correctly raised error: {e}")
    
    print("\n✓ Error handling tests passed!")
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("CONFIGURATION LOADER TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Algorithm Retrieval", test_algorithm_retrieval),
        ("Experiment Retrieval", test_experiment_retrieval),
        ("Test Program Retrieval", test_test_program_retrieval),
        ("Configuration Summary", test_config_summary),
        ("Error Handling", test_error_handling)
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