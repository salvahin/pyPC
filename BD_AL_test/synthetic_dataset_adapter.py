#!/usr/bin/env python3
"""
Synthetic Dataset Adapter
Bridges between synthetic functions and evaluation framework
Handles complex parameter types and execution safety
"""

import ast
import importlib.util
import time
import traceback
import threading
import signal
import os
import sys
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import yaml
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from synthetic_baseline_generators import SyntheticInputGenerator, TestSuite


@dataclass
class SyntheticFunctionResult:
    """Result from executing synthetic function"""
    success: bool
    result: Any = None
    execution_time: float = 0.0
    memory_usage: Optional[float] = None
    error: Optional[str] = None
    timeout: bool = False
    coverage_info: Optional[Dict[str, Any]] = None


@dataclass
class SyntheticFunction:
    """Metadata about a synthetic function"""
    name: str
    file_path: str
    function_name: str
    description: str
    target_coverage: float
    cyclomatic_complexity: int
    difficulty: str
    tags: List[str]
    parameter_ranges: List[Dict[str, Any]]
    bounds: Tuple[float, float] = (-999999, 999999)
    timeout_seconds: float = 30.0


class TimeoutException(Exception):
    """Exception raised when function execution times out"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("Function execution timed out")


class SafeExecutionContext:
    """Context manager for safe function execution with timeout"""
    
    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout_seconds = timeout_seconds
        self.old_handler = None
        
    def __enter__(self):
        # Set up timeout signal handler (Unix only)
        if hasattr(signal, 'SIGALRM'):
            self.old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.timeout_seconds))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original signal handler
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancel alarm
            if self.old_handler:
                signal.signal(signal.SIGALRM, self.old_handler)


class SyntheticDatasetAdapter:
    """Adapter for synthetic dataset functions"""
    
    def __init__(self, config_path: str, verbose: bool = True):
        self.config_path = config_path
        self.verbose = verbose
        self.functions: Dict[str, SyntheticFunction] = {}
        self.loaded_modules: Dict[str, Any] = {}
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'timeouts': 0,
            'errors': 0,
            'total_time': 0.0
        }
        
        self._load_configuration()
    
    def _load_configuration(self):
        """Load synthetic dataset configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            base_path = Path(self.config_path).parent.parent
            
            for func_name, func_config in config.items():
                if func_name in ['global_config', 'experiment_config']:
                    continue
                
                file_path = base_path / func_config['file']
                
                synthetic_func = SyntheticFunction(
                    name=func_name,
                    file_path=str(file_path),
                    function_name=func_config.get('function', 'target_function'),
                    description=func_config.get('description', ''),
                    target_coverage=func_config.get('target_coverage', 50.0),
                    cyclomatic_complexity=func_config.get('cyclomatic_complexity', 50),
                    difficulty=func_config.get('difficulty', 'medium'),
                    tags=func_config.get('tags', []),
                    parameter_ranges=func_config.get('parameter_ranges', []),
                    timeout_seconds=func_config.get('timeout_seconds', 30.0)
                )
                
                self.functions[func_name] = synthetic_func
                
                if self.verbose:
                    print(f"Loaded synthetic function: {func_name}")
        
        except Exception as e:
            raise RuntimeError(f"Failed to load synthetic dataset configuration: {e}")
    
    def get_function_names(self) -> List[str]:
        """Get list of available function names"""
        return list(self.functions.keys())
    
    def get_function_info(self, func_name: str) -> Optional[SyntheticFunction]:
        """Get information about a specific function"""
        return self.functions.get(func_name)
    
    def get_functions_by_category(self) -> Dict[str, List[str]]:
        """Group functions by category based on their names"""
        categories = {
            'Algorithmic Foundations': [],
            'Input Validation': [],
            'Concurrency & Resource Management': [],
            'Mathematical Stress Tests': [],
            'Real-World Applications': []
        }
        
        category_keywords = {
            'Algorithmic Foundations': ['cryptographic', 'avl_tree', 'numerical_solver'],
            'Input Validation': ['json_parser', 'protocol_state'],
            'Concurrency & Resource Management': ['resource_scheduler', 'cache_manager', 'event_processor', 'lock_free'],
            'Mathematical Stress Tests': ['matrix_optimizer', 'signal_processor', 'optimization_solver', 'statistical_analyzer'],
            'Real-World Applications': ['workflow_engine', 'distributed_system']
        }
        
        for func_name in self.functions:
            categorized = False
            for category, keywords in category_keywords.items():
                if any(keyword in func_name for keyword in keywords):
                    categories[category].append(func_name)
                    categorized = True
                    break
            
            if not categorized:
                categories['Real-World Applications'].append(func_name)
        
        return categories
    
    def load_function_module(self, func_name: str) -> bool:
        """Load and cache the module for a function"""
        if func_name not in self.functions:
            return False
        
        if func_name in self.loaded_modules:
            return True
        
        try:
            func_info = self.functions[func_name]
            
            if not os.path.exists(func_info.file_path):
                if self.verbose:
                    print(f"Warning: File not found for {func_name}: {func_info.file_path}")
                return False
            
            spec = importlib.util.spec_from_file_location(f"synthetic_{func_name}", func_info.file_path)
            if spec is None or spec.loader is None:
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Verify target function exists
            if not hasattr(module, func_info.function_name):
                if self.verbose:
                    print(f"Warning: Function {func_info.function_name} not found in {func_name}")
                return False
            
            self.loaded_modules[func_name] = module
            
            if self.verbose:
                print(f"Successfully loaded module for {func_name}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to load module for {func_name}: {e}")
            return False
    
    def execute_function(self, func_name: str, test_input: Tuple[Any, ...], 
                        timeout_override: Optional[float] = None) -> SyntheticFunctionResult:
        """Execute a synthetic function with given input"""
        
        self.execution_stats['total_executions'] += 1
        
        if not self.load_function_module(func_name):
            return SyntheticFunctionResult(
                success=False,
                error=f"Failed to load module for {func_name}"
            )
        
        func_info = self.functions[func_name]
        module = self.loaded_modules[func_name]
        target_function = getattr(module, func_info.function_name)
        
        timeout = timeout_override or func_info.timeout_seconds
        
        start_time = time.time()
        result = SyntheticFunctionResult(success=False)
        
        try:
            with SafeExecutionContext(timeout):
                # Execute the function
                if len(test_input) == 1:
                    output = target_function(test_input[0])
                elif len(test_input) == 2:
                    output = target_function(test_input[0], test_input[1])
                else:
                    output = target_function(*test_input)
                
                execution_time = time.time() - start_time
                
                result = SyntheticFunctionResult(
                    success=True,
                    result=output,
                    execution_time=execution_time
                )
                
                self.execution_stats['successful_executions'] += 1
                self.execution_stats['total_time'] += execution_time
                
                # Extract coverage info if available
                if isinstance(output, dict) and 'status' in output:
                    result.coverage_info = {
                        'status': output['status'],
                        'complexity_achieved': True
                    }
        
        except TimeoutException:
            result = SyntheticFunctionResult(
                success=False,
                error="Execution timeout",
                timeout=True,
                execution_time=time.time() - start_time
            )
            self.execution_stats['timeouts'] += 1
        
        except Exception as e:
            result = SyntheticFunctionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                execution_time=time.time() - start_time
            )
            self.execution_stats['errors'] += 1
        
        return result
    
    def generate_test_suite(self, func_name: str, method: str, n_tests: int, 
                           seed: Optional[int] = None) -> TestSuite:
        """Generate test suite for a specific function"""
        
        if func_name not in self.functions:
            raise ValueError(f"Unknown function: {func_name}")
        
        func_info = self.functions[func_name]
        
        # Import generator here to avoid circular imports
        from synthetic_baseline_generators import SyntheticDatasetBaselineGenerator
        
        generator = SyntheticDatasetBaselineGenerator(
            param_ranges=func_info.parameter_ranges,
            bounds=func_info.bounds,
            seed=seed
        )
        
        return generator.generate_synthetic_tests(method, n_tests)
    
    def evaluate_test_suite(self, func_name: str, test_suite: TestSuite, 
                          detailed_analysis: bool = False) -> Dict[str, Any]:
        """Evaluate a test suite on a synthetic function"""
        
        if not self.load_function_module(func_name):
            return {
                'error': f'Failed to load function {func_name}',
                'coverage': 0.0,
                'execution_count': 0
            }
        
        func_info = self.functions[func_name]
        
        # Execute all test cases
        results = []
        successful_executions = 0
        total_execution_time = 0.0
        coverage_statuses = set()
        
        start_evaluation = time.time()
        
        for i, test_case in enumerate(test_suite.test_cases):
            if self.verbose and i > 0 and i % 10 == 0:
                print(f"  Executed {i}/{len(test_suite.test_cases)} tests...")
            
            result = self.execute_function(func_name, test_case)
            results.append(result)
            
            if result.success:
                successful_executions += 1
                total_execution_time += result.execution_time
                
                # Track coverage information
                if result.coverage_info:
                    coverage_statuses.add(result.coverage_info.get('status', 'unknown'))
        
        evaluation_time = time.time() - start_evaluation
        
        # Calculate metrics
        success_rate = successful_executions / len(test_suite.test_cases) if test_suite.test_cases.size > 0 else 0
        avg_execution_time = total_execution_time / max(successful_executions, 1)
        
        # Estimate coverage based on diversity of outcomes
        unique_statuses = len(coverage_statuses)
        estimated_coverage = min(unique_statuses * 5, 100)  # Rough estimate
        
        # Collect error patterns
        error_patterns = {}
        timeout_count = 0
        
        for result in results:
            if not result.success:
                if result.timeout:
                    timeout_count += 1
                elif result.error:
                    error_type = result.error.split(':')[0]
                    error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
        
        evaluation_result = {
            'function_name': func_name,
            'method': test_suite.method,
            'test_count': len(test_suite.test_cases),
            'successful_executions': successful_executions,
            'success_rate': success_rate,
            'estimated_coverage': estimated_coverage,
            'unique_outcomes': unique_statuses,
            'avg_execution_time': avg_execution_time,
            'total_evaluation_time': evaluation_time,
            'timeout_count': timeout_count,
            'error_patterns': error_patterns,
            'target_coverage': func_info.target_coverage,
            'difficulty': func_info.difficulty,
            'tags': func_info.tags
        }
        
        if detailed_analysis:
            evaluation_result.update({
                'detailed_results': results,
                'execution_times': [r.execution_time for r in results if r.success],
                'coverage_statuses': list(coverage_statuses)
            })
        
        return evaluation_result
    
    def batch_evaluate(self, func_names: List[str], method: str, n_tests: int, 
                      n_runs: int = 1, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Evaluate multiple functions with multiple runs"""
        
        all_results = []
        
        for func_name in func_names:
            if self.verbose:
                print(f"\nEvaluating {func_name} with {method}")
                print("-" * 50)
            
            func_results = []
            
            for run in range(n_runs):
                run_seed = (seed + run) if seed is not None else None
                
                if self.verbose:
                    print(f"  Run {run + 1}/{n_runs}...")
                
                # Generate test suite
                test_suite = self.generate_test_suite(func_name, method, n_tests, run_seed)
                
                # Evaluate test suite
                result = self.evaluate_test_suite(func_name, test_suite)
                result['run'] = run + 1
                result['generation_time'] = test_suite.generation_time
                
                func_results.append(result)
                
                if self.verbose:
                    print(f"    Coverage: {result['estimated_coverage']:.1f}%, "
                          f"Success: {result['success_rate']:.1%}")
            
            # Aggregate results across runs
            aggregated = self._aggregate_results(func_results)
            all_results.append(aggregated)
        
        return all_results
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across multiple runs"""
        
        if not results:
            return {}
        
        # Get basic info from first result
        aggregated = {
            'function_name': results[0]['function_name'],
            'method': results[0]['method'],
            'runs': len(results),
            'target_coverage': results[0]['target_coverage'],
            'difficulty': results[0]['difficulty'],
            'tags': results[0]['tags']
        }
        
        # Aggregate numerical metrics
        metrics = ['estimated_coverage', 'success_rate', 'avg_execution_time', 
                  'total_evaluation_time', 'unique_outcomes', 'test_count']
        
        for metric in metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
        
        # Aggregate counts
        count_metrics = ['successful_executions', 'timeout_count']
        for metric in count_metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                aggregated[f'{metric}_total'] = np.sum(values)
                aggregated[f'{metric}_mean'] = np.mean(values)
        
        # Combine error patterns
        all_error_patterns = {}
        for result in results:
            for error_type, count in result.get('error_patterns', {}).items():
                all_error_patterns[error_type] = all_error_patterns.get(error_type, 0) + count
        
        aggregated['error_patterns'] = all_error_patterns
        
        return aggregated
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get overall execution statistics"""
        stats = self.execution_stats.copy()
        
        if stats['total_executions'] > 0:
            stats['success_rate'] = stats['successful_executions'] / stats['total_executions']
            stats['timeout_rate'] = stats['timeouts'] / stats['total_executions']
            stats['error_rate'] = stats['errors'] / stats['total_executions']
        else:
            stats['success_rate'] = 0.0
            stats['timeout_rate'] = 0.0
            stats['error_rate'] = 0.0
        
        if stats['successful_executions'] > 0:
            stats['avg_execution_time'] = stats['total_time'] / stats['successful_executions']
        else:
            stats['avg_execution_time'] = 0.0
        
        return stats
    
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate the synthetic dataset by running smoke tests"""
        
        validation_results = {
            'total_functions': len(self.functions),
            'loadable_functions': 0,
            'executable_functions': 0,
            'failed_functions': [],
            'warnings': []
        }
        
        for func_name in self.functions:
            # Try to load module
            if self.load_function_module(func_name):
                validation_results['loadable_functions'] += 1
                
                # Try basic execution with simple input
                try:
                    test_suite = self.generate_test_suite(func_name, "pure_random", 1, seed=42)
                    result = self.evaluate_test_suite(func_name, test_suite)
                    
                    if result.get('successful_executions', 0) > 0:
                        validation_results['executable_functions'] += 1
                    else:
                        validation_results['failed_functions'].append({
                            'function': func_name,
                            'reason': 'execution_failed',
                            'details': result.get('error_patterns', {})
                        })
                
                except Exception as e:
                    validation_results['failed_functions'].append({
                        'function': func_name,
                        'reason': 'test_generation_failed',
                        'error': str(e)
                    })
            else:
                validation_results['failed_functions'].append({
                    'function': func_name,
                    'reason': 'module_load_failed'
                })
        
        # Add warnings for functions with issues
        for func_name, func_info in self.functions.items():
            if not os.path.exists(func_info.file_path):
                validation_results['warnings'].append(f"File not found: {func_name}")
        
        return validation_results


if __name__ == "__main__":
    """Example usage and testing"""
    
    print("Synthetic Dataset Adapter - Validation Test")
    print("=" * 60)
    
    # Initialize adapter
    config_path = "config/synthetic_test_programs.yaml"
    
    try:
        adapter = SyntheticDatasetAdapter(config_path, verbose=True)
        
        print(f"\nLoaded {len(adapter.get_function_names())} synthetic functions")
        
        # Validate dataset
        validation = adapter.validate_dataset()
        print(f"\nValidation Results:")
        print(f"  Loadable: {validation['loadable_functions']}/{validation['total_functions']}")
        print(f"  Executable: {validation['executable_functions']}/{validation['total_functions']}")
        
        if validation['failed_functions']:
            print(f"  Failed functions: {len(validation['failed_functions'])}")
            for failure in validation['failed_functions'][:3]:
                print(f"    - {failure['function']}: {failure['reason']}")
        
        # Test a few functions
        available_functions = adapter.get_function_names()
        test_functions = available_functions[:3]  # Test first 3 functions
        
        print(f"\nTesting functions with baseline methods:")
        results = adapter.batch_evaluate(
            func_names=test_functions,
            method="pure_random",
            n_tests=10,
            n_runs=2,
            seed=42
        )
        
        for result in results:
            print(f"\n{result['function_name']}:")
            print(f"  Coverage: {result.get('estimated_coverage_mean', 0):.1f}% ± {result.get('estimated_coverage_std', 0):.1f}%")
            print(f"  Success Rate: {result.get('success_rate_mean', 0):.1%}")
            print(f"  Difficulty: {result.get('difficulty', 'unknown')}")
        
        # Show execution statistics
        stats = adapter.get_execution_statistics()
        print(f"\nExecution Statistics:")
        print(f"  Total executions: {stats['total_executions']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Average time: {stats['avg_execution_time']:.3f}s")
        
        print("\nAdapter validation complete!")
        
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()