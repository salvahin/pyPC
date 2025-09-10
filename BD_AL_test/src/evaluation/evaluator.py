#!/usr/bin/env python3
"""
Unified Test Evaluation Framework

This module consolidates all test evaluation functionality into a single,
comprehensive evaluator supporting multiple algorithms and test programs.
"""

import time
import subprocess
import threading
import json
import yaml
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import importlib.util
import sys
import traceback
import psutil
import os


@dataclass
class EvaluationResult:
    """Standardized evaluation result"""
    coverage: float = 0.0
    branch_distance: float = float('inf')
    approach_level: int = 0
    execution_time: float = 0.0
    memory_usage: float = 0.0
    test_cases_generated: int = 0
    unique_branches_hit: int = 0
    total_branches: int = 0
    success: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'coverage': self.coverage,
            'branch_distance': self.branch_distance,
            'approach_level': self.approach_level,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'test_cases_generated': self.test_cases_generated,
            'unique_branches_hit': self.unique_branches_hit,
            'total_branches': self.total_branches,
            'success': self.success,
            'error_message': self.error_message
        }


class ResourceMonitor:
    """Monitor CPU and memory usage during execution"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.max_memory = 0.0
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.monitoring = True
        self.max_memory = 0.0
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> float:
        """Stop monitoring and return peak memory usage in MB"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        return self.max_memory
    
    def _monitor_resources(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                self.max_memory = max(self.max_memory, memory_mb)
                time.sleep(0.1)
            except Exception:
                break


class SafeExecutionContext:
    """Safe execution environment with timeout and error handling"""
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.result = None
        self.exception = None
        self.timed_out = False
    
    def execute_with_timeout(self, func, *args, **kwargs):
        """Execute function with timeout protection"""
        def target():
            try:
                self.result = func(*args, **kwargs)
            except Exception as e:
                self.exception = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(self.timeout)
        
        if thread.is_alive():
            self.timed_out = True
            return None
        
        if self.exception:
            raise self.exception
        
        return self.result


class UnifiedTestEvaluator:
    """Comprehensive test evaluator for all algorithm types"""
    
    def __init__(self, test_programs_config: Optional[str] = None, verbose: bool = True):
        self.verbose = verbose
        self.test_programs = {}
        
        # Load test programs configuration
        if test_programs_config and Path(test_programs_config).exists():
            self.load_test_programs_config(test_programs_config)
        else:
            self.discover_test_programs()
    
    def load_test_programs_config(self, config_path: str):
        """Load test programs from YAML configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.test_programs = config
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load config {config_path}: {e}")
            self.discover_test_programs()
    
    def discover_test_programs(self):
        """Auto-discover test programs in test_programs directory"""
        test_dir = Path("test_programs")
        if not test_dir.exists():
            if self.verbose:
                print("Warning: test_programs directory not found")
            return
        
        for py_file in test_dir.glob("*.py"):
            program_name = py_file.stem
            self.test_programs[program_name] = {
                'file': str(py_file),
                'function': 'target_function',  # Default function name
                'timeout': 30.0
            }
    
    def evaluate_single_run(self, 
                           program_name: str, 
                           test_inputs: List[Any],
                           algorithm_name: str = "unknown",
                           timeout: Optional[float] = None) -> EvaluationResult:
        """Evaluate a single test run"""
        
        if program_name not in self.test_programs:
            return EvaluationResult(
                success=False,
                error_message=f"Unknown program: {program_name}"
            )
        
        program_config = self.test_programs[program_name]
        actual_timeout = timeout or program_config.get('timeout', 30.0)
        
        # Start resource monitoring
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        result = EvaluationResult()
        
        try:
            # Execute with timeout protection
            execution_context = SafeExecutionContext(actual_timeout)
            
            evaluation_result = execution_context.execute_with_timeout(
                self._execute_program,
                program_config,
                test_inputs
            )
            
            if execution_context.timed_out:
                result.error_message = f"Execution timed out after {actual_timeout}s"
            elif evaluation_result:
                result.coverage = evaluation_result.get('coverage', 0.0)
                result.branch_distance = evaluation_result.get('branch_distance', float('inf'))
                result.approach_level = evaluation_result.get('approach_level', 0)
                result.unique_branches_hit = evaluation_result.get('branches_hit', 0)
                result.total_branches = evaluation_result.get('total_branches', 1)
                result.success = True
            
        except Exception as e:
            result.error_message = str(e)
            if self.verbose:
                print(f"Evaluation error for {program_name}: {e}")
        
        # Finalize result
        result.execution_time = time.time() - start_time
        result.memory_usage = monitor.stop_monitoring()
        result.test_cases_generated = len(test_inputs) if test_inputs else 0
        
        return result
    
    def _execute_program(self, program_config: Dict[str, Any], test_inputs: List[Any]) -> Dict[str, Any]:
        """Execute the target program with test inputs"""
        
        program_file = program_config['file']
        function_name = program_config.get('function', 'target_function')
        
        # Load the program module dynamically
        spec = importlib.util.spec_from_file_location("target_module", program_file)
        if not spec or not spec.loader:
            raise RuntimeError(f"Could not load module from {program_file}")
        
        target_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(target_module)
        
        if not hasattr(target_module, function_name):
            raise RuntimeError(f"Function {function_name} not found in {program_file}")
        
        target_function = getattr(target_module, function_name)
        
        # Execute test inputs and measure coverage
        branches_hit = set()
        total_execution_count = 0
        total_branch_distance = 0.0
        
        for test_input in test_inputs:
            try:
                # Call the function with test input
                if isinstance(test_input, (list, tuple)):
                    result = target_function(*test_input)
                else:
                    result = target_function(test_input)
                
                # Simulate branch tracking (in real implementation, this would use instrumentation)
                total_execution_count += 1
                
                # Mock coverage calculation (would be replaced with actual instrumentation)
                simulated_branches = self._simulate_branch_coverage(test_input, result)
                branches_hit.update(simulated_branches)
                
            except Exception as e:
                if self.verbose:
                    print(f"Test input {test_input} caused exception: {e}")
                continue
        
        # Calculate final metrics
        total_branches = max(len(branches_hit), 10)  # Minimum realistic branch count
        coverage = len(branches_hit) / total_branches
        avg_branch_distance = total_branch_distance / max(total_execution_count, 1)
        
        return {
            'coverage': coverage,
            'branch_distance': avg_branch_distance,
            'approach_level': 0,  # Would be calculated with real instrumentation
            'branches_hit': len(branches_hit),
            'total_branches': total_branches
        }
    
    def _simulate_branch_coverage(self, test_input: Any, result: Any) -> List[int]:
        """Simulate branch coverage for demonstration purposes"""
        # This is a mock implementation - in reality, you'd use code instrumentation
        
        # Simple heuristic based on input characteristics
        branches = []
        
        if isinstance(test_input, (list, tuple)):
            for i, val in enumerate(test_input):
                if isinstance(val, (int, float)):
                    # Simulate branches based on value ranges
                    if val < 0:
                        branches.append(i * 2)
                    elif val > 0:
                        branches.append(i * 2 + 1)
                    
                    if abs(val) > 100:
                        branches.append(100 + i)
                    
                elif isinstance(val, str):
                    branches.append(200 + len(val) % 10)
        
        # Add some randomness to simulate different execution paths
        import random
        random_branches = [random.randint(300, 400) for _ in range(random.randint(1, 5))]
        branches.extend(random_branches)
        
        return branches
    
    def evaluate_batch(self, 
                      program_name: str,
                      test_batches: Dict[str, List[Any]],
                      timeout: Optional[float] = None) -> Dict[str, List[EvaluationResult]]:
        """Evaluate multiple test batches for different algorithms"""
        
        results = {}
        
        for algorithm_name, test_inputs in test_batches.items():
            if self.verbose:
                print(f"Evaluating {algorithm_name} on {program_name}...")
            
            algorithm_results = []
            
            # Split large batches into smaller chunks
            chunk_size = 50
            for i in range(0, len(test_inputs), chunk_size):
                chunk = test_inputs[i:i + chunk_size]
                
                result = self.evaluate_single_run(
                    program_name=program_name,
                    test_inputs=chunk,
                    algorithm_name=algorithm_name,
                    timeout=timeout
                )
                
                algorithm_results.append(result)
            
            results[algorithm_name] = algorithm_results
        
        return results
    
    def run_comprehensive_evaluation(self,
                                   algorithms: Dict[str, Any],
                                   programs: Optional[List[str]] = None,
                                   repetitions: int = 10,
                                   timeout: float = 30.0) -> Dict[str, Dict[str, List[EvaluationResult]]]:
        """Run comprehensive evaluation across programs and algorithms"""
        
        if programs is None:
            programs = list(self.test_programs.keys())
        
        if self.verbose:
            print(f"Running comprehensive evaluation:")
            print(f"  Programs: {len(programs)}")
            print(f"  Algorithms: {len(algorithms)}")
            print(f"  Repetitions: {repetitions}")
        
        results = {}
        
        for program_name in programs:
            if self.verbose:
                print(f"\nEvaluating program: {program_name}")
            
            program_results = {}
            
            for algorithm_name, algorithm_instance in algorithms.items():
                if self.verbose:
                    print(f"  Running {algorithm_name}...")
                
                algorithm_results = []
                
                for rep in range(repetitions):
                    try:
                        # Generate test cases using the algorithm
                        test_inputs = self._generate_test_cases(algorithm_instance, program_name)
                        
                        # Evaluate the generated tests
                        eval_result = self.evaluate_single_run(
                            program_name=program_name,
                            test_inputs=test_inputs,
                            algorithm_name=algorithm_name,
                            timeout=timeout
                        )
                        
                        algorithm_results.append(eval_result)
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"    Rep {rep+1} failed: {e}")
                        
                        algorithm_results.append(EvaluationResult(
                            success=False,
                            error_message=str(e)
                        ))
                
                program_results[algorithm_name] = algorithm_results
            
            results[program_name] = program_results
        
        return results
    
    def _generate_test_cases(self, algorithm_instance: Any, program_name: str) -> List[Any]:
        """Generate test cases using the algorithm instance"""
        
        # This is a simplified interface - would be expanded based on algorithm types
        if hasattr(algorithm_instance, 'generate'):
            return algorithm_instance.generate(n_tests=50)
        elif hasattr(algorithm_instance, 'generate_tests'):
            return algorithm_instance.generate_tests(50)
        elif callable(algorithm_instance):
            return algorithm_instance(n_tests=50)
        else:
            raise ValueError(f"Unknown algorithm interface: {type(algorithm_instance)}")
    
    def get_program_list(self) -> List[str]:
        """Get list of available test programs"""
        return list(self.test_programs.keys())
    
    def export_results(self, results: Dict[str, Any], output_path: str):
        """Export evaluation results to JSON file"""
        
        # Convert EvaluationResult objects to dictionaries
        exportable_results = {}
        
        for program_name, program_results in results.items():
            exportable_results[program_name] = {}
            
            for algorithm_name, algorithm_results in program_results.items():
                exportable_results[program_name][algorithm_name] = [
                    result.to_dict() if isinstance(result, EvaluationResult) else result
                    for result in algorithm_results
                ]
        
        with open(output_path, 'w') as f:
            json.dump(exportable_results, f, indent=2)
        
        if self.verbose:
            print(f"Results exported to: {output_path}")


# Simplified interface for backward compatibility
class TestEvaluator(UnifiedTestEvaluator):
    """Simplified test evaluator interface"""
    pass