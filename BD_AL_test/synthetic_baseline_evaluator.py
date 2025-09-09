#!/usr/bin/env python3
"""
Comprehensive Evaluation Engine for Synthetic Dataset
Advanced analysis and evaluation of baseline methods on synthetic functions
"""

import numpy as np
import pandas as pd
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import psutil
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from synthetic_dataset_adapter import SyntheticDatasetAdapter, SyntheticFunction
from synthetic_baseline_generators import SyntheticDatasetBaselineGenerator


@dataclass
class EvaluationConfig:
    """Configuration for evaluation experiments"""
    n_tests_per_method: int = 100
    n_runs_per_experiment: int = 10
    timeout_seconds: float = 30.0
    max_parallel_workers: int = 4
    memory_limit_gb: float = 8.0
    enable_detailed_analysis: bool = False
    enable_resource_monitoring: bool = True
    seed: Optional[int] = 42


@dataclass
class ResourceMetrics:
    """System resource usage metrics"""
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    avg_cpu_percent: float = 0.0
    execution_time: float = 0.0
    disk_io_mb: float = 0.0


@dataclass
class DetailedEvaluationResult:
    """Detailed result from baseline method evaluation"""
    function_name: str
    method_name: str
    config: EvaluationConfig
    
    # Performance metrics
    coverage_stats: Dict[str, float] = field(default_factory=dict)
    execution_stats: Dict[str, float] = field(default_factory=dict)
    success_stats: Dict[str, float] = field(default_factory=dict)
    
    # Resource usage
    resource_metrics: ResourceMetrics = field(default_factory=ResourceMetrics)
    
    # Quality metrics
    diversity_score: float = 0.0
    edge_case_coverage: float = 0.0
    error_resilience: float = 0.0
    
    # Detailed data
    run_results: List[Dict[str, Any]] = field(default_factory=list)
    convergence_curve: List[float] = field(default_factory=list)
    
    # Metadata
    complexity_level: str = "unknown"
    difficulty_category: str = "unknown"
    tags: List[str] = field(default_factory=list)
    
    # Status
    completed: bool = False
    error_message: Optional[str] = None


class ResourceMonitor:
    """Monitor system resource usage during evaluation"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.measurements = []
        self.monitor_thread = None
        self.process = psutil.Process()
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.measurements = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> ResourceMetrics:
        """Stop monitoring and return metrics"""
        if not self.monitoring:
            return ResourceMetrics()
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if not self.measurements:
            return ResourceMetrics()
        
        memory_values = [m['memory_mb'] for m in self.measurements]
        cpu_values = [m['cpu_percent'] for m in self.measurements]
        
        return ResourceMetrics(
            peak_memory_mb=max(memory_values),
            avg_memory_mb=np.mean(memory_values),
            peak_cpu_percent=max(cpu_values),
            avg_cpu_percent=np.mean(cpu_values),
            execution_time=self.measurements[-1]['timestamp'] - self.measurements[0]['timestamp'],
            disk_io_mb=0.0  # Could be implemented if needed
        )
    
    def _monitor_loop(self):
        """Internal monitoring loop"""
        start_time = time.time()
        
        while self.monitoring:
            try:
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()
                
                self.measurements.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_info.rss / 1024 / 1024,
                    'cpu_percent': cpu_percent,
                    'elapsed': time.time() - start_time
                })
                
                time.sleep(self.interval)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break


class SyntheticBaselineEvaluator:
    """Comprehensive evaluator for baseline methods on synthetic dataset"""
    
    def __init__(self, config_path: str, evaluation_config: Optional[EvaluationConfig] = None,
                 verbose: bool = True):
        self.config_path = config_path
        self.eval_config = evaluation_config or EvaluationConfig()
        self.verbose = verbose
        
        # Initialize adapter
        self.adapter = SyntheticDatasetAdapter(config_path, verbose=verbose)
        
        # Results storage
        self.results: List[DetailedEvaluationResult] = []
        self.experiment_metadata = {
            'start_time': None,
            'end_time': None,
            'total_experiments': 0,
            'completed_experiments': 0,
            'failed_experiments': 0
        }
    
    def get_baseline_methods(self) -> List[str]:
        """Get list of available baseline methods"""
        generator = SyntheticDatasetBaselineGenerator([], seed=self.eval_config.seed)
        return generator.get_all_methods() + [
            "grid_search_adaptive",
            "monte_carlo_uniform",
            "latin_hypercube_stratified",
            "random_walk",
            "hill_climbing_multi_start"
        ]
    
    def evaluate_single_function_method(self, func_name: str, method_name: str) -> DetailedEvaluationResult:
        """Evaluate single function with single method"""
        
        result = DetailedEvaluationResult(
            function_name=func_name,
            method_name=method_name,
            config=self.eval_config
        )
        
        try:
            func_info = self.adapter.get_function_info(func_name)
            if not func_info:
                result.error_message = f"Function {func_name} not found"
                return result
            
            result.complexity_level = self._categorize_complexity(func_info.cyclomatic_complexity)
            result.difficulty_category = func_info.difficulty
            result.tags = func_info.tags
            
            # Start resource monitoring
            monitor = ResourceMonitor() if self.eval_config.enable_resource_monitoring else None
            if monitor:
                monitor.start_monitoring()
            
            run_results = []
            coverage_values = []
            execution_times = []
            success_rates = []
            
            # Run multiple experiments
            for run in range(self.eval_config.n_runs_per_experiment):
                if self.verbose:
                    print(f"    Run {run + 1}/{self.eval_config.n_runs_per_experiment}")
                
                run_seed = (self.eval_config.seed + run) if self.eval_config.seed is not None else None
                
                # Generate and evaluate test suite
                try:
                    test_suite = self.adapter.generate_test_suite(
                        func_name, method_name, 
                        self.eval_config.n_tests_per_method, 
                        seed=run_seed
                    )
                    
                    evaluation_result = self.adapter.evaluate_test_suite(
                        func_name, test_suite, 
                        detailed_analysis=self.eval_config.enable_detailed_analysis
                    )
                    
                    run_results.append(evaluation_result)
                    coverage_values.append(evaluation_result.get('estimated_coverage', 0))
                    execution_times.append(evaluation_result.get('avg_execution_time', 0))
                    success_rates.append(evaluation_result.get('success_rate', 0))
                    
                except Exception as e:
                    if self.verbose:
                        print(f"      Run failed: {e}")
                    run_results.append({'error': str(e), 'run': run})
            
            # Stop resource monitoring
            if monitor:
                result.resource_metrics = monitor.stop_monitoring()
            
            # Aggregate results
            if coverage_values:
                result.coverage_stats = {
                    'mean': np.mean(coverage_values),
                    'std': np.std(coverage_values),
                    'min': np.min(coverage_values),
                    'max': np.max(coverage_values),
                    'median': np.median(coverage_values)
                }
                
                result.convergence_curve = coverage_values
            
            if execution_times:
                result.execution_stats = {
                    'mean_time': np.mean(execution_times),
                    'std_time': np.std(execution_times),
                    'min_time': np.min(execution_times),
                    'max_time': np.max(execution_times),
                    'total_time': np.sum(execution_times)
                }
            
            if success_rates:
                result.success_stats = {
                    'mean_success': np.mean(success_rates),
                    'std_success': np.std(success_rates),
                    'min_success': np.min(success_rates),
                    'max_success': np.max(success_rates)
                }
            
            # Calculate quality metrics
            result.diversity_score = self._calculate_diversity_score(run_results)
            result.edge_case_coverage = self._calculate_edge_case_coverage(run_results)
            result.error_resilience = self._calculate_error_resilience(run_results)
            
            result.run_results = run_results
            result.completed = True
            
            if self.verbose:
                coverage_mean = result.coverage_stats.get('mean', 0)
                success_mean = result.success_stats.get('mean', 0)
                print(f"    Result: {coverage_mean:.1f}% coverage, {success_mean:.1%} success")
        
        except Exception as e:
            result.error_message = str(e)
            if self.verbose:
                print(f"    Failed: {e}")
        
        return result
    
    def evaluate_all_methods_on_function(self, func_name: str, 
                                       methods: Optional[List[str]] = None) -> List[DetailedEvaluationResult]:
        """Evaluate all baseline methods on a single function"""
        
        if methods is None:
            methods = self.get_baseline_methods()
        
        if self.verbose:
            print(f"\nEvaluating {func_name} with {len(methods)} methods")
            print("-" * 60)
        
        results = []
        
        for method in methods:
            if self.verbose:
                print(f"  Method: {method}")
            
            result = self.evaluate_single_function_method(func_name, method)
            results.append(result)
            
            self.experiment_metadata['total_experiments'] += 1
            if result.completed:
                self.experiment_metadata['completed_experiments'] += 1
            else:
                self.experiment_metadata['failed_experiments'] += 1
        
        return results
    
    def evaluate_comprehensive(self, 
                             functions: Optional[List[str]] = None,
                             methods: Optional[List[str]] = None,
                             parallel: bool = True) -> List[DetailedEvaluationResult]:
        """Comprehensive evaluation of methods on functions"""
        
        self.experiment_metadata['start_time'] = time.time()
        
        if functions is None:
            functions = self.adapter.get_function_names()
        
        if methods is None:
            methods = self.get_baseline_methods()
        
        if self.verbose:
            print(f"Starting comprehensive evaluation:")
            print(f"  Functions: {len(functions)}")
            print(f"  Methods: {len(methods)}")
            print(f"  Total experiments: {len(functions) * len(methods)}")
            print(f"  Parallel execution: {parallel}")
            print()
        
        all_results = []
        
        if parallel and len(functions) > 1:
            all_results = self._evaluate_parallel(functions, methods)
        else:
            # Sequential evaluation
            for func_name in functions:
                func_results = self.evaluate_all_methods_on_function(func_name, methods)
                all_results.extend(func_results)
        
        self.results = all_results
        self.experiment_metadata['end_time'] = time.time()
        
        if self.verbose:
            total_time = self.experiment_metadata['end_time'] - self.experiment_metadata['start_time']
            print(f"\nEvaluation completed in {total_time:.1f} seconds")
            print(f"  Completed: {self.experiment_metadata['completed_experiments']}")
            print(f"  Failed: {self.experiment_metadata['failed_experiments']}")
        
        return all_results
    
    def _evaluate_parallel(self, functions: List[str], methods: List[str]) -> List[DetailedEvaluationResult]:
        """Parallel evaluation using thread pool"""
        
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.eval_config.max_parallel_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            
            for func_name in functions:
                for method in methods:
                    future = executor.submit(self.evaluate_single_function_method, func_name, method)
                    future_to_task[future] = (func_name, method)
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                func_name, method = future_to_task[future]
                
                try:
                    result = future.result()
                    all_results.append(result)
                    
                    self.experiment_metadata['total_experiments'] += 1
                    if result.completed:
                        self.experiment_metadata['completed_experiments'] += 1
                    else:
                        self.experiment_metadata['failed_experiments'] += 1
                    
                    if self.verbose:
                        progress = len(all_results)
                        total = len(functions) * len(methods)
                        coverage = result.coverage_stats.get('mean', 0) if result.coverage_stats else 0
                        print(f"  [{progress}/{total}] {func_name}/{method}: {coverage:.1f}%")
                
                except Exception as e:
                    if self.verbose:
                        print(f"  Error in {func_name}/{method}: {e}")
                    
                    # Create error result
                    error_result = DetailedEvaluationResult(
                        function_name=func_name,
                        method_name=method,
                        config=self.eval_config,
                        error_message=str(e)
                    )
                    all_results.append(error_result)
                    
                    self.experiment_metadata['total_experiments'] += 1
                    self.experiment_metadata['failed_experiments'] += 1
        
        return all_results
    
    def analyze_by_function_complexity(self) -> Dict[str, Any]:
        """Analyze results grouped by function complexity"""
        
        if not self.results:
            return {}
        
        complexity_groups = {
            'low': [],
            'medium': [],
            'high': [],
            'extreme': []
        }
        
        for result in self.results:
            if result.completed and result.coverage_stats:
                complexity = result.complexity_level
                if complexity in complexity_groups:
                    complexity_groups[complexity].append(result)
        
        analysis = {}
        
        for complexity, results in complexity_groups.items():
            if not results:
                continue
            
            coverage_values = [r.coverage_stats['mean'] for r in results]
            execution_times = [r.execution_stats.get('mean_time', 0) for r in results if r.execution_stats]
            
            analysis[complexity] = {
                'count': len(results),
                'avg_coverage': np.mean(coverage_values),
                'std_coverage': np.std(coverage_values),
                'avg_execution_time': np.mean(execution_times) if execution_times else 0,
                'best_method': max(results, key=lambda r: r.coverage_stats['mean']).method_name,
                'worst_method': min(results, key=lambda r: r.coverage_stats['mean']).method_name
            }
        
        return analysis
    
    def analyze_by_method_performance(self) -> Dict[str, Any]:
        """Analyze results grouped by method"""
        
        if not self.results:
            return {}
        
        method_groups = {}
        
        for result in self.results:
            if result.completed and result.coverage_stats:
                method = result.method_name
                if method not in method_groups:
                    method_groups[method] = []
                method_groups[method].append(result)
        
        analysis = {}
        
        for method, results in method_groups.items():
            coverage_values = [r.coverage_stats['mean'] for r in results]
            execution_times = [r.execution_stats.get('mean_time', 0) for r in results if r.execution_stats]
            success_rates = [r.success_stats.get('mean_success', 0) for r in results if r.success_stats]
            diversity_scores = [r.diversity_score for r in results]
            
            analysis[method] = {
                'functions_tested': len(results),
                'avg_coverage': np.mean(coverage_values),
                'std_coverage': np.std(coverage_values),
                'avg_execution_time': np.mean(execution_times) if execution_times else 0,
                'avg_success_rate': np.mean(success_rates) if success_rates else 0,
                'avg_diversity': np.mean(diversity_scores),
                'reliability_score': len([r for r in results if r.coverage_stats['mean'] > 10]) / len(results),
                'best_function': max(results, key=lambda r: r.coverage_stats['mean']).function_name,
                'worst_function': min(results, key=lambda r: r.coverage_stats['mean']).function_name
            }
        
        return analysis
    
    def generate_performance_matrix(self) -> pd.DataFrame:
        """Generate performance matrix (methods × functions)"""
        
        if not self.results:
            return pd.DataFrame()
        
        # Extract data for matrix
        matrix_data = []
        
        for result in self.results:
            if result.completed and result.coverage_stats:
                matrix_data.append({
                    'Function': result.function_name,
                    'Method': result.method_name,
                    'Coverage': result.coverage_stats['mean'],
                    'Execution_Time': result.execution_stats.get('mean_time', 0),
                    'Success_Rate': result.success_stats.get('mean_success', 0),
                    'Diversity': result.diversity_score,
                    'Complexity': result.complexity_level,
                    'Difficulty': result.difficulty_category
                })
        
        df = pd.DataFrame(matrix_data)
        
        # Create pivot table
        coverage_matrix = df.pivot_table(
            values='Coverage', 
            index='Method', 
            columns='Function', 
            aggfunc='mean'
        )
        
        return coverage_matrix
    
    def export_results(self, output_dir: str = "synthetic_baseline_results"):
        """Export results to files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export detailed results as JSON
        detailed_results = []
        for result in self.results:
            result_dict = {
                'function_name': result.function_name,
                'method_name': result.method_name,
                'completed': result.completed,
                'error_message': result.error_message,
                'coverage_stats': result.coverage_stats,
                'execution_stats': result.execution_stats,
                'success_stats': result.success_stats,
                'diversity_score': result.diversity_score,
                'edge_case_coverage': result.edge_case_coverage,
                'error_resilience': result.error_resilience,
                'complexity_level': result.complexity_level,
                'difficulty_category': result.difficulty_category,
                'tags': result.tags,
                'resource_metrics': {
                    'peak_memory_mb': result.resource_metrics.peak_memory_mb,
                    'avg_memory_mb': result.resource_metrics.avg_memory_mb,
                    'peak_cpu_percent': result.resource_metrics.peak_cpu_percent,
                    'avg_cpu_percent': result.resource_metrics.avg_cpu_percent,
                    'execution_time': result.resource_metrics.execution_time
                } if result.resource_metrics else {}
            }
            detailed_results.append(result_dict)
        
        with open(output_path / "detailed_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Export performance matrix as CSV
        performance_matrix = self.generate_performance_matrix()
        performance_matrix.to_csv(output_path / "performance_matrix.csv")
        
        # Export complexity analysis
        complexity_analysis = self.analyze_by_function_complexity()
        with open(output_path / "complexity_analysis.json", 'w') as f:
            json.dump(complexity_analysis, f, indent=2)
        
        # Export method analysis
        method_analysis = self.analyze_by_method_performance()
        with open(output_path / "method_analysis.json", 'w') as f:
            json.dump(method_analysis, f, indent=2)
        
        # Export experiment metadata
        with open(output_path / "experiment_metadata.json", 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2)
        
        if self.verbose:
            print(f"Results exported to {output_path}")
    
    # Helper methods
    
    def _categorize_complexity(self, cyclomatic_complexity: int) -> str:
        """Categorize function by complexity"""
        if cyclomatic_complexity < 10:
            return "low"
        elif cyclomatic_complexity < 30:
            return "medium"
        elif cyclomatic_complexity < 60:
            return "high"
        else:
            return "extreme"
    
    def _calculate_diversity_score(self, run_results: List[Dict[str, Any]]) -> float:
        """Calculate diversity score from run results"""
        if not run_results:
            return 0.0
        
        # Count unique outcomes
        unique_statuses = set()
        for result in run_results:
            if 'error_patterns' in result:
                for error_type in result['error_patterns'].keys():
                    unique_statuses.add(error_type)
            if result.get('successful_executions', 0) > 0:
                unique_statuses.add('success')
        
        return min(len(unique_statuses) * 10, 100)  # Scale 0-100
    
    def _calculate_edge_case_coverage(self, run_results: List[Dict[str, Any]]) -> float:
        """Calculate edge case coverage score"""
        if not run_results:
            return 0.0
        
        edge_case_indicators = 0
        total_indicators = 0
        
        for result in run_results:
            total_indicators += 1
            
            # Look for indicators of edge case coverage
            if result.get('timeout_count', 0) > 0:
                edge_case_indicators += 0.5  # Timeouts indicate complex paths
            
            error_patterns = result.get('error_patterns', {})
            if error_patterns:
                edge_case_indicators += min(len(error_patterns) * 0.2, 1.0)
            
            unique_outcomes = result.get('unique_outcomes', 0)
            if unique_outcomes > 5:
                edge_case_indicators += 0.3
        
        return (edge_case_indicators / max(total_indicators, 1)) * 100
    
    def _calculate_error_resilience(self, run_results: List[Dict[str, Any]]) -> float:
        """Calculate error resilience score"""
        if not run_results:
            return 0.0
        
        success_rates = [result.get('success_rate', 0) for result in run_results]
        if not success_rates:
            return 0.0
        
        # Resilience is inverse of variance in success rates
        variance = np.var(success_rates)
        return max(0, 100 - variance * 100)


if __name__ == "__main__":
    """Example usage of the evaluator"""
    
    print("Synthetic Baseline Evaluator - Test Run")
    print("=" * 60)
    
    # Configuration
    eval_config = EvaluationConfig(
        n_tests_per_method=20,
        n_runs_per_experiment=3,
        max_parallel_workers=2,
        enable_detailed_analysis=False,
        enable_resource_monitoring=True
    )
    
    # Initialize evaluator
    evaluator = SyntheticBaselineEvaluator(
        config_path="config/synthetic_test_programs.yaml",
        evaluation_config=eval_config,
        verbose=True
    )
    
    # Test with a subset of functions and methods
    test_functions = evaluator.adapter.get_function_names()[:2]  # First 2 functions
    test_methods = ["pure_random", "stratified_random"]  # Simple methods
    
    print(f"Testing {len(test_functions)} functions with {len(test_methods)} methods")
    
    # Run evaluation
    results = evaluator.evaluate_comprehensive(
        functions=test_functions,
        methods=test_methods,
        parallel=True
    )
    
    # Analyze results
    print(f"\nCompleted {len(results)} experiments")
    
    complexity_analysis = evaluator.analyze_by_function_complexity()
    method_analysis = evaluator.analyze_by_method_performance()
    
    print("\nComplexity Analysis:")
    for complexity, stats in complexity_analysis.items():
        print(f"  {complexity}: {stats['avg_coverage']:.1f}% avg coverage ({stats['count']} results)")
    
    print("\nMethod Analysis:")
    for method, stats in method_analysis.items():
        print(f"  {method}: {stats['avg_coverage']:.1f}% avg coverage, {stats['reliability_score']:.1%} reliability")
    
    # Export results
    evaluator.export_results("test_synthetic_baseline_results")
    
    print("\nEvaluation test completed!")