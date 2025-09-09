#!/usr/bin/env python3
"""
Simple Direct Baseline Experiment on Synthetic Functions

This script runs the available baseline methods on synthetic functions
and generates a comprehensive analysis report.
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from baseline_evaluator import BaselineEvaluator


def get_available_synthetic_functions():
    """Get list of available synthetic test functions"""
    test_programs_dir = Path("test_programs")
    
    # List of synthetic functions (complex ones suitable for baseline testing)
    synthetic_functions = []
    
    # Check which synthetic functions actually exist
    potential_functions = [
        "cryptographic_hash", "avl_tree_operations", "numerical_solver",
        "matrix_optimizer", "signal_processor", "json_parser_validator",
        "protocol_state_machine", "workflow_engine", "distributed_system",
        "optimization_solver", "resource_scheduler", "cache_manager",
        "event_processor", "lock_free_queue", "statistical_analyzer",
        # Also include some existing complex functions
        "complex_conditions", "deep_branching", "nested_loops",
        "binary_search_tree", "path_finder", "state_machine",
        "recursive_calc", "advanced_datastructure", "pattern_matcher",
        "constraint_solver", "graph_traversal"
    ]
    
    for func in potential_functions:
        if (test_programs_dir / f"{func}.py").exists():
            synthetic_functions.append(func)
    
    print(f"Found {len(synthetic_functions)} synthetic functions to evaluate")
    return synthetic_functions


def run_baseline_evaluation_experiment():
    """Run comprehensive baseline evaluation experiment"""
    
    print("=" * 80)
    print("SYNTHETIC DATASET BASELINE EVALUATION EXPERIMENT")
    print("=" * 80)
    
    start_time = time.time()
    
    # Create output directory
    output_dir = Path("synthetic_experiment_results")
    output_dir.mkdir(exist_ok=True)
    
    # Get available functions
    synthetic_functions = get_available_synthetic_functions()
    
    if not synthetic_functions:
        print("No synthetic functions found. Please ensure test_programs/ contains synthetic functions.")
        return
    
    # Initialize baseline evaluator
    evaluator = BaselineEvaluator()
    
    # Baseline methods to test
    baseline_methods = [
        "random_testing", "adaptive_random_testing", "hill_climbing",
        "simulated_annealing", "coverage_guided", "boundary_value",
        "equivalence_class", "systematic_testing"
    ]
    
    print(f"Evaluating {len(baseline_methods)} baseline methods on {len(synthetic_functions)} synthetic functions")
    print(f"Methods: {', '.join(baseline_methods)}")
    print(f"Functions: {', '.join(synthetic_functions[:5])}{'...' if len(synthetic_functions) > 5 else ''}")
    print()
    
    # Store all results
    experiment_results = {
        'metadata': {
            'start_time': datetime.now().isoformat(),
            'experiment_type': 'synthetic_baseline_evaluation',
            'functions': synthetic_functions,
            'methods': baseline_methods,
            'repetitions_per_method': 10
        },
        'results': {}
    }
    
    total_evaluations = 0
    completed_evaluations = 0
    
    # Run evaluation for each function
    for i, func_name in enumerate(synthetic_functions):
        print(f"[{i+1}/{len(synthetic_functions)}] Evaluating function: {func_name}")
        
        func_results = {}
        
        for j, method in enumerate(baseline_methods):
            print(f"  [{j+1}/{len(baseline_methods)}] Running method: {method}")
            
            try:
                # Run baseline method with 10 repetitions
                repetitions = 10
                method_results = []
                
                for rep in range(repetitions):
                    try:
                        result = evaluator.evaluate_single_run(
                            function_name=func_name,
                            method=method,
                            timeout=30.0
                        )
                        if result:
                            method_results.append(result)
                        total_evaluations += 1
                    except Exception as e:
                        print(f"    Run {rep+1} failed: {e}")
                        total_evaluations += 1
                
                if method_results:
                    # Calculate statistics
                    coverages = [r.get('coverage', 0) for r in method_results]
                    exec_times = [r.get('execution_time', 0) for r in method_results]
                    
                    func_results[method] = {
                        'raw_results': method_results,
                        'statistics': {
                            'mean_coverage': float(np.mean(coverages)),
                            'std_coverage': float(np.std(coverages)),
                            'max_coverage': float(np.max(coverages)),
                            'min_coverage': float(np.min(coverages)),
                            'mean_time': float(np.mean(exec_times)),
                            'successful_runs': len(method_results),
                            'total_runs': repetitions
                        },
                        'status': 'completed'
                    }
                    completed_evaluations += len(method_results)
                    print(f"    Completed: {len(method_results)}/{repetitions} runs, avg coverage: {np.mean(coverages):.3f}")
                else:
                    func_results[method] = {
                        'status': 'failed',
                        'successful_runs': 0,
                        'total_runs': repetitions
                    }
                    print(f"    Failed: All runs failed")
                    
            except Exception as e:
                print(f"    Method failed entirely: {e}")
                func_results[method] = {
                    'status': 'error',
                    'error': str(e),
                    'successful_runs': 0,
                    'total_runs': 10
                }
                total_evaluations += 10
        
        experiment_results['results'][func_name] = func_results
        print()
    
    # Add final metadata
    end_time = time.time()
    experiment_results['metadata'].update({
        'end_time': datetime.now().isoformat(),
        'duration_seconds': end_time - start_time,
        'total_evaluations': total_evaluations,
        'completed_evaluations': completed_evaluations,
        'success_rate': completed_evaluations / total_evaluations if total_evaluations > 0 else 0
    })
    
    # Save detailed results
    results_file = output_dir / f"baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    # Generate summary report
    generate_experiment_report(experiment_results, output_dir)
    
    print("=" * 80)
    print("EXPERIMENT COMPLETED!")
    print("=" * 80)
    print(f"Runtime: {(end_time - start_time)/60:.1f} minutes")
    print(f"Total evaluations: {total_evaluations}")
    print(f"Successful evaluations: {completed_evaluations}")
    print(f"Success rate: {completed_evaluations/total_evaluations*100:.1f}%")
    print(f"Results saved to: {results_file}")
    print()
    
    return experiment_results


def generate_experiment_report(results, output_dir):
    """Generate comprehensive HTML report"""
    
    # Calculate method performance across all functions
    method_performance = {}
    
    for func_name, func_results in results['results'].items():
        for method, method_data in func_results.items():
            if method_data.get('status') == 'completed':
                stats = method_data.get('statistics', {})
                if method not in method_performance:
                    method_performance[method] = []
                method_performance[method].append({
                    'function': func_name,
                    'coverage': stats.get('mean_coverage', 0),
                    'time': stats.get('mean_time', 0),
                    'runs': stats.get('successful_runs', 0)
                })
    
    # Calculate overall rankings
    method_rankings = {}
    for method, performances in method_performance.items():
        coverages = [p['coverage'] for p in performances]
        times = [p['time'] for p in performances]
        
        method_rankings[method] = {
            'overall_mean_coverage': float(np.mean(coverages)),
            'overall_std_coverage': float(np.std(coverages)),
            'functions_evaluated': len(performances),
            'total_runs': sum(p['runs'] for p in performances),
            'mean_execution_time': float(np.mean(times))
        }
    
    # Sort by coverage
    ranked_methods = sorted(method_rankings.items(), 
                           key=lambda x: x[1]['overall_mean_coverage'], 
                           reverse=True)
    
    # Generate HTML report
    metadata = results['metadata']
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Synthetic Dataset Baseline Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .summary {{ background: #e6f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .section {{ margin: 30px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
        th {{ background: #f2f2f2; font-weight: bold; }}
        .metric {{ color: #2c5aa0; font-weight: bold; }}
        .good {{ color: green; }}
        .average {{ color: orange; }}
        .poor {{ color: red; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Synthetic Dataset Baseline Evaluation Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Duration:</strong> {metadata.get('duration_seconds', 0)/60:.1f} minutes</p>
    </div>

    <div class="summary">
        <h2>Experiment Summary</h2>
        <ul>
            <li><strong>Functions Evaluated:</strong> {len(metadata.get('functions', []))}</li>
            <li><strong>Methods Tested:</strong> {len(metadata.get('methods', []))}</li>
            <li><strong>Total Evaluations:</strong> {metadata.get('total_evaluations', 0)}</li>
            <li><strong>Successful Evaluations:</strong> {metadata.get('completed_evaluations', 0)}</li>
            <li><strong>Success Rate:</strong> {metadata.get('success_rate', 0)*100:.1f}%</li>
        </ul>
    </div>

    <div class="section">
        <h2>Method Performance Rankings</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Method</th>
                <th>Mean Coverage</th>
                <th>Std Dev</th>
                <th>Functions</th>
                <th>Total Runs</th>
                <th>Avg Time (s)</th>
            </tr>
    """
    
    for rank, (method, stats) in enumerate(ranked_methods, 1):
        coverage = stats['overall_mean_coverage']
        coverage_class = "good" if coverage > 0.7 else "average" if coverage > 0.4 else "poor"
        
        html_content += f"""
            <tr>
                <td>{rank}</td>
                <td><strong>{method}</strong></td>
                <td class="{coverage_class}">{coverage:.3f}</td>
                <td>{stats['overall_std_coverage']:.3f}</td>
                <td>{stats['functions_evaluated']}</td>
                <td>{stats['total_runs']}</td>
                <td>{stats['mean_execution_time']:.2f}</td>
            </tr>
        """
    
    html_content += """
        </table>
    </div>

    <div class="section">
        <h2>Per-Function Results Summary</h2>
        <table>
            <tr>
                <th>Function</th>
                <th>Best Method</th>
                <th>Best Coverage</th>
                <th>Methods Completed</th>
            </tr>
    """
    
    for func_name, func_results in results['results'].items():
        best_method = None
        best_coverage = 0
        completed_methods = 0
        
        for method, method_data in func_results.items():
            if method_data.get('status') == 'completed':
                completed_methods += 1
                coverage = method_data.get('statistics', {}).get('mean_coverage', 0)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_method = method
        
        coverage_class = "good" if best_coverage > 0.7 else "average" if best_coverage > 0.4 else "poor"
        
        html_content += f"""
            <tr>
                <td>{func_name}</td>
                <td>{best_method or 'None'}</td>
                <td class="{coverage_class}">{best_coverage:.3f}</td>
                <td>{completed_methods}/{len(func_results)}</td>
            </tr>
        """
    
    html_content += f"""
        </table>
    </div>

    <div class="section">
        <h2>Key Findings</h2>
        <ul>
            <li><strong>Best Overall Method:</strong> {ranked_methods[0][0] if ranked_methods else 'None'} 
                (Coverage: {ranked_methods[0][1]['overall_mean_coverage']:.3f})</li>
            <li><strong>Most Consistent:</strong> {min(ranked_methods, key=lambda x: x[1]['overall_std_coverage'])[0] if ranked_methods else 'None'}</li>
            <li><strong>Fastest Method:</strong> {min(ranked_methods, key=lambda x: x[1]['mean_execution_time'])[0] if ranked_methods else 'None'}</li>
        </ul>
    </div>

    <div class="section">
        <h2>Data Files</h2>
        <p>Detailed experimental data available in JSON format for further analysis.</p>
    </div>

</body>
</html>
    """
    
    # Save HTML report
    report_file = output_dir / f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {report_file}")
    
    return str(report_file)


if __name__ == "__main__":
    run_baseline_evaluation_experiment()