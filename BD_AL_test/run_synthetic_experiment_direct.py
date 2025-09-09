#!/usr/bin/env python3
"""
Direct Synthetic Baseline Experiment Runner

This is a streamlined version that directly executes the baseline evaluation
experiment using the existing framework components.
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Import baseline generators
from baseline_test_generators import BaselineTestGenerator
from baseline_evaluator import BaselineEvaluator
from compare_baselines_vs_mo import BaselineVsMOComparator


def setup_logging() -> logging.Logger:
    """Setup comprehensive logging"""
    logger = logging.getLogger('SyntheticExperiment')
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def run_baseline_evaluation_on_synthetic_functions(output_dir: str = "synthetic_experiment_results") -> Dict[str, Any]:
    """
    Run baseline evaluation on synthetic functions using existing baseline framework
    """
    logger = setup_logging()
    logger.info("Starting Synthetic Dataset Baseline Evaluation")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get list of synthetic functions from test_programs directory
    synthetic_functions = [
        "cryptographic_hash", "avl_tree_operations", "numerical_solver", 
        "matrix_optimizer", "signal_processor", "json_parser_validator",
        "protocol_state_machine", "workflow_engine", "distributed_system",
        "optimization_solver", "resource_scheduler", "cache_manager",
        "event_processor", "lock_free_queue", "statistical_analyzer"
    ]
    
    # Baseline methods to evaluate
    baseline_methods = [
        "random_testing", "adaptive_random_testing", "hill_climbing",
        "simulated_annealing", "genetic_algorithm", "particle_swarm",
        "coverage_guided", "boundary_value", "equivalence_class", "systematic_testing"
    ]
    
    logger.info(f"Evaluating {len(baseline_methods)} baseline methods on {len(synthetic_functions)} synthetic functions")
    
    # Initialize baseline evaluator
    evaluator = BaselineEvaluator()
    
    # Execute baseline evaluation for each function
    all_results = {}
    total_evaluations = 0
    successful_evaluations = 0
    
    for func_name in synthetic_functions:
        logger.info(f"Processing function: {func_name}")
        func_results = {}
        
        for method in baseline_methods:
            logger.info(f"  Running method: {method}")
            
            try:
                # Run baseline method on this function
                # Using 10 repetitions for faster execution
                repetitions = 10
                results = evaluator.run_baseline_method(
                    function_name=func_name,
                    method=method,
                    repetitions=repetitions,
                    timeout=30.0
                )
                
                func_results[method] = {
                    'results': results,
                    'repetitions': repetitions,
                    'status': 'completed'
                }
                
                successful_evaluations += repetitions
                total_evaluations += repetitions
                
                logger.info(f"    Completed {method} on {func_name}: {repetitions} runs")
                
            except Exception as e:
                logger.error(f"    Failed {method} on {func_name}: {e}")
                func_results[method] = {
                    'status': 'failed',
                    'error': str(e),
                    'repetitions': 0
                }
                total_evaluations += repetitions
        
        all_results[func_name] = func_results
    
    # Save baseline results
    baseline_results_file = output_path / f"baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    experiment_summary = {
        'metadata': {
            'experiment_type': 'synthetic_baseline_evaluation',
            'timestamp': datetime.now().isoformat(),
            'functions_evaluated': len(synthetic_functions),
            'methods_evaluated': len(baseline_methods),
            'total_evaluations': total_evaluations,
            'successful_evaluations': successful_evaluations,
            'success_rate': successful_evaluations / total_evaluations if total_evaluations > 0 else 0
        },
        'functions': synthetic_functions,
        'methods': baseline_methods,
        'results': all_results
    }
    
    with open(baseline_results_file, 'w') as f:
        json.dump(experiment_summary, f, indent=2)
    
    logger.info(f"Baseline evaluation completed!")
    logger.info(f"Results saved to: {baseline_results_file}")
    logger.info(f"Success rate: {successful_evaluations}/{total_evaluations} ({successful_evaluations/total_evaluations*100:.1f}%)")
    
    return experiment_summary


def run_comparison_with_mo_results(baseline_results: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Compare baseline results with existing MO results
    """
    logger = setup_logging()
    logger.info("Starting comparison with MO results")
    
    output_path = Path(output_dir)
    
    # Load latest MO results
    mo_results_dirs = list(Path("parallel_mo_results").glob("parallel_run_*"))
    if not mo_results_dirs:
        logger.warning("No MO results found for comparison")
        return {"status": "no_mo_results"}
    
    latest_mo_dir = max(mo_results_dirs, key=lambda x: x.name)
    logger.info(f"Using MO results from: {latest_mo_dir}")
    
    # Initialize comparison framework
    try:
        comparator = BaselineVsMOComparator()
        
        # Perform comparison
        comparison_results = comparator.compare_methods(
            baseline_results=baseline_results,
            mo_results_dir=str(latest_mo_dir),
            output_dir=str(output_path)
        )
        
        # Save comparison results
        comparison_file = output_path / f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        logger.info(f"Comparison completed! Results saved to: {comparison_file}")
        return comparison_results
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return {"status": "comparison_failed", "error": str(e)}


def generate_comprehensive_report(baseline_results: Dict[str, Any], 
                                comparison_results: Dict[str, Any], 
                                output_dir: str) -> str:
    """
    Generate a comprehensive experiment report
    """
    logger = setup_logging()
    logger.info("Generating comprehensive experiment report")
    
    output_path = Path(output_dir)
    
    # Generate summary statistics
    metadata = baseline_results.get('metadata', {})
    
    # Method performance summary
    method_performance = {}
    for func_name, func_results in baseline_results.get('results', {}).items():
        for method, method_data in func_results.items():
            if method_data.get('status') == 'completed':
                results = method_data.get('results', [])
                if results and isinstance(results, list) and len(results) > 0:
                    # Calculate average coverage
                    coverages = [r.get('coverage', 0) for r in results if isinstance(r, dict)]
                    if coverages:
                        avg_coverage = np.mean(coverages)
                        if method not in method_performance:
                            method_performance[method] = []
                        method_performance[method].append(avg_coverage)
    
    # Calculate overall method rankings
    method_rankings = {}
    for method, coverages in method_performance.items():
        method_rankings[method] = {
            'mean_coverage': float(np.mean(coverages)),
            'median_coverage': float(np.median(coverages)),
            'std_coverage': float(np.std(coverages)),
            'functions_evaluated': len(coverages)
        }
    
    # Sort methods by mean coverage
    ranked_methods = sorted(method_rankings.items(), key=lambda x: x[1]['mean_coverage'], reverse=True)
    
    # Generate HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Synthetic Dataset Baseline Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .summary {{ background: #e8f4fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .section {{ margin: 30px 0; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
            th {{ background: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Synthetic Dataset Baseline Evaluation Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <h2>Experiment Summary</h2>
            <ul>
                <li>Functions Evaluated: {metadata.get('functions_evaluated', 0)}</li>
                <li>Methods Evaluated: {metadata.get('methods_evaluated', 0)}</li>
                <li>Total Evaluations: {metadata.get('total_evaluations', 0)}</li>
                <li>Success Rate: {metadata.get('success_rate', 0):.1%}</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>Method Performance Rankings</h2>
            <table>
                <thead>
                    <tr><th>Rank</th><th>Method</th><th>Mean Coverage</th><th>Std Dev</th><th>Functions</th></tr>
                </thead>
                <tbody>
    """
    
    for rank, (method, stats) in enumerate(ranked_methods, 1):
        html_report += f"""
                    <tr>
                        <td>{rank}</td>
                        <td>{method}</td>
                        <td>{stats['mean_coverage']:.3f}</td>
                        <td>{stats['std_coverage']:.3f}</td>
                        <td>{stats['functions_evaluated']}</td>
                    </tr>
        """
    
    html_report += """
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>Function Coverage Analysis</h2>
            <p>Detailed per-function results saved in JSON format.</p>
        </div>
        
        <div class="section">
            <h2>Comparison with Multi-Objective Algorithms</h2>
    """
    
    if comparison_results.get('status') == 'no_mo_results':
        html_report += "<p>No multi-objective results available for comparison.</p>"
    elif comparison_results.get('status') == 'comparison_failed':
        html_report += f"<p>Comparison failed: {comparison_results.get('error', 'Unknown error')}</p>"
    else:
        html_report += "<p>Multi-objective comparison completed. See detailed comparison results file.</p>"
    
    html_report += """
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    report_file = output_path / f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(report_file, 'w') as f:
        f.write(html_report)
    
    logger.info(f"Comprehensive report generated: {report_file}")
    return str(report_file)


def main():
    """Main execution function"""
    print("=" * 80)
    print("SYNTHETIC DATASET BASELINE EVALUATION EXPERIMENT")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Step 1: Run baseline evaluation
        print("Phase 1: Running baseline evaluation on synthetic functions...")
        baseline_results = run_baseline_evaluation_on_synthetic_functions()
        
        # Step 2: Compare with MO results
        print("Phase 2: Comparing with multi-objective results...")
        comparison_results = run_comparison_with_mo_results(
            baseline_results, 
            "synthetic_experiment_results"
        )
        
        # Step 3: Generate comprehensive report
        print("Phase 3: Generating comprehensive report...")
        report_file = generate_comprehensive_report(
            baseline_results, 
            comparison_results, 
            "synthetic_experiment_results"
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Total runtime: {duration/60:.1f} minutes")
        print(f"Report available at: {report_file}")
        print(f"Results directory: synthetic_experiment_results/")
        
        # Print summary statistics
        metadata = baseline_results.get('metadata', {})
        print(f"Functions evaluated: {metadata.get('functions_evaluated', 0)}")
        print(f"Methods evaluated: {metadata.get('methods_evaluated', 0)}")
        print(f"Total evaluations: {metadata.get('total_evaluations', 0)}")
        print(f"Success rate: {metadata.get('success_rate', 0):.1%}")
        
    except Exception as e:
        print(f"EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()