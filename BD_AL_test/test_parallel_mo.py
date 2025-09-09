#!/usr/bin/env python3
"""
Test script for parallel MO experiment runner
Demonstrates parallelization and comprehensive analysis
"""

import time
from parallel_mo_experiment_runner import ParallelMOExperimentRunner
from mo_statistical_analysis import perform_comprehensive_analysis, generate_statistical_report
from mo_visualization_dashboard import MOVisualizationDashboard
import pandas as pd
import numpy as np

def main():
    print("="*80)
    print("PARALLEL MULTI-OBJECTIVE EXPERIMENT DEMONSTRATION")
    print("="*80)
    
    # Configure experiment
    print("\n1. CONFIGURATION")
    print("-"*40)
    algorithms = ['NSGA2', 'NSGA3', 'MOEAD', 'CTAEA']
    test_suite = 'basic'
    runs = 3
    generations = 30
    pop_size = 30
    workers = 4
    
    print(f"  Algorithms: {algorithms}")
    print(f"  Test suite: {test_suite}")
    print(f"  Runs per config: {runs}")
    print(f"  Generations: {generations}")
    print(f"  Population: {pop_size}")
    print(f"  Workers: {workers}")
    
    # Create runner
    runner = ParallelMOExperimentRunner(
        algorithms=algorithms,
        test_suite=test_suite,
        n_workers=workers,
        runs_per_config=runs,
        generations=generations,
        pop_size=pop_size,
        objective_type='conflicting',
        verbose=False
    )
    
    # Run experiments
    print("\n2. PARALLEL EXECUTION")
    print("-"*40)
    start_time = time.time()
    results = runner.run_experiments()
    parallel_time = time.time() - start_time
    
    # Calculate sequential time estimate
    total_experiments = len(algorithms) * len(runner.test_programs) * runs
    avg_time_per_exp = parallel_time / workers  # Rough estimate
    sequential_estimate = total_experiments * avg_time_per_exp
    
    print(f"  Total experiments: {total_experiments}")
    print(f"  Parallel execution time: {parallel_time:.1f}s")
    print(f"  Sequential estimate: {sequential_estimate:.1f}s")
    print(f"  Speedup: {sequential_estimate/parallel_time:.1f}x")
    
    # Analyze results
    print("\n3. STATISTICAL ANALYSIS")
    print("-"*40)
    analysis_df = runner.analyze_and_compare()
    
    # Show summary by algorithm
    summary = analysis_df.groupby('Algorithm').agg({
        'Success_Rate': 'mean',
        'HV_Mean': 'mean',
        'IGD_Mean': 'mean',
        'Solutions_Mean': 'mean',
        'Time_Mean': 'mean'
    }).round(3)
    
    print("\nAlgorithm Performance Summary:")
    print(summary.to_string())
    
    # Perform statistical tests
    from mo_statistical_analysis import MOStatisticalAnalyzer
    analyzer = MOStatisticalAnalyzer()
    
    # Collect HV values for each algorithm
    hv_by_alg = {}
    for alg in algorithms:
        hv_values = []
        for prog_results in results[alg].values():
            for result in prog_results:
                if result.is_successful():
                    hv_values.append(result.metrics.get('hypervolume', 0))
        if hv_values:
            hv_by_alg[alg] = np.array(hv_values)
    
    # Friedman test
    if len(hv_by_alg) >= 3:
        # Make samples same length
        min_len = min(len(v) for v in hv_by_alg.values())
        samples = [v[:min_len] for v in hv_by_alg.values()]
        
        friedman_result = analyzer.friedman_test(*samples)
        print(f"\nFriedman Test:")
        print(f"  p-value: {friedman_result.get('p_value', 'N/A'):.4f}")
        print(f"  Significant: {friedman_result.get('significant', False)}")
    
    # Best algorithm identification
    print("\n4. BEST ALGORITHM IDENTIFICATION")
    print("-"*40)
    
    # Rank by different metrics
    rankings = {}
    for metric in ['HV_Mean', 'Solutions_Mean', 'Time_Mean']:
        if metric in analysis_df.columns:
            # Higher is better for HV and Solutions, lower for Time
            ascending = (metric == 'Time_Mean')
            metric_summary = analysis_df.groupby('Algorithm')[metric].mean().sort_values(ascending=ascending)
            rankings[metric] = list(metric_summary.index)
    
    print("Rankings by metric:")
    for metric, ranking in rankings.items():
        print(f"  {metric}: {' > '.join(ranking)}")
    
    # Overall winner (by average rank)
    alg_ranks = {alg: [] for alg in algorithms}
    for metric_ranking in rankings.values():
        for rank, alg in enumerate(metric_ranking, 1):
            alg_ranks[alg].append(rank)
    
    avg_ranks = {alg: np.mean(ranks) for alg, ranks in alg_ranks.items()}
    overall_ranking = sorted(avg_ranks.items(), key=lambda x: x[1])
    
    print(f"\nOverall Winner: {overall_ranking[0][0]} (avg rank: {overall_ranking[0][1]:.1f})")
    
    # Save results
    print("\n5. SAVING RESULTS")
    print("-"*40)
    output_dir = runner.save_results()
    print(f"  Results saved to: {output_dir}")
    
    # Create simple visualization
    print("\n6. VISUALIZATION")
    print("-"*40)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: HV comparison
        ax1 = axes[0]
        alg_names = list(hv_by_alg.keys())
        hv_means = [np.mean(hv_by_alg[alg]) for alg in alg_names]
        hv_stds = [np.std(hv_by_alg[alg]) for alg in alg_names]
        
        x_pos = np.arange(len(alg_names))
        ax1.bar(x_pos, hv_means, yerr=hv_stds, capsize=5)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(alg_names)
        ax1.set_ylabel('Hypervolume')
        ax1.set_title('Hypervolume Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Success rates
        ax2 = axes[1]
        success_rates = analysis_df.groupby('Algorithm')['Success_Rate'].mean().values
        ax2.bar(x_pos, success_rates * 100)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(alg_names)
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Algorithm Reliability')
        ax2.set_ylim([0, 105])
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Multi-Objective Algorithm Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
        print("  Visualization saved")
        
    except Exception as e:
        print(f"  Could not create visualization: {e}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    return output_dir

if __name__ == "__main__":
    output_dir = main()
    print(f"\nAll results available in: {output_dir}")