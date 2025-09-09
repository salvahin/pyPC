#!/usr/bin/env python3
"""
Demonstration of Enhanced Multi-Objective Analysis System
Shows all new metrics and generates comprehensive reports
"""

import numpy as np
import pandas as pd
from datetime import datetime


def create_demo_results():
    """Create demonstration results with realistic metrics"""
    
    # Simulate experiment results for 4 algorithms and 3 programs
    algorithms = ['NSGA2', 'NSGA3', 'MOEAD', 'CTAEA']
    programs = ['minimum', 'bubble_sort', 'trig_area']
    
    data = []
    
    for alg in algorithms:
        for prog in programs:
            # Base performance varies by algorithm and program
            base_hv = {'NSGA2': 8.5, 'NSGA3': 8.3, 'MOEAD': 7.8, 'CTAEA': 8.4}[alg]
            base_cov = {'minimum': 0.95, 'bubble_sort': 0.75, 'trig_area': 0.60}[prog]
            
            # Add some realistic variation
            hv = base_hv + np.random.normal(0, 0.2)
            coverage = base_cov + np.random.normal(0, 0.05)
            coverage = max(0, min(1, coverage))  # Clamp to [0, 1]
            
            # Time to targets based on program complexity
            if prog == 'minimum':
                time_50 = np.random.randint(5, 10)
                time_75 = np.random.randint(10, 15)
                time_90 = np.random.randint(15, 25)
            elif prog == 'bubble_sort':
                time_50 = np.random.randint(10, 20)
                time_75 = np.random.randint(20, 35)
                time_90 = np.random.randint(35, 50)
            else:
                time_50 = np.random.randint(15, 30)
                time_75 = np.random.randint(30, 45)
                time_90 = -1  # Doesn't reach 90%
            
            # Algorithm-specific characteristics
            if alg == 'NSGA2':
                unique_sols = np.random.randint(40, 60)
                stagnation = np.random.randint(5, 15)
                exec_time = np.random.uniform(0.8, 1.2)
            elif alg == 'NSGA3':
                unique_sols = np.random.randint(45, 65)
                stagnation = np.random.randint(8, 18)
                exec_time = np.random.uniform(1.0, 1.5)
            elif alg == 'MOEAD':
                unique_sols = np.random.randint(60, 80)
                stagnation = np.random.randint(10, 25)
                exec_time = np.random.uniform(1.5, 2.5)
            else:  # CTAEA
                unique_sols = np.random.randint(35, 55)
                stagnation = np.random.randint(3, 12)
                exec_time = np.random.uniform(1.2, 1.8)
            
            row = {
                'Algorithm': alg,
                'Program': prog,
                'Success_Rate': 1.0,
                # Original metrics
                'HV_Mean': hv,
                'HV_Std': np.random.uniform(0.05, 0.15),
                'HV_Max': hv + np.random.uniform(0.1, 0.3),
                'IGD_Mean': np.random.uniform(0.1, 0.3),
                'IGD_Std': np.random.uniform(0.01, 0.05),
                'Solutions_Mean': np.random.randint(10, 50),
                'Solutions_Std': np.random.uniform(2, 8),
                'Time_Mean': exec_time,
                'Time_Std': exec_time * 0.1,
                # Coverage metrics
                'coverage_mean': coverage,
                'coverage_std': np.random.uniform(0.01, 0.03),
                'coverage_max': min(1.0, coverage + 0.05),
                'coverage_efficiency': coverage / (50 * 50),  # coverage per evaluation
                'improvement_rate': coverage / 50,  # coverage gain per generation
                # Time to targets
                'time_to_50pct': time_50,
                'time_to_75pct': time_75,
                'time_to_90pct': time_90,
                'time_to_95pct': -1,
                'time_to_100pct': -1 if prog != 'minimum' else np.random.randint(40, 50),
                # Diversity metrics
                'unique_solutions_mean': unique_sols,
                'unique_solutions_total': unique_sols * 5,  # 5 runs
                'solution_redundancy': 1 - (unique_sols / (50 * 50)),
                'diversity_trend': np.random.uniform(-0.01, 0.01),
                # Robustness metrics
                'performance_cv': np.random.uniform(0.05, 0.15),
                'performance_iqr': np.random.uniform(0.1, 0.3),
                'convergence_gen_mean': np.random.randint(20, 40),
                'convergence_gen_std': np.random.uniform(2, 5),
                # Stagnation
                'avg_stagnation': stagnation,
                'max_stagnation': stagnation + np.random.randint(5, 10),
                # Trade-off metrics
                'spread_mean': np.random.uniform(0.3, 0.6),
                'spread_std': np.random.uniform(0.05, 0.1),
                'spacing_mean': np.random.uniform(0.2, 0.4),
                'spacing_std': np.random.uniform(0.03, 0.08),
                'front_size_mean': np.random.randint(15, 35),
                'front_size_std': np.random.uniform(2, 5),
            }
            
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Add rankings
    for metric in ['HV_Mean', 'Solutions_Mean']:
        df[f'{metric}_Rank'] = df.groupby('Program')[metric].rank(
            ascending=False, method='average'
        )
    
    df['IGD_Mean_Rank'] = df.groupby('Program')['IGD_Mean'].rank(
        ascending=True, method='average'
    )
    
    return df


def demonstrate_enhanced_analysis():
    """Demonstrate the enhanced analysis capabilities"""
    
    print("=" * 80)
    print("ENHANCED MULTI-OBJECTIVE ANALYSIS DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create demo data
    print("Creating demonstration data...")
    df = create_demo_results()
    print(f"  Generated results for {df['Algorithm'].nunique()} algorithms")
    print(f"  Tested on {df['Program'].nunique()} programs")
    print(f"  Total metrics tracked: {len(df.columns)}")
    print()
    
    # Show sample of enhanced metrics
    print("Sample of Enhanced Metrics:")
    print("-" * 80)
    
    sample_alg = 'NSGA2'
    sample_prog = 'bubble_sort'
    sample_row = df[(df['Algorithm'] == sample_alg) & (df['Program'] == sample_prog)].iloc[0]
    
    print(f"\nAlgorithm: {sample_alg}, Program: {sample_prog}")
    print("\nCoverage Metrics:")
    print(f"  • Final Coverage: {sample_row['coverage_mean']:.2%}")
    print(f"  • Coverage Efficiency: {sample_row['coverage_efficiency']:.4f} per evaluation")
    print(f"  • Time to 50% coverage: {sample_row['time_to_50pct']:.0f} generations")
    print(f"  • Time to 75% coverage: {sample_row['time_to_75pct']:.0f} generations")
    
    print("\nDiversity Metrics:")
    print(f"  • Unique Solutions: {sample_row['unique_solutions_mean']:.0f}")
    print(f"  • Solution Redundancy: {sample_row['solution_redundancy']:.2%}")
    print(f"  • Diversity Trend: {sample_row['diversity_trend']:.4f}")
    
    print("\nRobustness Metrics:")
    print(f"  • Performance CV: {sample_row['performance_cv']:.3f}")
    print(f"  • Convergence Generation: {sample_row['convergence_gen_mean']:.0f} ± {sample_row['convergence_gen_std']:.1f}")
    print(f"  • Average Stagnation: {sample_row['avg_stagnation']:.0f} generations")
    
    print("\nTrade-off Quality:")
    print(f"  • Spread: {sample_row['spread_mean']:.3f} ± {sample_row['spread_std']:.3f}")
    print(f"  • Spacing: {sample_row['spacing_mean']:.3f} ± {sample_row['spacing_std']:.3f}")
    print(f"  • Front Size: {sample_row['front_size_mean']:.0f} solutions")
    
    # Algorithm comparison
    print("\n" + "=" * 80)
    print("ALGORITHM COMPARISON")
    print("=" * 80)
    
    # Coverage comparison
    print("\nCoverage Performance:")
    coverage_summary = df.groupby('Algorithm')['coverage_mean'].agg(['mean', 'std'])
    coverage_summary = coverage_summary.sort_values('mean', ascending=False)
    for alg, row in coverage_summary.iterrows():
        print(f"  {alg:8s}: {row['mean']:.2%} ± {row['std']:.2%}")
    
    # Efficiency comparison
    print("\nEfficiency Ranking:")
    efficiency_summary = df.groupby('Algorithm')['coverage_efficiency'].mean().sort_values(ascending=False)
    for alg, eff in efficiency_summary.items():
        print(f"  {alg:8s}: {eff:.5f} coverage/evaluation")
    
    # Speed to targets
    print("\nSpeed to 75% Coverage (generations):")
    speed_summary = df[df['time_to_75pct'] > 0].groupby('Algorithm')['time_to_75pct'].mean().sort_values()
    for alg, time in speed_summary.items():
        print(f"  {alg:8s}: {time:.1f}")
    
    # Diversity comparison
    print("\nSolution Diversity:")
    diversity_summary = df.groupby('Algorithm')['unique_solutions_mean'].mean().sort_values(ascending=False)
    for alg, unique in diversity_summary.items():
        redundancy = df[df['Algorithm'] == alg]['solution_redundancy'].mean()
        print(f"  {alg:8s}: {unique:.0f} unique ({redundancy:.1%} redundancy)")
    
    # Robustness comparison
    print("\nAlgorithm Stability (lower CV = more stable):")
    stability_summary = df.groupby('Algorithm')['performance_cv'].mean().sort_values()
    for alg, cv in stability_summary.items():
        print(f"  {alg:8s}: CV = {cv:.3f}")
    
    # Program complexity analysis
    print("\n" + "=" * 80)
    print("PROGRAM COMPLEXITY ANALYSIS")
    print("=" * 80)
    
    for prog in df['Program'].unique():
        prog_data = df[df['Program'] == prog]
        avg_coverage = prog_data['coverage_mean'].mean()
        avg_time_75 = prog_data[prog_data['time_to_75pct'] > 0]['time_to_75pct'].mean()
        
        print(f"\n{prog}:")
        print(f"  Average Coverage: {avg_coverage:.2%}")
        print(f"  Average Time to 75%: {avg_time_75:.0f} generations" if not pd.isna(avg_time_75) else "  Average Time to 75%: Not reached")
        print(f"  Best Algorithm: {prog_data.loc[prog_data['coverage_mean'].idxmax(), 'Algorithm']}")
        print(f"  Fastest Algorithm: {prog_data.loc[prog_data['Time_Mean'].idxmin(), 'Algorithm']}")
    
    # Insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    # Find complementary algorithms
    print("\nAlgorithm Complementarity:")
    
    # Check which algorithms excel at different aspects
    best_coverage = df.groupby('Algorithm')['coverage_mean'].mean().idxmax()
    best_speed = df.groupby('Algorithm')['Time_Mean'].mean().idxmin()
    best_diversity = df.groupby('Algorithm')['unique_solutions_mean'].mean().idxmax()
    best_stability = df.groupby('Algorithm')['performance_cv'].mean().idxmin()
    
    print(f"  • Best Coverage: {best_coverage}")
    print(f"  • Fastest: {best_speed}")
    print(f"  • Most Diverse: {best_diversity}")
    print(f"  • Most Stable: {best_stability}")
    
    if len(set([best_coverage, best_speed, best_diversity, best_stability])) > 1:
        print("\n  → Different algorithms excel at different objectives")
        print("  → Consider ensemble approach or algorithm selection based on requirements")
    
    # Stagnation analysis
    print("\nStagnation Analysis:")
    high_stagnation = df[df['avg_stagnation'] > 15]
    if not high_stagnation.empty:
        problematic = high_stagnation.groupby('Algorithm').size()
        for alg, count in problematic.items():
            print(f"  • {alg}: High stagnation in {count} test cases")
            print(f"    → Consider increasing mutation rate or diversity mechanisms")
    
    # Generate report
    print("\n" + "=" * 80)
    print("REPORT GENERATION")
    print("=" * 80)
    
    from mo_comprehensive_report import MOComprehensiveReporter
    
    reporter = MOComprehensiveReporter(df, verbose=False)
    
    # Generate and save report
    report_path = "demo_comprehensive_report.txt"
    report = reporter.generate_full_report(report_path)
    
    print(f"\nComprehensive report generated: {report_path}")
    print(f"Report size: {len(report)} characters")
    print(f"Report sections: Executive Summary, Algorithm Profiles, Statistical Analysis,")
    print(f"                 Key Insights, and Recommendations")
    
    # Show a snippet of the executive summary
    print("\n" + "-" * 80)
    print("EXECUTIVE SUMMARY (excerpt):")
    print("-" * 80)
    lines = report.split('\n')
    summary_start = next(i for i, line in enumerate(lines) if 'EXECUTIVE SUMMARY' in line)
    for line in lines[summary_start:summary_start+15]:
        print(line)
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_enhanced_analysis()