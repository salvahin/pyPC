#!/usr/bin/env python3
"""
Comprehensive Report Generator for Multi-Objective Algorithm Analysis
Generates detailed insights and recommendations from experiment results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import scipy.stats as stats
from datetime import datetime


class MOComprehensiveReporter:
    """Generate comprehensive analysis reports for MO algorithm comparison"""
    
    def __init__(self, results_df: pd.DataFrame, verbose: bool = True):
        """
        Initialize report generator
        
        Args:
            results_df: DataFrame from analyze_and_compare
            verbose: Print progress messages
        """
        self.df = results_df
        self.verbose = verbose
        self.report_sections = []
        
    def generate_executive_summary(self) -> str:
        """Generate executive summary with key findings"""
        summary = []
        summary.append("=" * 80)
        summary.append("EXECUTIVE SUMMARY")
        summary.append("=" * 80)
        summary.append("")
        
        # Overall best algorithm
        if 'HV_Mean' in self.df.columns:
            best_hv = self.df.groupby('Algorithm')['HV_Mean'].mean().idxmax()
            summary.append(f"• Best Overall Algorithm (Hypervolume): {best_hv}")
        
        if 'IGD_Mean' in self.df.columns:
            best_igd = self.df.groupby('Algorithm')['IGD_Mean'].mean().idxmin()
            summary.append(f"• Best Overall Algorithm (IGD): {best_igd}")
        
        if 'coverage_mean' in self.df.columns:
            best_cov = self.df.groupby('Algorithm')['coverage_mean'].mean().idxmax()
            avg_cov = self.df.groupby('Algorithm')['coverage_mean'].mean().max()
            summary.append(f"• Best Coverage Algorithm: {best_cov} ({avg_cov:.2%} average)")
        
        if 'Time_Mean' in self.df.columns:
            fastest = self.df.groupby('Algorithm')['Time_Mean'].mean().idxmin()
            summary.append(f"• Fastest Algorithm: {fastest}")
        
        # Coverage milestones
        summary.append("")
        summary.append("Coverage Milestone Performance:")
        for target in [50, 75, 90]:
            col = f'time_to_{target}pct'
            if col in self.df.columns:
                valid_times = self.df[self.df[col] > 0]
                if not valid_times.empty:
                    best_alg = valid_times.groupby('Algorithm')[col].mean().idxmin()
                    best_time = valid_times.groupby('Algorithm')[col].mean().min()
                    summary.append(f"  • Fastest to {target}% coverage: {best_alg} ({best_time:.1f} generations)")
        
        # Robustness
        if 'performance_cv' in self.df.columns:
            most_stable = self.df.groupby('Algorithm')['performance_cv'].mean().idxmin()
            summary.append(f"\n• Most Stable Algorithm: {most_stable}")
        
        # Diversity
        if 'unique_solutions_mean' in self.df.columns:
            most_diverse = self.df.groupby('Algorithm')['unique_solutions_mean'].mean().idxmax()
            summary.append(f"• Most Diverse Solutions: {most_diverse}")
        
        return "\n".join(summary)
    
    def generate_algorithm_profiles(self) -> str:
        """Generate detailed profiles for each algorithm"""
        profiles = []
        profiles.append("\n" + "=" * 80)
        profiles.append("ALGORITHM PROFILES")
        profiles.append("=" * 80)
        
        for algorithm in self.df['Algorithm'].unique():
            alg_data = self.df[self.df['Algorithm'] == algorithm]
            
            profiles.append(f"\n{algorithm}")
            profiles.append("-" * len(algorithm))
            
            # Strengths
            strengths = []
            weaknesses = []
            
            # Performance analysis
            if 'HV_Mean' in alg_data.columns:
                hv_rank = alg_data['HV_Mean_Rank'].mean() if 'HV_Mean_Rank' in alg_data.columns else None
                if hv_rank and hv_rank <= 2:
                    strengths.append("Strong hypervolume performance")
                elif hv_rank and hv_rank > 3:
                    weaknesses.append("Poor hypervolume performance")
            
            # Coverage analysis
            if 'coverage_mean' in alg_data.columns:
                avg_coverage = alg_data['coverage_mean'].mean()
                if avg_coverage > 0.8:
                    strengths.append(f"Excellent coverage ({avg_coverage:.1%})")
                elif avg_coverage < 0.5:
                    weaknesses.append(f"Low coverage ({avg_coverage:.1%})")
            
            # Efficiency analysis
            if 'coverage_efficiency' in alg_data.columns:
                efficiency = alg_data['coverage_efficiency'].mean()
                if efficiency > alg_data['coverage_efficiency'].median():
                    strengths.append("High efficiency")
            
            # Diversity analysis
            if 'solution_redundancy' in alg_data.columns:
                redundancy = alg_data['solution_redundancy'].mean()
                if redundancy < 0.5:
                    strengths.append("Good solution diversity")
                elif redundancy > 0.8:
                    weaknesses.append("High solution redundancy")
            
            # Convergence speed
            if 'time_to_75pct' in alg_data.columns:
                time_75 = alg_data[alg_data['time_to_75pct'] > 0]['time_to_75pct'].mean()
                if not pd.isna(time_75) and time_75 < 30:
                    strengths.append(f"Fast convergence to 75% coverage ({time_75:.0f} gen)")
            
            # Stagnation
            if 'avg_stagnation' in alg_data.columns:
                stagnation = alg_data['avg_stagnation'].mean()
                if stagnation > 20:
                    weaknesses.append(f"Prone to stagnation ({stagnation:.0f} gen average)")
            
            profiles.append("\nStrengths:")
            for s in strengths:
                profiles.append(f"  ✓ {s}")
            
            if weaknesses:
                profiles.append("\nWeaknesses:")
                for w in weaknesses:
                    profiles.append(f"  ✗ {w}")
            
            # Best suited for
            profiles.append("\nBest suited for:")
            if 'coverage_mean' in alg_data.columns and alg_data['coverage_mean'].mean() > 0.7:
                profiles.append("  • High coverage requirements")
            if 'Time_Mean' in alg_data.columns and alg_data['Time_Mean'].mean() < alg_data['Time_Mean'].median():
                profiles.append("  • Time-constrained testing")
            if 'performance_cv' in alg_data.columns and alg_data['performance_cv'].mean() < 0.2:
                profiles.append("  • Consistent results needed")
        
        return "\n".join(profiles)
    
    def generate_statistical_analysis(self) -> str:
        """Generate statistical comparison between algorithms"""
        analysis = []
        analysis.append("\n" + "=" * 80)
        analysis.append("STATISTICAL ANALYSIS")
        analysis.append("=" * 80)
        
        # Prepare data for statistical tests
        algorithms = self.df['Algorithm'].unique()
        programs = self.df['Program'].unique()
        
        # Friedman test for overall comparison
        if len(algorithms) >= 3 and 'HV_Mean' in self.df.columns:
            analysis.append("\nFriedman Test Results (Hypervolume):")
            
            # Prepare data matrix
            data_matrix = []
            for prog in programs:
                row = []
                for alg in algorithms:
                    val = self.df[(self.df['Algorithm'] == alg) & (self.df['Program'] == prog)]['HV_Mean'].values
                    if len(val) > 0:
                        row.append(val[0])
                    else:
                        row.append(0)
                if len(row) == len(algorithms):
                    data_matrix.append(row)
            
            if data_matrix:
                data_matrix = np.array(data_matrix)
                try:
                    statistic, p_value = stats.friedmanchisquare(*data_matrix.T)
                    analysis.append(f"  Statistic: {statistic:.4f}")
                    analysis.append(f"  P-value: {p_value:.4f}")
                    if p_value < 0.05:
                        analysis.append("  Result: Significant differences detected (p < 0.05)")
                    else:
                        analysis.append("  Result: No significant differences (p >= 0.05)")
                except:
                    analysis.append("  Could not perform Friedman test")
        
        # Pairwise comparisons
        if len(algorithms) >= 2:
            analysis.append("\nPairwise Comparisons (Win/Tie/Loss):")
            
            for i, alg1 in enumerate(algorithms):
                for alg2 in algorithms[i+1:]:
                    wins, ties, losses = 0, 0, 0
                    
                    for prog in programs:
                        val1 = self.df[(self.df['Algorithm'] == alg1) & (self.df['Program'] == prog)]['HV_Mean'].values
                        val2 = self.df[(self.df['Algorithm'] == alg2) & (self.df['Program'] == prog)]['HV_Mean'].values
                        
                        if len(val1) > 0 and len(val2) > 0:
                            diff = val1[0] - val2[0]
                            if abs(diff) < 0.01:  # Tie threshold
                                ties += 1
                            elif diff > 0:
                                wins += 1
                            else:
                                losses += 1
                    
                    analysis.append(f"  {alg1} vs {alg2}: {wins}W / {ties}T / {losses}L")
        
        # Effect sizes
        if 'coverage_mean' in self.df.columns:
            analysis.append("\nEffect Sizes (Coverage):")
            
            for i, alg1 in enumerate(algorithms):
                for alg2 in algorithms[i+1:]:
                    data1 = self.df[self.df['Algorithm'] == alg1]['coverage_mean'].dropna()
                    data2 = self.df[self.df['Algorithm'] == alg2]['coverage_mean'].dropna()
                    
                    if len(data1) > 0 and len(data2) > 0:
                        # Cohen's d
                        pooled_std = np.sqrt((np.std(data1)**2 + np.std(data2)**2) / 2)
                        if pooled_std > 0:
                            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                            
                            # Interpret effect size
                            if abs(cohens_d) < 0.2:
                                interpretation = "negligible"
                            elif abs(cohens_d) < 0.5:
                                interpretation = "small"
                            elif abs(cohens_d) < 0.8:
                                interpretation = "medium"
                            else:
                                interpretation = "large"
                            
                            analysis.append(f"  {alg1} vs {alg2}: d={cohens_d:.3f} ({interpretation})")
        
        return "\n".join(analysis)
    
    def generate_recommendations(self) -> str:
        """Generate recommendations based on analysis"""
        recommendations = []
        recommendations.append("\n" + "=" * 80)
        recommendations.append("RECOMMENDATIONS")
        recommendations.append("=" * 80)
        
        # Group programs by complexity if available
        program_groups = {}
        for prog in self.df['Program'].unique():
            prog_data = self.df[self.df['Program'] == prog]
            avg_coverage = prog_data['coverage_mean'].mean() if 'coverage_mean' in prog_data.columns else 0
            
            if avg_coverage > 0.8:
                complexity = "Simple"
            elif avg_coverage > 0.5:
                complexity = "Moderate"
            else:
                complexity = "Complex"
            
            if complexity not in program_groups:
                program_groups[complexity] = []
            program_groups[complexity].append(prog)
        
        recommendations.append("\nAlgorithm Selection Guidelines:")
        
        # Recommendations by program complexity
        for complexity, programs in program_groups.items():
            recommendations.append(f"\nFor {complexity} Programs ({', '.join(programs[:3])}):")
            
            # Find best algorithm for this complexity
            prog_data = self.df[self.df['Program'].isin(programs)]
            
            if not prog_data.empty:
                # Best by coverage
                if 'coverage_mean' in prog_data.columns:
                    best_cov = prog_data.groupby('Algorithm')['coverage_mean'].mean().idxmax()
                    recommendations.append(f"  • Best Coverage: {best_cov}")
                
                # Fastest
                if 'Time_Mean' in prog_data.columns:
                    fastest = prog_data.groupby('Algorithm')['Time_Mean'].mean().idxmin()
                    recommendations.append(f"  • Fastest: {fastest}")
                
                # Most reliable
                if 'performance_cv' in prog_data.columns:
                    most_reliable = prog_data.groupby('Algorithm')['performance_cv'].mean().idxmin()
                    recommendations.append(f"  • Most Reliable: {most_reliable}")
        
        # General recommendations
        recommendations.append("\nGeneral Guidelines:")
        
        # Coverage-focused recommendation
        if 'coverage_mean' in self.df.columns:
            best_coverage_alg = self.df.groupby('Algorithm')['coverage_mean'].mean().idxmax()
            recommendations.append(f"  • For maximum coverage: Use {best_coverage_alg}")
        
        # Speed-focused recommendation
        if 'Time_Mean' in self.df.columns:
            fastest_alg = self.df.groupby('Algorithm')['Time_Mean'].mean().idxmin()
            recommendations.append(f"  • For quick testing: Use {fastest_alg}")
        
        # Diversity-focused recommendation
        if 'unique_solutions_mean' in self.df.columns:
            most_diverse_alg = self.df.groupby('Algorithm')['unique_solutions_mean'].mean().idxmax()
            recommendations.append(f"  • For diverse test cases: Use {most_diverse_alg}")
        
        # Trade-off recommendation
        if 'spread_mean' in self.df.columns:
            best_tradeoff_alg = self.df.groupby('Algorithm')['spread_mean'].mean().idxmin()
            recommendations.append(f"  • For balanced trade-offs: Use {best_tradeoff_alg}")
        
        # Parameter tuning suggestions
        recommendations.append("\nParameter Tuning Suggestions:")
        
        for algorithm in self.df['Algorithm'].unique():
            alg_data = self.df[self.df['Algorithm'] == algorithm]
            
            if 'avg_stagnation' in alg_data.columns:
                avg_stag = alg_data['avg_stagnation'].mean()
                if avg_stag > 20:
                    recommendations.append(f"  • {algorithm}: Consider increasing mutation rate (high stagnation)")
            
            if 'solution_redundancy' in alg_data.columns:
                redundancy = alg_data['solution_redundancy'].mean()
                if redundancy > 0.7:
                    recommendations.append(f"  • {algorithm}: Increase population diversity mechanisms")
        
        return "\n".join(recommendations)
    
    def generate_insights(self) -> str:
        """Generate insights about algorithm behavior"""
        insights = []
        insights.append("\n" + "=" * 80)
        insights.append("KEY INSIGHTS")
        insights.append("=" * 80)
        
        # Coverage patterns
        if 'coverage_mean' in self.df.columns:
            insights.append("\nCoverage Patterns:")
            
            # Check if coverage correlates with program complexity
            simple_progs = self.df[self.df['coverage_mean'] > 0.8]['Program'].unique()
            complex_progs = self.df[self.df['coverage_mean'] < 0.5]['Program'].unique()
            
            if len(simple_progs) > 0:
                insights.append(f"  • Easy programs ({len(simple_progs)}): All algorithms achieve >80% coverage")
            if len(complex_progs) > 0:
                insights.append(f"  • Hard programs ({len(complex_progs)}): All algorithms struggle (<50% coverage)")
            
            # Coverage efficiency
            if 'coverage_efficiency' in self.df.columns:
                best_eff = self.df.groupby('Algorithm')['coverage_efficiency'].mean().idxmax()
                worst_eff = self.df.groupby('Algorithm')['coverage_efficiency'].mean().idxmin()
                ratio = (self.df.groupby('Algorithm')['coverage_efficiency'].mean().max() / 
                        self.df.groupby('Algorithm')['coverage_efficiency'].mean().min())
                insights.append(f"  • {best_eff} is {ratio:.1f}x more efficient than {worst_eff}")
        
        # Convergence patterns
        insights.append("\nConvergence Patterns:")
        
        for target in [50, 75, 90]:
            col = f'time_to_{target}pct'
            if col in self.df.columns:
                valid_data = self.df[self.df[col] > 0]
                if not valid_data.empty:
                    fastest = valid_data.groupby('Algorithm')[col].mean().idxmin()
                    slowest = valid_data.groupby('Algorithm')[col].mean().idxmax()
                    fast_time = valid_data.groupby('Algorithm')[col].mean().min()
                    slow_time = valid_data.groupby('Algorithm')[col].mean().max()
                    insights.append(f"  • To {target}% coverage: {fastest} ({fast_time:.0f} gen) vs {slowest} ({slow_time:.0f} gen)")
        
        # Diversity insights
        if 'diversity_trend' in self.df.columns:
            insights.append("\nDiversity Evolution:")
            
            for algorithm in self.df['Algorithm'].unique():
                alg_data = self.df[self.df['Algorithm'] == algorithm]
                if 'diversity_trend' in alg_data.columns:
                    trend = alg_data['diversity_trend'].mean()
                    if not pd.isna(trend):
                        if trend > 0:
                            insights.append(f"  • {algorithm}: Maintains/increases diversity over time")
                        else:
                            insights.append(f"  • {algorithm}: Loses diversity over time")
        
        # Trade-off quality
        if 'spread_mean' in self.df.columns and 'spacing_mean' in self.df.columns:
            insights.append("\nPareto Front Quality:")
            
            best_spread = self.df.groupby('Algorithm')['spread_mean'].mean().idxmin()
            best_spacing = self.df.groupby('Algorithm')['spacing_mean'].mean().idxmin()
            
            insights.append(f"  • Best spread (diversity): {best_spread}")
            insights.append(f"  • Best spacing (uniformity): {best_spacing}")
            
            if best_spread == best_spacing:
                insights.append(f"  • {best_spread} achieves both good diversity and uniformity")
        
        # Robustness insights
        if 'performance_cv' in self.df.columns:
            insights.append("\nRobustness Analysis:")
            
            cv_values = self.df.groupby('Algorithm')['performance_cv'].mean()
            most_stable = cv_values.idxmin()
            least_stable = cv_values.idxmax()
            
            insights.append(f"  • Most consistent: {most_stable} (CV={cv_values[most_stable]:.3f})")
            insights.append(f"  • Most variable: {least_stable} (CV={cv_values[least_stable]:.3f})")
        
        return "\n".join(insights)
    
    def generate_full_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate complete comprehensive report
        
        Args:
            save_path: Optional path to save report
            
        Returns:
            Complete report as string
        """
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("MULTI-OBJECTIVE ALGORITHM COMPARISON REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        # Add all sections
        report.append(self.generate_executive_summary())
        report.append(self.generate_algorithm_profiles())
        report.append(self.generate_statistical_analysis())
        report.append(self.generate_insights())
        report.append(self.generate_recommendations())
        
        # Footer
        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        full_report = "\n".join(report)
        
        # Save if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(full_report)
            if self.verbose:
                print(f"Report saved to: {save_path}")
        
        return full_report


def generate_comparison_table(df: pd.DataFrame, metrics: List[str], 
                             save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate a comparison table for selected metrics
    
    Args:
        df: Results DataFrame
        metrics: List of metric names to include
        save_path: Optional path to save table
        
    Returns:
        Comparison DataFrame
    """
    # Filter to only requested metrics that exist
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print(f"Warning: No requested metrics found in data")
        return pd.DataFrame()
    
    # Create pivot table
    comparison = df.pivot_table(
        index='Algorithm',
        values=available_metrics,
        aggfunc='mean'
    ).round(4)
    
    # Add rankings
    for metric in available_metrics:
        if metric in ['IGD_Mean', 'Time_Mean', 'performance_cv', 'solution_redundancy']:
            # Lower is better
            comparison[f'{metric}_Rank'] = comparison[metric].rank(ascending=True)
        else:
            # Higher is better
            comparison[f'{metric}_Rank'] = comparison[metric].rank(ascending=False)
    
    if save_path:
        comparison.to_csv(save_path)
    
    return comparison


if __name__ == "__main__":
    # Example usage
    print("Multi-Objective Comprehensive Report Generator")
    print("Load your analysis DataFrame and create a reporter instance:")
    print("  reporter = MOComprehensiveReporter(results_df)")
    print("  report = reporter.generate_full_report('report.txt')")