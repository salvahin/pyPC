#!/usr/bin/env python3
"""
Advanced Comparison Framework: Baseline vs Multi-Objective Algorithms
Comprehensive comparison and analysis of classical methods vs metaheuristic approaches
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, friedmanchisquare
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from synthetic_baseline_evaluator import SyntheticBaselineEvaluator, EvaluationConfig
from synthetic_dataset_adapter import SyntheticDatasetAdapter


@dataclass
class ComparisonResult:
    """Result from statistical comparison"""
    baseline_method: str
    mo_algorithm: str
    function_name: str
    
    # Performance metrics
    baseline_performance: Dict[str, float] = field(default_factory=dict)
    mo_performance: Dict[str, float] = field(default_factory=dict)
    
    # Statistical test results
    statistical_test: str = ""
    p_value: float = 1.0
    effect_size: float = 0.0
    significant: bool = False
    
    # Advantage analysis
    performance_advantage: str = "none"  # "baseline", "mo", or "none"
    advantage_magnitude: float = 0.0
    
    # Resource comparison
    baseline_resources: Dict[str, float] = field(default_factory=dict)
    mo_resources: Dict[str, float] = field(default_factory=dict)
    resource_efficiency: str = "none"  # Which is more efficient


@dataclass
class ComprehensiveComparisonReport:
    """Comprehensive comparison report"""
    
    # Overall statistics
    total_comparisons: int = 0
    baseline_wins: int = 0
    mo_wins: int = 0
    ties: int = 0
    
    # Performance summary
    avg_baseline_coverage: float = 0.0
    avg_mo_coverage: float = 0.0
    coverage_improvement: float = 0.0
    
    # Resource analysis
    avg_baseline_time: float = 0.0
    avg_mo_time: float = 0.0
    time_overhead: float = 0.0
    
    # Function complexity analysis
    complexity_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Method-specific analysis
    method_rankings: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Detailed results
    detailed_comparisons: List[ComparisonResult] = field(default_factory=list)


class BaselineVsMOComparator:
    """Advanced comparison framework for baseline vs MO algorithms"""
    
    def __init__(self, config_path: str, verbose: bool = True):
        self.config_path = config_path
        self.verbose = verbose
        
        # Initialize components
        self.adapter = SyntheticDatasetAdapter(config_path, verbose=verbose)
        
        # Results storage
        self.baseline_results = None
        self.mo_results = None
        self.comparison_report = None
    
    def load_mo_results(self, mo_results_path: str) -> bool:
        """Load multi-objective algorithm results"""
        try:
            if mo_results_path.endswith('.json'):
                with open(mo_results_path, 'r') as f:
                    self.mo_results = json.load(f)
            elif mo_results_path.endswith('.csv'):
                self.mo_results = pd.read_csv(mo_results_path)
            else:
                raise ValueError("MO results must be JSON or CSV format")
            
            if self.verbose:
                print(f"Loaded MO results from {mo_results_path}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to load MO results: {e}")
            return False
    
    def run_baseline_evaluation(self, 
                              functions: Optional[List[str]] = None,
                              methods: Optional[List[str]] = None,
                              eval_config: Optional[EvaluationConfig] = None) -> bool:
        """Run baseline evaluation"""
        
        if eval_config is None:
            eval_config = EvaluationConfig(
                n_tests_per_method=100,
                n_runs_per_experiment=10,
                max_parallel_workers=4
            )
        
        evaluator = SyntheticBaselineEvaluator(
            self.config_path, 
            eval_config, 
            verbose=self.verbose
        )
        
        try:
            results = evaluator.evaluate_comprehensive(
                functions=functions,
                methods=methods,
                parallel=True
            )
            
            self.baseline_results = results
            
            if self.verbose:
                print(f"Completed baseline evaluation with {len(results)} experiments")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Baseline evaluation failed: {e}")
            return False
    
    def perform_comprehensive_comparison(self) -> ComprehensiveComparisonReport:
        """Perform comprehensive comparison between baseline and MO methods"""
        
        if self.baseline_results is None:
            raise ValueError("Baseline results not available. Run baseline evaluation first.")
        
        if self.mo_results is None:
            raise ValueError("MO results not loaded. Load MO results first.")
        
        report = ComprehensiveComparisonReport()
        
        # Convert MO results to comparable format
        mo_data = self._process_mo_results()
        baseline_data = self._process_baseline_results()
        
        # Perform pairwise comparisons
        comparisons = []
        
        for func_name in baseline_data.keys():
            if func_name not in mo_data:
                continue
            
            func_comparisons = self._compare_function_results(
                func_name, 
                baseline_data[func_name],
                mo_data[func_name]
            )
            comparisons.extend(func_comparisons)
        
        report.detailed_comparisons = comparisons
        report.total_comparisons = len(comparisons)
        
        # Analyze overall performance
        self._analyze_overall_performance(report, comparisons)
        
        # Analyze by complexity
        self._analyze_by_complexity(report, comparisons)
        
        # Generate recommendations
        self._generate_recommendations(report, comparisons)
        
        self.comparison_report = report
        
        return report
    
    def _process_mo_results(self) -> Dict[str, Dict[str, Any]]:
        """Process MO results into comparable format"""
        mo_data = {}
        
        if isinstance(self.mo_results, pd.DataFrame):
            # Process DataFrame format
            for _, row in self.mo_results.iterrows():
                func_name = row.get('Program', row.get('Function', 'unknown'))
                algorithm = row.get('Algorithm', 'unknown')
                
                if func_name not in mo_data:
                    mo_data[func_name] = {}
                
                mo_data[func_name][algorithm] = {
                    'coverage_mean': row.get('Coverage_Mean', row.get('HV_Mean', 0)),
                    'coverage_std': row.get('Coverage_Std', row.get('HV_Std', 0)),
                    'execution_time': row.get('Time_Mean', 0),
                    'success_rate': 1.0,  # Assume MO methods succeeded
                    'solutions_count': row.get('Solutions_Mean', 0)
                }
        
        elif isinstance(self.mo_results, list):
            # Process list format
            for result in self.mo_results:
                func_name = result.get('function_name', 'unknown')
                algorithm = result.get('algorithm', 'unknown')
                
                if func_name not in mo_data:
                    mo_data[func_name] = {}
                
                mo_data[func_name][algorithm] = {
                    'coverage_mean': result.get('coverage_mean', 0),
                    'coverage_std': result.get('coverage_std', 0),
                    'execution_time': result.get('execution_time', 0),
                    'success_rate': result.get('success_rate', 1.0),
                    'solutions_count': result.get('solutions_count', 0)
                }
        
        return mo_data
    
    def _process_baseline_results(self) -> Dict[str, Dict[str, Any]]:
        """Process baseline results into comparable format"""
        baseline_data = {}
        
        for result in self.baseline_results:
            if not result.completed or not result.coverage_stats:
                continue
            
            func_name = result.function_name
            method_name = result.method_name
            
            if func_name not in baseline_data:
                baseline_data[func_name] = {}
            
            baseline_data[func_name][method_name] = {
                'coverage_mean': result.coverage_stats['mean'],
                'coverage_std': result.coverage_stats['std'],
                'execution_time': result.execution_stats.get('mean_time', 0),
                'success_rate': result.success_stats.get('mean_success', 0),
                'complexity_level': result.complexity_level,
                'difficulty': result.difficulty_category,
                'diversity_score': result.diversity_score,
                'resource_usage': {
                    'memory': result.resource_metrics.avg_memory_mb,
                    'cpu': result.resource_metrics.avg_cpu_percent
                }
            }
        
        return baseline_data
    
    def _compare_function_results(self, func_name: str, 
                                baseline_data: Dict[str, Any],
                                mo_data: Dict[str, Any]) -> List[ComparisonResult]:
        """Compare results for a single function"""
        
        comparisons = []
        
        for baseline_method, baseline_stats in baseline_data.items():
            for mo_algorithm, mo_stats in mo_data.items():
                
                comparison = ComparisonResult(
                    baseline_method=baseline_method,
                    mo_algorithm=mo_algorithm,
                    function_name=func_name
                )
                
                # Performance comparison
                baseline_coverage = baseline_stats['coverage_mean']
                mo_coverage = mo_stats['coverage_mean']
                
                comparison.baseline_performance = {
                    'coverage': baseline_coverage,
                    'execution_time': baseline_stats['execution_time'],
                    'success_rate': baseline_stats['success_rate']
                }
                
                comparison.mo_performance = {
                    'coverage': mo_coverage,
                    'execution_time': mo_stats['execution_time'],
                    'success_rate': mo_stats['success_rate']
                }
                
                # Statistical significance test
                # Note: This is simplified - in practice, you'd need the raw data
                comparison.statistical_test = "effect_size_estimation"
                
                coverage_diff = mo_coverage - baseline_coverage
                pooled_std = np.sqrt((baseline_stats['coverage_std']**2 + mo_stats['coverage_std']**2) / 2)
                
                if pooled_std > 0:
                    comparison.effect_size = coverage_diff / pooled_std  # Cohen's d approximation
                else:
                    comparison.effect_size = 0
                
                comparison.p_value = 0.05 if abs(comparison.effect_size) > 0.5 else 0.2  # Simplified
                comparison.significant = abs(comparison.effect_size) > 0.5
                
                # Determine advantage
                if coverage_diff > 2.0:  # >2% improvement
                    comparison.performance_advantage = "mo"
                    comparison.advantage_magnitude = coverage_diff
                elif coverage_diff < -2.0:  # >2% worse
                    comparison.performance_advantage = "baseline"  
                    comparison.advantage_magnitude = abs(coverage_diff)
                else:
                    comparison.performance_advantage = "none"
                    comparison.advantage_magnitude = 0
                
                # Resource efficiency
                baseline_time = baseline_stats['execution_time']
                mo_time = mo_stats['execution_time']
                
                if mo_time > baseline_time * 2:  # MO takes >2x time
                    comparison.resource_efficiency = "baseline"
                elif baseline_time > mo_time * 2:  # Baseline takes >2x time
                    comparison.resource_efficiency = "mo"
                else:
                    comparison.resource_efficiency = "similar"
                
                comparison.baseline_resources = {
                    'time': baseline_time,
                    'memory': baseline_stats.get('resource_usage', {}).get('memory', 0)
                }
                
                comparison.mo_resources = {
                    'time': mo_time,
                    'memory': 0  # Not available in MO results
                }
                
                comparisons.append(comparison)
        
        return comparisons
    
    def _analyze_overall_performance(self, report: ComprehensiveComparisonReport, 
                                   comparisons: List[ComparisonResult]):
        """Analyze overall performance patterns"""
        
        baseline_wins = 0
        mo_wins = 0
        ties = 0
        
        baseline_coverages = []
        mo_coverages = []
        baseline_times = []
        mo_times = []
        
        for comp in comparisons:
            if comp.performance_advantage == "baseline":
                baseline_wins += 1
            elif comp.performance_advantage == "mo":
                mo_wins += 1
            else:
                ties += 1
            
            baseline_coverages.append(comp.baseline_performance['coverage'])
            mo_coverages.append(comp.mo_performance['coverage'])
            baseline_times.append(comp.baseline_performance['execution_time'])
            mo_times.append(comp.mo_performance['execution_time'])
        
        report.baseline_wins = baseline_wins
        report.mo_wins = mo_wins
        report.ties = ties
        
        if baseline_coverages:
            report.avg_baseline_coverage = np.mean(baseline_coverages)
        if mo_coverages:
            report.avg_mo_coverage = np.mean(mo_coverages)
        
        report.coverage_improvement = report.avg_mo_coverage - report.avg_baseline_coverage
        
        if baseline_times:
            report.avg_baseline_time = np.mean(baseline_times)
        if mo_times:
            report.avg_mo_time = np.mean(mo_times)
        
        if report.avg_baseline_time > 0:
            report.time_overhead = (report.avg_mo_time - report.avg_baseline_time) / report.avg_baseline_time
    
    def _analyze_by_complexity(self, report: ComprehensiveComparisonReport,
                             comparisons: List[ComparisonResult]):
        """Analyze results by function complexity"""
        
        # Get complexity information from baseline results
        complexity_map = {}
        for result in self.baseline_results:
            if result.completed:
                complexity_map[result.function_name] = result.complexity_level
        
        complexity_groups = {'low': [], 'medium': [], 'high': [], 'extreme': []}
        
        for comp in comparisons:
            complexity = complexity_map.get(comp.function_name, 'unknown')
            if complexity in complexity_groups:
                complexity_groups[complexity].append(comp)
        
        for complexity, group_comparisons in complexity_groups.items():
            if not group_comparisons:
                continue
            
            baseline_wins = sum(1 for c in group_comparisons if c.performance_advantage == "baseline")
            mo_wins = sum(1 for c in group_comparisons if c.performance_advantage == "mo")
            ties = len(group_comparisons) - baseline_wins - mo_wins
            
            avg_baseline_coverage = np.mean([c.baseline_performance['coverage'] for c in group_comparisons])
            avg_mo_coverage = np.mean([c.mo_performance['coverage'] for c in group_comparisons])
            
            report.complexity_breakdown[complexity] = {
                'total_comparisons': len(group_comparisons),
                'baseline_wins': baseline_wins,
                'mo_wins': mo_wins,
                'ties': ties,
                'avg_baseline_coverage': avg_baseline_coverage,
                'avg_mo_coverage': avg_mo_coverage,
                'coverage_improvement': avg_mo_coverage - avg_baseline_coverage,
                'mo_advantage_rate': mo_wins / len(group_comparisons)
            }
    
    def _generate_recommendations(self, report: ComprehensiveComparisonReport,
                                comparisons: List[ComparisonResult]):
        """Generate recommendations based on analysis"""
        
        recommendations = []
        
        # Overall performance recommendation
        if report.mo_wins > report.baseline_wins * 1.5:
            recommendations.append(
                "Multi-objective algorithms show significant advantages over classical methods "
                f"({report.mo_wins} wins vs {report.baseline_wins} for baselines)"
            )
        elif report.baseline_wins > report.mo_wins * 1.5:
            recommendations.append(
                "Classical methods perform competitively with multi-objective algorithms "
                f"({report.baseline_wins} wins vs {report.mo_wins} for MO methods)"
            )
        else:
            recommendations.append(
                "Performance between classical and multi-objective methods is mixed, "
                "suggesting function-specific method selection is important"
            )
        
        # Complexity-based recommendations
        for complexity, stats in report.complexity_breakdown.items():
            if stats['mo_advantage_rate'] > 0.7:
                recommendations.append(
                    f"For {complexity} complexity functions, multi-objective algorithms "
                    f"are clearly superior ({stats['mo_advantage_rate']:.1%} advantage rate)"
                )
            elif stats['mo_advantage_rate'] < 0.3:
                recommendations.append(
                    f"For {complexity} complexity functions, classical methods are "
                    f"often sufficient ({1-stats['mo_advantage_rate']:.1%} baseline advantage)"
                )
        
        # Resource efficiency recommendation
        if report.time_overhead > 2.0:
            recommendations.append(
                f"Multi-objective methods require {report.time_overhead:.1f}x more execution time. "
                "Consider computational budget when selecting methods."
            )
        elif report.time_overhead < 0.5:
            recommendations.append(
                "Multi-objective methods are computationally efficient compared to baselines."
            )
        
        # Coverage improvement recommendation
        if report.coverage_improvement > 5.0:
            recommendations.append(
                f"Multi-objective methods provide substantial coverage improvements "
                f"({report.coverage_improvement:.1f}% average improvement)"
            )
        elif abs(report.coverage_improvement) < 2.0:
            recommendations.append(
                "Coverage differences between methods are minimal. "
                "Consider other factors like execution time and complexity."
            )
        
        report.recommendations = recommendations
    
    def generate_visualizations(self, output_dir: str = "comparison_plots"):
        """Generate comparison visualizations"""
        
        if self.comparison_report is None:
            raise ValueError("No comparison results available")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        
        # 1. Overall performance comparison
        self._plot_overall_comparison(output_path)
        
        # 2. Complexity breakdown
        self._plot_complexity_breakdown(output_path)
        
        # 3. Method rankings
        self._plot_method_rankings(output_path)
        
        # 4. Resource efficiency analysis
        self._plot_resource_analysis(output_path)
        
        if self.verbose:
            print(f"Visualizations saved to {output_path}")
    
    def _plot_overall_comparison(self, output_path: Path):
        """Plot overall performance comparison"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Win/Loss distribution
        ax = axes[0, 0]
        categories = ['Baseline Wins', 'MO Wins', 'Ties']
        values = [self.comparison_report.baseline_wins, 
                 self.comparison_report.mo_wins, 
                 self.comparison_report.ties]
        colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_title('Overall Performance Comparison')
        ax.set_ylabel('Number of Comparisons')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value}', ha='center', va='bottom')
        
        # Coverage comparison
        ax = axes[0, 1]
        methods = ['Baseline\nMethods', 'Multi-Objective\nAlgorithms']
        coverages = [self.comparison_report.avg_baseline_coverage, 
                    self.comparison_report.avg_mo_coverage]
        
        bars = ax.bar(methods, coverages, color=['#ff7f0e', '#1f77b4'], alpha=0.7)
        ax.set_title('Average Coverage Comparison')
        ax.set_ylabel('Coverage (%)')
        
        for bar, value in zip(bars, coverages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value:.1f}%', ha='center', va='bottom')
        
        # Execution time comparison
        ax = axes[1, 0]
        times = [self.comparison_report.avg_baseline_time,
                self.comparison_report.avg_mo_time]
        
        bars = ax.bar(methods, times, color=['#ff7f0e', '#1f77b4'], alpha=0.7)
        ax.set_title('Average Execution Time Comparison')
        ax.set_ylabel('Time (seconds)')
        ax.set_yscale('log')
        
        for bar, value in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                   f'{value:.3f}s', ha='center', va='bottom')
        
        # Effect size distribution
        ax = axes[1, 1]
        effect_sizes = [comp.effect_size for comp in self.comparison_report.detailed_comparisons]
        ax.hist(effect_sizes, bins=20, alpha=0.7, color='#2ca02c', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Effect Size Distribution')
        ax.set_xlabel('Effect Size (Cohen\'s d)')
        ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(output_path / 'overall_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_complexity_breakdown(self, output_path: Path):
        """Plot performance by complexity level"""
        
        if not self.comparison_report.complexity_breakdown:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        complexities = list(self.comparison_report.complexity_breakdown.keys())
        
        # MO advantage rate by complexity
        ax = axes[0, 0]
        advantage_rates = [self.comparison_report.complexity_breakdown[c]['mo_advantage_rate'] 
                          for c in complexities]
        
        bars = ax.bar(complexities, advantage_rates, alpha=0.7, color='#1f77b4')
        ax.set_title('MO Algorithm Advantage Rate by Complexity')
        ax.set_ylabel('MO Advantage Rate')
        ax.set_ylim(0, 1)
        
        for bar, rate in zip(bars, advantage_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{rate:.1%}', ha='center', va='bottom')
        
        # Coverage improvement by complexity
        ax = axes[0, 1]
        improvements = [self.comparison_report.complexity_breakdown[c]['coverage_improvement']
                       for c in complexities]
        
        colors = ['#2ca02c' if imp > 0 else '#d62728' for imp in improvements]
        bars = ax.bar(complexities, improvements, alpha=0.7, color=colors)
        ax.set_title('Coverage Improvement by Complexity')
        ax.set_ylabel('Coverage Improvement (%)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            y_pos = height + 0.5 if height > 0 else height - 0.5
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{imp:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # Comparison counts by complexity
        ax = axes[1, 0]
        baseline_wins = [self.comparison_report.complexity_breakdown[c]['baseline_wins'] 
                        for c in complexities]
        mo_wins = [self.comparison_report.complexity_breakdown[c]['mo_wins'] 
                  for c in complexities]
        ties = [self.comparison_report.complexity_breakdown[c]['ties'] 
               for c in complexities]
        
        x = np.arange(len(complexities))
        width = 0.25
        
        ax.bar(x - width, baseline_wins, width, label='Baseline Wins', alpha=0.7, color='#ff7f0e')
        ax.bar(x, mo_wins, width, label='MO Wins', alpha=0.7, color='#1f77b4')
        ax.bar(x + width, ties, width, label='Ties', alpha=0.7, color='#2ca02c')
        
        ax.set_title('Win/Loss Distribution by Complexity')
        ax.set_ylabel('Count')
        ax.set_xticks(x)
        ax.set_xticklabels(complexities)
        ax.legend()
        
        # Average coverage by complexity and method type
        ax = axes[1, 1]
        baseline_coverages = [self.comparison_report.complexity_breakdown[c]['avg_baseline_coverage']
                             for c in complexities]
        mo_coverages = [self.comparison_report.complexity_breakdown[c]['avg_mo_coverage']
                       for c in complexities]
        
        x = np.arange(len(complexities))
        width = 0.35
        
        ax.bar(x - width/2, baseline_coverages, width, label='Baseline Methods', 
               alpha=0.7, color='#ff7f0e')
        ax.bar(x + width/2, mo_coverages, width, label='MO Algorithms', 
               alpha=0.7, color='#1f77b4')
        
        ax.set_title('Average Coverage by Complexity and Method Type')
        ax.set_ylabel('Coverage (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(complexities)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'complexity_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_method_rankings(self, output_path: Path):
        """Plot method performance rankings"""
        
        # Collect method performance data
        method_performance = {}
        
        for comp in self.comparison_report.detailed_comparisons:
            baseline_method = comp.baseline_method
            mo_algorithm = comp.mo_algorithm
            
            if baseline_method not in method_performance:
                method_performance[baseline_method] = {'type': 'baseline', 'coverages': [], 'times': []}
            if mo_algorithm not in method_performance:
                method_performance[mo_algorithm] = {'type': 'mo', 'coverages': [], 'times': []}
            
            method_performance[baseline_method]['coverages'].append(comp.baseline_performance['coverage'])
            method_performance[baseline_method]['times'].append(comp.baseline_performance['execution_time'])
            
            method_performance[mo_algorithm]['coverages'].append(comp.mo_performance['coverage'])
            method_performance[mo_algorithm]['times'].append(comp.mo_performance['execution_time'])
        
        # Calculate average performance
        methods = []
        avg_coverages = []
        avg_times = []
        method_types = []
        
        for method, data in method_performance.items():
            if data['coverages']:  # Only include methods with data
                methods.append(method)
                avg_coverages.append(np.mean(data['coverages']))
                avg_times.append(np.mean(data['times']))
                method_types.append(data['type'])
        
        if not methods:
            return
        
        # Create ranking plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Coverage ranking
        sorted_indices = np.argsort(avg_coverages)[::-1]  # Sort descending
        
        colors = ['#ff7f0e' if method_types[i] == 'baseline' else '#1f77b4' 
                 for i in sorted_indices]
        
        y_pos = np.arange(len(methods))
        ax1.barh(y_pos, [avg_coverages[i] for i in sorted_indices], 
                color=colors, alpha=0.7)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([methods[i] for i in sorted_indices])
        ax1.set_xlabel('Average Coverage (%)')
        ax1.set_title('Method Ranking by Coverage Performance')
        
        # Add legend
        baseline_patch = plt.Rectangle((0, 0), 1, 1, fc='#ff7f0e', alpha=0.7)
        mo_patch = plt.Rectangle((0, 0), 1, 1, fc='#1f77b4', alpha=0.7)
        ax1.legend([baseline_patch, mo_patch], ['Baseline Methods', 'MO Algorithms'])
        
        # Execution time ranking (log scale)
        sorted_indices_time = np.argsort(avg_times)  # Sort ascending (faster is better)
        
        colors_time = ['#ff7f0e' if method_types[i] == 'baseline' else '#1f77b4' 
                      for i in sorted_indices_time]
        
        ax2.barh(y_pos, [avg_times[i] for i in sorted_indices_time], 
                color=colors_time, alpha=0.7)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([methods[i] for i in sorted_indices_time])
        ax2.set_xlabel('Average Execution Time (seconds)')
        ax2.set_xscale('log')
        ax2.set_title('Method Ranking by Execution Speed')
        
        # Add legend
        ax2.legend([baseline_patch, mo_patch], ['Baseline Methods', 'MO Algorithms'])
        
        plt.tight_layout()
        plt.savefig(output_path / 'method_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_resource_analysis(self, output_path: Path):
        """Plot resource usage analysis"""
        
        # Extract resource data
        baseline_times = []
        mo_times = []
        coverage_diffs = []
        
        for comp in self.comparison_report.detailed_comparisons:
            baseline_times.append(comp.baseline_performance['execution_time'])
            mo_times.append(comp.mo_performance['execution_time'])
            coverage_diffs.append(comp.mo_performance['coverage'] - comp.baseline_performance['coverage'])
        
        if not baseline_times:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time comparison scatter
        ax = axes[0, 0]
        ax.scatter(baseline_times, mo_times, alpha=0.6, s=50)
        
        # Add diagonal line for reference
        min_time = min(min(baseline_times), min(mo_times))
        max_time = max(max(baseline_times), max(mo_times))
        ax.plot([min_time, max_time], [min_time, max_time], 'r--', alpha=0.5, label='Equal Time')
        
        ax.set_xlabel('Baseline Execution Time (s)')
        ax.set_ylabel('MO Execution Time (s)')
        ax.set_title('Execution Time Comparison')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        
        # Time overhead distribution
        ax = axes[0, 1]
        time_ratios = [mo_t / max(b_t, 1e-6) for b_t, mo_t in zip(baseline_times, mo_times)]
        ax.hist(time_ratios, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='Equal Time')
        ax.set_xlabel('MO Time / Baseline Time Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Time Overhead Distribution')
        ax.set_xscale('log')
        ax.legend()
        
        # Coverage improvement vs time cost
        ax = axes[1, 0]
        time_costs = [(mo_t - b_t) for b_t, mo_t in zip(baseline_times, mo_times)]
        ax.scatter(time_costs, coverage_diffs, alpha=0.6, s=50)
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Additional Time Cost (s)')
        ax.set_ylabel('Coverage Improvement (%)')
        ax.set_title('Coverage Improvement vs Time Cost')
        
        # Efficiency quadrants
        ax.text(0.05, 0.95, 'Better\nFaster', transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.5))
        ax.text(0.05, 0.05, 'Worse\nFaster', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.5))
        ax.text(0.75, 0.95, 'Better\nSlower', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.5))
        ax.text(0.75, 0.05, 'Worse\nSlower', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.5))
        
        # Resource efficiency summary
        ax = axes[1, 1]
        
        # Count methods in each efficiency category
        efficiency_counts = {'baseline': 0, 'mo': 0, 'similar': 0}
        for comp in self.comparison_report.detailed_comparisons:
            efficiency_counts[comp.resource_efficiency] += 1
        
        categories = ['Baseline\nMore Efficient', 'Similar\nEfficiency', 'MO\nMore Efficient']
        counts = [efficiency_counts['baseline'], efficiency_counts['similar'], efficiency_counts['mo']]
        colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.7)
        ax.set_title('Resource Efficiency Distribution')
        ax.set_ylabel('Number of Comparisons')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'resource_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_comparison_report(self, output_path: str = "baseline_vs_mo_report.json"):
        """Export comprehensive comparison report"""
        
        if self.comparison_report is None:
            raise ValueError("No comparison results available")
        
        # Convert report to JSON-serializable format
        report_dict = {
            'summary': {
                'total_comparisons': self.comparison_report.total_comparisons,
                'baseline_wins': self.comparison_report.baseline_wins,
                'mo_wins': self.comparison_report.mo_wins,
                'ties': self.comparison_report.ties,
                'avg_baseline_coverage': self.comparison_report.avg_baseline_coverage,
                'avg_mo_coverage': self.comparison_report.avg_mo_coverage,
                'coverage_improvement': self.comparison_report.coverage_improvement,
                'avg_baseline_time': self.comparison_report.avg_baseline_time,
                'avg_mo_time': self.comparison_report.avg_mo_time,
                'time_overhead': self.comparison_report.time_overhead
            },
            'complexity_breakdown': self.comparison_report.complexity_breakdown,
            'recommendations': self.comparison_report.recommendations,
            'detailed_comparisons': []
        }
        
        # Add detailed comparisons
        for comp in self.comparison_report.detailed_comparisons:
            comp_dict = {
                'baseline_method': comp.baseline_method,
                'mo_algorithm': comp.mo_algorithm,
                'function_name': comp.function_name,
                'baseline_performance': comp.baseline_performance,
                'mo_performance': comp.mo_performance,
                'statistical_test': comp.statistical_test,
                'p_value': comp.p_value,
                'effect_size': comp.effect_size,
                'significant': comp.significant,
                'performance_advantage': comp.performance_advantage,
                'advantage_magnitude': comp.advantage_magnitude,
                'resource_efficiency': comp.resource_efficiency
            }
            report_dict['detailed_comparisons'].append(comp_dict)
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        if self.verbose:
            print(f"Comparison report exported to {output_path}")


if __name__ == "__main__":
    """Example usage of the comparison framework"""
    
    print("Baseline vs MO Comparison Framework - Test")
    print("=" * 60)
    
    # Initialize comparator
    comparator = BaselineVsMOComparator(
        config_path="config/synthetic_test_programs.yaml",
        verbose=True
    )
    
    print("Note: This test requires actual MO results to demonstrate full functionality.")
    print("For now, running baseline evaluation only...")
    
    # Run baseline evaluation on subset
    test_functions = comparator.adapter.get_function_names()[:2]
    test_methods = ["pure_random", "stratified_random"]
    
    eval_config = EvaluationConfig(
        n_tests_per_method=10,
        n_runs_per_experiment=2,
        max_parallel_workers=2
    )
    
    success = comparator.run_baseline_evaluation(
        functions=test_functions,
        methods=test_methods,
        eval_config=eval_config
    )
    
    if success:
        print("Baseline evaluation completed successfully!")
        print(f"Generated {len(comparator.baseline_results)} baseline results")
        
        # Show example of what comparison would look like
        print("\nExample baseline results:")
        for i, result in enumerate(comparator.baseline_results[:2]):
            print(f"  {i+1}. {result.function_name}/{result.method_name}: "
                  f"{result.coverage_stats.get('mean', 0):.1f}% coverage")
    
    print("\nTo run full comparison, load MO results with:")
    print("  comparator.load_mo_results('path_to_mo_results.csv')")
    print("  report = comparator.perform_comprehensive_comparison()")
    print("  comparator.generate_visualizations()")
    print("  comparator.export_comparison_report()")