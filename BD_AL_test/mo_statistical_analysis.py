#!/usr/bin/env python3
"""
Statistical Analysis Tools for Multi-Objective Optimization
Provides statistical tests and effect size calculations for MO algorithm comparison
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class MOStatisticalAnalyzer:
    """Statistical analysis for multi-objective optimization results"""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize statistical analyzer
        
        Args:
            significance_level: Significance level for hypothesis tests
        """
        self.alpha = significance_level
        self.test_results = {}
        
    def wilcoxon_test(self, data1: np.ndarray, data2: np.ndarray,
                      alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Perform Wilcoxon signed-rank test
        
        Args:
            data1: First sample
            data2: Second sample
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Test results dictionary
        """
        if len(data1) != len(data2):
            raise ValueError("Samples must have same length for paired test")
        
        if len(data1) < 5:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'warning': 'Sample size too small (n < 5)'
            }
        
        try:
            statistic, p_value = stats.wilcoxon(data1, data2, 
                                               alternative=alternative)
            
            return {
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'effect_size': self.calculate_effect_size(data1, data2)
            }
        except Exception as e:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'error': str(e)
            }
    
    def friedman_test(self, *samples) -> Dict[str, Any]:
        """
        Perform Friedman test for multiple related samples
        
        Args:
            samples: Multiple related samples
            
        Returns:
            Test results dictionary
        """
        if len(samples) < 3:
            raise ValueError("Need at least 3 samples for Friedman test")
        
        # Check all samples have same length
        n = len(samples[0])
        if not all(len(s) == n for s in samples):
            raise ValueError("All samples must have same length")
        
        if n < 5:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'warning': 'Sample size too small (n < 5)'
            }
        
        try:
            statistic, p_value = stats.friedmanchisquare(*samples)
            
            result = {
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'n_samples': len(samples),
                'sample_size': n
            }
            
            # Perform post-hoc tests if significant
            if result['significant']:
                result['post_hoc'] = self.nemenyi_post_hoc(samples)
            
            return result
            
        except Exception as e:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'error': str(e)
            }
    
    def nemenyi_post_hoc(self, samples: List[np.ndarray]) -> Dict[str, Any]:
        """
        Perform Nemenyi post-hoc test after Friedman test
        
        Args:
            samples: List of samples
            
        Returns:
            Pairwise comparison results
        """
        k = len(samples)  # Number of algorithms
        n = len(samples[0])  # Number of test cases
        
        # Calculate average ranks
        ranks = np.array([stats.rankdata(-s) for s in samples]).T
        avg_ranks = np.mean(ranks, axis=0)
        
        # Critical difference for Nemenyi test
        q_alpha = 2.807  # For alpha=0.05, k=4 (approximate)
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
        
        # Pairwise comparisons
        comparisons = {}
        for i, j in combinations(range(k), 2):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            comparisons[f"{i}_vs_{j}"] = {
                'rank_diff': diff,
                'critical_diff': cd,
                'significant': diff > cd
            }
        
        return {
            'average_ranks': avg_ranks.tolist(),
            'critical_difference': cd,
            'comparisons': comparisons
        }
    
    def kruskal_wallis_test(self, *samples) -> Dict[str, Any]:
        """
        Perform Kruskal-Wallis H test for independent samples
        
        Args:
            samples: Multiple independent samples
            
        Returns:
            Test results dictionary
        """
        if len(samples) < 2:
            raise ValueError("Need at least 2 samples for Kruskal-Wallis test")
        
        try:
            statistic, p_value = stats.kruskal(*samples)
            
            return {
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'n_groups': len(samples)
            }
        except Exception as e:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'error': str(e)
            }
    
    def calculate_effect_size(self, data1: np.ndarray, 
                             data2: np.ndarray) -> Dict[str, float]:
        """
        Calculate various effect size measures
        
        Args:
            data1: First sample
            data2: Second sample
            
        Returns:
            Dictionary of effect size measures
        """
        # Cohen's d
        pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
        else:
            cohens_d = 0.0
        
        # Cliff's delta (non-parametric)
        cliffs_delta = self._cliffs_delta(data1, data2)
        
        # Vargha-Delaney A measure
        vargha_delaney = self._vargha_delaney_a(data1, data2)
        
        return {
            'cohens_d': cohens_d,
            'cliffs_delta': cliffs_delta,
            'vargha_delaney_a': vargha_delaney,
            'interpretation': self._interpret_effect_size(cohens_d, cliffs_delta)
        }
    
    def _cliffs_delta(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Calculate Cliff's delta effect size
        
        Args:
            data1: First sample
            data2: Second sample
            
        Returns:
            Cliff's delta value
        """
        n1, n2 = len(data1), len(data2)
        if n1 == 0 or n2 == 0:
            return 0.0
        
        # Count dominance
        dominance = 0
        for x1 in data1:
            for x2 in data2:
                if x1 > x2:
                    dominance += 1
                elif x1 < x2:
                    dominance -= 1
        
        return dominance / (n1 * n2)
    
    def _vargha_delaney_a(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Calculate Vargha-Delaney A measure
        
        Args:
            data1: First sample
            data2: Second sample
            
        Returns:
            A measure (0.5 = no effect, >0.5 = data1 better)
        """
        n1, n2 = len(data1), len(data2)
        if n1 == 0 or n2 == 0:
            return 0.5
        
        r1 = 0
        for x1 in data1:
            for x2 in data2:
                if x1 > x2:
                    r1 += 1
                elif x1 == x2:
                    r1 += 0.5
        
        return r1 / (n1 * n2)
    
    def _interpret_effect_size(self, cohens_d: float, 
                               cliffs_delta: float) -> str:
        """
        Interpret effect size magnitude
        
        Args:
            cohens_d: Cohen's d value
            cliffs_delta: Cliff's delta value
            
        Returns:
            Interpretation string
        """
        # Use Cliff's delta for interpretation (more robust)
        abs_delta = abs(cliffs_delta)
        
        if abs_delta < 0.147:
            return "negligible"
        elif abs_delta < 0.33:
            return "small"
        elif abs_delta < 0.474:
            return "medium"
        else:
            return "large"
    
    def compare_algorithms_pairwise(self, results_df: pd.DataFrame,
                                   metric: str = 'HV_Mean',
                                   algorithms: List[str] = None) -> pd.DataFrame:
        """
        Perform pairwise comparison of algorithms
        
        Args:
            results_df: DataFrame with algorithm results
            metric: Metric to compare
            algorithms: List of algorithms to compare (None = all)
            
        Returns:
            DataFrame with pairwise comparison results
        """
        if algorithms is None:
            algorithms = results_df['Algorithm'].unique()
        
        comparison_results = []
        
        for alg1, alg2 in combinations(algorithms, 2):
            data1 = results_df[results_df['Algorithm'] == alg1][metric].values
            data2 = results_df[results_df['Algorithm'] == alg2][metric].values
            
            # Ensure same length by matching programs
            programs = results_df['Program'].unique()
            paired_data1 = []
            paired_data2 = []
            
            for prog in programs:
                val1 = results_df[(results_df['Algorithm'] == alg1) & 
                                 (results_df['Program'] == prog)][metric].values
                val2 = results_df[(results_df['Algorithm'] == alg2) & 
                                 (results_df['Program'] == prog)][metric].values
                
                if len(val1) > 0 and len(val2) > 0:
                    paired_data1.append(val1[0])
                    paired_data2.append(val2[0])
            
            if len(paired_data1) >= 5:
                test_result = self.wilcoxon_test(
                    np.array(paired_data1), 
                    np.array(paired_data2)
                )
                
                comparison_results.append({
                    'Algorithm_1': alg1,
                    'Algorithm_2': alg2,
                    'Metric': metric,
                    'P_Value': test_result['p_value'],
                    'Significant': test_result['significant'],
                    'Effect_Size': test_result['effect_size']['cliffs_delta'],
                    'Effect_Magnitude': test_result['effect_size']['interpretation'],
                    'Winner': alg1 if np.mean(paired_data1) > np.mean(paired_data2) else alg2
                })
        
        return pd.DataFrame(comparison_results)
    
    def calculate_critical_difference(self, results_df: pd.DataFrame,
                                     metric: str = 'HV_Mean') -> Dict[str, Any]:
        """
        Calculate critical difference for algorithm ranking
        
        Args:
            results_df: DataFrame with algorithm results
            metric: Metric for ranking
            
        Returns:
            Critical difference analysis
        """
        algorithms = results_df['Algorithm'].unique()
        programs = results_df['Program'].unique()
        
        # Create matrix of algorithm performances
        performance_matrix = []
        for prog in programs:
            prog_values = []
            for alg in algorithms:
                val = results_df[(results_df['Algorithm'] == alg) & 
                               (results_df['Program'] == prog)][metric].values
                if len(val) > 0:
                    prog_values.append(val[0])
                else:
                    prog_values.append(np.nan)
            performance_matrix.append(prog_values)
        
        performance_matrix = np.array(performance_matrix)
        
        # Remove rows with NaN
        valid_rows = ~np.any(np.isnan(performance_matrix), axis=1)
        performance_matrix = performance_matrix[valid_rows]
        
        if len(performance_matrix) < 5:
            return {
                'error': 'Insufficient data for critical difference calculation'
            }
        
        # Calculate average ranks
        ranks = np.array([stats.rankdata(-row) for row in performance_matrix])
        avg_ranks = np.mean(ranks, axis=0)
        
        # Critical difference
        k = len(algorithms)
        n = len(performance_matrix)
        q_alpha = 2.807  # For alpha=0.05, approximate
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
        
        # Create ranking
        ranking = sorted(zip(algorithms, avg_ranks), key=lambda x: x[1])
        
        return {
            'ranking': ranking,
            'average_ranks': dict(zip(algorithms, avg_ranks)),
            'critical_difference': cd,
            'n_datasets': n,
            'significant_differences': self._find_significant_differences(
                algorithms, avg_ranks, cd
            )
        }
    
    def _find_significant_differences(self, algorithms: List[str],
                                     avg_ranks: np.ndarray,
                                     cd: float) -> List[Tuple[str, str]]:
        """
        Find significantly different algorithm pairs
        
        Args:
            algorithms: List of algorithm names
            avg_ranks: Average ranks
            cd: Critical difference
            
        Returns:
            List of significantly different pairs
        """
        sig_pairs = []
        
        for i, j in combinations(range(len(algorithms)), 2):
            if abs(avg_ranks[i] - avg_ranks[j]) > cd:
                sig_pairs.append((algorithms[i], algorithms[j]))
        
        return sig_pairs


def perform_comprehensive_analysis(results_df: pd.DataFrame,
                                  metrics: List[str] = None) -> Dict[str, Any]:
    """
    Perform comprehensive statistical analysis
    
    Args:
        results_df: DataFrame with experimental results
        metrics: List of metrics to analyze
        
    Returns:
        Comprehensive analysis results
    """
    if metrics is None:
        metrics = ['HV_Mean', 'IGD_Mean', 'Solutions_Mean']
    
    analyzer = MOStatisticalAnalyzer()
    analysis_results = {}
    
    for metric in metrics:
        if metric not in results_df.columns:
            continue
        
        # Pairwise comparisons
        pairwise = analyzer.compare_algorithms_pairwise(results_df, metric)
        
        # Critical difference
        cd_analysis = analyzer.calculate_critical_difference(results_df, metric)
        
        # Friedman test
        algorithms = results_df['Algorithm'].unique()
        samples = []
        for alg in algorithms:
            alg_data = results_df[results_df['Algorithm'] == alg][metric].values
            if len(alg_data) > 0:
                samples.append(alg_data)
        
        if len(samples) >= 3:
            friedman = analyzer.friedman_test(*samples)
        else:
            friedman = None
        
        analysis_results[metric] = {
            'pairwise_comparisons': pairwise,
            'critical_difference': cd_analysis,
            'friedman_test': friedman
        }
    
    return analysis_results


def generate_statistical_report(analysis_results: Dict[str, Any],
                               output_file: str = None) -> str:
    """
    Generate statistical analysis report
    
    Args:
        analysis_results: Results from comprehensive analysis
        output_file: Optional file to save report
        
    Returns:
        Report string
    """
    report = []
    report.append("=" * 80)
    report.append("STATISTICAL ANALYSIS REPORT")
    report.append("=" * 80)
    
    for metric, results in analysis_results.items():
        report.append(f"\n## Metric: {metric}")
        report.append("-" * 40)
        
        # Ranking
        if 'critical_difference' in results and 'ranking' in results['critical_difference']:
            report.append("\n### Algorithm Ranking:")
            for i, (alg, rank) in enumerate(results['critical_difference']['ranking'], 1):
                report.append(f"  {i}. {alg}: {rank:.3f}")
            
            cd = results['critical_difference']['critical_difference']
            report.append(f"\nCritical Difference: {cd:.3f}")
        
        # Friedman test
        if results.get('friedman_test') and results['friedman_test'].get('p_value'):
            ft = results['friedman_test']
            report.append(f"\n### Friedman Test:")
            report.append(f"  Statistic: {ft['statistic']:.3f}")
            report.append(f"  P-value: {ft['p_value']:.4f}")
            report.append(f"  Significant: {ft['significant']}")
        
        # Significant pairwise differences
        if 'pairwise_comparisons' in results:
            sig_pairs = results['pairwise_comparisons'][
                results['pairwise_comparisons']['Significant'] == True
            ]
            if len(sig_pairs) > 0:
                report.append("\n### Significant Pairwise Differences:")
                for _, row in sig_pairs.iterrows():
                    report.append(f"  {row['Algorithm_1']} vs {row['Algorithm_2']}: "
                                f"p={row['P_Value']:.4f}, "
                                f"effect={row['Effect_Magnitude']} "
                                f"(winner: {row['Winner']})")
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
    
    return report_text


if __name__ == "__main__":
    # Test with sample data
    print("Statistical Analysis Tools for MO Optimization")
    print("This module should be imported and used with experimental results")