#!/usr/bin/env python3
"""
Unified Statistical Analysis Framework

This module consolidates all statistical analysis capabilities into a single,
comprehensive framework supporting both multi-objective and baseline method comparisons.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import kruskal, mannwhitneyu, friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import yaml
from pathlib import Path
import warnings
from dataclasses import dataclass, asdict
from enum import Enum
import itertools
from datetime import datetime
import logging

warnings.filterwarnings('ignore')


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)


class TestType(Enum):
    """Statistical test types"""
    KRUSKAL_WALLIS = "kruskal_wallis"
    MANN_WHITNEY = "mann_whitney"
    FRIEDMAN = "friedman"
    WILCOXON = "wilcoxon"
    PAIRED_T = "paired_t"
    INDEPENDENT_T = "independent_t"


class CorrectionMethod(Enum):
    """Multiple comparison correction methods"""
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    HOCHBERG = "hochberg"
    FDR_BH = "fdr_bh"
    FDR_BY = "fdr_by"


class EffectSizeType(Enum):
    """Effect size calculation types"""
    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    GLASS_DELTA = "glass_delta"
    CLIFF_DELTA = "cliff_delta"
    VARGHA_DELANEY = "vargha_delaney"


@dataclass
class StatisticalResult:
    """Standardized statistical test result"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    effect_size_type: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size: int = 0
    degrees_of_freedom: Optional[int] = None
    power: Optional[float] = None
    interpretation: str = ""
    significant: bool = False
    corrected_p_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ComparisonResult:
    """Pairwise comparison result"""
    method1: str
    method2: str
    metric: str
    function_name: str
    statistical_test: StatisticalResult
    descriptive_stats: Dict[str, Dict[str, float]]
    bootstrap_ci: Optional[Tuple[float, float]] = None
    practical_significance: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['statistical_test'] = self.statistical_test.to_dict()
        return result


class EffectSizeCalculator:
    """Unified effect size calculations"""
    
    @staticmethod
    def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        if len(x1) == 0 or len(x2) == 0:
            return 0.0
            
        n1, n2 = len(x1), len(x2)
        s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
        
        if n1 < 2 or n2 < 2:
            return 0.0
            
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        return (np.mean(x1) - np.mean(x2)) / pooled_std if pooled_std > 0 else 0.0
    
    @staticmethod
    def hedges_g(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Hedges' g effect size (bias-corrected Cohen's d)"""
        cohens_d = EffectSizeCalculator.cohens_d(x1, x2)
        n = len(x1) + len(x2)
        
        if n <= 9:
            return cohens_d
            
        correction = 1 - (3 / (4 * n - 9))
        return cohens_d * correction
    
    @staticmethod
    def glass_delta(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Glass's delta effect size"""
        if len(x2) < 2:
            return 0.0
            
        s2 = np.std(x2, ddof=1)
        return (np.mean(x1) - np.mean(x2)) / s2 if s2 > 0 else 0.0
    
    @staticmethod
    def cliff_delta(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Cliff's delta (non-parametric effect size)"""
        n1, n2 = len(x1), len(x2)
        if n1 == 0 or n2 == 0:
            return 0.0
            
        dominance = sum(xi > yj for xi in x1 for yj in x2)
        return (2 * dominance) / (n1 * n2) - 1
    
    @staticmethod
    def vargha_delaney_a(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Vargha-Delaney A statistic"""
        n1, n2 = len(x1), len(x2)
        if n1 == 0 or n2 == 0:
            return 0.5
            
        dominance = sum(xi > yj for xi in x1 for yj in x2) + 0.5 * sum(xi == yj for xi in x1 for yj in x2)
        return dominance / (n1 * n2)
    
    @staticmethod
    def interpret_effect_size(effect_size: float, effect_type: EffectSizeType) -> str:
        """Interpret effect size magnitude"""
        abs_effect = abs(effect_size)
        
        if effect_type in [EffectSizeType.COHENS_D, EffectSizeType.HEDGES_G, EffectSizeType.GLASS_DELTA]:
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"
        elif effect_type == EffectSizeType.CLIFF_DELTA:
            if abs_effect < 0.147:
                return "negligible"
            elif abs_effect < 0.33:
                return "small"
            elif abs_effect < 0.474:
                return "medium"
            else:
                return "large"
        elif effect_type == EffectSizeType.VARGHA_DELANEY:
            if abs(abs_effect - 0.5) < 0.06:
                return "negligible"
            elif abs(abs_effect - 0.5) < 0.14:
                return "small"
            elif abs(abs_effect - 0.5) < 0.21:
                return "medium"
            else:
                return "large"
        
        return "unknown"


class BootstrapAnalyzer:
    """Bootstrap analysis for confidence intervals"""
    
    def __init__(self, n_bootstrap: int = 10000, confidence_level: float = 0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def bootstrap_mean_difference(self, x1: np.ndarray, x2: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """Bootstrap confidence interval for mean difference"""
        if len(x1) == 0 or len(x2) == 0:
            return 0.0, (0.0, 0.0)
            
        def bootstrap_sample():
            sample1 = np.random.choice(x1, size=len(x1), replace=True)
            sample2 = np.random.choice(x2, size=len(x2), replace=True)
            return np.mean(sample1) - np.mean(sample2)
        
        bootstrap_diffs = [bootstrap_sample() for _ in range(self.n_bootstrap)]
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_diffs, lower_percentile)
        ci_upper = np.percentile(bootstrap_diffs, upper_percentile)
        
        return np.mean(bootstrap_diffs), (ci_lower, ci_upper)
    
    def bootstrap_effect_size(self, x1: np.ndarray, x2: np.ndarray, 
                            effect_type: EffectSizeType) -> Tuple[float, Tuple[float, float]]:
        """Bootstrap confidence interval for effect size"""
        if len(x1) == 0 or len(x2) == 0:
            return 0.0, (0.0, 0.0)
            
        effect_func = getattr(EffectSizeCalculator, effect_type.value)
        
        def bootstrap_effect():
            sample1 = np.random.choice(x1, size=len(x1), replace=True)
            sample2 = np.random.choice(x2, size=len(x2), replace=True)
            return effect_func(sample1, sample2)
        
        bootstrap_effects = [bootstrap_effect() for _ in range(self.n_bootstrap)]
        bootstrap_effects = np.array(bootstrap_effects)
        
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_effects, lower_percentile)
        ci_upper = np.percentile(bootstrap_effects, upper_percentile)
        
        return np.mean(bootstrap_effects), (ci_lower, ci_upper)


class StatisticalAnalyzer:
    """Unified statistical analyzer for both MO and baseline comparisons"""
    
    def __init__(self, 
                 alpha: float = 0.05,
                 correction_method: CorrectionMethod = CorrectionMethod.FDR_BH,
                 effect_size_type: EffectSizeType = EffectSizeType.HEDGES_G,
                 min_effect_size: float = 0.3,
                 bootstrap_samples: int = 10000,
                 power_analysis: bool = True):
        
        self.alpha = alpha
        self.correction_method = correction_method
        self.effect_size_type = effect_size_type
        self.min_effect_size = min_effect_size
        self.bootstrap_analyzer = BootstrapAnalyzer(bootstrap_samples)
        self.effect_calculator = EffectSizeCalculator()
        self.power_analysis_enabled = power_analysis
        
        self.logger = self._setup_logger()
        
        # Results storage
        self.comparison_results: List[ComparisonResult] = []
        self.meta_analysis_results: Dict[str, Any] = {}
        
        # Power analysis storage
        self.power_results: Dict[str, float] = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('StatisticalAnalyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def perform_statistical_test(self, 
                                data1: np.ndarray, 
                                data2: np.ndarray, 
                                test_type: TestType = TestType.MANN_WHITNEY,
                                paired: bool = False) -> StatisticalResult:
        """Perform statistical significance test"""
        
        if len(data1) == 0 or len(data2) == 0:
            return StatisticalResult(
                test_name=test_type.value,
                statistic=0.0,
                p_value=1.0,
                interpretation="insufficient_data"
            )
        
        try:
            if test_type == TestType.MANN_WHITNEY:
                statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                test_name = "Mann-Whitney U"
                
            elif test_type == TestType.WILCOXON and paired:
                statistic, p_value = wilcoxon(data1, data2)
                test_name = "Wilcoxon Signed-Rank"
                
            elif test_type == TestType.INDEPENDENT_T:
                statistic, p_value = stats.ttest_ind(data1, data2)
                test_name = "Independent t-test"
                
            elif test_type == TestType.PAIRED_T and paired:
                statistic, p_value = stats.ttest_rel(data1, data2)
                test_name = "Paired t-test"
                
            elif test_type == TestType.FRIEDMAN:
                # For Friedman test, data1 and data2 should be part of larger dataset
                statistic, p_value = friedmanchisquare(data1, data2)
                test_name = "Friedman"
                
            elif test_type == TestType.KRUSKAL_WALLIS:
                statistic, p_value = kruskal(data1, data2)
                test_name = "Kruskal-Wallis"
                
            else:
                statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                test_name = "Mann-Whitney U (default)"
            
            # Calculate effect size
            effect_size = getattr(self.effect_calculator, self.effect_size_type.value)(data1, data2)
            effect_interpretation = self.effect_calculator.interpret_effect_size(
                effect_size, self.effect_size_type
            )
            
            # Bootstrap confidence interval for effect size
            _, effect_ci = self.bootstrap_analyzer.bootstrap_effect_size(
                data1, data2, self.effect_size_type
            )
            
            return StatisticalResult(
                test_name=test_name,
                statistic=float(statistic),
                p_value=float(p_value),
                effect_size=float(effect_size),
                effect_size_type=self.effect_size_type.value,
                confidence_interval=effect_ci,
                sample_size=len(data1) + len(data2),
                interpretation=effect_interpretation,
                significant=p_value < self.alpha
            )
            
        except Exception as e:
            self.logger.error(f"Statistical test error: {e}")
            return StatisticalResult(
                test_name=test_type.value,
                statistic=0.0,
                p_value=1.0,
                interpretation="test_failed"
            )
    
    def compare_algorithms_pairwise(self, results_df: pd.DataFrame,
                                   metric: str = 'coverage',
                                   algorithms: List[str] = None) -> pd.DataFrame:
        """Perform pairwise comparison of algorithms"""
        if algorithms is None:
            algorithms = results_df['Algorithm'].unique() if 'Algorithm' in results_df.columns else []
            if len(algorithms) == 0:
                algorithms = results_df.columns[results_df.columns != metric].tolist()
        
        comparison_results = []
        
        for alg1, alg2 in itertools.combinations(algorithms, 2):
            if 'Algorithm' in results_df.columns:
                data1 = results_df[results_df['Algorithm'] == alg1][metric].values
                data2 = results_df[results_df['Algorithm'] == alg2][metric].values
            else:
                data1 = results_df[alg1].values if alg1 in results_df.columns else np.array([])
                data2 = results_df[alg2].values if alg2 in results_df.columns else np.array([])
            
            # Remove NaN values
            data1 = data1[~np.isnan(data1)]
            data2 = data2[~np.isnan(data2)]
            
            if len(data1) >= 5 and len(data2) >= 5:
                test_result = self.perform_statistical_test(data1, data2)
                
                comparison_results.append({
                    'Algorithm_1': alg1,
                    'Algorithm_2': alg2,
                    'Metric': metric,
                    'P_Value': test_result.p_value,
                    'Significant': test_result.significant,
                    'Effect_Size': test_result.effect_size,
                    'Effect_Magnitude': test_result.interpretation,
                    'Winner': alg1 if np.mean(data1) > np.mean(data2) else alg2,
                    'Mean_1': np.mean(data1),
                    'Mean_2': np.mean(data2)
                })
        
        return pd.DataFrame(comparison_results)
    
    def calculate_critical_difference(self, results_df: pd.DataFrame,
                                     metric: str = 'coverage') -> Dict[str, Any]:
        """Calculate critical difference for algorithm ranking"""
        
        if 'Algorithm' in results_df.columns and 'Program' in results_df.columns:
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
        else:
            # Assume columns are algorithms
            algorithms = [col for col in results_df.columns if col != metric]
            performance_matrix = results_df[algorithms].values
        
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
        """Find significantly different algorithm pairs"""
        sig_pairs = []
        
        for i, j in itertools.combinations(range(len(algorithms)), 2):
            if abs(avg_ranks[i] - avg_ranks[j]) > cd:
                sig_pairs.append((algorithms[i], algorithms[j]))
        
        return sig_pairs
    
    def perform_multiple_comparisons(self, data: Dict[str, Any], 
                                   metrics: List[str] = None) -> List[ComparisonResult]:
        """Perform comprehensive pairwise comparisons with multiple correction"""
        
        if metrics is None:
            metrics = ['coverage', 'branch_distance', 'approach_level', 'execution_time']
        
        results = []
        p_values = []
        
        # Extract method combinations from data structure
        all_methods = set()
        functions = []
        
        for func_name, func_data in data.items():
            functions.append(func_name)
            all_methods.update(func_data.keys())
        
        method_pairs = list(itertools.combinations(all_methods, 2))
        
        self.logger.info(f"Comparing {len(method_pairs)} method pairs across {len(metrics)} metrics")
        
        # Perform all pairwise comparisons
        for func_name in functions:
            for metric in metrics:
                for method1, method2 in method_pairs:
                    
                    # Extract data for both methods
                    data1 = self._extract_metric_data(data, func_name, method1, metric)
                    data2 = self._extract_metric_data(data, func_name, method2, metric)
                    
                    if len(data1) == 0 or len(data2) == 0:
                        continue
                    
                    # Perform statistical test
                    stat_result = self.perform_statistical_test(data1, data2)
                    
                    # Calculate descriptive statistics
                    desc_stats = {
                        method1: self._calculate_descriptive_stats(data1),
                        method2: self._calculate_descriptive_stats(data2)
                    }
                    
                    # Bootstrap confidence interval for mean difference
                    _, bootstrap_ci = self.bootstrap_analyzer.bootstrap_mean_difference(data1, data2)
                    
                    # Determine practical significance
                    practical_sig = (stat_result.effect_size is not None and 
                                   abs(stat_result.effect_size) >= self.min_effect_size)
                    
                    comparison = ComparisonResult(
                        method1=method1,
                        method2=method2,
                        metric=metric,
                        function_name=func_name,
                        statistical_test=stat_result,
                        descriptive_stats=desc_stats,
                        bootstrap_ci=bootstrap_ci,
                        practical_significance=practical_sig
                    )
                    
                    results.append(comparison)
                    p_values.append(stat_result.p_value)
        
        # Apply multiple comparison correction
        if p_values:
            corrected_significant, corrected_p_values, _, _ = multipletests(
                p_values, method=self.correction_method.value, alpha=self.alpha
            )
            
            for i, result in enumerate(results):
                result.statistical_test.corrected_p_value = float(corrected_p_values[i])
                result.statistical_test.significant = corrected_significant[i]
        
        self.comparison_results = results
        self.logger.info(f"Completed {len(results)} statistical comparisons")
        
        return results
    
    def _extract_metric_data(self, data: Dict[str, Any], 
                           func_name: str, method: str, metric: str) -> np.ndarray:
        """Extract metric data for a specific function and method"""
        
        if (func_name in data and 
            method in data[func_name] and
            'results' in data[func_name][method]):
            
            results = data[func_name][method]['results']
            if isinstance(results, list) and len(results) > 0:
                values = []
                for result in results:
                    if isinstance(result, dict) and metric in result:
                        values.append(result[metric])
                    elif hasattr(result, metric):
                        values.append(getattr(result, metric))
                return np.array(values, dtype=float)
        
        return np.array([])
    
    def _calculate_descriptive_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate descriptive statistics for data"""
        if len(data) == 0:
            return {}
            
        return {
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'q1': float(np.percentile(data, 25)),
            'q3': float(np.percentile(data, 75)),
            'n': len(data)
        }
    
    def generate_statistical_report(self, output_path: str = "statistical_analysis_report") -> Path:
        """Generate comprehensive statistical analysis report"""
        
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate summary statistics
        summary = self._generate_summary_statistics()
        
        # Save detailed results
        results_file = output_dir / f"detailed_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'comparison_results': [r.to_dict() for r in self.comparison_results],
                'meta_analysis': self.meta_analysis_results,
                'summary_statistics': summary,
                'analysis_config': {
                    'alpha': self.alpha,
                    'correction_method': self.correction_method.value,
                    'effect_size_type': self.effect_size_type.value,
                    'min_effect_size': self.min_effect_size
                }
            }, f, indent=2, cls=NumpyEncoder)
        
        # Generate text report
        text_report = self._generate_text_report(summary)
        text_file = output_dir / f"statistical_report_{timestamp}.txt"
        with open(text_file, 'w') as f:
            f.write(text_report)
        
        self.logger.info(f"Statistical analysis report generated in {output_dir}")
        return output_dir
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the analysis"""
        
        if not self.comparison_results:
            return {}
        
        # Count significant results
        significant_results = [r for r in self.comparison_results if r.statistical_test.significant]
        practically_significant = [r for r in self.comparison_results if r.practical_significance]
        
        # Effect size distribution
        effect_sizes = [r.statistical_test.effect_size for r in self.comparison_results 
                       if r.statistical_test.effect_size is not None]
        
        # Method performance summary
        method_wins = {}
        for result in significant_results:
            if result.statistical_test.effect_size is not None:
                if result.statistical_test.effect_size > 0:
                    winner = result.method1
                else:
                    winner = result.method2
                
                method_wins[winner] = method_wins.get(winner, 0) + 1
        
        return {
            'total_comparisons': len(self.comparison_results),
            'significant_comparisons': len(significant_results),
            'practically_significant': len(practically_significant),
            'significance_rate': len(significant_results) / len(self.comparison_results) if self.comparison_results else 0,
            'practical_significance_rate': len(practically_significant) / len(self.comparison_results) if self.comparison_results else 0,
            'effect_size_distribution': {
                'mean': float(np.mean(effect_sizes)) if effect_sizes else 0,
                'median': float(np.median(effect_sizes)) if effect_sizes else 0,
                'std': float(np.std(effect_sizes)) if effect_sizes else 0,
                'min': float(np.min(effect_sizes)) if effect_sizes else 0,
                'max': float(np.max(effect_sizes)) if effect_sizes else 0
            },
            'method_performance_ranking': dict(sorted(method_wins.items(), key=lambda x: x[1], reverse=True)),
            'analysis_timestamp': datetime.now().isoformat(),
            'multiple_correction_method': self.correction_method.value,
            'effect_size_threshold': self.min_effect_size
        }
    
    def _generate_text_report(self, summary: Dict[str, Any]) -> str:
        """Generate text statistical report"""
        
        report = []
        report.append("=" * 80)
        report.append("UNIFIED STATISTICAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {summary.get('analysis_timestamp', 'Unknown')}")
        report.append(f"Configuration: α={self.alpha}, Correction={self.correction_method.value}, Effect Size={self.effect_size_type.value}")
        report.append("")
        
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Comparisons: {summary.get('total_comparisons', 0)}")
        report.append(f"Statistically Significant: {summary.get('significant_comparisons', 0)} ({summary.get('significance_rate', 0):.1%})")
        report.append(f"Practically Significant: {summary.get('practically_significant', 0)} ({summary.get('practical_significance_rate', 0):.1%})")
        report.append(f"Average Effect Size: {summary.get('effect_size_distribution', {}).get('mean', 0):.3f}")
        report.append("")
        
        # Method performance ranking
        ranking = summary.get('method_performance_ranking', {})
        if ranking:
            report.append("METHOD PERFORMANCE RANKING")
            report.append("-" * 40)
            for i, (method, wins) in enumerate(ranking.items(), 1):
                report.append(f"{i:2d}. {method}: {wins} significant wins")
            report.append("")
        
        # Significant results by metric
        if self.comparison_results:
            metrics = list(set(r.metric for r in self.comparison_results))
            report.append("SIGNIFICANCE BY METRIC")
            report.append("-" * 40)
            
            for metric in metrics:
                metric_results = [r for r in self.comparison_results if r.metric == metric]
                sig_count = sum(1 for r in metric_results if r.statistical_test.significant)
                sig_rate = sig_count / len(metric_results) if metric_results else 0
                report.append(f"{metric}: {sig_count}/{len(metric_results)} ({sig_rate:.1%})")
        
        report_text = "\n".join(report)
        return report_text
    
    def calculate_power_analysis(self, x1: np.ndarray, x2: np.ndarray, 
                                effect_size: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate statistical power for given samples
        
        Post-hoc power analysis as specified in methodology Section 7.4
        """
        if len(x1) == 0 or len(x2) == 0:
            return {'power': 0.0, 'required_n': 0, 'achieved_effect_size': 0.0}
        
        # Calculate achieved effect size if not provided
        if effect_size is None:
            effect_size = self.effect_calculator.hedges_g(x1, x2)
        
        n1, n2 = len(x1), len(x2)
        
        try:
            # Calculate power using normal approximation
            # This is a simplified implementation - would use specialized library in practice
            pooled_std = np.sqrt(((n1 - 1) * np.var(x1, ddof=1) + (n2 - 1) * np.var(x2, ddof=1)) / (n1 + n2 - 2))
            
            if pooled_std == 0:
                return {'power': 1.0 if effect_size > 0 else 0.0, 'required_n': n1, 'achieved_effect_size': effect_size}
            
            # Cohen's approximation for power
            delta = effect_size * np.sqrt(n1 * n2 / (n1 + n2))
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)  # Two-tailed test
            z_beta = stats.norm.cdf(delta - z_alpha)
            
            power = z_beta
            
            # Calculate required sample size for 80% power
            z_80 = stats.norm.ppf(0.8)  # 80% power
            required_n_per_group = 2 * ((z_alpha + z_80) / effect_size) ** 2 if effect_size > 0 else float('inf')
            
            return {
                'power': max(0.0, min(1.0, power)),
                'required_n_per_group': int(required_n_per_group) if required_n_per_group != float('inf') else 0,
                'achieved_effect_size': effect_size,
                'sample_size_n1': n1,
                'sample_size_n2': n2
            }
            
        except Exception as e:
            self.logger.warning(f"Power analysis failed: {e}")
            return {'power': 0.0, 'required_n_per_group': 0, 'achieved_effect_size': effect_size}
    
    def perform_sensitivity_analysis(self, data: Dict[str, Any], 
                                   parameter_variations: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Perform sensitivity analysis by varying statistical parameters
        
        Tests robustness of results to different parameter choices
        """
        sensitivity_results = {}
        original_alpha = self.alpha
        original_correction = self.correction_method
        
        for param_name, values in parameter_variations.items():
            param_results = []
            
            for value in values:
                try:
                    # Temporarily change parameter
                    if param_name == 'alpha':
                        self.alpha = value
                    elif param_name == 'correction_method':
                        if isinstance(value, str):
                            self.correction_method = CorrectionMethod(value)
                    
                    # Re-run analysis with new parameter
                    temp_results = self.perform_multiple_comparisons(data)
                    
                    # Extract key metrics
                    significant_count = sum(1 for r in temp_results if r.statistical_test.significant)
                    avg_effect_size = np.mean([r.statistical_test.effect_size 
                                             for r in temp_results 
                                             if r.statistical_test.effect_size is not None])
                    
                    param_results.append({
                        'parameter_value': value,
                        'significant_results': significant_count,
                        'total_comparisons': len(temp_results),
                        'significance_rate': significant_count / len(temp_results) if temp_results else 0,
                        'average_effect_size': avg_effect_size
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Sensitivity analysis failed for {param_name}={value}: {e}")
            
            sensitivity_results[param_name] = param_results
        
        # Restore original parameters
        self.alpha = original_alpha
        self.correction_method = original_correction
        
        return sensitivity_results
    
    def calculate_confidence_intervals_matrix(self, data: Dict[str, Any], 
                                             confidence_level: float = 0.95) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Calculate confidence intervals for all method comparisons
        
        Returns matrix of confidence intervals for mean differences
        """
        ci_matrix = {}
        
        for func_name, func_data in data.items():
            ci_matrix[func_name] = {}
            methods = list(func_data.keys())
            
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i != j:
                        key = f"{method1}_vs_{method2}"
                        
                        # Extract coverage data (example metric)
                        data1 = self._extract_metric_data(data, func_name, method1, 'coverage')
                        data2 = self._extract_metric_data(data, func_name, method2, 'coverage')
                        
                        if len(data1) > 0 and len(data2) > 0:
                            _, ci = self.bootstrap_analyzer.bootstrap_mean_difference(data1, data2)
                            ci_matrix[func_name][key] = ci
                        else:
                            ci_matrix[func_name][key] = (0.0, 0.0)
        
        return ci_matrix


class AdvancedStatisticalAnalyzer(StatisticalAnalyzer):
    """Advanced statistical analyzer with additional methodology features"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bayesian_analysis_enabled = kwargs.get('bayesian_analysis', False)
    
    def calculate_bayes_factor(self, x1: np.ndarray, x2: np.ndarray) -> Dict[str, float]:
        """
        Calculate Bayes factor for hypothesis testing
        
        Provides alternative to p-values as specified in advanced methodology
        """
        try:
            # Simplified Bayes factor calculation
            # In practice, would use specialized Bayesian libraries
            
            n1, n2 = len(x1), len(x2)
            if n1 == 0 or n2 == 0:
                return {'bayes_factor': 1.0, 'interpretation': 'no_evidence'}
            
            # Calculate t-statistic
            mean_diff = np.mean(x1) - np.mean(x2)
            pooled_std = np.sqrt(((n1-1)*np.var(x1, ddof=1) + (n2-1)*np.var(x2, ddof=1)) / (n1+n2-2))
            
            if pooled_std == 0:
                return {'bayes_factor': float('inf') if mean_diff != 0 else 1.0, 
                       'interpretation': 'extreme_evidence' if mean_diff != 0 else 'no_evidence'}
            
            t_stat = mean_diff / (pooled_std * np.sqrt(1/n1 + 1/n2))
            
            # Approximate Bayes factor using BIC approximation
            log_bf = -0.5 * (n1 + n2 - 2) * np.log(1 + t_stat**2 / (n1 + n2 - 2))
            bayes_factor = np.exp(log_bf)
            
            # Interpret Bayes factor
            if bayes_factor > 100:
                interpretation = 'extreme_evidence'
            elif bayes_factor > 10:
                interpretation = 'strong_evidence'
            elif bayes_factor > 3:
                interpretation = 'moderate_evidence'
            elif bayes_factor > 1:
                interpretation = 'weak_evidence'
            else:
                interpretation = 'no_evidence'
            
            return {
                'bayes_factor': bayes_factor,
                'log_bayes_factor': log_bf,
                'interpretation': interpretation,
                't_statistic': t_stat
            }
            
        except Exception as e:
            self.logger.warning(f"Bayes factor calculation failed: {e}")
            return {'bayes_factor': 1.0, 'interpretation': 'calculation_failed'}
    
    def perform_meta_analysis(self, study_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform meta-analysis across multiple studies/experiments
        
        Combines effect sizes across different experimental conditions
        """
        if not study_results:
            return {'error': 'No study results provided'}
        
        effect_sizes = []
        weights = []
        
        for study in study_results:
            if 'effect_size' in study and 'sample_size' in study:
                effect_sizes.append(study['effect_size'])
                # Weight by sample size (simplified - would use variance in practice)
                weights.append(study['sample_size'])
        
        if not effect_sizes:
            return {'error': 'No valid effect sizes found'}
        
        effect_sizes = np.array(effect_sizes)
        weights = np.array(weights)
        
        # Calculate weighted mean effect size
        weighted_mean_effect = np.average(effect_sizes, weights=weights)
        
        # Calculate heterogeneity
        total_weight = np.sum(weights)
        q_statistic = np.sum(weights * (effect_sizes - weighted_mean_effect)**2)
        
        # I-squared statistic for heterogeneity
        k = len(effect_sizes)
        i_squared = max(0, (q_statistic - (k - 1)) / q_statistic) if q_statistic > 0 else 0
        
        return {
            'weighted_mean_effect_size': weighted_mean_effect,
            'q_statistic': q_statistic,
            'i_squared': i_squared,
            'heterogeneity_interpretation': 'high' if i_squared > 0.75 else 'moderate' if i_squared > 0.5 else 'low',
            'number_of_studies': k,
            'total_sample_size': int(np.sum([s.get('sample_size', 0) for s in study_results]))
        }


# Factory functions for backward compatibility
def create_mo_analyzer(**kwargs) -> StatisticalAnalyzer:
    """Create analyzer configured for MO optimization analysis"""
    return StatisticalAnalyzer(**kwargs)


def create_baseline_analyzer(**kwargs) -> StatisticalAnalyzer:
    """Create analyzer configured for baseline method analysis"""
    return StatisticalAnalyzer(**kwargs)


# Main interface for comprehensive analysis
def perform_comprehensive_analysis(results_df: pd.DataFrame,
                                  metrics: List[str] = None,
                                  **kwargs) -> Dict[str, Any]:
    """Perform comprehensive statistical analysis on results DataFrame"""
    
    if metrics is None:
        # Auto-detect metrics from DataFrame
        potential_metrics = ['coverage', 'HV_Mean', 'IGD_Mean', 'Solutions_Mean', 
                           'branch_distance', 'approach_level', 'execution_time']
        metrics = [m for m in potential_metrics if m in results_df.columns]
    
    analyzer = StatisticalAnalyzer(**kwargs)
    analysis_results = {}
    
    for metric in metrics:
        if metric not in results_df.columns:
            continue
        
        # Pairwise comparisons
        pairwise = analyzer.compare_algorithms_pairwise(results_df, metric)
        
        # Critical difference
        cd_analysis = analyzer.calculate_critical_difference(results_df, metric)
        
        analysis_results[metric] = {
            'pairwise_comparisons': pairwise,
            'critical_difference': cd_analysis
        }
    
    return analysis_results


if __name__ == "__main__":
    """Example usage and testing"""
    print("Unified Statistical Analysis Framework")
    print("This module provides comprehensive statistical analysis capabilities")
    print("for both multi-objective optimization and baseline method comparisons.")