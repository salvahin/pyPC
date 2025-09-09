#!/usr/bin/env python3
"""
Advanced Statistical Analysis & Reporting System for Synthetic Dataset Baseline Evaluation

This module provides comprehensive statistical analysis tools for comparing baseline methods
against multi-objective algorithms on synthetic datasets. It implements rigorous statistical
testing with multiple comparison corrections and effect size analysis.

Features:
- Non-parametric statistical tests (Kruskal-Wallis, Mann-Whitney U, Friedman)
- Multiple comparison corrections (Bonferroni, Holm, FDR)
- Effect size calculations (Cohen's d, Glass's delta, Hedges' g)
- Bootstrap confidence intervals
- Power analysis and sample size estimation
- Comprehensive reporting with LaTeX and HTML output
- Meta-analysis across function categories
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import kruskal, mannwhitneyu, friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from datetime import datetime
import hashlib


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
    """Statistical test result container"""
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
    """Advanced effect size calculations"""
    
    @staticmethod
    def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(x1), len(x2)
        s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        return (np.mean(x1) - np.mean(x2)) / pooled_std if pooled_std > 0 else 0.0
    
    @staticmethod
    def hedges_g(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Hedges' g effect size (bias-corrected Cohen's d)"""
        cohens_d = EffectSizeCalculator.cohens_d(x1, x2)
        n = len(x1) + len(x2)
        correction = 1 - (3 / (4 * n - 9))
        return cohens_d * correction
    
    @staticmethod
    def glass_delta(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Glass's delta effect size"""
        s2 = np.std(x2, ddof=1)
        return (np.mean(x1) - np.mean(x2)) / s2 if s2 > 0 else 0.0
    
    @staticmethod
    def cliff_delta(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Cliff's delta (non-parametric effect size)"""
        n1, n2 = len(x1), len(x2)
        dominance = sum(xi > yj for xi in x1 for yj in x2)
        return (2 * dominance) / (n1 * n2) - 1
    
    @staticmethod
    def vargha_delaney_a(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Vargha-Delaney A statistic"""
        n1, n2 = len(x1), len(x2)
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
            if abs_effect - 0.5 < 0.06:
                return "negligible"
            elif abs_effect - 0.5 < 0.14:
                return "small"
            elif abs_effect - 0.5 < 0.21:
                return "medium"
            else:
                return "large"
        
        return "unknown"


class BootstrapAnalyzer:
    """Bootstrap analysis for confidence intervals and hypothesis testing"""
    
    def __init__(self, n_bootstrap: int = 10000, confidence_level: float = 0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def bootstrap_mean_difference(self, x1: np.ndarray, x2: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """Bootstrap confidence interval for mean difference"""
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


class SyntheticStatisticalAnalyzer:
    """Comprehensive statistical analyzer for synthetic dataset evaluation"""
    
    def __init__(self, 
                 alpha: float = 0.05,
                 correction_method: CorrectionMethod = CorrectionMethod.FDR_BH,
                 effect_size_type: EffectSizeType = EffectSizeType.HEDGES_G,
                 min_effect_size: float = 0.3,
                 bootstrap_samples: int = 10000):
        
        self.alpha = alpha
        self.correction_method = correction_method
        self.effect_size_type = effect_size_type
        self.min_effect_size = min_effect_size
        self.bootstrap_analyzer = BootstrapAnalyzer(bootstrap_samples)
        self.effect_calculator = EffectSizeCalculator()
        
        self.logger = self._setup_logger()
        
        # Results storage
        self.comparison_results: List[ComparisonResult] = []
        self.meta_analysis_results: Dict[str, Any] = {}
        self.power_analysis_results: Dict[str, Any] = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('SyntheticStatisticalAnalyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_experimental_data(self, baseline_results_path: str, mo_results_path: str) -> Tuple[Dict, Dict]:
        """Load experimental results from both baseline and MO experiments"""
        try:
            with open(baseline_results_path, 'r') as f:
                baseline_data = json.load(f)
            
            with open(mo_results_path, 'r') as f:
                mo_data = json.load(f)
            
            self.logger.info(f"Loaded baseline data: {len(baseline_data)} functions")
            self.logger.info(f"Loaded MO data: {len(mo_data)} functions")
            
            return baseline_data, mo_data
            
        except Exception as e:
            self.logger.error(f"Error loading experimental data: {e}")
            raise
    
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
    
    def perform_multiple_comparisons(self, 
                                   baseline_data: Dict, 
                                   mo_data: Dict,
                                   metrics: List[str] = None) -> List[ComparisonResult]:
        """Perform comprehensive pairwise comparisons with multiple correction"""
        
        if metrics is None:
            metrics = ['coverage', 'branch_distance', 'approach_level', 'execution_time']
        
        results = []
        p_values = []
        comparison_info = []
        
        # Extract all function names
        common_functions = set(baseline_data.keys()) & set(mo_data.keys())
        self.logger.info(f"Analyzing {len(common_functions)} common functions")
        
        # Get all method combinations
        baseline_methods = set()
        mo_methods = set()
        
        for func_name in common_functions:
            if func_name in baseline_data:
                baseline_methods.update(baseline_data[func_name].keys())
            if func_name in mo_data:
                mo_methods.update(mo_data[func_name].keys())
        
        all_methods = list(baseline_methods) + list(mo_methods)
        method_pairs = list(itertools.combinations(all_methods, 2))
        
        self.logger.info(f"Comparing {len(method_pairs)} method pairs across {len(metrics)} metrics")
        
        # Perform all pairwise comparisons
        for func_name in common_functions:
            for metric in metrics:
                for method1, method2 in method_pairs:
                    
                    # Extract data for both methods
                    data1 = self._extract_metric_data(baseline_data, mo_data, func_name, method1, metric)
                    data2 = self._extract_metric_data(baseline_data, mo_data, func_name, method2, metric)
                    
                    if len(data1) == 0 or len(data2) == 0:
                        continue
                    
                    # Perform statistical test
                    stat_result = self.perform_statistical_test(data1, data2)
                    
                    # Calculate descriptive statistics
                    desc_stats = {
                        method1: {
                            'mean': float(np.mean(data1)),
                            'median': float(np.median(data1)),
                            'std': float(np.std(data1)),
                            'min': float(np.min(data1)),
                            'max': float(np.max(data1)),
                            'q1': float(np.percentile(data1, 25)),
                            'q3': float(np.percentile(data1, 75)),
                            'n': len(data1)
                        },
                        method2: {
                            'mean': float(np.mean(data2)),
                            'median': float(np.median(data2)),
                            'std': float(np.std(data2)),
                            'min': float(np.min(data2)),
                            'max': float(np.max(data2)),
                            'q1': float(np.percentile(data2, 25)),
                            'q3': float(np.percentile(data2, 75)),
                            'n': len(data2)
                        }
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
                    comparison_info.append((func_name, method1, method2, metric))
        
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
    
    def _extract_metric_data(self, baseline_data: Dict, mo_data: Dict, 
                           func_name: str, method: str, metric: str) -> np.ndarray:
        """Extract metric data for a specific function and method"""
        
        # Try baseline data first
        if (func_name in baseline_data and 
            method in baseline_data[func_name] and
            'results' in baseline_data[func_name][method]):
            
            results = baseline_data[func_name][method]['results']
            if isinstance(results, list) and len(results) > 0:
                values = []
                for result in results:
                    if isinstance(result, dict) and metric in result:
                        values.append(result[metric])
                    elif hasattr(result, metric):
                        values.append(getattr(result, metric))
                return np.array(values, dtype=float)
        
        # Try MO data
        if (func_name in mo_data and 
            method in mo_data[func_name] and
            'results' in mo_data[func_name][method]):
            
            results = mo_data[func_name][method]['results']
            if isinstance(results, list) and len(results) > 0:
                values = []
                for result in results:
                    if isinstance(result, dict) and metric in result:
                        values.append(result[metric])
                    elif hasattr(result, metric):
                        values.append(getattr(result, metric))
                return np.array(values, dtype=float)
        
        return np.array([])
    
    def perform_meta_analysis(self) -> Dict[str, Any]:
        """Perform meta-analysis across function categories"""
        
        # Load function categories from config
        config_path = Path("config/synthetic_test_programs.yaml")
        if not config_path.exists():
            self.logger.warning("Synthetic config not found, using default categories")
            categories = {"default": list(set(r.function_name for r in self.comparison_results))}
        else:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                categories = self._extract_function_categories(config)
        
        meta_results = {}
        
        for category, functions in categories.items():
            category_results = [r for r in self.comparison_results if r.function_name in functions]
            
            if not category_results:
                continue
            
            # Aggregate effect sizes by metric and method pair
            effect_sizes_by_comparison = {}
            
            for result in category_results:
                key = f"{result.method1}_vs_{result.method2}_{result.metric}"
                if key not in effect_sizes_by_comparison:
                    effect_sizes_by_comparison[key] = []
                
                if result.statistical_test.effect_size is not None:
                    effect_sizes_by_comparison[key].append(result.statistical_test.effect_size)
            
            # Calculate meta-analysis statistics
            meta_stats = {}
            for comparison, effects in effect_sizes_by_comparison.items():
                if len(effects) > 1:
                    effects_array = np.array(effects)
                    meta_stats[comparison] = {
                        'mean_effect': float(np.mean(effects_array)),
                        'se_effect': float(np.std(effects_array) / np.sqrt(len(effects_array))),
                        'ci_lower': float(np.mean(effects_array) - 1.96 * np.std(effects_array) / np.sqrt(len(effects_array))),
                        'ci_upper': float(np.mean(effects_array) + 1.96 * np.std(effects_array) / np.sqrt(len(effects_array))),
                        'heterogeneity_i2': self._calculate_i2(effects_array),
                        'n_studies': len(effects),
                        'interpretation': EffectSizeCalculator.interpret_effect_size(
                            np.mean(effects_array), self.effect_size_type
                        )
                    }
            
            meta_results[category] = meta_stats
        
        self.meta_analysis_results = meta_results
        return meta_results
    
    def _extract_function_categories(self, config: Dict) -> Dict[str, List[str]]:
        """Extract function categories from config"""
        categories = {}
        
        for func_name, func_config in config.items():
            if isinstance(func_config, dict) and 'tags' in func_config:
                for tag in func_config['tags']:
                    if tag not in categories:
                        categories[tag] = []
                    categories[tag].append(func_name)
        
        return categories
    
    def _calculate_i2(self, effects: np.ndarray) -> float:
        """Calculate I² heterogeneity statistic"""
        if len(effects) < 2:
            return 0.0
        
        q = np.sum((effects - np.mean(effects))**2)
        df = len(effects) - 1
        i2 = max(0, (q - df) / q) * 100 if q > 0 else 0.0
        return float(i2)
    
    def generate_comprehensive_report(self, output_path: str = "synthetic_statistical_report"):
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
            }, f, indent=2)
        
        # Generate HTML report
        html_report = self._generate_html_report(summary)
        html_file = output_dir / f"statistical_report_{timestamp}.html"
        with open(html_file, 'w') as f:
            f.write(html_report)
        
        # Generate LaTeX report
        latex_report = self._generate_latex_report(summary)
        latex_file = output_dir / f"statistical_report_{timestamp}.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_report)
        
        # Generate visualizations
        self._generate_statistical_plots(output_dir, timestamp)
        
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
            'significance_rate': len(significant_results) / len(self.comparison_results),
            'practical_significance_rate': len(practically_significant) / len(self.comparison_results),
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
    
    def _generate_html_report(self, summary: Dict[str, Any]) -> str:
        """Generate HTML statistical report"""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Synthetic Dataset Statistical Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .summary { background: #e8f4fd; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .section { margin: 30px 0; }
                .table { width: 100%; border-collapse: collapse; }
                .table th, .table td { padding: 8px; border: 1px solid #ddd; text-align: left; }
                .table th { background: #f2f2f2; }
                .significant { background: #ffeb3b; }
                .practical { background: #4caf50; color: white; }
                .effect-large { font-weight: bold; color: #f44336; }
                .effect-medium { font-weight: bold; color: #ff9800; }
                .effect-small { color: #9e9e9e; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Synthetic Dataset Statistical Analysis Report</h1>
                <p>Generated: {timestamp}</p>
                <p>Analysis Configuration: α = {alpha}, Correction: {correction}, Effect Size: {effect_type}</p>
            </div>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <ul>
                    <li>Total Comparisons: {total_comparisons}</li>
                    <li>Statistically Significant: {significant} ({significance_rate:.1%})</li>
                    <li>Practically Significant: {practical} ({practical_rate:.1%})</li>
                    <li>Average Effect Size: {avg_effect:.3f}</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Method Performance Ranking</h2>
                <table class="table">
                    <thead>
                        <tr><th>Method</th><th>Significant Wins</th><th>Win Rate</th></tr>
                    </thead>
                    <tbody>
                        {method_ranking}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>Detailed Results Summary</h2>
                <p>Effect Size Interpretation: Negligible (&lt;0.2), Small (0.2-0.5), Medium (0.5-0.8), Large (&gt;0.8)</p>
                <p>Statistical significance corrected using {correction} method</p>
            </div>
        </body>
        </html>
        """.format(
            timestamp=summary.get('analysis_timestamp', 'Unknown'),
            alpha=self.alpha,
            correction=self.correction_method.value,
            effect_type=self.effect_size_type.value,
            total_comparisons=summary.get('total_comparisons', 0),
            significant=summary.get('significant_comparisons', 0),
            significance_rate=summary.get('significance_rate', 0),
            practical=summary.get('practically_significant', 0),
            practical_rate=summary.get('practical_significance_rate', 0),
            avg_effect=summary.get('effect_size_distribution', {}).get('mean', 0),
            method_ranking=self._format_method_ranking_html(summary.get('method_performance_ranking', {}))
        )
        
        return html_template
    
    def _format_method_ranking_html(self, ranking: Dict[str, int]) -> str:
        """Format method ranking for HTML table"""
        if not ranking:
            return "<tr><td colspan='3'>No significant results found</td></tr>"
        
        total_wins = sum(ranking.values())
        rows = []
        
        for method, wins in ranking.items():
            win_rate = wins / total_wins if total_wins > 0 else 0
            rows.append(f"<tr><td>{method}</td><td>{wins}</td><td>{win_rate:.1%}</td></tr>")
        
        return "".join(rows)
    
    def _generate_latex_report(self, summary: Dict[str, Any]) -> str:
        """Generate LaTeX statistical report"""
        
        latex_template = r"""
        \documentclass{article}
        \usepackage{booktabs}
        \usepackage{array}
        \usepackage{xcolor}
        \usepackage{geometry}
        \geometry{margin=1in}
        
        \title{Synthetic Dataset Statistical Analysis Report}
        \author{Automated Statistical Analysis System}
        \date{\today}
        
        \begin{document}
        \maketitle
        
        \section{Executive Summary}
        This report presents the results of comprehensive statistical analysis comparing baseline methods against multi-objective algorithms on synthetic test functions.
        
        \begin{itemize}
            \item Total Comparisons: """ + str(summary.get('total_comparisons', 0)) + r"""
            \item Statistically Significant: """ + str(summary.get('significant_comparisons', 0)) + r""" (""" + f"{summary.get('significance_rate', 0):.1%}" + r""")
            \item Practically Significant: """ + str(summary.get('practically_significant', 0)) + r""" (""" + f"{summary.get('practical_significance_rate', 0):.1%}" + r""")
            \item Average Effect Size: """ + f"{summary.get('effect_size_distribution', {}).get('mean', 0):.3f}" + r"""
        \end{itemize}
        
        \section{Analysis Configuration}
        \begin{itemize}
            \item Significance Level ($\alpha$): """ + str(self.alpha) + r"""
            \item Multiple Comparison Correction: """ + self.correction_method.value + r"""
            \item Effect Size Measure: """ + self.effect_size_type.value + r"""
            \item Practical Significance Threshold: """ + str(self.min_effect_size) + r"""
        \end{itemize}
        
        \section{Method Performance Ranking}
        """ + self._format_method_ranking_latex(summary.get('method_performance_ranking', {})) + r"""
        
        \section{Statistical Interpretation}
        Effect sizes are interpreted according to Cohen's conventions:
        \begin{itemize}
            \item Negligible: $|d| < 0.2$
            \item Small: $0.2 \leq |d| < 0.5$
            \item Medium: $0.5 \leq |d| < 0.8$
            \item Large: $|d| \geq 0.8$
        \end{itemize}
        
        All p-values have been corrected for multiple comparisons using the """ + self.correction_method.value + r""" method.
        
        \end{document}
        """
        
        return latex_template
    
    def _format_method_ranking_latex(self, ranking: Dict[str, int]) -> str:
        """Format method ranking for LaTeX table"""
        if not ranking:
            return "No significant results found."
        
        total_wins = sum(ranking.values())
        
        latex_table = r"""
        \begin{table}[h]
        \centering
        \begin{tabular}{lcc}
        \toprule
        Method & Significant Wins & Win Rate \\
        \midrule
        """
        
        for method, wins in ranking.items():
            win_rate = wins / total_wins if total_wins > 0 else 0
            latex_table += f"{method} & {wins} & {win_rate:.1%} \\\\\n"
        
        latex_table += r"""
        \bottomrule
        \end{tabular}
        \caption{Method performance ranking based on significant wins}
        \end{table}
        """
        
        return latex_table
    
    def _generate_statistical_plots(self, output_dir: Path, timestamp: str):
        """Generate statistical visualization plots"""
        
        if not self.comparison_results:
            return
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Plot 1: Effect size distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        effect_sizes = [r.statistical_test.effect_size for r in self.comparison_results 
                       if r.statistical_test.effect_size is not None]
        
        if effect_sizes:
            axes[0, 0].hist(effect_sizes, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(np.mean(effect_sizes), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(effect_sizes):.3f}')
            axes[0, 0].axvline(self.min_effect_size, color='orange', linestyle=':', 
                              label=f'Threshold: {self.min_effect_size}')
            axes[0, 0].set_xlabel('Effect Size')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Effect Size Distribution')
            axes[0, 0].legend()
        
        # Plot 2: P-value distribution
        p_values = [r.statistical_test.p_value for r in self.comparison_results]
        
        axes[0, 1].hist(p_values, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(self.alpha, color='red', linestyle='--', 
                          label=f'α = {self.alpha}')
        axes[0, 1].set_xlabel('P-value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('P-value Distribution')
        axes[0, 1].legend()
        
        # Plot 3: Significance by metric
        metrics = list(set(r.metric for r in self.comparison_results))
        significance_by_metric = []
        
        for metric in metrics:
            metric_results = [r for r in self.comparison_results if r.metric == metric]
            sig_count = sum(1 for r in metric_results if r.statistical_test.significant)
            significance_by_metric.append(sig_count / len(metric_results) if metric_results else 0)
        
        axes[1, 0].bar(metrics, significance_by_metric, alpha=0.7)
        axes[1, 0].set_xlabel('Metric')
        axes[1, 0].set_ylabel('Significance Rate')
        axes[1, 0].set_title('Statistical Significance by Metric')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Method comparison matrix
        methods = list(set([r.method1 for r in self.comparison_results] + 
                          [r.method2 for r in self.comparison_results]))
        
        if len(methods) <= 10:  # Only show if reasonable number of methods
            comparison_matrix = np.zeros((len(methods), len(methods)))
            
            for result in self.comparison_results:
                if result.statistical_test.significant:
                    i = methods.index(result.method1)
                    j = methods.index(result.method2)
                    
                    if result.statistical_test.effect_size > 0:
                        comparison_matrix[i, j] += 1
                    else:
                        comparison_matrix[j, i] += 1
            
            im = axes[1, 1].imshow(comparison_matrix, cmap='RdYlBu_r', aspect='auto')
            axes[1, 1].set_xticks(range(len(methods)))
            axes[1, 1].set_yticks(range(len(methods)))
            axes[1, 1].set_xticklabels(methods, rotation=45)
            axes[1, 1].set_yticklabels(methods)
            axes[1, 1].set_title('Method Wins Matrix')
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / f"statistical_plots_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate effect size by function complexity plot
        self._plot_effect_by_complexity(output_dir, timestamp)
    
    def _plot_effect_by_complexity(self, output_dir: Path, timestamp: str):
        """Plot effect sizes by function complexity"""
        
        # Load complexity information from config
        config_path = Path("config/synthetic_test_programs.yaml")
        if not config_path.exists():
            return
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            complexity_map = {}
            for func_name, func_config in config.items():
                if isinstance(func_config, dict) and 'cyclomatic_complexity' in func_config:
                    complexity_map[func_name] = func_config['cyclomatic_complexity']
            
            # Extract effect sizes with complexity
            data_points = []
            for result in self.comparison_results:
                if (result.function_name in complexity_map and 
                    result.statistical_test.effect_size is not None):
                    
                    data_points.append({
                        'complexity': complexity_map[result.function_name],
                        'effect_size': abs(result.statistical_test.effect_size),
                        'significant': result.statistical_test.significant,
                        'practical': result.practical_significance,
                        'metric': result.metric,
                        'function': result.function_name
                    })
            
            if not data_points:
                return
            
            df = pd.DataFrame(data_points)
            
            # Create scatter plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot non-significant points
            non_sig = df[~df['significant']]
            if not non_sig.empty:
                ax.scatter(non_sig['complexity'], non_sig['effect_size'], 
                          alpha=0.3, color='gray', label='Non-significant', s=50)
            
            # Plot significant points
            sig = df[df['significant']]
            if not sig.empty:
                scatter = ax.scatter(sig['complexity'], sig['effect_size'], 
                                   c=sig['practical'].astype(int), 
                                   cmap='RdYlGn', alpha=0.7, s=80, 
                                   edgecolors='black', linewidth=0.5)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Practical Significance')
                cbar.set_ticks([0, 1])
                cbar.set_ticklabels(['No', 'Yes'])
            
            # Add trend line
            if len(df) > 10:
                z = np.polyfit(df['complexity'], df['effect_size'], 1)
                p = np.poly1d(z)
                ax.plot(df['complexity'].sort_values(), p(df['complexity'].sort_values()), 
                       "r--", alpha=0.8, linewidth=2, label=f'Trend (slope={z[0]:.4f})')
            
            ax.axhline(self.min_effect_size, color='orange', linestyle=':', 
                      linewidth=2, label=f'Practical Threshold ({self.min_effect_size})')
            
            ax.set_xlabel('Cyclomatic Complexity')
            ax.set_ylabel('Absolute Effect Size')
            ax.set_title('Effect Size vs Function Complexity')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"effect_complexity_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting complexity analysis: {e}")


def main():
    """Main function for running comprehensive statistical analysis"""
    
    analyzer = SyntheticStatisticalAnalyzer(
        alpha=0.05,
        correction_method=CorrectionMethod.FDR_BH,
        effect_size_type=EffectSizeType.HEDGES_G,
        min_effect_size=0.3
    )
    
    # Example usage
    try:
        # Load experimental data (paths would be provided by user)
        baseline_path = "synthetic_baseline_results.json"
        mo_path = "synthetic_mo_results.json"
        
        if Path(baseline_path).exists() and Path(mo_path).exists():
            baseline_data, mo_data = analyzer.load_experimental_data(baseline_path, mo_path)
            
            # Perform comprehensive statistical analysis
            comparison_results = analyzer.perform_multiple_comparisons(baseline_data, mo_data)
            
            # Perform meta-analysis
            meta_results = analyzer.perform_meta_analysis()
            
            # Generate comprehensive report
            report_dir = analyzer.generate_comprehensive_report()
            
            print(f"Statistical analysis completed. Report generated in: {report_dir}")
            
        else:
            print("Example data files not found. Use analyzer.load_experimental_data() with actual data files.")
            print("Analyzer ready for use with actual experimental data.")
            
    except Exception as e:
        print(f"Error in statistical analysis: {e}")


if __name__ == "__main__":
    main()