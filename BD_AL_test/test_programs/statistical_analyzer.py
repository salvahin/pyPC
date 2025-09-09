"""
Advanced Statistical Analysis with Hypothesis Testing and Distribution Fitting
Target: 25-40% coverage, ~70 cyclomatic complexity
"""

import math
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

class DistributionType(Enum):
    NORMAL = "normal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    BETA = "beta"
    WEIBULL = "weibull"
    POISSON = "poisson"
    BINOMIAL = "binomial"

class HypothesisTest(Enum):
    T_TEST = "t_test"
    WILCOXON = "wilcoxon"
    MANN_WHITNEY = "mann_whitney"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    CHI_SQUARE = "chi_square"
    F_TEST = "f_test"
    KRUSKAL_WALLIS = "kruskal_wallis"

@dataclass
class StatisticalResult:
    test_statistic: float
    p_value: float
    critical_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    power: float
    degrees_of_freedom: int
    interpretation: str

class StatisticalAnalyzer:
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.analysis_cache: Dict[str, Any] = {}
        self.computation_history: List[str] = []
        
    def analyze_dataset(self, data: List[float], analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive statistical analysis of dataset
        """
        if not self._validate_data(data):
            return {'error': 'invalid_data'}
        
        if not isinstance(analysis_config, dict):
            return {'error': 'invalid_analysis_config'}
        
        analysis_type = analysis_config.get('type', 'descriptive')
        
        results = {}
        
        try:
            if analysis_type == 'descriptive':
                results = self._descriptive_analysis(data, analysis_config)
            elif analysis_type == 'inferential':
                results = self._inferential_analysis(data, analysis_config)
            elif analysis_type == 'distribution_fitting':
                results = self._distribution_fitting(data, analysis_config)
            elif analysis_type == 'hypothesis_testing':
                results = self._hypothesis_testing(data, analysis_config)
            elif analysis_type == 'regression_analysis':
                results = self._regression_analysis(data, analysis_config)
            elif analysis_type == 'time_series':
                results = self._time_series_analysis(data, analysis_config)
            elif analysis_type == 'multivariate':
                results = self._multivariate_analysis(data, analysis_config)
            else:
                return {'error': 'unknown_analysis_type'}
            
            # Add metadata
            results['metadata'] = {
                'sample_size': len(data),
                'analysis_type': analysis_type,
                'significance_level': self.significance_level,
                'computation_time': time.time()
            }
            
            return results
            
        except Exception as e:
            return {'error': f'analysis_failed: {str(e)}'}
    
    def _validate_data(self, data: List[float]) -> bool:
        """Validate input data"""
        if not data or not isinstance(data, list):
            return False
        
        if len(data) > 100000:  # Reasonable size limit
            return False
        
        for value in data:
            if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                return False
        
        return True
    
    def _descriptive_analysis(self, data: List[float], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute comprehensive descriptive statistics
        """
        n = len(data)
        
        # Central tendency
        mean = sum(data) / n
        sorted_data = sorted(data)
        
        if n % 2 == 0:
            median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        else:
            median = sorted_data[n//2]
        
        # Mode calculation (simplified)
        mode = self._calculate_mode(data)
        
        # Variability measures
        variance = sum((x - mean) ** 2 for x in data) / (n - 1) if n > 1 else 0
        std_dev = math.sqrt(variance)
        
        # Range statistics
        range_val = max(data) - min(data)
        iqr = self._calculate_iqr(sorted_data)
        
        # Shape measures
        skewness = self._calculate_skewness(data, mean, std_dev)
        kurtosis = self._calculate_kurtosis(data, mean, std_dev)
        
        # Robust statistics
        mad = self._calculate_mad(data, median)  # Median Absolute Deviation
        
        # Percentiles
        percentiles = {}
        for p in [5, 10, 25, 50, 75, 90, 95]:
            percentiles[f'p{p}'] = self._calculate_percentile(sorted_data, p)
        
        # Outlier detection
        outliers = self._detect_outliers(data, config.get('outlier_method', 'iqr'))
        
        # Confidence intervals
        confidence_level = config.get('confidence_level', 0.95)
        ci_mean = self._confidence_interval_mean(data, confidence_level)
        ci_std = self._confidence_interval_std(data, confidence_level)
        
        return {
            'central_tendency': {
                'mean': mean,
                'median': median,
                'mode': mode,
                'trimmed_mean': self._trimmed_mean(data, 0.1)
            },
            'variability': {
                'variance': variance,
                'standard_deviation': std_dev,
                'range': range_val,
                'interquartile_range': iqr,
                'coefficient_of_variation': std_dev / abs(mean) if mean != 0 else float('inf'),
                'median_absolute_deviation': mad
            },
            'shape': {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'skewness_interpretation': self._interpret_skewness(skewness),
                'kurtosis_interpretation': self._interpret_kurtosis(kurtosis)
            },
            'percentiles': percentiles,
            'outliers': {
                'count': len(outliers),
                'values': outliers[:10],  # Limit output
                'percentage': len(outliers) / n * 100
            },
            'confidence_intervals': {
                'mean': ci_mean,
                'standard_deviation': ci_std
            }
        }
    
    def _inferential_analysis(self, data: List[float], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inferential statistical analysis
        """
        sample_size = len(data)
        
        # Population parameter estimation
        sample_mean = sum(data) / sample_size
        sample_var = sum((x - sample_mean) ** 2 for x in data) / (sample_size - 1) if sample_size > 1 else 0
        sample_std = math.sqrt(sample_var)
        
        # Standard error calculations
        se_mean = sample_std / math.sqrt(sample_size)
        se_proportion = math.sqrt((sample_mean * (1 - sample_mean)) / sample_size) if 0 <= sample_mean <= 1 else 0
        
        # Hypothesis testing for mean
        hypothesized_mean = config.get('hypothesized_mean', 0)
        t_statistic = (sample_mean - hypothesized_mean) / se_mean if se_mean != 0 else 0
        
        # Degrees of freedom
        df = sample_size - 1
        
        # Critical values (approximated)
        alpha = self.significance_level
        t_critical = self._t_critical_value(df, alpha/2)  # Two-tailed
        
        # P-value approximation
        p_value = self._calculate_t_p_value(t_statistic, df)
        
        # Confidence intervals
        confidence_level = config.get('confidence_level', 0.95)
        margin_error = t_critical * se_mean
        ci_lower = sample_mean - margin_error
        ci_upper = sample_mean + margin_error
        
        # Effect size (Cohen's d)
        if config.get('comparison_mean') is not None:
            comparison_mean = config['comparison_mean']
            cohens_d = (sample_mean - comparison_mean) / sample_std if sample_std != 0 else 0
        else:
            cohens_d = sample_mean / sample_std if sample_std != 0 else 0
        
        # Power analysis (simplified)
        power = self._estimate_power(sample_size, cohens_d, alpha)
        
        return {
            'sample_statistics': {
                'sample_mean': sample_mean,
                'sample_std': sample_std,
                'sample_size': sample_size,
                'standard_error': se_mean
            },
            'hypothesis_test': {
                'null_hypothesis': f'μ = {hypothesized_mean}',
                'alternative_hypothesis': f'μ ≠ {hypothesized_mean}',
                't_statistic': t_statistic,
                't_critical': t_critical,
                'p_value': p_value,
                'reject_null': abs(t_statistic) > t_critical,
                'degrees_of_freedom': df
            },
            'confidence_interval': {
                'level': confidence_level,
                'lower_bound': ci_lower,
                'upper_bound': ci_upper,
                'margin_of_error': margin_error
            },
            'effect_size': {
                'cohens_d': cohens_d,
                'interpretation': self._interpret_cohens_d(cohens_d)
            },
            'power_analysis': {
                'statistical_power': power,
                'recommended_sample_size': self._recommend_sample_size(cohens_d, alpha, 0.8)
            }
        }
    
    def _distribution_fitting(self, data: List[float], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fit various probability distributions to data
        """
        distributions_to_test = config.get('distributions', ['normal', 'uniform', 'exponential'])
        
        results = {}
        best_fit = None
        best_fit_score = float('inf')
        
        for dist_name in distributions_to_test:
            if dist_name not in [d.value for d in DistributionType]:
                continue
            
            try:
                # Estimate parameters for each distribution
                params = self._estimate_distribution_parameters(data, dist_name)
                
                # Goodness of fit tests
                ks_statistic, ks_p_value = self._kolmogorov_smirnov_test(data, dist_name, params)
                chi_square_stat, chi_square_p = self._chi_square_goodness_of_fit(data, dist_name, params)
                
                # AIC/BIC for model comparison
                log_likelihood = self._calculate_log_likelihood(data, dist_name, params)
                num_params = self._get_num_parameters(dist_name)
                aic = 2 * num_params - 2 * log_likelihood
                bic = num_params * math.log(len(data)) - 2 * log_likelihood
                
                # Track best fit
                if aic < best_fit_score:
                    best_fit_score = aic
                    best_fit = {
                        'distribution': dist_name,
                        'parameters': params,
                        'aic': aic,
                        'bic': bic
                    }
                
                results[dist_name] = {
                    'parameters': params,
                    'goodness_of_fit': {
                        'kolmogorov_smirnov': {
                            'statistic': ks_statistic,
                            'p_value': ks_p_value,
                            'accept_fit': ks_p_value > self.significance_level
                        },
                        'chi_square': {
                            'statistic': chi_square_stat,
                            'p_value': chi_square_p,
                            'accept_fit': chi_square_p > self.significance_level
                        }
                    },
                    'model_selection': {
                        'log_likelihood': log_likelihood,
                        'aic': aic,
                        'bic': bic,
                        'num_parameters': num_params
                    },
                    'predicted_values': self._generate_predicted_values(dist_name, params, len(data))
                }
                
            except Exception as e:
                results[dist_name] = {'error': f'fitting_failed: {str(e)}'}
        
        return {
            'fitted_distributions': results,
            'best_fit': best_fit,
            'model_comparison': self._rank_models_by_aic(results)
        }
    
    def _hypothesis_testing(self, data: List[float], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform various hypothesis tests
        """
        test_type = config.get('test_type', 't_test')
        comparison_data = config.get('comparison_data')
        
        if test_type == 't_test':
            return self._perform_t_test(data, config)
        elif test_type == 'wilcoxon' and comparison_data:
            return self._perform_wilcoxon_test(data, comparison_data, config)
        elif test_type == 'mann_whitney' and comparison_data:
            return self._perform_mann_whitney_test(data, comparison_data, config)
        elif test_type == 'kolmogorov_smirnov':
            return self._perform_ks_test(data, config)
        elif test_type == 'chi_square':
            return self._perform_chi_square_test(data, config)
        else:
            return {'error': 'unsupported_test_type_or_missing_data'}
    
    def _regression_analysis(self, data: List[float], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Regression analysis (simplified)
        """
        y_data = config.get('dependent_variable', data)
        x_data = config.get('independent_variable', list(range(len(data))))
        
        if len(x_data) != len(y_data):
            return {'error': 'mismatched_variable_lengths'}
        
        # Simple linear regression
        n = len(x_data)
        sum_x = sum(x_data)
        sum_y = sum(y_data)
        sum_xy = sum(x_data[i] * y_data[i] for i in range(n))
        sum_x2 = sum(x * x for x in x_data)
        sum_y2 = sum(y * y for y in y_data)
        
        # Calculate slope and intercept
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return {'error': 'perfect_multicollinearity'}
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in y_data)
        ss_res = sum((y_data[i] - (slope * x_data[i] + intercept)) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Standard errors and t-statistics
        mse = ss_res / (n - 2) if n > 2 else 0
        se_slope = math.sqrt(mse / sum((x - sum_x/n) ** 2 for x in x_data)) if sum((x - sum_x/n) ** 2 for x in x_data) != 0 else 0
        se_intercept = math.sqrt(mse * (1/n + (sum_x/n)**2 / sum((x - sum_x/n) ** 2 for x in x_data))) if sum((x - sum_x/n) ** 2 for x in x_data) != 0 else 0
        
        t_slope = slope / se_slope if se_slope != 0 else 0
        t_intercept = intercept / se_intercept if se_intercept != 0 else 0
        
        return {
            'regression_coefficients': {
                'slope': slope,
                'intercept': intercept,
                'slope_std_error': se_slope,
                'intercept_std_error': se_intercept
            },
            'model_fit': {
                'r_squared': r_squared,
                'adjusted_r_squared': 1 - (1 - r_squared) * (n - 1) / (n - 2) if n > 2 else 0,
                'residual_standard_error': math.sqrt(mse)
            },
            'significance_tests': {
                'slope_t_statistic': t_slope,
                'intercept_t_statistic': t_intercept,
                'slope_p_value': self._calculate_t_p_value(t_slope, n - 2),
                'intercept_p_value': self._calculate_t_p_value(t_intercept, n - 2)
            },
            'predictions': [slope * x + intercept for x in x_data]
        }
    
    # Helper methods for statistical calculations
    def _calculate_mode(self, data: List[float]) -> Optional[float]:
        """Calculate mode (most frequent value)"""
        if not data:
            return None
        
        frequency = {}
        for value in data:
            frequency[value] = frequency.get(value, 0) + 1
        
        max_count = max(frequency.values())
        modes = [value for value, count in frequency.items() if count == max_count]
        
        return modes[0] if len(modes) == 1 else None  # Return None for multimodal
    
    def _calculate_iqr(self, sorted_data: List[float]) -> float:
        """Calculate interquartile range"""
        n = len(sorted_data)
        q1_pos = n // 4
        q3_pos = 3 * n // 4
        
        q1 = sorted_data[q1_pos]
        q3 = sorted_data[q3_pos]
        
        return q3 - q1
    
    def _calculate_skewness(self, data: List[float], mean: float, std_dev: float) -> float:
        """Calculate skewness (measure of asymmetry)"""
        if std_dev == 0:
            return 0.0
        
        n = len(data)
        skew = sum(((x - mean) / std_dev) ** 3 for x in data) / n
        
        return skew
    
    def _calculate_kurtosis(self, data: List[float], mean: float, std_dev: float) -> float:
        """Calculate kurtosis (measure of tail heaviness)"""
        if std_dev == 0:
            return 0.0
        
        n = len(data)
        kurt = sum(((x - mean) / std_dev) ** 4 for x in data) / n - 3  # Excess kurtosis
        
        return kurt
    
    def _calculate_mad(self, data: List[float], median: float) -> float:
        """Calculate Median Absolute Deviation"""
        deviations = [abs(x - median) for x in data]
        deviations.sort()
        
        n = len(deviations)
        if n % 2 == 0:
            return (deviations[n//2 - 1] + deviations[n//2]) / 2
        else:
            return deviations[n//2]
    
    def _calculate_percentile(self, sorted_data: List[float], percentile: float) -> float:
        """Calculate percentile value"""
        if not sorted_data:
            return 0.0
        
        n = len(sorted_data)
        index = (percentile / 100.0) * (n - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            fraction = index - int(index)
            return lower + fraction * (upper - lower)
    
    def _detect_outliers(self, data: List[float], method: str) -> List[float]:
        """Detect outliers using specified method"""
        if method == 'iqr':
            sorted_data = sorted(data)
            q1 = self._calculate_percentile(sorted_data, 25)
            q3 = self._calculate_percentile(sorted_data, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            return [x for x in data if x < lower_bound or x > upper_bound]
        
        elif method == 'z_score':
            mean = sum(data) / len(data)
            std_dev = math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))
            
            threshold = 3.0  # Standard threshold for outliers
            return [x for x in data if abs((x - mean) / std_dev) > threshold if std_dev != 0]
        
        return []
    
    def _trimmed_mean(self, data: List[float], trim_percent: float) -> float:
        """Calculate trimmed mean"""
        sorted_data = sorted(data)
        n = len(sorted_data)
        trim_count = int(n * trim_percent)
        
        if trim_count * 2 >= n:
            return sum(data) / len(data)  # Fall back to regular mean
        
        trimmed_data = sorted_data[trim_count:n-trim_count]
        return sum(trimmed_data) / len(trimmed_data) if trimmed_data else 0.0
    
    def _confidence_interval_mean(self, data: List[float], confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for mean"""
        n = len(data)
        mean = sum(data) / n
        std_dev = math.sqrt(sum((x - mean) ** 2 for x in data) / (n - 1)) if n > 1 else 0
        se = std_dev / math.sqrt(n)
        
        alpha = 1 - confidence_level
        t_critical = self._t_critical_value(n - 1, alpha / 2)
        margin_error = t_critical * se
        
        return (mean - margin_error, mean + margin_error)
    
    def _t_critical_value(self, df: int, alpha: float) -> float:
        """Approximate t-critical value"""
        # Simplified approximation for common cases
        if df >= 30:
            # Use normal approximation for large df
            if alpha <= 0.025:
                return 1.96
            elif alpha <= 0.05:
                return 1.645
        
        # Rough approximation for small df
        return 2.0 + (30 - df) * 0.1 if df < 30 else 2.0
    
    def _calculate_t_p_value(self, t_stat: float, df: int) -> float:
        """Approximate p-value for t-statistic"""
        # Very simplified approximation
        abs_t = abs(t_stat)
        
        if abs_t > 3:
            return 0.001
        elif abs_t > 2:
            return 0.05
        elif abs_t > 1.5:
            return 0.1
        else:
            return 0.2
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness value"""
        if abs(skewness) < 0.5:
            return "approximately symmetric"
        elif skewness > 0.5:
            return "positively skewed (right tail)"
        else:
            return "negatively skewed (left tail)"
    
    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpret kurtosis value"""
        if abs(kurtosis) < 0.5:
            return "mesokurtic (normal-like)"
        elif kurtosis > 0.5:
            return "leptokurtic (heavy-tailed)"
        else:
            return "platykurtic (light-tailed)"
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible effect"
        elif abs_d < 0.5:
            return "small effect"
        elif abs_d < 0.8:
            return "medium effect"
        else:
            return "large effect"
    
    # Additional helper methods (simplified implementations)
    def _estimate_distribution_parameters(self, data: List[float], dist_name: str) -> Dict[str, float]:
        """Estimate parameters for given distribution"""
        if dist_name == 'normal':
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            return {'mean': mean, 'std_dev': math.sqrt(variance)}
        elif dist_name == 'uniform':
            return {'min': min(data), 'max': max(data)}
        elif dist_name == 'exponential':
            mean = sum(data) / len(data)
            return {'lambda': 1 / mean if mean != 0 else 1}
        else:
            return {}
    
    def _kolmogorov_smirnov_test(self, data: List[float], dist_name: str, params: Dict[str, float]) -> Tuple[float, float]:
        """Simplified K-S test"""
        return 0.1, 0.5  # Placeholder values
    
    def _chi_square_goodness_of_fit(self, data: List[float], dist_name: str, params: Dict[str, float]) -> Tuple[float, float]:
        """Simplified chi-square test"""
        return 5.0, 0.3  # Placeholder values

def target_function(dataset: List[float], analysis_specification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Target function for test case generation with high complexity
    """
    # Validate dataset
    if not isinstance(dataset, list) or not dataset:
        return {'error': 'invalid_dataset'}
    
    if len(dataset) > 10000:
        return {'error': 'dataset_too_large'}
    
    # Validate data values
    for i, value in enumerate(dataset):
        if not isinstance(value, (int, float)):
            return {'error': f'invalid_value_at_index_{i}'}
        
        if math.isnan(value) or math.isinf(value):
            return {'error': f'invalid_numeric_value_at_{i}'}
    
    # Validate analysis specification
    if not isinstance(analysis_specification, dict):
        return {'error': 'invalid_analysis_specification'}
    
    # Extract analysis parameters
    analysis_type = analysis_specification.get('type', 'descriptive')
    significance_level = analysis_specification.get('significance_level', 0.05)
    
    if not (0 < significance_level < 1):
        return {'error': 'invalid_significance_level'}
    
    # Check minimum sample size requirements
    if len(dataset) < 3:
        return {'error': 'insufficient_sample_size'}
    
    # Create analyzer and perform analysis
    try:
        analyzer = StatisticalAnalyzer(significance_level)
        result = analyzer.analyze_dataset(dataset, analysis_specification)
        
        if 'error' in result:
            return result
        
        # Determine analysis outcome complexity
        if analysis_type == 'descriptive':
            # Check for interesting statistical properties
            if 'shape' in result and abs(result['shape']['skewness']) > 2:
                status = 'highly_skewed_distribution'
            elif 'outliers' in result and result['outliers']['percentage'] > 10:
                status = 'outlier_rich_dataset'
            elif 'variability' in result and result['variability']['coefficient_of_variation'] > 1:
                status = 'high_variability'
            else:
                status = 'normal_distribution_characteristics'
        
        elif analysis_type == 'inferential':
            # Check hypothesis test results
            if 'hypothesis_test' in result and result['hypothesis_test']['reject_null']:
                if result['hypothesis_test']['p_value'] < 0.001:
                    status = 'highly_significant_result'
                else:
                    status = 'significant_result'
            else:
                status = 'non_significant_result'
        
        elif analysis_type == 'distribution_fitting':
            # Check model fitting quality
            if 'best_fit' in result and result['best_fit']:
                if result['best_fit']['aic'] < 100:
                    status = 'excellent_model_fit'
                else:
                    status = 'acceptable_model_fit'
            else:
                status = 'poor_model_fit'
        
        elif analysis_type == 'hypothesis_testing':
            # Complex decision based on test results
            if 'error' in result:
                status = 'test_execution_error'
            else:
                # Placeholder for test-specific logic
                status = 'hypothesis_test_completed'
        
        elif analysis_type == 'regression_analysis':
            # Check regression quality
            if 'model_fit' in result:
                r_squared = result['model_fit']['r_squared']
                if r_squared > 0.9:
                    status = 'strong_linear_relationship'
                elif r_squared > 0.5:
                    status = 'moderate_linear_relationship'
                else:
                    status = 'weak_linear_relationship'
            else:
                status = 'regression_analysis_failed'
        
        else:
            status = 'analysis_completed'
        
        return {
            'status': status,
            'analysis_results': result,
            'dataset_properties': {
                'sample_size': len(dataset),
                'min_value': min(dataset),
                'max_value': max(dataset),
                'mean': sum(dataset) / len(dataset)
            }
        }
        
    except Exception as e:
        return {'error': f'statistical_analysis_exception: {str(e)}'}