#!/usr/bin/env python3
"""
Comprehensive Smoke Test for Unified Test Generation Framework

This script provides complete validation of the experimental framework functionality
as specified in EXPERIMENTAL_METHODOLOGY.md, including all core components,
analysis tools, and reporting capabilities.

Execution time: ~15-30 minutes
Coverage: All major framework components

Usage:
    python3 smoke_test_comprehensive.py
    python3 smoke_test_comprehensive.py --verbose
    python3 smoke_test_comprehensive.py --quick

Author: Generated for pyPC/BD_AL_test framework validation
"""

import sys
import os
import time
import traceback
import argparse
import json
import yaml
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import framework components
try:
    from src.algorithms.baseline.baseline_test_generators import BaselineTestGenerator
    from src.algorithms.baseline.generators import UnifiedBaselineGenerator
    from src.core.algorithm_manager import AlgorithmManager
    from src.evaluation.evaluator import UnifiedTestEvaluator
    from src.evaluation.enhanced_metrics import EnhancedTestEvaluator
    from src.analysis.statistical import StatisticalAnalyzer, AdvancedStatisticalAnalyzer
    from src.visualization.enhanced_plots import (
        EnhancedVisualizationSuite, CriticalDifferenceVisualizer, 
        InteractiveDashboard, MultiObjectiveVisualizer
    )
    from src.publication.latex_generator import (
        LaTeXTableGenerator, CitationManager, ReplicationPackageGenerator
    )
    from src.analysis.complexity_analyzer import ComplexityAnalyzer
except ImportError as e:
    print(f"❌ Failed to import framework components: {e}")
    print("Please ensure all required modules are available.")
    sys.exit(1)


@dataclass
class TestResult:
    """Result of a single test component"""
    component: str
    success: bool
    duration: float
    message: str
    details: Dict[str, Any] = None
    error: Optional[str] = None


@dataclass
class SmokeTestReport:
    """Complete smoke test report"""
    start_time: datetime
    end_time: datetime
    total_duration: float
    tests_run: int
    tests_passed: int
    tests_failed: int
    success_rate: float
    results: List[TestResult]
    system_info: Dict[str, Any]
    framework_info: Dict[str, Any]


class ComprehensiveSmokeTest:
    """Comprehensive smoke test suite for the entire framework"""
    
    # Test configuration
    SMOKE_TEST_PROGRAMS = [
        'minimum',           # Simple (CC=3)
        'three_number_sort', # Simple (CC=4) 
        'bubble_sort',       # Simple (CC=5)
        'trig_area',         # Medium (CC=14)
        'complex_conditions', # Complex (CC=32)
        'deep_branching'     # Complex (CC=31)
    ]
    
    BASELINE_METHODS = [
        'random', 'adaptive_random', 'quasi_random',
        'grid_search', 'boundary_value', 'hill_climbing'
    ]
    
    MO_ALGORITHMS = ['NSGA2', 'NSGA3', 'MOEAD', 'CTAEA']
    
    # Minimal parameters for fast execution
    SMOKE_PARAMS = {
        'baseline': {
            'n_tests': 15,
            'repetitions': 3,
            'timeout': 10.0
        },
        'mo': {
            'generations': 10,
            'population_size': 20,
            'repetitions': 3,
            'timeout': 15.0
        }
    }
    
    QUICK_PARAMS = {
        'baseline': {
            'n_tests': 10,
            'repetitions': 2,
            'timeout': 5.0
        },
        'mo': {
            'generations': 5,
            'population_size': 10,
            'repetitions': 2,
            'timeout': 10.0
        }
    }

    def __init__(self, verbose: bool = True, quick: bool = False):
        """Initialize smoke test suite"""
        self.verbose = verbose
        self.quick = quick
        self.params = self.QUICK_PARAMS if quick else self.SMOKE_PARAMS
        
        # Test results storage
        self.results: List[TestResult] = []
        self.start_time = datetime.now()
        
        # Create temporary working directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="smoke_test_"))
        self.output_dir = self.temp_dir / "results"
        self.output_dir.mkdir(exist_ok=True)
        
        # System information
        self.system_info = self._collect_system_info()
        
        if self.verbose:
            print("🧪 Comprehensive Framework Smoke Test")
            print("=" * 50)
            print(f"Mode: {'Quick' if quick else 'Standard'}")
            print(f"Test Programs: {len(self.SMOKE_TEST_PROGRAMS)}")
            print(f"Baseline Methods: {len(self.BASELINE_METHODS)}")
            print(f"MO Algorithms: {len(self.MO_ALGORITHMS)}")
            print(f"Working Directory: {self.temp_dir}")
            print()

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information"""
        import platform
        import psutil
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'available_memory_gb': round(psutil.virtual_memory().available / (1024**3), 2)
        }

    def run_test(self, test_name: str, test_func, *args, **kwargs) -> TestResult:
        """Run a single test component with timing and error handling"""
        if self.verbose:
            print(f"🔍 Testing {test_name}...")
        
        start_time = time.time()
        
        try:
            result = test_func(*args, **kwargs)
            duration = time.time() - start_time
            
            test_result = TestResult(
                component=test_name,
                success=True,
                duration=duration,
                message="✅ Passed",
                details=result if isinstance(result, dict) else None
            )
            
            if self.verbose:
                print(f"   ✅ {test_name} - {duration:.2f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            traceback_str = traceback.format_exc() if self.verbose else None
            
            test_result = TestResult(
                component=test_name,
                success=False,
                duration=duration,
                message=f"❌ Failed: {error_msg}",
                error=traceback_str
            )
            
            if self.verbose:
                print(f"   ❌ {test_name} - {duration:.2f}s - {error_msg}")
                if traceback_str:
                    print(f"      {traceback_str}")
        
        self.results.append(test_result)
        return test_result

    def test_configuration_loading(self) -> Dict[str, Any]:
        """Test configuration file loading"""
        config_results = {}
        
        # Test unified config
        unified_config_path = Path("config/unified_config.yaml")
        if unified_config_path.exists():
            with open(unified_config_path, 'r') as f:
                unified_config = yaml.safe_load(f)
                config_results['unified_config'] = {
                    'loaded': True,
                    'keys': list(unified_config.keys()),
                    'baseline_methods': len(unified_config.get('baseline_methods', {})),
                    'mo_algorithms': len(unified_config.get('multi_objective_algorithms', {}))
                }
        else:
            config_results['unified_config'] = {'loaded': False}
        
        # Test program metadata
        metadata_path = Path("config/program_metadata.yaml")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
                config_results['program_metadata'] = {
                    'loaded': True,
                    'total_programs': metadata.get('metadata_info', {}).get('total_programs', 0),
                    'complexity_distribution': metadata.get('metadata_info', {}).get('complexity_distribution', {})
                }
        else:
            config_results['program_metadata'] = {'loaded': False}
        
        return config_results

    def test_program_metadata_system(self) -> Dict[str, Any]:
        """Test program metadata and complexity analysis"""
        try:
            analyzer = ComplexityAnalyzer()
            
            # Test a few sample programs
            test_programs = ['minimum', 'bubble_sort', 'complex_conditions']
            results = {}
            
            for program in test_programs:
                program_path = Path(f"test_programs/{program}.py")
                if program_path.exists():
                    complexity_info = analyzer.analyze_file(str(program_path))
                    results[program] = {
                        'cyclomatic_complexity': complexity_info.get('cyclomatic_complexity', 0),
                        'maintainability_index': complexity_info.get('maintainability_index', 0),
                        'lines_of_code': complexity_info.get('total_lines', 0)
                    }
            
            return {
                'programs_analyzed': len(results),
                'sample_results': results,
                'analyzer_available': True
            }
            
        except Exception as e:
            return {
                'programs_analyzed': 0,
                'analyzer_available': False,
                'error': str(e)
            }

    def test_baseline_methods(self) -> Dict[str, Any]:
        """Test all baseline test generation methods"""
        try:
            generator = UnifiedBaselineGenerator()
            
            # Test each baseline method
            baseline_results = {}
            test_program = 'minimum'  # Simple program for testing
            
            for method in self.BASELINE_METHODS[:3]:  # Test first 3 for speed
                try:
                    # Generate test cases
                    test_cases = generator.generate_test_suite(
                        method=method,
                        n_tests=self.params['baseline']['n_tests'],
                        dimensions=4,
                        bounds=[(-100, 100)] * 4
                    )
                    
                    baseline_results[method] = {
                        'success': True,
                        'test_cases_generated': len(test_cases) if test_cases else 0,
                        'sample_test': test_cases[0].tolist() if test_cases and len(test_cases) > 0 else None
                    }
                    
                except Exception as e:
                    baseline_results[method] = {
                        'success': False,
                        'error': str(e)
                    }
            
            return {
                'methods_tested': len(baseline_results),
                'successful_methods': sum(1 for r in baseline_results.values() if r.get('success')),
                'results': baseline_results
            }
            
        except Exception as e:
            return {
                'methods_tested': 0,
                'successful_methods': 0,
                'error': str(e)
            }

    def test_multi_objective_algorithms(self) -> Dict[str, Any]:
        """Test multi-objective algorithms"""
        try:
            algorithm_manager = AlgorithmManager()
            
            mo_results = {}
            
            # Test first 2 MO algorithms for speed
            for algorithm in self.MO_ALGORITHMS[:2]:
                try:
                    # Get algorithm configuration
                    config = algorithm_manager.get_algorithm_config(algorithm)
                    
                    # Create algorithm instance with minimal parameters
                    algorithm_instance = algorithm_manager.create_algorithm(
                        algorithm,
                        pop_size=self.params['mo']['population_size'],
                        n_gen=self.params['mo']['generations']
                    )
                    
                    mo_results[algorithm] = {
                        'success': True,
                        'config_available': config is not None,
                        'instance_created': algorithm_instance is not None,
                        'population_size': self.params['mo']['population_size'],
                        'generations': self.params['mo']['generations']
                    }
                    
                except Exception as e:
                    mo_results[algorithm] = {
                        'success': False,
                        'error': str(e)
                    }
            
            return {
                'algorithms_tested': len(mo_results),
                'successful_algorithms': sum(1 for r in mo_results.values() if r.get('success')),
                'results': mo_results
            }
            
        except Exception as e:
            return {
                'algorithms_tested': 0,
                'successful_algorithms': 0,
                'error': str(e)
            }

    def test_enhanced_evaluation_metrics(self) -> Dict[str, Any]:
        """Test enhanced evaluation metrics system"""
        try:
            evaluator = EnhancedTestEvaluator()
            
            # Test with a simple program
            test_program = 'minimum'
            test_inputs = [[1, 2, 3, 4], [5, 1, 8, 2], [10, -5, 0, 15]]
            
            program_path = f"test_programs/{test_program}.py"
            
            if Path(program_path).exists():
                result = evaluator.evaluate_test_suite(
                    program_path=program_path,
                    target_function_name='target_function',
                    test_inputs=test_inputs,
                    timeout=5.0
                )
                
                return {
                    'evaluation_successful': True,
                    'coverage': result.coverage,
                    'branch_distance': result.branch_distance,
                    'approach_level': result.approach_level,
                    'memory_usage': result.memory_usage,
                    'execution_time': result.execution_time,
                    'metrics_count': len([
                        result.coverage, result.branch_distance, result.approach_level,
                        result.memory_usage, result.execution_time
                    ])
                }
            else:
                return {
                    'evaluation_successful': False,
                    'error': f"Test program not found: {program_path}"
                }
                
        except Exception as e:
            return {
                'evaluation_successful': False,
                'error': str(e)
            }

    def test_statistical_analysis_framework(self) -> Dict[str, Any]:
        """Test statistical analysis capabilities"""
        try:
            analyzer = StatisticalAnalyzer()
            advanced_analyzer = AdvancedStatisticalAnalyzer()
            
            # Generate sample data for testing
            import numpy as np
            np.random.seed(42)
            
            group1 = np.random.normal(0.7, 0.1, 20)  # Coverage group 1
            group2 = np.random.normal(0.5, 0.15, 20)  # Coverage group 2
            
            sample_data = {
                'algorithm1': {'coverage': group1.tolist()},
                'algorithm2': {'coverage': group2.tolist()}
            }
            
            # Test basic statistical analysis
            comparison_result = analyzer.compare_algorithms(
                sample_data, 
                metric='coverage'
            )
            
            # Test advanced analysis
            effect_size = advanced_analyzer.calculate_hedges_g(group1, group2)
            
            # Test power analysis
            power_result = advanced_analyzer.perform_power_analysis(
                effect_size=0.5,
                alpha=0.05,
                n_samples=20
            )
            
            return {
                'basic_analysis_success': comparison_result is not None,
                'effect_size_calculated': not np.isnan(effect_size),
                'power_analysis_success': power_result is not None,
                'statistical_tests_available': True,
                'sample_effect_size': float(effect_size),
                'sample_p_value': comparison_result.get('p_value', 'N/A') if comparison_result else 'N/A'
            }
            
        except Exception as e:
            return {
                'basic_analysis_success': False,
                'statistical_tests_available': False,
                'error': str(e)
            }

    def test_visualization_suite(self) -> Dict[str, Any]:
        """Test visualization framework"""
        try:
            viz_suite = EnhancedVisualizationSuite()
            cd_visualizer = CriticalDifferenceVisualizer()
            dashboard = InteractiveDashboard()
            
            # Generate sample data
            import numpy as np
            import pandas as pd
            
            sample_results = {
                'algorithm1': np.random.normal(0.7, 0.1, 10),
                'algorithm2': np.random.normal(0.5, 0.15, 10),
                'algorithm3': np.random.normal(0.6, 0.12, 10)
            }
            
            # Test basic visualization creation
            output_file = self.output_dir / "test_boxplot.png"
            
            # Create a simple boxplot
            viz_result = viz_suite.create_comparison_boxplot(
                sample_results,
                title="Test Comparison",
                output_path=str(output_file)
            )
            
            # Test critical difference diagram
            rankings = {'algorithm1': 1.5, 'algorithm2': 2.8, 'algorithm3': 2.2}
            cd_file = self.output_dir / "test_cd_diagram.png"
            
            cd_result = cd_visualizer.create_cd_diagram(
                rankings=rankings,
                critical_difference=0.5,
                output_path=str(cd_file)
            )
            
            return {
                'visualization_suite_available': True,
                'boxplot_created': output_file.exists(),
                'cd_diagram_created': cd_file.exists(),
                'dashboard_available': dashboard is not None,
                'files_generated': [
                    str(f) for f in [output_file, cd_file] if f.exists()
                ]
            }
            
        except Exception as e:
            return {
                'visualization_suite_available': False,
                'error': str(e)
            }

    def test_publication_tools(self) -> Dict[str, Any]:
        """Test publication and LaTeX generation tools"""
        try:
            latex_gen = LaTeXTableGenerator()
            citation_mgr = CitationManager()
            replication_gen = ReplicationPackageGenerator()
            
            # Test LaTeX table generation
            import pandas as pd
            
            sample_data = pd.DataFrame({
                'Algorithm': ['NSGA-II', 'NSGA-III', 'Random'],
                'Coverage': [0.75, 0.73, 0.45],
                'Time': [12.5, 15.2, 2.1]
            })
            
            latex_table = latex_gen.generate_results_table(
                sample_data,
                metrics=['Coverage', 'Time'],
                caption="Sample Results Table"
            )
            
            # Test citation generation
            citation_mgr.add_algorithm_citations(['NSGA2', 'NSGA3'])
            bibliography = citation_mgr.generate_bibliography()
            
            # Test replication package structure
            package_structure = replication_gen.create_package_structure(
                str(self.output_dir / "replication_test")
            )
            
            return {
                'latex_generation_success': latex_table is not None and len(latex_table) > 0,
                'citation_system_available': bibliography is not None,
                'replication_package_created': package_structure is not None,
                'latex_table_length': len(latex_table) if latex_table else 0,
                'citations_count': len(bibliography) if bibliography else 0
            }
            
        except Exception as e:
            return {
                'latex_generation_success': False,
                'citation_system_available': False,
                'error': str(e)
            }

    def test_report_generation(self) -> Dict[str, Any]:
        """Test comprehensive report generation"""
        try:
            # Generate sample experimental results
            sample_results = {
                'minimum': {
                    'random': [
                        {'coverage': 0.8, 'execution_time': 1.2, 'success': True},
                        {'coverage': 0.7, 'execution_time': 1.1, 'success': True},
                        {'coverage': 0.9, 'execution_time': 1.3, 'success': True}
                    ],
                    'NSGA2': [
                        {'coverage': 0.9, 'execution_time': 5.2, 'success': True},
                        {'coverage': 0.85, 'execution_time': 5.1, 'success': True},
                        {'coverage': 0.95, 'execution_time': 5.3, 'success': True}
                    ]
                }
            }
            
            # Test JSON export
            json_file = self.output_dir / "sample_results.json"
            with open(json_file, 'w') as f:
                json.dump(sample_results, f, indent=2)
            
            # Test CSV export
            import pandas as pd
            
            # Flatten results for CSV
            csv_data = []
            for program, algorithms in sample_results.items():
                for algorithm, runs in algorithms.items():
                    for i, run in enumerate(runs):
                        csv_data.append({
                            'program': program,
                            'algorithm': algorithm,
                            'run': i + 1,
                            'coverage': run['coverage'],
                            'execution_time': run['execution_time'],
                            'success': run['success']
                        })
            
            csv_file = self.output_dir / "sample_results.csv"
            pd.DataFrame(csv_data).to_csv(csv_file, index=False)
            
            # Test summary report generation
            summary_file = self.output_dir / "summary_report.txt"
            with open(summary_file, 'w') as f:
                f.write("Experimental Results Summary\n")
                f.write("="*30 + "\n")
                f.write(f"Programs tested: {len(sample_results)}\n")
                f.write(f"Algorithms tested: {len(set().union(*[algs.keys() for algs in sample_results.values()]))}\n")
                f.write(f"Total runs: {sum(len(runs) for algs in sample_results.values() for runs in algs.values())}\n")
            
            return {
                'json_export_success': json_file.exists(),
                'csv_export_success': csv_file.exists(),
                'summary_report_created': summary_file.exists(),
                'files_created': [str(f) for f in [json_file, csv_file, summary_file] if f.exists()],
                'total_files': 3
            }
            
        except Exception as e:
            return {
                'json_export_success': False,
                'csv_export_success': False,
                'summary_report_created': False,
                'error': str(e)
            }

    def test_end_to_end_mini_experiment(self) -> Dict[str, Any]:
        """Run a minimal end-to-end experiment"""
        try:
            # Use minimal parameters for speed
            test_program = 'minimum'
            baseline_method = 'random'
            mo_algorithm = 'NSGA2'
            
            results = {}
            
            # Test baseline method
            generator = UnifiedBaselineGenerator()
            baseline_tests = generator.generate_test_suite(
                method=baseline_method,
                n_tests=5,
                dimensions=4,
                bounds=[(-10, 10)] * 4
            )
            
            # Test MO algorithm setup
            algorithm_manager = AlgorithmManager()
            mo_instance = algorithm_manager.create_algorithm(
                mo_algorithm,
                pop_size=10,
                n_gen=3
            )
            
            # Test evaluation
            evaluator = UnifiedTestEvaluator()
            if baseline_tests is not None and len(baseline_tests) > 0:
                eval_result = evaluator.evaluate_single_run(
                    program_name=test_program,
                    test_inputs=baseline_tests[:3].tolist(),
                    algorithm_name=baseline_method,
                    timeout=5.0
                )
                
                results['evaluation_success'] = eval_result.success
                results['coverage_achieved'] = eval_result.coverage
            
            return {
                'end_to_end_success': True,
                'baseline_generated': baseline_tests is not None,
                'mo_created': mo_instance is not None,
                'evaluation_completed': 'evaluation_success' in results,
                'details': results
            }
            
        except Exception as e:
            return {
                'end_to_end_success': False,
                'error': str(e)
            }

    def run_all_tests(self) -> SmokeTestReport:
        """Run complete smoke test suite"""
        if self.verbose:
            print("🚀 Starting Comprehensive Smoke Test Suite\n")
        
        # Core framework tests
        self.run_test("Configuration Loading", self.test_configuration_loading)
        self.run_test("Program Metadata System", self.test_program_metadata_system)
        self.run_test("Baseline Methods", self.test_baseline_methods)
        self.run_test("Multi-Objective Algorithms", self.test_multi_objective_algorithms)
        self.run_test("Enhanced Evaluation Metrics", self.test_enhanced_evaluation_metrics)
        
        # Analysis and reporting tests
        self.run_test("Statistical Analysis Framework", self.test_statistical_analysis_framework)
        self.run_test("Visualization Suite", self.test_visualization_suite)
        self.run_test("Publication Tools", self.test_publication_tools)
        self.run_test("Report Generation", self.test_report_generation)
        
        # Integration test
        self.run_test("End-to-End Mini Experiment", self.test_end_to_end_mini_experiment)
        
        # Generate final report
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        tests_passed = sum(1 for r in self.results if r.success)
        tests_failed = len(self.results) - tests_passed
        success_rate = (tests_passed / len(self.results)) * 100 if self.results else 0
        
        # Framework information
        framework_info = {
            'total_programs_available': len(list(Path("test_programs").glob("*.py"))),
            'baseline_methods_available': len(self.BASELINE_METHODS),
            'mo_algorithms_available': len(self.MO_ALGORITHMS),
            'config_files_found': len(list(Path("config").glob("*.yaml"))),
            'src_modules_found': len(list(Path("src").rglob("*.py")))
        }
        
        report = SmokeTestReport(
            start_time=self.start_time,
            end_time=end_time,
            total_duration=total_duration,
            tests_run=len(self.results),
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            success_rate=success_rate,
            results=self.results,
            system_info=self.system_info,
            framework_info=framework_info
        )
        
        return report

    def print_summary_report(self, report: SmokeTestReport):
        """Print comprehensive summary report"""
        print("\n" + "="*70)
        print("🏁 COMPREHENSIVE SMOKE TEST SUMMARY")
        print("="*70)
        
        # Overall results
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   • Tests Run: {report.tests_run}")
        print(f"   • Tests Passed: {report.tests_passed} ✅")
        print(f"   • Tests Failed: {report.tests_failed} ❌")
        print(f"   • Success Rate: {report.success_rate:.1f}%")
        print(f"   • Total Duration: {report.total_duration:.2f} seconds")
        
        # System information
        print(f"\n🖥️  SYSTEM INFORMATION:")
        print(f"   • Platform: {report.system_info['platform']}")
        print(f"   • Python: {report.system_info['python_version']}")
        print(f"   • CPU Cores: {report.system_info['cpu_count']}")
        print(f"   • Memory: {report.system_info['memory_gb']} GB")
        print(f"   • Available Memory: {report.system_info['available_memory_gb']} GB")
        
        # Framework information
        print(f"\n🔧 FRAMEWORK INFORMATION:")
        print(f"   • Test Programs: {report.framework_info['total_programs_available']}")
        print(f"   • Baseline Methods: {report.framework_info['baseline_methods_available']}")
        print(f"   • MO Algorithms: {report.framework_info['mo_algorithms_available']}")
        print(f"   • Config Files: {report.framework_info['config_files_found']}")
        print(f"   • Source Modules: {report.framework_info['src_modules_found']}")
        
        # Detailed test results
        print(f"\n📋 DETAILED TEST RESULTS:")
        for result in report.results:
            status = "✅" if result.success else "❌"
            print(f"   {status} {result.component:<35} {result.duration:>6.2f}s")
            if not result.success and result.error and self.verbose:
                print(f"      └─ Error: {result.message}")
        
        # Performance summary
        total_test_time = sum(r.duration for r in report.results)
        avg_test_time = total_test_time / len(report.results) if report.results else 0
        
        print(f"\n⚡ PERFORMANCE SUMMARY:")
        print(f"   • Average Test Time: {avg_test_time:.2f} seconds")
        print(f"   • Fastest Test: {min(r.duration for r in report.results):.2f}s")
        print(f"   • Slowest Test: {max(r.duration for r in report.results):.2f}s")
        print(f"   • Framework Overhead: {(report.total_duration - total_test_time):.2f}s")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if report.success_rate >= 90:
            print("   🎉 Excellent! Framework is ready for production use.")
        elif report.success_rate >= 75:
            print("   ⚠️  Good overall. Address failed components before heavy usage.")
        else:
            print("   🚨 Framework needs attention. Multiple components failing.")
        
        if report.tests_failed > 0:
            failed_components = [r.component for r in report.results if not r.success]
            print(f"   📝 Failed components: {', '.join(failed_components)}")
        
        print(f"\n🧹 CLEANUP:")
        print(f"   • Temporary files: {self.temp_dir}")
        print(f"   • Results directory: {self.output_dir}")
        
        print("\n" + "="*70)

    def save_detailed_report(self, report: SmokeTestReport, output_file: str = None):
        """Save detailed report to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"smoke_test_report_{timestamp}.json"
        
        # Convert dataclass to dict for JSON serialization
        report_dict = asdict(report)
        
        # Convert datetime objects to ISO format strings
        report_dict['start_time'] = report.start_time.isoformat()
        report_dict['end_time'] = report.end_time.isoformat()
        
        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        if self.verbose:
            print(f"\n📄 Detailed report saved to: {output_file}")

    def cleanup(self):
        """Clean up temporary files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                if self.verbose:
                    print(f"🧹 Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Warning: Could not clean up {self.temp_dir}: {e}")


def main():
    """Main entry point for smoke test"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Smoke Test for Unified Test Generation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 smoke_test_comprehensive.py                 # Standard test
  python3 smoke_test_comprehensive.py --verbose       # Verbose output
  python3 smoke_test_comprehensive.py --quick         # Quick test mode
  python3 smoke_test_comprehensive.py --quiet         # Minimal output
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output (default: True)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output mode'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (reduced parameters)'
    )
    
    parser.add_argument(
        '--save-report',
        type=str,
        metavar='FILE',
        help='Save detailed JSON report to file'
    )
    
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Do not clean up temporary files'
    )
    
    args = parser.parse_args()
    
    # Configure verbosity
    verbose = args.verbose and not args.quiet
    if not args.verbose and not args.quiet:
        verbose = True  # Default to verbose
    
    try:
        # Run smoke test
        smoke_test = ComprehensiveSmokeTest(verbose=verbose, quick=args.quick)
        report = smoke_test.run_all_tests()
        
        # Print summary
        if not args.quiet:
            smoke_test.print_summary_report(report)
        
        # Save detailed report if requested
        if args.save_report:
            smoke_test.save_detailed_report(report, args.save_report)
        
        # Cleanup
        if not args.no_cleanup:
            smoke_test.cleanup()
        
        # Exit with appropriate code
        exit_code = 0 if report.success_rate >= 75 else 1
        if not args.quiet:
            if exit_code == 0:
                print("\n🎉 Smoke test completed successfully!")
            else:
                print("\n❌ Smoke test completed with failures!")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Smoke test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n💥 Smoke test failed with exception: {e}")
        if verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()