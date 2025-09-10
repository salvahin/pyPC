#!/usr/bin/env python3
"""
Unified Test Generation Framework - Main Entry Point

This is the single entry point for running all test generation experiments,
comparisons, and analyses in the unified framework.

Usage:
    python main.py --help                           # Show all available commands
    python main.py baseline --method random        # Run baseline experiment
    python main.py multi-objective --algorithm NSGA2  # Run MO experiment
    python main.py compare                          # Compare baseline vs MO
    python main.py analyze                          # Statistical analysis
    python main.py demo                             # Run demo experiment
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.algorithms.baseline.generators import UnifiedBaselineGenerator
from src.evaluation.evaluator import UnifiedTestEvaluator
from src.analysis.statistical import StatisticalAnalyzer

# Import existing runners for compatibility
from algorithm_manager import AlgorithmManager


class UnifiedFramework:
    """Main framework orchestrator"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = self._setup_logger()
        
        # Core components
        self.baseline_generator = UnifiedBaselineGenerator()
        self.evaluator = UnifiedTestEvaluator(verbose=verbose)
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Configuration
        self.config_dir = Path("config")
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('UnifiedFramework')
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_baseline_experiment(self, 
                               methods: List[str] = None,
                               programs: List[str] = None,
                               n_tests: int = 50,
                               repetitions: int = 10,
                               output_file: str = None) -> Dict[str, Any]:
        """Run comprehensive baseline experiment"""
        
        if methods is None:
            methods = self.baseline_generator.get_available_methods()
        
        if programs is None:
            programs = self.evaluator.get_program_list()
        
        if not programs:
            self.logger.warning("No test programs found. Using demo program.")
            programs = ["demo_program"]
        
        self.logger.info(f"Running baseline experiment:")
        self.logger.info(f"  Methods: {methods}")
        self.logger.info(f"  Programs: {len(programs)}")
        self.logger.info(f"  Tests per method: {n_tests}")
        self.logger.info(f"  Repetitions: {repetitions}")
        
        results = {}
        
        for program in programs:
            self.logger.info(f"Evaluating program: {program}")
            program_results = {}
            
            for method in methods:
                self.logger.info(f"  Running method: {method}")
                method_results = []
                
                for rep in range(repetitions):
                    try:
                        # Generate test cases
                        test_cases = self.baseline_generator.generate_tests(
                            method=method,
                            n_tests=n_tests,
                            dimensions=2  # Default dimensions
                        )
                        
                        # Evaluate test cases
                        eval_result = self.evaluator.evaluate_single_run(
                            program_name=program,
                            test_inputs=test_cases,
                            algorithm_name=method
                        )
                        
                        method_results.append(eval_result.to_dict())
                        
                    except Exception as e:
                        self.logger.error(f"    Rep {rep+1} failed: {e}")
                        method_results.append({
                            'success': False,
                            'error_message': str(e),
                            'coverage': 0.0,
                            'execution_time': 0.0
                        })
                
                program_results[method] = {
                    'results': method_results,
                    'summary': self._calculate_summary_stats(method_results)
                }
            
            results[program] = program_results
        
        # Save results
        if output_file is None:
            output_file = self.results_dir / "baseline_experiment_results.json"
        
        self.evaluator.export_results(results, str(output_file))
        
        return results
    
    def run_multi_objective_experiment(self,
                                      algorithms: List[str] = None,
                                      programs: List[str] = None,
                                      generations: int = 100,
                                      population_size: int = 50,
                                      repetitions: int = 10,
                                      output_file: str = None) -> Dict[str, Any]:
        """Run multi-objective optimization experiment"""
        
        if algorithms is None:
            algorithms = ['NSGA2', 'NSGA3', 'MOEAD', 'CTAEA']
        
        if programs is None:
            programs = self.evaluator.get_program_list()
        
        if not programs:
            self.logger.warning("No test programs found. Using demo program.")
            programs = ["demo_program"]
        
        self.logger.info(f"Running multi-objective experiment:")
        self.logger.info(f"  Algorithms: {algorithms}")
        self.logger.info(f"  Programs: {len(programs)}")
        self.logger.info(f"  Generations: {generations}")
        self.logger.info(f"  Population: {population_size}")
        self.logger.info(f"  Repetitions: {repetitions}")
        
        results = {}
        
        # Use existing MO framework
        try:
            manager = AlgorithmManager()
            
            for program in programs:
                self.logger.info(f"Evaluating program: {program}")
                program_results = {}
                
                for algorithm in algorithms:
                    self.logger.info(f"  Running algorithm: {algorithm}")
                    algorithm_results = []
                    
                    for rep in range(repetitions):
                        try:
                            # Run MO algorithm
                            mo_result = manager.run_algorithm(
                                algorithm_name=algorithm,
                                target_program=program,
                                generations=generations,
                                population_size=population_size
                            )
                            
                            algorithm_results.append(mo_result)
                            
                        except Exception as e:
                            self.logger.error(f"    Rep {rep+1} failed: {e}")
                            algorithm_results.append({
                                'success': False,
                                'error_message': str(e),
                                'hypervolume': 0.0,
                                'igd': float('inf'),
                                'execution_time': 0.0
                            })
                    
                    program_results[algorithm] = {
                        'results': algorithm_results,
                        'summary': self._calculate_summary_stats(algorithm_results)
                    }
                
                results[program] = program_results
            
        except ImportError as e:
            self.logger.error(f"Multi-objective framework not available: {e}")
            return {}
        
        # Save results
        if output_file is None:
            output_file = self.results_dir / "mo_experiment_results.json"
        
        # Export results directly to JSON
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_comparison_analysis(self,
                               baseline_results: str = None,
                               mo_results: str = None,
                               output_dir: str = None) -> str:
        """Run comprehensive comparison analysis"""
        
        if baseline_results is None:
            baseline_results = self.results_dir / "baseline_experiment_results.json"
        
        if mo_results is None:
            mo_results = self.results_dir / "mo_experiment_results.json"
        
        if output_dir is None:
            output_dir = self.results_dir / "comparison_analysis"
        
        self.logger.info("Running comparison analysis...")
        
        # Check if result files exist
        baseline_path = Path(baseline_results)
        mo_path = Path(mo_results)
        
        if not baseline_path.exists():
            self.logger.error(f"Baseline results not found: {baseline_path}")
            return ""
        
        if not mo_path.exists():
            self.logger.error(f"MO results not found: {mo_path}")
            return ""
        
        # Load data
        import json
        
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
        
        with open(mo_path, 'r') as f:
            mo_data = json.load(f)
        
        # Perform statistical comparison
        combined_data = {**baseline_data, **mo_data}
        comparison_results = self.statistical_analyzer.perform_multiple_comparisons(combined_data)
        
        # Generate report
        report_dir = self.statistical_analyzer.generate_statistical_report(str(output_dir))
        
        self.logger.info(f"Comparison analysis completed. Report: {report_dir}")
        return str(report_dir)
    
    def run_demo_experiment(self) -> str:
        """Run a quick demo experiment"""
        
        self.logger.info("Running demo experiment...")
        
        # Run small baseline experiment
        baseline_results = self.run_baseline_experiment(
            methods=['random', 'adaptive_random'],
            programs=None,  # Use available programs
            n_tests=20,
            repetitions=3,
            output_file=self.results_dir / "demo_baseline_results.json"
        )
        
        # Run small MO experiment
        mo_results = self.run_multi_objective_experiment(
            algorithms=['NSGA2'],
            programs=None,
            generations=50,
            population_size=20,
            repetitions=3,
            output_file=self.results_dir / "demo_mo_results.json"
        )
        
        # Run comparison
        if baseline_results and mo_results:
            report_dir = self.run_comparison_analysis(
                baseline_results=str(self.results_dir / "demo_baseline_results.json"),
                mo_results=str(self.results_dir / "demo_mo_results.json"),
                output_dir=str(self.results_dir / "demo_comparison")
            )
            
            self.logger.info(f"Demo completed successfully! Check: {report_dir}")
            return report_dir
        else:
            self.logger.warning("Demo completed with limited results")
            return str(self.results_dir)
    
    def _calculate_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate summary statistics for results"""
        import numpy as np
        
        if not results:
            return {}
        
        # Extract common metrics
        coverage_values = []
        execution_times = []
        success_count = 0
        
        for result in results:
            if isinstance(result, dict):
                if result.get('success', False):
                    success_count += 1
                    
                coverage = result.get('coverage', 0.0)
                if coverage is not None and not np.isnan(coverage):
                    coverage_values.append(coverage)
                
                exec_time = result.get('execution_time', 0.0)
                if exec_time is not None and not np.isnan(exec_time):
                    execution_times.append(exec_time)
        
        summary = {
            'success_rate': success_count / len(results),
            'total_runs': len(results)
        }
        
        if coverage_values:
            summary.update({
                'coverage_mean': float(np.mean(coverage_values)),
                'coverage_std': float(np.std(coverage_values)),
                'coverage_median': float(np.median(coverage_values)),
                'coverage_min': float(np.min(coverage_values)),
                'coverage_max': float(np.max(coverage_values))
            })
        
        if execution_times:
            summary.update({
                'execution_time_mean': float(np.mean(execution_times)),
                'execution_time_std': float(np.std(execution_times)),
                'execution_time_median': float(np.median(execution_times))
            })
        
        return summary


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Unified Test Generation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py baseline --method random adaptive_random --programs simple_calc
  python main.py multi-objective --algorithm NSGA2 NSGA3 --generations 50
  python main.py compare --baseline results/baseline.json --mo results/mo.json
  python main.py analyze --data results/experiment.json
  python main.py demo
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Baseline experiment
    baseline_parser = subparsers.add_parser('baseline', help='Run baseline experiment')
    baseline_parser.add_argument('--method', nargs='+', 
                                help='Baseline methods to run')
    baseline_parser.add_argument('--programs', nargs='+',
                                help='Test programs to evaluate')
    baseline_parser.add_argument('--n-tests', type=int, default=50,
                                help='Number of test cases per method')
    baseline_parser.add_argument('--repetitions', type=int, default=10,
                                help='Number of repetitions')
    baseline_parser.add_argument('--output', '-o',
                                help='Output file for results')
    
    # Multi-objective experiment
    mo_parser = subparsers.add_parser('multi-objective', help='Run MO experiment')
    mo_parser.add_argument('--algorithm', nargs='+',
                          help='MO algorithms to run')
    mo_parser.add_argument('--programs', nargs='+',
                          help='Test programs to evaluate')
    mo_parser.add_argument('--generations', type=int, default=100,
                          help='Number of generations')
    mo_parser.add_argument('--population', type=int, default=50,
                          help='Population size')
    mo_parser.add_argument('--repetitions', type=int, default=10,
                          help='Number of repetitions')
    mo_parser.add_argument('--output', '-o',
                          help='Output file for results')
    
    # Comparison analysis
    compare_parser = subparsers.add_parser('compare', help='Compare baseline vs MO')
    compare_parser.add_argument('--baseline', 
                               help='Baseline results file')
    compare_parser.add_argument('--mo',
                               help='MO results file')
    compare_parser.add_argument('--output', '-o',
                               help='Output directory for analysis')
    
    # Statistical analysis
    analyze_parser = subparsers.add_parser('analyze', help='Perform statistical analysis')
    analyze_parser.add_argument('--data', required=True,
                               help='Experimental data file')
    analyze_parser.add_argument('--output', '-o',
                               help='Output directory for analysis')
    
    # Demo
    demo_parser = subparsers.add_parser('demo', help='Run demo experiment')
    
    # List available methods/algorithms
    list_parser = subparsers.add_parser('list', help='List available methods/algorithms')
    list_parser.add_argument('--type', choices=['baseline', 'mo', 'programs'],
                            default='baseline', help='What to list')
    
    # Visualization
    viz_parser = subparsers.add_parser('visualize', help='Generate visualizations')
    viz_parser.add_argument('--input', '-i', required=True,
                           help='Input directory or JSON file with statistical results')
    viz_parser.add_argument('--output', '-o', default='visualizations',
                           help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Initialize framework
    framework = UnifiedFramework(verbose=args.verbose)
    
    try:
        if args.command == 'baseline':
            results = framework.run_baseline_experiment(
                methods=args.method,
                programs=args.programs,
                n_tests=args.n_tests,
                repetitions=args.repetitions,
                output_file=args.output
            )
            print(f"Baseline experiment completed. Results saved.")
            
        elif args.command == 'multi-objective':
            results = framework.run_multi_objective_experiment(
                algorithms=args.algorithm,
                programs=args.programs,
                generations=args.generations,
                population_size=args.population,
                repetitions=args.repetitions,
                output_file=args.output
            )
            print(f"Multi-objective experiment completed. Results saved.")
            
        elif args.command == 'compare':
            report_dir = framework.run_comparison_analysis(
                baseline_results=args.baseline,
                mo_results=args.mo,
                output_dir=args.output
            )
            print(f"Comparison analysis completed. Report: {report_dir}")
            
        elif args.command == 'analyze':
            # Load and analyze data
            analyzer = StatisticalAnalyzer()
            
            import json
            with open(args.data, 'r') as f:
                data = json.load(f)
            
            results = analyzer.perform_multiple_comparisons(data)
            report_dir = analyzer.generate_statistical_report(args.output or "analysis_report")
            
            print(f"Statistical analysis completed. Report: {report_dir}")
            
        elif args.command == 'demo':
            report_dir = framework.run_demo_experiment()
            print(f"Demo completed successfully! Check results: {report_dir}")
            
        elif args.command == 'list':
            if args.type == 'baseline':
                methods = framework.baseline_generator.get_available_methods()
                print("Available baseline methods:")
                for method in methods:
                    print(f"  - {method}")
            elif args.type == 'programs':
                programs = framework.evaluator.get_program_list()
                print("Available test programs:")
                for program in programs:
                    print(f"  - {program}")
            elif args.type == 'mo':
                print("Available MO algorithms:")
                print("  - NSGA2")
                print("  - NSGA3")
                print("  - MOEAD")
                print("  - CTAEA")
        
        elif args.command == 'visualize':
            # Import and run visualization generator
            import subprocess
            import sys
            
            cmd = [
                sys.executable, 'generate_visualizations.py',
                '--input', args.input,
                '--output', args.output,
                '--no-show'  # Don't display plots in CLI mode
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Visualizations generated successfully!")
                print(f"📁 Output directory: {Path(args.output).absolute()}")
                print(f"🌐 Open {Path(args.output) / 'index.html'} to view all charts")
            else:
                print(f"❌ Visualization generation failed:")
                print(result.stderr)
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()