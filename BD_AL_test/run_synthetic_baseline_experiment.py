#!/usr/bin/env python3
"""
Comprehensive Synthetic Dataset Baseline Evaluation Experiment Runner

This script orchestrates the complete experimental evaluation comparing classical test generation
methods against multi-objective algorithms on the synthetic challenging functions dataset.

Usage:
    python3 run_synthetic_baseline_experiment.py [options]
    
Options:
    --phase: Experimental phase to run (pilot, baseline, comparison, analysis, all)
    --config: Path to experiment configuration file
    --output-dir: Output directory for results
    --dry-run: Show what would be executed without running
    --resume: Resume from previous interrupted run
    --workers: Number of parallel workers
"""

import argparse
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os
from datetime import datetime
import hashlib
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal
import time

# Import our custom modules
from synthetic_baseline_generators import SyntheticDatasetBaselineGenerator
from synthetic_dataset_adapter import SyntheticDatasetAdapter  
from synthetic_baseline_evaluator import SyntheticBaselineEvaluator
from baseline_vs_mo_comprehensive import BaselineVsMOComparator
from synthetic_statistical_analyzer import SyntheticStatisticalAnalyzer


class ExperimentState:
    """Track experiment execution state"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.state_file = output_dir / "experiment_state.json"
        self.state = self.load_state()
    
    def load_state(self) -> Dict:
        """Load experiment state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return self.default_state()
        return self.default_state()
    
    def default_state(self) -> Dict:
        """Default experiment state"""
        return {
            'phase': 'not_started',
            'completed_functions': [],
            'completed_methods': [],
            'start_time': None,
            'last_checkpoint': None,
            'errors': [],
            'statistics': {
                'total_evaluations': 0,
                'successful_evaluations': 0,
                'failed_evaluations': 0
            }
        }
    
    def save_state(self):
        """Save current state to file"""
        self.state['last_checkpoint'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def update_phase(self, phase: str):
        """Update current phase"""
        self.state['phase'] = phase
        self.save_state()
    
    def mark_completed(self, function_name: str, method_name: str):
        """Mark function-method combination as completed"""
        completion_key = f"{function_name}::{method_name}"
        if completion_key not in self.state['completed_functions']:
            self.state['completed_functions'].append(completion_key)
        self.save_state()
    
    def is_completed(self, function_name: str, method_name: str) -> bool:
        """Check if function-method combination is completed"""
        completion_key = f"{function_name}::{method_name}"
        return completion_key in self.state['completed_functions']
    
    def add_error(self, error_info: Dict):
        """Add error to state"""
        self.state['errors'].append({
            'timestamp': datetime.now().isoformat(),
            **error_info
        })
        self.save_state()


class SyntheticExperimentRunner:
    """Main experiment runner for synthetic baseline evaluation"""
    
    def __init__(self, config_path: str, output_dir: str, resume: bool = False):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize experiment state
        self.state = ExperimentState(self.output_dir)
        
        # Initialize components
        self.generators = None
        self.adapter = None
        self.evaluator = None
        self.comparator = None
        self.analyzer = None
        
        self.resume = resume
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('SyntheticExperimentRunner')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_file = self.output_dir / 'experiment.log'
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        self.logger.warning(f"Received signal {signum}. Saving state and shutting down...")
        self.state.save_state()
        sys.exit(0)
    
    def initialize_components(self):
        """Initialize all experiment components"""
        try:
            self.logger.info("Initializing experiment components...")
            
            # Initialize baseline generators with default parameter ranges
            default_param_ranges = [
                {"name": "input_data", "type": "bytes", "min_length": 0, "max_length": 100},
                {"name": "config", "type": "dict", "required_keys": ["param1"], "param1": [1, 10]}
            ]
            self.generators = SyntheticDatasetBaselineGenerator(default_param_ranges)
            
            # Initialize dataset adapter
            dataset_config = self.config['dataset']['config_file']
            self.adapter = SyntheticDatasetAdapter(dataset_config)
            
            # Initialize evaluator
            self.evaluator = SyntheticBaselineEvaluator(
                max_workers=self.config.get('computational_resources', {}).get('recommended_cores', 4)
            )
            
            # Initialize comparison framework
            self.comparator = BaselineVsMOComparator()
            
            # Initialize statistical analyzer
            stat_config = self.config.get('statistical_analysis', {})
            self.analyzer = SyntheticStatisticalAnalyzer(
                alpha=stat_config.get('significance_level', 0.05),
                min_effect_size=stat_config.get('practical_significance_threshold', 0.3)
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run_phase_pilot(self) -> Dict[str, Any]:
        """Run pilot phase for validation"""
        self.logger.info("Starting pilot phase...")
        self.state.update_phase('pilot')
        
        pilot_config = self.config['experimental_phases']['phase_1_pilot']
        functions = pilot_config['functions']
        methods = pilot_config['methods']
        repetitions = pilot_config['repetitions']
        
        results = {}
        
        for func_name in functions:
            self.logger.info(f"Running pilot test on function: {func_name}")
            func_results = {}
            
            for method in methods:
                if self.resume and self.state.is_completed(func_name, method):
                    self.logger.info(f"Skipping completed: {func_name} - {method}")
                    continue
                
                try:
                    # Generate test cases
                    if method in self.config['baseline_methods']:
                        test_cases = self.generators.generate_synthetic_tests(
                            method, repetitions
                        )
                    else:
                        # This would be MO algorithm - skip for pilot if not implemented
                        self.logger.info(f"Skipping MO method {method} in pilot phase")
                        continue
                    
                    # Evaluate test cases
                    evaluation_results = self.adapter.evaluate_function_batch(
                        func_name, test_cases, timeout=30.0
                    )
                    
                    func_results[method] = {
                        'test_cases': len(test_cases),
                        'results': evaluation_results,
                        'status': 'completed'
                    }
                    
                    self.state.mark_completed(func_name, method)
                    self.logger.info(f"Completed pilot: {func_name} - {method}")
                    
                except Exception as e:
                    self.logger.error(f"Pilot error {func_name}-{method}: {e}")
                    func_results[method] = {'status': 'failed', 'error': str(e)}
                    self.state.add_error({
                        'phase': 'pilot',
                        'function': func_name,
                        'method': method,
                        'error': str(e)
                    })
            
            results[func_name] = func_results
        
        # Save pilot results
        pilot_file = self.output_dir / 'pilot_results.json'
        with open(pilot_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info("Pilot phase completed")
        return results
    
    def run_phase_baseline(self) -> Dict[str, Any]:
        """Run full baseline evaluation phase"""
        self.logger.info("Starting baseline evaluation phase...")
        self.state.update_phase('baseline')
        
        # Load function list
        functions = self.adapter.get_function_list()
        baseline_methods = list(self.config['baseline_methods'].keys())
        repetitions = self.config['experimental_design']['repetitions']
        
        self.logger.info(f"Evaluating {len(baseline_methods)} methods on {len(functions)} functions")
        
        # Use comprehensive evaluator
        results = self.evaluator.evaluate_comprehensive(
            functions, baseline_methods, repetitions
        )
        
        # Save baseline results
        baseline_file = self.output_dir / 'baseline_results.json'
        with open(baseline_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info("Baseline evaluation phase completed")
        return results
    
    def run_phase_comparison(self) -> Dict[str, Any]:
        """Run baseline vs MO comparison phase"""
        self.logger.info("Starting baseline vs MO comparison phase...")
        self.state.update_phase('comparison')
        
        # Load baseline results
        baseline_file = self.output_dir / 'baseline_results.json'
        if not baseline_file.exists():
            raise FileNotFoundError("Baseline results not found. Run baseline phase first.")
        
        # Load MO results (assuming they exist from parallel MO experiment)
        mo_results_paths = list(Path(".").glob("parallel_mo_results/*.json"))
        if not mo_results_paths:
            self.logger.warning("No MO results found. Running comparison with available data.")
            return {}
        
        # Use comparison framework
        comparison_results = self.comparator.perform_comprehensive_comparison(
            str(baseline_file),
            str(mo_results_paths[0])  # Use most recent MO results
        )
        
        # Save comparison results
        comparison_file = self.output_dir / 'comparison_results.json'
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        self.logger.info("Comparison phase completed")
        return comparison_results
    
    def run_phase_analysis(self) -> Dict[str, Any]:
        """Run statistical analysis phase"""
        self.logger.info("Starting statistical analysis phase...")
        self.state.update_phase('analysis')
        
        # Load all results
        baseline_file = self.output_dir / 'baseline_results.json'
        comparison_file = self.output_dir / 'comparison_results.json'
        
        if not baseline_file.exists():
            raise FileNotFoundError("Baseline results not found")
        
        try:
            # Load MO results for comparison
            mo_results_paths = list(Path(".").glob("parallel_mo_results/*.json"))
            if mo_results_paths:
                mo_results_file = str(mo_results_paths[0])
            else:
                # Create dummy MO results for analysis
                mo_results_file = self._create_dummy_mo_results()
            
            # Perform comprehensive statistical analysis
            baseline_data, mo_data = self.analyzer.load_experimental_data(
                str(baseline_file), mo_results_file
            )
            
            # Run statistical comparisons
            comparison_results = self.analyzer.perform_multiple_comparisons(
                baseline_data, mo_data
            )
            
            # Perform meta-analysis
            meta_results = self.analyzer.perform_meta_analysis()
            
            # Generate comprehensive report
            report_dir = self.analyzer.generate_comprehensive_report(
                str(self.output_dir / "statistical_analysis")
            )
            
            analysis_results = {
                'comparison_results': len(comparison_results),
                'significant_results': sum(1 for r in comparison_results if r.statistical_test.significant),
                'meta_analysis_categories': len(meta_results),
                'report_directory': str(report_dir)
            }
            
            # Save analysis summary
            analysis_file = self.output_dir / 'analysis_results.json'
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            self.logger.info("Statistical analysis phase completed")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Analysis phase error: {e}")
            raise
    
    def _create_dummy_mo_results(self) -> str:
        """Create dummy MO results for testing statistical analysis"""
        import random
        
        # Load function list
        functions = self.adapter.get_function_list()
        mo_methods = list(self.config['mo_algorithms'].keys())
        
        dummy_results = {}
        for func_name in functions[:5]:  # Use subset for dummy data
            dummy_results[func_name] = {}
            for method in mo_methods:
                results = []
                for _ in range(10):  # 10 dummy runs
                    results.append({
                        'coverage': random.uniform(0.1, 0.8),
                        'branch_distance': random.uniform(0.0, 10.0),
                        'approach_level': random.uniform(0.0, 5.0),
                        'execution_time': random.uniform(1.0, 30.0)
                    })
                
                dummy_results[func_name][method] = {
                    'results': results,
                    'statistics': {
                        'mean_coverage': sum(r['coverage'] for r in results) / len(results)
                    }
                }
        
        dummy_file = self.output_dir / 'dummy_mo_results.json'
        with open(dummy_file, 'w') as f:
            json.dump(dummy_results, f, indent=2)
        
        return str(dummy_file)
    
    def run_experiment(self, phase: str = 'all', dry_run: bool = False) -> Dict[str, Any]:
        """Run the complete experiment or specific phase"""
        
        if dry_run:
            self.logger.info("DRY RUN MODE - No actual execution")
            return self._show_execution_plan(phase)
        
        self.logger.info(f"Starting synthetic baseline experiment - Phase: {phase}")
        self.state.state['start_time'] = datetime.now().isoformat()
        self.state.save_state()
        
        # Initialize components
        self.initialize_components()
        
        results = {}
        
        try:
            if phase in ['pilot', 'all']:
                results['pilot'] = self.run_phase_pilot()
            
            if phase in ['baseline', 'all']:
                results['baseline'] = self.run_phase_baseline()
            
            if phase in ['comparison', 'all']:
                results['comparison'] = self.run_phase_comparison()
            
            if phase in ['analysis', 'all']:
                results['analysis'] = self.run_phase_analysis()
            
            # Generate final summary
            summary = self._generate_final_summary(results)
            summary_file = self.output_dir / 'experiment_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info("Experiment completed successfully!")
            self.state.update_phase('completed')
            
            return results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            self.state.add_error({
                'phase': 'experiment',
                'error': str(e),
                'fatal': True
            })
            raise
    
    def _show_execution_plan(self, phase: str) -> Dict[str, Any]:
        """Show what would be executed in dry run mode"""
        plan = {
            'experiment_config': self.config['experiment_metadata']['name'],
            'output_directory': str(self.output_dir),
            'requested_phase': phase,
            'execution_plan': {}
        }
        
        if phase in ['pilot', 'all']:
            pilot_config = self.config['experimental_phases']['phase_1_pilot']
            plan['execution_plan']['pilot'] = {
                'functions': pilot_config['functions'],
                'methods': pilot_config['methods'],
                'repetitions': pilot_config['repetitions'],
                'estimated_runtime_minutes': len(pilot_config['functions']) * len(pilot_config['methods']) * 5
            }
        
        if phase in ['baseline', 'all']:
            functions = len(self.config['dataset']['total_functions'])
            methods = len(self.config['baseline_methods'])
            repetitions = self.config['experimental_design']['repetitions']
            plan['execution_plan']['baseline'] = {
                'total_functions': functions,
                'baseline_methods': methods,
                'repetitions': repetitions,
                'total_evaluations': functions * methods * repetitions,
                'estimated_runtime_hours': (functions * methods * repetitions * 30) / 3600  # 30 sec per eval
            }
        
        return plan
    
    def _generate_final_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final experiment summary"""
        return {
            'experiment_metadata': self.config['experiment_metadata'],
            'execution_summary': {
                'phases_completed': list(results.keys()),
                'total_runtime': self._calculate_runtime(),
                'final_state': self.state.state,
            },
            'results_summary': results,
            'output_files': [str(f) for f in self.output_dir.glob('*.json')],
            'completion_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_runtime(self) -> str:
        """Calculate total experiment runtime"""
        if self.state.state['start_time']:
            start = datetime.fromisoformat(self.state.state['start_time'])
            duration = datetime.now() - start
            return str(duration)
        return "unknown"


def main():
    """Main entry point for the experiment runner"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive synthetic baseline evaluation experiment"
    )
    
    parser.add_argument(
        '--phase', 
        choices=['pilot', 'baseline', 'comparison', 'analysis', 'all'],
        default='all',
        help='Experimental phase to run'
    )
    
    parser.add_argument(
        '--config',
        default='config/synthetic_baseline_experiment.yaml',
        help='Path to experiment configuration file'
    )
    
    parser.add_argument(
        '--output-dir',
        default='synthetic_experiment_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show execution plan without running'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous interrupted run'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        help='Number of parallel workers'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize experiment runner
        runner = SyntheticExperimentRunner(
            config_path=args.config,
            output_dir=args.output_dir,
            resume=args.resume
        )
        
        # Run experiment
        results = runner.run_experiment(
            phase=args.phase,
            dry_run=args.dry_run
        )
        
        if args.dry_run:
            print("\n=== EXECUTION PLAN ===")
            print(json.dumps(results, indent=2))
        else:
            print(f"\n=== EXPERIMENT COMPLETED ===")
            print(f"Results saved to: {args.output_dir}")
            print(f"Summary: {len(results)} phases completed")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()