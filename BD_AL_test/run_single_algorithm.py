#!/usr/bin/env python3
"""
Single Algorithm Runner Script
Run one metaheuristic algorithm on all test programs with detailed analysis
"""

import argparse
import sys
import time
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

# Framework imports
from algorithm_factory import AlgorithmFactory
from config_loader import ConfigLoader
from metrics_collector import MetricsCollector, RunMetrics

# Existing project imports
from tree_converter import TreeVisitor
from test_fitness import Fitness
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination


class FitnessProblem(Problem):
    """Problem wrapper for pymoo optimization"""
    
    def __init__(self, fitness, dimensions):
        xl = -999999
        xu = 999999
        self.fitness = fitness
        super().__init__(n_var=dimensions, n_obj=1, xl=xl, xu=xu)
    
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.fitness.fitness_function(x)


class SingleAlgorithmRunner:
    """Runner for single algorithm experiments"""
    
    def __init__(self, algorithm_name: str, variant: Optional[str] = None,
                 custom_params: Optional[Dict[str, Any]] = None,
                 num_runs: int = 1, max_generations: int = 100,
                 verbose: bool = True):
        """
        Initialize the runner
        
        Args:
            algorithm_name: Name of the algorithm to run
            variant: Optional variant name
            custom_params: Custom parameters for the algorithm
            num_runs: Number of runs per test program
            max_generations: Maximum generations for optimization
            verbose: Whether to print progress
        """
        self.algorithm_name = algorithm_name
        self.variant = variant
        self.custom_params = custom_params or {}
        self.num_runs = num_runs
        self.max_generations = max_generations
        self.verbose = verbose
        
        # Initialize components
        self.config_loader = ConfigLoader()
        self.factory = AlgorithmFactory(self.config_loader)
        self.metrics = MetricsCollector()
        
        # Load configurations
        self.config_loader.load_all()
        self.test_programs = self.config_loader.test_programs_config['test_programs']
        
    def convert_tree(self, path: str) -> TreeVisitor:
        """Convert Python file to tree structure"""
        with open(path, 'r') as f:
            lines = f.readlines()
            tree = ast.parse(''.join(lines))
        
        visitor = TreeVisitor()
        visitor.visit(tree)
        return visitor
    
    def run_single_test(self, test_program_name: str, test_config: Dict,
                        run_id: int, seed: int) -> RunMetrics:
        """
        Run algorithm on a single test program
        
        Args:
            test_program_name: Name of the test program
            test_config: Test program configuration
            run_id: Run identifier
            seed: Random seed
            
        Returns:
            RunMetrics for the run
        """
        # Set random seed
        np.random.seed(seed)
        
        # Start metrics collection
        run_metrics = self.metrics.start_run(
            self.algorithm_name + (f"_{self.variant}" if self.variant else ""),
            test_program_name,
            run_id,
            seed
        )
        
        try:
            # Convert tree
            visitor = self.convert_tree(test_config['path'])
            fitness = Fitness(visitor)
            dimensions = test_config['dimensions']
            
            # Create algorithm
            algorithm = self.factory.create_algorithm(
                self.algorithm_name,
                variant=self.variant,
                custom_params=self.custom_params,
                dimensions=dimensions
            )
            
            # Create problem
            problem = FitnessProblem(fitness, dimensions)
            
            # Custom callback for metrics collection
            class MetricsCallback:
                def __init__(self, runner, metrics_collector, fitness_obj):
                    self.runner = runner
                    self.metrics = metrics_collector
                    self.fitness = fitness_obj
                    self.best_overall = float('inf')
                
                def __call__(self, algorithm):
                    gen = algorithm.n_gen
                    best_f = algorithm.pop.get("F").min()
                    
                    # Update metrics
                    self.metrics.update_generation(
                        generation=gen,
                        best_fitness=best_f,
                        population=algorithm.pop.get("X")
                    )
                    
                    # Check coverage if improved
                    if best_f < self.best_overall:
                        self.best_overall = best_f
                        best_x = algorithm.pop.get("X")[algorithm.pop.get("F").argmin()]
                        
                        # Get coverage
                        particle_pos = np.array([best_x], np.float32)
                        self.fitness.resolve_path(particle_pos)
                        coverage = len(list(set(self.fitness.walked_tree))) / len(self.fitness.whole_tree) if self.fitness.whole_tree else 0
                        
                        self.metrics.update_coverage(
                            coverage=coverage,
                            walked_tree=list(set(self.fitness.walked_tree)),
                            whole_tree=list(set(self.fitness.whole_tree))
                        )
                        
                        if self.runner.verbose:
                            print(f"    Gen {gen}: Fitness={best_f:.6f}, Coverage={coverage:.2%}")
            
            callback = MetricsCallback(self, self.metrics, fitness)
            
            # Run optimization
            if self.verbose:
                print(f"  Run {run_id}: ", end="")
            
            result = minimize(
                problem,
                algorithm,
                termination=('n_gen', self.max_generations),
                callback=callback,
                verbose=False
            )
            
            # Final update
            self.metrics.update_solution(result.X, result.F[0])
            
            # Calculate final coverage
            particle_pos = np.array([result.X], np.float32)
            fitness.resolve_path(particle_pos)
            final_coverage = len(list(set(fitness.walked_tree))) / len(fitness.whole_tree) if fitness.whole_tree else 0
            
            self.metrics.update_coverage(
                coverage=final_coverage,
                walked_tree=list(set(fitness.walked_tree)),
                whole_tree=list(set(fitness.whole_tree))
            )
            
            if self.verbose:
                print(f"Fitness={result.F[0]:.6f}, Coverage={final_coverage:.2%}")
            
        except Exception as e:
            print(f"    Error in run {run_id}: {e}")
        
        finally:
            self.metrics.end_run()
        
        return run_metrics
    
    def run_all_tests(self) -> Dict[str, List[RunMetrics]]:
        """
        Run algorithm on all test programs
        
        Returns:
            Dictionary mapping test program names to list of run metrics
        """
        results = {}
        
        print("\n" + "=" * 80)
        print(f"RUNNING {self.algorithm_name}" + 
              (f" ({self.variant})" if self.variant else "") +
              f" ON ALL TEST PROGRAMS")
        print("=" * 80)
        
        total_tests = len(self.test_programs)
        
        for idx, (prog_name, prog_config) in enumerate(self.test_programs.items(), 1):
            print(f"\n[{idx}/{total_tests}] Testing: {prog_name}")
            print(f"  Path: {prog_config['path']}")
            print(f"  Dimensions: {prog_config['dimensions']}")
            print(f"  Category: {prog_config['category']}")
            
            prog_results = []
            
            # Multiple runs with different seeds
            seeds = [42 + i * 100 for i in range(self.num_runs)]
            
            for run_id in range(self.num_runs):
                run_metrics = self.run_single_test(
                    prog_name,
                    prog_config,
                    run_id + 1,
                    seeds[run_id]
                )
                prog_results.append(run_metrics)
            
            results[prog_name] = prog_results
            
            # Print summary for this test program
            if self.num_runs > 1:
                fitnesses = [r.best_fitness for r in prog_results]
                coverages = [r.final_coverage for r in prog_results]
                print(f"  Summary: Fitness={np.mean(fitnesses):.6f}±{np.std(fitnesses):.6f}, "
                      f"Coverage={np.mean(coverages):.2%}±{np.std(coverages):.2%}")
        
        return results
    
    def print_summary(self, results: Dict[str, List[RunMetrics]]) -> None:
        """Print summary table of results"""
        print("\n" + "=" * 80)
        print("SUMMARY RESULTS")
        print("=" * 80)
        
        # Prepare data for table
        summary_data = []
        for prog_name, runs in results.items():
            fitnesses = [r.best_fitness for r in runs]
            coverages = [r.final_coverage for r in runs]
            times = [r.execution_time for r in runs]
            
            summary_data.append({
                'Test Program': prog_name,
                'Runs': len(runs),
                'Best Fitness': f"{np.min(fitnesses):.6f}",
                'Mean Fitness': f"{np.mean(fitnesses):.6f}",
                'Std Fitness': f"{np.std(fitnesses):.6f}",
                'Best Coverage': f"{np.max(coverages):.2%}",
                'Mean Coverage': f"{np.mean(coverages):.2%}",
                'Mean Time (s)': f"{np.mean(times):.2f}"
            })
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # Overall statistics
        all_fitnesses = [r.best_fitness for runs in results.values() for r in runs]
        all_coverages = [r.final_coverage for runs in results.values() for r in runs]
        all_times = [r.execution_time for runs in results.values() for r in runs]
        
        print("\n" + "-" * 80)
        print("OVERALL STATISTICS")
        print("-" * 80)
        print(f"Algorithm: {self.algorithm_name}" + 
              (f" ({self.variant})" if self.variant else ""))
        print(f"Total runs: {len(all_fitnesses)}")
        print(f"Mean fitness: {np.mean(all_fitnesses):.6f} ± {np.std(all_fitnesses):.6f}")
        print(f"Mean coverage: {np.mean(all_coverages):.2%} ± {np.std(all_coverages):.2%}")
        print(f"Total execution time: {np.sum(all_times):.2f}s")
        print(f"Programs with 100% coverage: {sum(1 for c in all_coverages if c >= 1.0)}/{len(all_coverages)}")
    
    def save_results(self, results: Dict[str, List[RunMetrics]], 
                    output_dir: str = "results") -> None:
        """Save results to files"""
        # Create output directory
        output_path = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algo_name = self.algorithm_name + (f"_{self.variant}" if self.variant else "")
        run_dir = output_path / f"{algo_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        csv_file = run_dir / "results.csv"
        self.metrics.export_to_csv(str(csv_file))
        print(f"\nResults saved to: {csv_file}")
        
        # Save to JSON
        json_file = run_dir / "results.json"
        self.metrics.export_to_json(str(json_file))
        
        # Save convergence data
        convergence_file = run_dir / "convergence_data.json"
        convergence_data = {}
        for prog_name, runs in results.items():
            convergence_data[prog_name] = {
                f"run_{i+1}": run.convergence_history 
                for i, run in enumerate(runs)
            }
        
        with open(convergence_file, 'w') as f:
            json.dump(convergence_data, f, indent=2)
        
        # Save configuration used
        config_file = run_dir / "config.json"
        config = {
            'algorithm': self.algorithm_name,
            'variant': self.variant,
            'custom_params': self.custom_params,
            'num_runs': self.num_runs,
            'max_generations': self.max_generations,
            'timestamp': timestamp
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"All results saved to: {run_dir}")
    
    def plot_results(self, results: Dict[str, List[RunMetrics]], 
                    save_path: Optional[str] = None) -> None:
        """Create visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Coverage bar chart
        ax1 = axes[0, 0]
        prog_names = list(results.keys())
        mean_coverages = [np.mean([r.final_coverage for r in runs]) * 100 
                         for runs in results.values()]
        colors = ['green' if c >= 95 else 'orange' if c >= 75 else 'red' 
                 for c in mean_coverages]
        
        bars = ax1.bar(range(len(prog_names)), mean_coverages, color=colors)
        ax1.set_xticks(range(len(prog_names)))
        ax1.set_xticklabels(prog_names, rotation=45, ha='right')
        ax1.set_ylabel('Coverage (%)')
        ax1.set_title('Code Coverage by Test Program')
        ax1.axhline(y=100, color='g', linestyle='--', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, mean_coverages):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # 2. Convergence curves
        ax2 = axes[0, 1]
        for prog_name, runs in list(results.items())[:5]:  # Show first 5 programs
            for run in runs[:1]:  # Show first run only for clarity
                if run.convergence_history:
                    ax2.plot(run.convergence_history, label=prog_name, alpha=0.7)
        
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Best Fitness')
        ax2.set_title('Convergence Curves')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 3. Execution time distribution
        ax3 = axes[1, 0]
        prog_names_short = [name[:15] + '...' if len(name) > 15 else name 
                           for name in prog_names]
        mean_times = [np.mean([r.execution_time for r in runs]) 
                     for runs in results.values()]
        
        bars = ax3.barh(range(len(prog_names)), mean_times, color='steelblue')
        ax3.set_yticks(range(len(prog_names)))
        ax3.set_yticklabels(prog_names_short)
        ax3.set_xlabel('Execution Time (seconds)')
        ax3.set_title('Execution Time by Test Program')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, mean_times):
            ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}s', ha='left', va='center', fontsize=8)
        
        # 4. Fitness box plot (if multiple runs)
        ax4 = axes[1, 1]
        if self.num_runs > 1:
            fitness_data = [
                [r.best_fitness for r in runs]
                for runs in results.values()
            ]
            box_plot = ax4.boxplot(fitness_data, labels=prog_names_short)
            ax4.set_xticklabels(prog_names_short, rotation=45, ha='right')
            ax4.set_ylabel('Best Fitness')
            ax4.set_title('Fitness Distribution Across Runs')
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')
        else:
            # Single run - show fitness values
            fitnesses = [runs[0].best_fitness for runs in results.values()]
            bars = ax4.bar(range(len(prog_names)), fitnesses, color='coral')
            ax4.set_xticks(range(len(prog_names)))
            ax4.set_xticklabels(prog_names_short, rotation=45, ha='right')
            ax4.set_ylabel('Best Fitness')
            ax4.set_title('Best Fitness by Test Program')
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')
        
        plt.suptitle(f'{self.algorithm_name}' + 
                    (f' ({self.variant})' if self.variant else '') + 
                    ' - Performance Analysis',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run a single metaheuristic algorithm on all test programs'
    )
    
    # Required arguments
    parser.add_argument('--algorithm', '-a', type=str, required=True,
                       help='Algorithm name (e.g., PSO, GA, DE)')
    
    # Optional arguments
    parser.add_argument('--variant', '-v', type=str,
                       help='Algorithm variant (e.g., aggressive, conservative)')
    parser.add_argument('--runs', '-r', type=int, default=1,
                       help='Number of runs per test program (default: 1)')
    parser.add_argument('--generations', '-g', type=int, default=100,
                       help='Maximum generations (default: 100)')
    parser.add_argument('--pop-size', type=int,
                       help='Population size (overrides default)')
    parser.add_argument('--output', '-o', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    parser.add_argument('--list-algorithms', action='store_true',
                       help='List available algorithms and exit')
    
    args = parser.parse_args()
    
    # List algorithms if requested
    if args.list_algorithms:
        config_loader = ConfigLoader()
        factory = AlgorithmFactory(config_loader)
        manager = factory.algorithm_manager
        print(manager.get_algorithm_summary())
        return 0
    
    # Prepare custom parameters
    custom_params = {}
    if args.pop_size:
        custom_params['pop_size'] = args.pop_size
    
    # Create and run
    try:
        runner = SingleAlgorithmRunner(
            algorithm_name=args.algorithm,
            variant=args.variant,
            custom_params=custom_params,
            num_runs=args.runs,
            max_generations=args.generations,
            verbose=not args.quiet
        )
        
        # Run tests
        results = runner.run_all_tests()
        
        # Display results
        runner.print_summary(results)
        
        # Save results
        runner.save_results(results, args.output)
        
        # Plot if requested
        if not args.no_plot:
            plot_path = Path(args.output) / f"{args.algorithm}_analysis.png"
            runner.plot_results(results, str(plot_path))
        
        # Print final statistics
        print(runner.metrics.get_statistics())
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())