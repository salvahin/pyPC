#!/usr/bin/env python3
"""
Multi-Objective Algorithm Runner Script
Run multi-objective optimization algorithms on test programs
Optimizes both fitness (minimize) and coverage (maximize)
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
from multi_objective_fitness import (
    MultiObjectiveFitness,
    MultiObjectiveProblem,
    MOFitnessFactory,
    calculate_hypervolume,
    calculate_adaptive_reference_point,
    get_fixed_reference_point
)

# Existing project imports
from tree_converter import TreeVisitor
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class MultiObjectiveRunner:
    """Runner for multi-objective optimization experiments"""
    
    def __init__(self, algorithm_name: str, n_objectives: int = 2,
                 num_runs: int = 1, max_generations: int = 100,
                 pop_size: int = 100, verbose: bool = True,
                 objective_type: str = 'traditional',
                 test_suite: str = None, max_branch_distance: float = 1000.0):
        """
        Initialize the multi-objective runner
        
        Args:
            algorithm_name: Name of the MO algorithm (NSGA2, NSGA3, MOEAD, CTAEA)
            n_objectives: Number of objectives (2 or 3)
            num_runs: Number of runs per test program
            max_generations: Maximum generations
            pop_size: Population size
            verbose: Whether to print progress
            objective_type: 'traditional' or 'conflicting'
        """
        self.algorithm_name = algorithm_name
        self.n_objectives = n_objectives
        self.num_runs = num_runs
        self.max_generations = max_generations
        self.pop_size = pop_size
        self.verbose = verbose
        self.objective_type = objective_type
        self.test_suite = test_suite
        self.max_branch_distance = max_branch_distance
        
        # Initialize components
        self.config_loader = ConfigLoader()
        self.factory = AlgorithmFactory(self.config_loader)
        
        # Metrics tracking
        self.coverage_history = {}
        self.path_coverage_history = {}
        self.execution_times = {}
        
        # Load configurations
        self.config_loader.load_all()
        
        # Select test programs based on suite
        if self.test_suite:
            suites = self.config_loader.test_programs_config.get('test_suites', {})
            if self.test_suite in suites:
                suite_programs = suites[self.test_suite]['programs']
                all_programs = self.config_loader.test_programs_config['test_programs']
                self.test_programs = {k: v for k, v in all_programs.items() if k in suite_programs}
                if self.verbose:
                    print(f"Using test suite '{self.test_suite}' with {len(self.test_programs)} programs")
            else:
                print(f"Warning: Test suite '{self.test_suite}' not found. Using all programs.")
                self.test_programs = self.config_loader.test_programs_config['test_programs']
        else:
            self.test_programs = self.config_loader.test_programs_config['test_programs']
        
        # Store results
        self.all_results = {}
        
    def convert_tree(self, path: str) -> TreeVisitor:
        """Convert Python file to tree structure"""
        with open(path, 'r') as f:
            lines = f.readlines()
            tree = ast.parse(''.join(lines))
        
        visitor = TreeVisitor()
        visitor.visit(tree)
        return visitor
    
    def create_algorithm(self, n_obj: int) -> Any:
        """
        Create multi-objective algorithm instance
        
        Args:
            n_obj: Number of objectives
            
        Returns:
            Algorithm instance
        """
        # Get algorithm from factory
        algo_factory = self.factory.create_algorithm(
            self.algorithm_name,
            custom_params={'pop_size': self.pop_size}
        )
        
        # If it's a lambda (needs n_obj), call it
        if callable(algo_factory) and not hasattr(algo_factory, 'pop'):
            return algo_factory(n_obj)
        
        return algo_factory
    
    def run_single_test(self, test_program_name: str, test_config: Dict,
                        run_id: int, seed: int) -> Dict[str, Any]:
        """
        Run MO algorithm on a single test program
        
        Args:
            test_program_name: Name of the test program
            test_config: Test program configuration
            run_id: Run identifier
            seed: Random seed
            
        Returns:
            Dictionary with run results
        """
        np.random.seed(seed)
        
        start_time = time.time()
        
        try:
            # Convert tree
            visitor = self.convert_tree(test_config['path'])
            
            # Create multi-objective problem with configurable max_branch_distance
            if self.n_objectives == 2:
                problem = MOFitnessFactory.create_dual_objective(
                    visitor, test_config['dimensions'],
                    objective_type=self.objective_type
                )
                # Set max_branch_distance if applicable
                if hasattr(problem.mo_fitness, 'max_branch_distance'):
                    problem.mo_fitness.max_branch_distance = self.max_branch_distance
            else:
                problem = MOFitnessFactory.create_three_objective(
                    visitor, test_config['dimensions']
                )
                if hasattr(problem.mo_fitness, 'max_branch_distance'):
                    problem.mo_fitness.max_branch_distance = self.max_branch_distance
            
            # Create algorithm
            algorithm = self.create_algorithm(self.n_objectives)
            
            # Create callback with enhanced metrics tracking
            callback = None
            prog_key = f"{test_program_name}_run{run_id}"
            self.coverage_history[prog_key] = []
            
            if self.verbose:
                def progress_callback(algorithm):
                    if algorithm.n_gen % 10 == 0:
                        # Get current Pareto front
                        F = algorithm.pop.get("F")
                        nds = NonDominatedSorting()
                        fronts = nds.do(F)
                        n_pareto = len(fronts[0]) if fronts else 0
                        
                        # Track best coverage in population
                        if self.objective_type == 'conflicting':
                            best_coverage = -np.min(F[:, 0]) if len(F) > 0 else 0
                        else:
                            best_coverage = -np.min(F[:, 1]) if len(F) > 0 else 0
                        
                        self.coverage_history[prog_key].append(best_coverage)
                        
                        print(f"    Gen {algorithm.n_gen}: {n_pareto} Pareto solutions, Best coverage: {best_coverage:.2%}")
                callback = progress_callback
            
            # Run optimization
            if self.verbose:
                print(f"  Run {run_id}: Starting optimization...")
            
            result = minimize(
                problem,
                algorithm,
                ('n_gen', self.max_generations),
                callback=callback,
                verbose=False
            )
            
            # Extract results
            pareto_front = result.F
            pareto_set = result.X
            
            # Calculate metrics based on objective type
            if self.objective_type == 'conflicting':
                # Obj 1: negative coverage, Obj 2: complexity
                coverage_values = -pareto_front[:, 0]  # Convert back from negative
                complexity_values = pareto_front[:, 1]
                # Use complexity as "fitness" for compatibility
                fitness_values = complexity_values
            else:
                # Traditional: Obj 1: fitness, Obj 2: negative coverage
                fitness_values = pareto_front[:, 0]
                coverage_values = -pareto_front[:, 1]  # Convert back from negative
            
            best_coverage_idx = np.argmax(coverage_values)
            best_fitness_idx = np.argmin(fitness_values)
            
            # Find knee point (best trade-off)
            knee_idx = self._find_knee_point(pareto_front)
            
            # Calculate additional metrics
            pareto_diversity = self._calculate_diversity(pareto_front)
            convergence_gen = self._find_convergence_generation(prog_key)
            
            # Get path coverage if available
            path_coverage = 0
            if hasattr(problem.mo_fitness, 'unique_paths'):
                path_coverage = len(problem.mo_fitness.unique_paths)
            
            end_time = time.time()
            
            results = {
                'run_id': run_id,
                'seed': seed,
                'execution_time': end_time - start_time,
                'n_pareto_solutions': len(pareto_front),
                'pareto_front': pareto_front.tolist(),
                'pareto_set': pareto_set.tolist(),
                'best_coverage': {
                    'coverage': float(coverage_values[best_coverage_idx]),
                    'fitness': float(fitness_values[best_coverage_idx]),
                    'solution': pareto_set[best_coverage_idx].tolist()
                },
                'best_fitness': {
                    'coverage': float(coverage_values[best_fitness_idx]),
                    'fitness': float(fitness_values[best_fitness_idx]),
                    'solution': pareto_set[best_fitness_idx].tolist()
                },
                'knee_point': {
                    'coverage': float(coverage_values[knee_idx]),
                    'fitness': float(fitness_values[knee_idx]),
                    'solution': pareto_set[knee_idx].tolist()
                },
                'hypervolume': self._calculate_hypervolume(pareto_front),
                'pareto_diversity': pareto_diversity,
                'convergence_generation': convergence_gen,
                'unique_paths_explored': path_coverage,
                'coverage_history': self.coverage_history.get(prog_key, [])
            }
            
            if self.verbose:
                print(f"    Completed: {len(pareto_front)} Pareto solutions found")
                print(f"    Best coverage: {results['best_coverage']['coverage']:.2%}")
                print(f"    Best fitness: {results['best_fitness']['fitness']:.6f}")
                
        except Exception as e:
            print(f"    Error in run {run_id}: {e}")
            import traceback
            traceback.print_exc()
            results = {
                'run_id': run_id,
                'seed': seed,
                'error': str(e),
                'execution_time': 0,
                'n_pareto_solutions': 0
            }
        
        return results
    
    def _find_knee_point(self, pareto_front: np.ndarray) -> int:
        """
        Find knee point in Pareto front (best trade-off)
        
        Args:
            pareto_front: Pareto front array
            
        Returns:
            Index of knee point
        """
        if len(pareto_front) == 1:
            return 0
        
        # Normalize objectives
        f_min = pareto_front.min(axis=0)
        f_max = pareto_front.max(axis=0)
        f_norm = (pareto_front - f_min) / (f_max - f_min + 1e-8)
        
        # Find point closest to ideal point (0, 0)
        distances = np.sqrt(np.sum(f_norm ** 2, axis=1))
        return np.argmin(distances)
    
    def _calculate_hypervolume(self, pareto_front: np.ndarray) -> float:
        """Calculate hypervolume indicator"""
        try:
            # Use fixed reference point based on objective type
            ref_point = get_fixed_reference_point(self.objective_type)
            return calculate_hypervolume(pareto_front, ref_point)
        except:
            return 0.0
    
    def _calculate_diversity(self, pareto_front: np.ndarray) -> Dict[str, float]:
        """Calculate diversity metrics for Pareto front"""
        if len(pareto_front) < 2:
            return {'spread': 0.0, 'spacing': 0.0}
        
        # Calculate spread (extent of Pareto front)
        spread = np.std(pareto_front, axis=0).mean()
        
        # Calculate spacing (uniformity of distribution)
        distances = []
        for i in range(len(pareto_front)):
            min_dist = float('inf')
            for j in range(len(pareto_front)):
                if i != j:
                    dist = np.linalg.norm(pareto_front[i] - pareto_front[j])
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)
        
        spacing = np.std(distances) if distances else 0.0
        
        return {'spread': float(spread), 'spacing': float(spacing)}
    
    def _find_convergence_generation(self, prog_key: str) -> int:
        """Find generation where best coverage was first achieved"""
        history = self.coverage_history.get(prog_key, [])
        if not history:
            return -1
        
        best_coverage = max(history)
        threshold = best_coverage * 0.95  # Within 95% of best
        
        for gen, coverage in enumerate(history):
            if coverage >= threshold:
                return (gen + 1) * 10  # Convert to actual generation number
        
        return self.max_generations
    
    def run_all_tests(self) -> Dict[str, List[Dict]]:
        """Run algorithm on all test programs"""
        results = {}
        
        print("\n" + "=" * 80)
        print(f"RUNNING {self.algorithm_name} (Multi-Objective) ON ALL TEST PROGRAMS")
        print(f"Objectives: {self.n_objectives} | Population: {self.pop_size} | Generations: {self.max_generations}")
        print("=" * 80)
        
        total_tests = len(self.test_programs)
        
        for idx, (prog_name, prog_config) in enumerate(self.test_programs.items(), 1):
            print(f"\n[{idx}/{total_tests}] Testing: {prog_name}")
            print(f"  Path: {prog_config['path']}")
            print(f"  Dimensions: {prog_config['dimensions']}")
            
            prog_results = []
            seeds = [42 + i * 100 for i in range(self.num_runs)]
            
            for run_id in range(self.num_runs):
                run_results = self.run_single_test(
                    prog_name,
                    prog_config,
                    run_id + 1,
                    seeds[run_id]
                )
                prog_results.append(run_results)
            
            results[prog_name] = prog_results
            
            # Print summary
            if prog_results and 'n_pareto_solutions' in prog_results[0]:
                avg_pareto = np.mean([r['n_pareto_solutions'] for r in prog_results])
                print(f"  Average Pareto solutions: {avg_pareto:.1f}")
        
        self.all_results = results
        return results
    
    def print_summary(self, results: Dict[str, List[Dict]]) -> None:
        """Print enhanced summary of results"""
        print("\n" + "=" * 80)
        print("MULTI-OBJECTIVE OPTIMIZATION SUMMARY")
        print(f"Algorithm: {self.algorithm_name} | Objective Type: {self.objective_type}")
        if self.test_suite:
            print(f"Test Suite: {self.test_suite}")
        print("=" * 80)
        
        summary_data = []
        
        for prog_name, runs in results.items():
            valid_runs = [r for r in runs if 'n_pareto_solutions' in r]
            
            if valid_runs:
                # Check if best_coverage exists in the run results
                if 'best_coverage' in valid_runs[0]:
                    best_coverage = max([r['best_coverage']['coverage'] for r in valid_runs])
                    best_fitness = min([r['best_fitness']['fitness'] for r in valid_runs])
                else:
                    # Handle case where these fields don't exist
                    best_coverage = 0.0
                    best_fitness = float('inf')
                
                # Get program complexity from config
                prog_config = self.config_loader.test_programs_config['test_programs'].get(prog_name, {})
                
                summary_data.append({
                    'Test Program': prog_name[:20],
                    'Complexity': prog_config.get('complexity', 'unknown'),
                    'Runs': len(valid_runs),
                    'Avg Pareto': np.mean([r['n_pareto_solutions'] for r in valid_runs]),
                    'Best Cov %': best_coverage * 100,
                    'Avg HV': np.mean([r.get('hypervolume', 0) for r in valid_runs]),
                    'Conv. Gen': np.mean([r.get('convergence_generation', -1) for r in valid_runs]),
                    'Paths': np.mean([r.get('unique_paths_explored', 0) for r in valid_runs]),
                    'Time (s)': np.mean([r['execution_time'] for r in valid_runs])
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            print(df.to_string(index=False, float_format='%.4f'))
        
        # Overall statistics
        all_pareto = [r['n_pareto_solutions'] for runs in results.values() 
                     for r in runs if 'n_pareto_solutions' in r]
        all_coverage = [r['best_coverage']['coverage'] for runs in results.values() 
                       for r in runs if 'best_coverage' in r]
        all_diversity = [r.get('pareto_diversity', {}).get('spread', 0) for runs in results.values()
                         for r in runs if 'pareto_diversity' in r]
        
        print("\n" + "-" * 80)
        print("OVERALL STATISTICS")
        print("-" * 80)
        print(f"Algorithm: {self.algorithm_name}")
        print(f"Total successful runs: {len(all_pareto)}")
        print(f"Average Pareto solutions: {np.mean(all_pareto):.1f}")
        print(f"Average Pareto diversity (spread): {np.mean(all_diversity):.3f}" if all_diversity else "")
        print(f"Programs with 100% coverage solution: {sum(1 for c in all_coverage if c >= 1.0)}/{len(all_coverage)}")
        print(f"Programs with <50% best coverage: {sum(1 for c in all_coverage if c < 0.5)}/{len(all_coverage)}")
    
    def plot_pareto_fronts(self, save_path: Optional[str] = None) -> None:
        """Plot Pareto fronts for all test programs with enhanced visualization"""
        n_programs = len(self.all_results)
        n_cols = 3
        n_rows = (n_programs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_programs > 1 else [axes]
        
        for idx, (prog_name, runs) in enumerate(self.all_results.items()):
            ax = axes[idx]
            
            all_coverages = []
            all_fitnesses = []
            
            # Plot all Pareto fronts from different runs
            for run_idx, run in enumerate(runs):
                if 'pareto_front' in run:
                    front = np.array(run['pareto_front'])
                    if len(front) > 0:
                        if self.objective_type == 'conflicting':
                            # Obj 1: negative coverage, Obj 2: complexity
                            coverage = -front[:, 0] * 100
                            fitness = front[:, 1]  # complexity
                            xlabel = 'Complexity (minimize)'
                        else:
                            # Traditional: fitness and coverage
                            coverage = -front[:, 1] * 100
                            fitness = front[:, 0]
                            xlabel = 'Fitness (minimize)'
                        
                        all_coverages.extend(coverage)
                        all_fitnesses.extend(fitness)
                        
                        # Color code by coverage level
                        colors = ['red' if c < 30 else 'yellow' if c < 70 else 'green' for c in coverage]
                        ax.scatter(fitness, coverage, c=colors, alpha=0.6, s=30, 
                                 label=f'Run {run_idx+1}' if run_idx < 3 else '')
            
            # Add expected coverage line if available
            prog_config = self.config_loader.test_programs_config['test_programs'].get(prog_name, {})
            expected_cov = prog_config.get('expected_coverage', 1.0) * 100
            ax.axhline(y=expected_cov, color='blue', linestyle='--', alpha=0.5, 
                      label=f'Expected: {expected_cov:.0f}%')
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Coverage % (maximize)')
            ax.set_title(f"{prog_name[:20]}{'...' if len(prog_name) > 20 else ''}\n" +
                        f"Complexity: {prog_config.get('complexity', 'unknown')}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_programs, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{self.algorithm_name} - Pareto Fronts', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Pareto fronts saved to: {save_path}")
        
        plt.show()
    
    def plot_coverage_progression(self, save_path: Optional[str] = None) -> None:
        """Plot coverage progression over generations"""
        if not self.coverage_history:
            print("No coverage history to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for prog_key, history in self.coverage_history.items():
            if history:
                generations = [(i+1) * 10 for i in range(len(history))]
                ax.plot(generations, [h * 100 for h in history], 
                       marker='o', label=prog_key, alpha=0.7)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Coverage (%)')
        ax.set_title(f'{self.algorithm_name} - Coverage Progression\nObjective Type: {self.objective_type}')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Coverage progression saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, output_dir: str = "results") -> None:
        """Save results to files"""
        output_path = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_path / f"MO_{self.algorithm_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        results_file = run_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        # Save Pareto fronts as CSV
        for prog_name, runs in self.all_results.items():
            for run_idx, run in enumerate(runs):
                if 'pareto_front' in run:
                    front = np.array(run['pareto_front'])
                    if len(front) > 0:
                        df = pd.DataFrame({
                            'fitness': front[:, 0],
                            'coverage': -front[:, 1],  # Convert back to positive
                            'complexity': front[:, 2] if self.n_objectives == 3 else None
                        })
                        csv_file = run_dir / f"pareto_{prog_name}_run{run_idx+1}.csv"
                        df.to_csv(csv_file, index=False)
        
        # Save configuration
        config_file = run_dir / "config.json"
        config = {
            'algorithm': self.algorithm_name,
            'n_objectives': self.n_objectives,
            'num_runs': self.num_runs,
            'max_generations': self.max_generations,
            'pop_size': self.pop_size,
            'timestamp': timestamp
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nResults saved to: {run_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run multi-objective optimization algorithms on test programs'
    )
    
    parser.add_argument('--algorithm', '-a', type=str, default='NSGA2',
                       choices=['NSGA2', 'NSGA3', 'MOEAD', 'CTAEA'],
                       help='Multi-objective algorithm (default: NSGA2)')
    parser.add_argument('--objectives', type=int, default=2, choices=[2, 3],
                       help='Number of objectives (default: 2)')
    parser.add_argument('--objective-type', type=str, default='conflicting',
                       choices=['traditional', 'conflicting'],
                       help='Objective type: traditional or conflicting (default: conflicting)')
    parser.add_argument('--runs', '-r', type=int, default=1,
                       help='Number of runs per test program (default: 1)')
    parser.add_argument('--generations', '-g', type=int, default=100,
                       help='Maximum generations (default: 100)')
    parser.add_argument('--pop-size', type=int, default=100,
                       help='Population size (default: 100)')
    parser.add_argument('--output', '-o', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    parser.add_argument('--test-suite', '-t', type=str, default=None,
                       choices=['basic', 'complex', 'deep_complex', 'all'],
                       help='Test suite to use (default: all programs)')
    parser.add_argument('--max-branch-distance', type=float, default=1000.0,
                       help='Maximum branch distance for normalization (default: 1000.0)')
    
    args = parser.parse_args()
    
    try:
        runner = MultiObjectiveRunner(
            algorithm_name=args.algorithm,
            n_objectives=args.objectives,
            num_runs=args.runs,
            max_generations=args.generations,
            pop_size=args.pop_size,
            verbose=not args.quiet,
            objective_type=args.objective_type,
            test_suite=args.test_suite,
            max_branch_distance=args.max_branch_distance
        )
        
        # Run tests
        results = runner.run_all_tests()
        
        # Display results
        runner.print_summary(results)
        
        # Save results
        runner.save_results(args.output)
        
        # Plot if requested
        if not args.no_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = Path(args.output) / f"MO_{args.algorithm}_{args.objective_type}_{timestamp}_pareto.png"
            runner.plot_pareto_fronts(str(plot_path))
            
            # Also create coverage progression plot
            runner.plot_coverage_progression(str(Path(args.output) / f"MO_{args.algorithm}_{timestamp}_coverage.png"))
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())