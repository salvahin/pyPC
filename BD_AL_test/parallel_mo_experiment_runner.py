#!/usr/bin/env python3
"""
Parallel Multi-Objective Experiment Runner
Comprehensive framework for parallel execution and comparison of MO algorithms
"""

import os
import sys
import time
import ast
import json
import pickle
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from itertools import product
import traceback

# Framework imports
from algorithm_factory import AlgorithmFactory
from config_loader import ConfigLoader
from multi_objective_fitness import (
    MultiObjectiveFitness,
    MOFitnessFactory,
    get_fixed_reference_point
)
from tree_converter import TreeVisitor

# Pymoo imports
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.ref_dirs import get_reference_directions


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    algorithm: str
    program: str
    program_path: str
    dimensions: int
    run_id: int
    seed: int
    generations: int
    pop_size: int
    objective_type: str = 'conflicting'
    n_objectives: int = 2
    
    def to_dict(self) -> Dict:
        return {
            'algorithm': self.algorithm,
            'program': self.program,
            'run_id': self.run_id,
            'seed': self.seed
        }


@dataclass
class ExperimentResult:
    """Result from a single experiment with comprehensive metrics"""
    config: ExperimentConfig
    pareto_front: Optional[np.ndarray] = None
    pareto_set: Optional[np.ndarray] = None
    execution_time: float = 0.0
    n_evaluations: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    convergence_history: List[float] = field(default_factory=list)
    
    # Enhanced coverage metrics
    coverage_history: List[float] = field(default_factory=list)
    final_coverage: float = 0.0
    branch_coverage_map: Dict[str, bool] = field(default_factory=dict)
    time_to_targets: Dict[float, int] = field(default_factory=dict)  # {coverage: generation}
    
    # Diversity and exploration metrics
    diversity_history: List[float] = field(default_factory=list)
    unique_solutions_count: int = 0
    unique_paths: Set[str] = field(default_factory=set)
    solution_redundancy: float = 0.0
    
    # Convergence and efficiency metrics
    stagnation_count: int = 0
    convergence_generation: int = -1
    coverage_efficiency: float = 0.0  # coverage per evaluation
    improvement_rate: float = 0.0
    
    error: Optional[str] = None
    
    def is_successful(self) -> bool:
        return self.error is None and self.pareto_front is not None


class ParallelMOExperimentRunner:
    """Parallel runner for multi-objective optimization experiments"""
    
    def __init__(self,
                 algorithms: List[str] = None,
                 test_suite: str = 'basic',
                 n_workers: int = None,
                 runs_per_config: int = 10,
                 generations: int = 50,
                 pop_size: int = 50,
                 objective_type: str = 'conflicting',
                 verbose: bool = True):
        """
        Initialize parallel experiment runner
        
        Args:
            algorithms: List of MO algorithms to test
            test_suite: Test program suite to use
            n_workers: Number of parallel workers (None = CPU count - 1)
            runs_per_config: Number of runs per (algorithm, program) pair
            generations: Number of generations per run
            pop_size: Population size
            objective_type: Type of objectives
            verbose: Print progress information
        """
        self.algorithms = algorithms or ['NSGA2', 'NSGA3', 'MOEAD', 'CTAEA']
        self.test_suite = test_suite
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.runs_per_config = runs_per_config
        self.generations = generations
        self.pop_size = pop_size
        self.objective_type = objective_type
        self.verbose = verbose
        
        # Load configurations
        self.config_loader = ConfigLoader()
        self.config_loader.load_all()
        self._load_test_programs()
        
        # Storage for results
        self.results: Dict[str, Dict[str, List[ExperimentResult]]] = {}
        self.reference_fronts: Dict[str, np.ndarray] = {}
        
        if self.verbose:
            print(f"Initialized ParallelMOExperimentRunner")
            print(f"  Workers: {self.n_workers}")
            print(f"  Algorithms: {self.algorithms}")
            print(f"  Test suite: {self.test_suite} ({len(self.test_programs)} programs)")
            print(f"  Runs per config: {self.runs_per_config}")
    
    def _load_test_programs(self):
        """Load test programs based on suite selection"""
        suites = self.config_loader.test_programs_config.get('test_suites', {})
        
        if self.test_suite in suites:
            suite_programs = suites[self.test_suite]['programs']
            all_programs = self.config_loader.test_programs_config['test_programs']
            self.test_programs = {
                k: v for k, v in all_programs.items() 
                if k in suite_programs
            }
        else:
            # Use all programs if suite not found
            self.test_programs = self.config_loader.test_programs_config['test_programs']
    
    def create_experiment_configs(self) -> List[ExperimentConfig]:
        """Create all experiment configurations"""
        configs = []
        
        for algorithm in self.algorithms:
            for prog_name, prog_config in self.test_programs.items():
                for run_id in range(self.runs_per_config):
                    config = ExperimentConfig(
                        algorithm=algorithm,
                        program=prog_name,
                        program_path=prog_config['path'],
                        dimensions=prog_config['dimensions'],
                        run_id=run_id + 1,
                        seed=42 + run_id * 100 + hash(algorithm) % 1000,
                        generations=self.generations,
                        pop_size=self.pop_size,
                        objective_type=self.objective_type
                    )
                    configs.append(config)
        
        return configs
    
    def run_experiments(self) -> Dict[str, Dict[str, List[ExperimentResult]]]:
        """
        Run all experiments in parallel
        
        Returns:
            Nested dict: algorithm -> program -> list of results
        """
        configs = self.create_experiment_configs()
        total_experiments = len(configs)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"RUNNING {total_experiments} EXPERIMENTS IN PARALLEL")
            print(f"{'='*80}")
        
        # Initialize result structure
        for algorithm in self.algorithms:
            self.results[algorithm] = {}
            for prog_name in self.test_programs:
                self.results[algorithm][prog_name] = []
        
        # Run experiments in parallel
        completed = 0
        failed = 0
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all jobs
            future_to_config = {
                executor.submit(run_single_experiment, config): config
                for config in configs
            }
            
            # Process completed jobs
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                completed += 1
                
                try:
                    result = future.result(timeout=300)  # 5 min timeout
                    self.results[config.algorithm][config.program].append(result)
                    
                    if self.verbose:
                        status = "✓" if result.is_successful() else "✗"
                        print(f"[{completed}/{total_experiments}] {status} "
                              f"{config.algorithm}/{config.program}/run{config.run_id} "
                              f"(time: {result.execution_time:.1f}s)")
                    
                    if not result.is_successful():
                        failed += 1
                        
                except Exception as e:
                    failed += 1
                    error_result = ExperimentResult(
                        config=config,
                        error=str(e)
                    )
                    self.results[config.algorithm][config.program].append(error_result)
                    
                    if self.verbose:
                        print(f"[{completed}/{total_experiments}] ✗ "
                              f"{config.algorithm}/{config.program}/run{config.run_id} "
                              f"- Error: {str(e)[:50]}")
        
        elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"EXPERIMENTS COMPLETED")
            print(f"  Total time: {elapsed:.1f}s")
            print(f"  Successful: {completed - failed}/{total_experiments}")
            print(f"  Average time per experiment: {elapsed/total_experiments:.1f}s")
            print(f"  Speedup vs sequential: {total_experiments*elapsed/total_experiments/elapsed:.1f}x")
            print(f"{'='*80}")
        
        # Calculate reference fronts
        self._calculate_reference_fronts()
        
        return self.results
    
    def _calculate_reference_fronts(self):
        """Calculate reference fronts for each program"""
        for prog_name in self.test_programs:
            all_fronts = []
            
            # Collect all Pareto fronts for this program
            for algorithm in self.algorithms:
                for result in self.results[algorithm][prog_name]:
                    if result.is_successful() and result.pareto_front is not None:
                        all_fronts.append(result.pareto_front)
            
            # Create reference front as non-dominated set of all fronts
            if all_fronts:
                combined = np.vstack(all_fronts)
                nds = NonDominatedSorting()
                fronts = nds.do(combined)
                if fronts and len(fronts[0]) > 0:
                    self.reference_fronts[prog_name] = combined[fronts[0]]
    
    def calculate_convergence_metrics(self, results: List[ExperimentResult]) -> Dict[str, float]:
        """Calculate convergence and efficiency metrics"""
        metrics = {}
        
        # Coverage metrics
        final_coverages = [r.final_coverage for r in results if r.is_successful()]
        if final_coverages:
            metrics['coverage_mean'] = np.mean(final_coverages)
            metrics['coverage_std'] = np.std(final_coverages)
            metrics['coverage_max'] = np.max(final_coverages)
        
        # Time to targets
        for target in [0.5, 0.75, 0.9, 0.95, 1.0]:
            times = [r.time_to_targets.get(target, -1) for r in results if r.is_successful()]
            valid_times = [t for t in times if t >= 0]
            if valid_times:
                metrics[f'time_to_{int(target*100)}pct'] = np.mean(valid_times)
            else:
                metrics[f'time_to_{int(target*100)}pct'] = -1
        
        # Efficiency metrics
        efficiencies = [r.coverage_efficiency for r in results if r.is_successful()]
        if efficiencies:
            metrics['coverage_efficiency'] = np.mean(efficiencies)
            metrics['improvement_rate'] = np.mean([r.improvement_rate for r in results if r.is_successful()])
        
        # Stagnation
        stagnations = [r.stagnation_count for r in results if r.is_successful()]
        if stagnations:
            metrics['avg_stagnation'] = np.mean(stagnations)
            metrics['max_stagnation'] = np.max(stagnations)
        
        return metrics
    
    def calculate_diversity_metrics(self, results: List[ExperimentResult]) -> Dict[str, float]:
        """Calculate diversity and exploration metrics"""
        metrics = {}
        
        # Solution uniqueness
        unique_counts = [r.unique_solutions_count for r in results if r.is_successful()]
        if unique_counts:
            metrics['unique_solutions_mean'] = np.mean(unique_counts)
            metrics['unique_solutions_total'] = np.sum(unique_counts)
        
        # Redundancy
        redundancies = [r.solution_redundancy for r in results if r.is_successful()]
        if redundancies:
            metrics['solution_redundancy'] = np.mean(redundancies)
        
        # Diversity evolution
        for r in results:
            if r.is_successful() and r.diversity_history:
                # Calculate diversity trend (increasing/decreasing)
                if len(r.diversity_history) > 1:
                    trend = np.polyfit(range(len(r.diversity_history)), r.diversity_history, 1)[0]
                    if 'diversity_trend' not in metrics:
                        metrics['diversity_trend'] = []
                    metrics['diversity_trend'].append(trend)
        
        if 'diversity_trend' in metrics:
            metrics['diversity_trend'] = np.mean(metrics['diversity_trend'])
        
        return metrics
    
    def calculate_robustness_metrics(self, results: List[ExperimentResult]) -> Dict[str, float]:
        """Calculate robustness and stability metrics"""
        metrics = {}
        
        if not results:
            return metrics
        
        # Performance consistency
        performances = [r.metrics.get('hypervolume', 0) for r in results if r.is_successful()]
        if len(performances) > 1:
            metrics['performance_cv'] = np.std(performances) / np.mean(performances) if np.mean(performances) > 0 else 0
            metrics['performance_iqr'] = np.percentile(performances, 75) - np.percentile(performances, 25)
        
        # Success stability
        metrics['success_rate'] = len([r for r in results if r.is_successful()]) / len(results)
        
        # Convergence stability
        conv_gens = [r.convergence_generation for r in results if r.is_successful() and r.convergence_generation > 0]
        if conv_gens:
            metrics['convergence_gen_mean'] = np.mean(conv_gens)
            metrics['convergence_gen_std'] = np.std(conv_gens)
        
        return metrics
    
    def calculate_tradeoff_metrics(self, pareto_fronts: List[np.ndarray]) -> Dict[str, float]:
        """Calculate trade-off quality metrics"""
        metrics = {}
        
        if not pareto_fronts:
            return metrics
        
        # Spread and spacing
        spreads = []
        spacings = []
        
        for front in pareto_fronts:
            if len(front) > 1:
                # Calculate spread
                sorted_front = front[front[:, 0].argsort()]
                distances = [np.linalg.norm(sorted_front[i] - sorted_front[i+1]) 
                           for i in range(len(sorted_front)-1)]
                if distances:
                    mean_dist = np.mean(distances)
                    spread = np.std(distances) / mean_dist if mean_dist > 0 else 0
                    spreads.append(spread)
                
                # Calculate spacing
                min_distances = []
                for i in range(len(front)):
                    dists = [np.linalg.norm(front[i] - front[j]) 
                            for j in range(len(front)) if i != j]
                    if dists:
                        min_distances.append(min(dists))
                if min_distances:
                    mean_min_dist = np.mean(min_distances)
                    spacing = np.sqrt(np.sum((np.array(min_distances) - mean_min_dist) ** 2) / len(min_distances))
                    spacings.append(spacing)
        
        if spreads:
            metrics['spread_mean'] = np.mean(spreads)
            metrics['spread_std'] = np.std(spreads)
        
        if spacings:
            metrics['spacing_mean'] = np.mean(spacings)
            metrics['spacing_std'] = np.std(spacings)
        
        # Front size variation
        front_sizes = [len(f) for f in pareto_fronts]
        if front_sizes:
            metrics['front_size_mean'] = np.mean(front_sizes)
            metrics['front_size_std'] = np.std(front_sizes)
        
        return metrics
    
    def analyze_and_compare(self) -> pd.DataFrame:
        """
        Analyze and compare algorithm performance
        
        Returns:
            DataFrame with comparative analysis
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("ANALYZING RESULTS")
            print(f"{'='*80}")
        
        analysis_data = []
        
        for algorithm in self.algorithms:
            for prog_name in self.test_programs:
                results = self.results[algorithm][prog_name]
                successful_results = [r for r in results if r.is_successful()]
                
                if successful_results:
                    # Basic statistics
                    hvs = [r.metrics.get('hypervolume', 0) for r in successful_results]
                    n_solutions = [len(r.pareto_front) for r in successful_results]
                    exec_times = [r.execution_time for r in successful_results]
                    
                    # Calculate IGD if reference front exists
                    igd_values = []
                    if prog_name in self.reference_fronts:
                        ref_front = self.reference_fronts[prog_name]
                        for r in successful_results:
                            if r.pareto_front is not None:
                                igd = calculate_igd(r.pareto_front, ref_front)
                                igd_values.append(igd)
                    
                    # Calculate enhanced metrics
                    conv_metrics = self.calculate_convergence_metrics(successful_results)
                    div_metrics = self.calculate_diversity_metrics(successful_results)
                    rob_metrics = self.calculate_robustness_metrics(results)
                    pareto_fronts = [r.pareto_front for r in successful_results if r.pareto_front is not None]
                    trade_metrics = self.calculate_tradeoff_metrics(pareto_fronts)
                    
                    row_data = {
                        'Algorithm': algorithm,
                        'Program': prog_name,
                        'Success_Rate': len(successful_results) / len(results),
                        # Original metrics
                        'HV_Mean': np.mean(hvs),
                        'HV_Std': np.std(hvs),
                        'HV_Max': np.max(hvs),
                        'IGD_Mean': np.mean(igd_values) if igd_values else np.nan,
                        'IGD_Std': np.std(igd_values) if igd_values else np.nan,
                        'Solutions_Mean': np.mean(n_solutions),
                        'Solutions_Std': np.std(n_solutions),
                        'Time_Mean': np.mean(exec_times),
                        'Time_Std': np.std(exec_times),
                    }
                    
                    # Add all enhanced metrics
                    row_data.update(conv_metrics)
                    row_data.update(div_metrics)
                    row_data.update(rob_metrics)
                    row_data.update(trade_metrics)
                    
                    analysis_data.append(row_data)
                else:
                    analysis_data.append({
                        'Algorithm': algorithm,
                        'Program': prog_name,
                        'Success_Rate': 0.0,
                        'HV_Mean': 0.0,
                        'HV_Std': 0.0,
                        'HV_Max': 0.0,
                        'IGD_Mean': np.nan,
                        'IGD_Std': np.nan,
                        'Solutions_Mean': 0.0,
                        'Solutions_Std': 0.0,
                        'Time_Mean': 0.0,
                        'Time_Std': 0.0
                    })
        
        df = pd.DataFrame(analysis_data)
        
        # Add rankings
        for metric in ['HV_Mean', 'Solutions_Mean']:
            df[f'{metric}_Rank'] = df.groupby('Program')[metric].rank(
                ascending=False, method='average'
            )
        
        # For IGD, lower is better
        df['IGD_Mean_Rank'] = df.groupby('Program')['IGD_Mean'].rank(
            ascending=True, method='average'
        )
        
        return df
    
    def save_results(self, output_dir: str = "parallel_mo_results"):
        """Save all results to disk"""
        output_path = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_path / f"parallel_run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        results_file = run_dir / "results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump({
                'results': self.results,
                'reference_fronts': self.reference_fronts,
                'config': {
                    'algorithms': self.algorithms,
                    'test_suite': self.test_suite,
                    'runs_per_config': self.runs_per_config,
                    'generations': self.generations,
                    'pop_size': self.pop_size,
                    'objective_type': self.objective_type
                }
            }, f)
        
        # Save analysis
        analysis_df = self.analyze_and_compare()
        analysis_df.to_csv(run_dir / "analysis.csv", index=False)
        
        # Save summary statistics
        summary = analysis_df.groupby('Algorithm').agg({
            'Success_Rate': 'mean',
            'HV_Mean': 'mean',
            'IGD_Mean': 'mean',
            'Solutions_Mean': 'mean',
            'Time_Mean': 'mean',
            'HV_Mean_Rank': 'mean',
            'IGD_Mean_Rank': 'mean'
        }).round(4)
        summary.to_csv(run_dir / "summary.csv")
        
        if self.verbose:
            print(f"\nResults saved to: {run_dir}")
            print("\nSummary Statistics:")
            print(summary)
        
        return run_dir


def run_single_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Run a single experiment (for parallel execution)
    
    Args:
        config: Experiment configuration
        
    Returns:
        ExperimentResult object
    """
    np.random.seed(config.seed)
    start_time = time.time()
    
    try:
        # Load and parse program
        with open(config.program_path, 'r') as f:
            tree = ast.parse(f.read())
        
        visitor = TreeVisitor()
        visitor.visit(tree)
        
        # Create problem
        problem = MOFitnessFactory.create_dual_objective(
            visitor, 
            config.dimensions,
            objective_type=config.objective_type
        )
        
        # Create algorithm
        algorithm = create_mo_algorithm(
            config.algorithm,
            config.pop_size,
            config.n_objectives
        )
        
        # Track convergence and enhanced metrics
        convergence_history = []
        coverage_history = []
        diversity_history = []
        unique_solutions = set()
        unique_paths = set()
        time_to_targets = {}
        target_coverages = [0.5, 0.75, 0.9, 0.95, 1.0]
        last_improvement_gen = 0
        stagnation_count = 0
        
        def callback_func(alg):
            F = alg.pop.get("F")
            X = alg.pop.get("X")
            
            if len(F) > 0:
                # Track best objective
                if config.objective_type == 'conflicting':
                    best_obj = -np.min(F[:, 0])  # Best coverage
                    current_coverage = best_obj
                else:
                    best_obj = np.min(F[:, 0])  # Best fitness
                    # Estimate coverage from fitness (inverse relationship)
                    current_coverage = max(0, 1.0 - best_obj / 100.0)
                
                convergence_history.append(float(best_obj))
                coverage_history.append(float(current_coverage))
                
                # Track time to coverage targets
                for target in target_coverages:
                    if target not in time_to_targets and current_coverage >= target:
                        time_to_targets[target] = alg.n_gen
                
                # Calculate diversity
                if len(X) > 1:
                    # Genotypic diversity: average pairwise distance
                    distances = []
                    for i in range(min(len(X), 20)):  # Sample for efficiency
                        for j in range(i+1, min(len(X), 20)):
                            distances.append(np.linalg.norm(X[i] - X[j]))
                    diversity = np.mean(distances) if distances else 0
                    diversity_history.append(float(diversity))
                
                # Track unique solutions
                for sol in X[:10]:  # Sample for efficiency
                    sol_tuple = tuple(np.round(sol, 2))
                    unique_solutions.add(sol_tuple)
                
                # Detect stagnation
                if len(convergence_history) > 1:
                    if abs(convergence_history[-1] - convergence_history[-2]) < 0.001:
                        stagnation_count += 1
                    else:
                        stagnation_count = 0
                        last_improvement_gen = alg.n_gen
        
        # Run optimization
        result = minimize(
            problem,
            algorithm,
            ('n_gen', config.generations),
            callback=callback_func,
            verbose=False,
            seed=config.seed
        )
        
        # Extract results
        pareto_front = result.F
        pareto_set = result.X
        
        # Calculate metrics
        metrics = calculate_metrics(
            pareto_front,
            objective_type=config.objective_type
        )
        
        elapsed = time.time() - start_time
        
        # Calculate final metrics
        final_coverage = coverage_history[-1] if coverage_history else 0.0
        coverage_efficiency = final_coverage / (config.generations * config.pop_size) if config.generations > 0 else 0
        unique_solutions_count = len(unique_solutions)
        solution_redundancy = 1.0 - (unique_solutions_count / (config.generations * config.pop_size))
        
        # Calculate improvement rate
        if len(coverage_history) > 1:
            improvement_rate = (coverage_history[-1] - coverage_history[0]) / len(coverage_history)
        else:
            improvement_rate = 0.0
        
        return ExperimentResult(
            config=config,
            pareto_front=pareto_front,
            pareto_set=pareto_set,
            execution_time=elapsed,
            n_evaluations=config.generations * config.pop_size,
            metrics=metrics,
            convergence_history=convergence_history,
            # Enhanced metrics
            coverage_history=coverage_history,
            final_coverage=final_coverage,
            time_to_targets=time_to_targets,
            diversity_history=diversity_history,
            unique_solutions_count=unique_solutions_count,
            unique_paths=unique_paths,
            solution_redundancy=solution_redundancy,
            stagnation_count=stagnation_count,
            convergence_generation=last_improvement_gen,
            coverage_efficiency=coverage_efficiency,
            improvement_rate=improvement_rate
        )
        
    except Exception as e:
        return ExperimentResult(
            config=config,
            error=f"{str(e)}\n{traceback.format_exc()}",
            execution_time=time.time() - start_time
        )


def create_mo_algorithm(name: str, pop_size: int, n_obj: int = 2):
    """
    Create multi-objective algorithm instance
    
    Args:
        name: Algorithm name
        pop_size: Population size
        n_obj: Number of objectives
        
    Returns:
        Algorithm instance
    """
    if name == 'NSGA2':
        return NSGA2(pop_size=pop_size)
    
    elif name == 'NSGA3':
        ref_dirs = get_reference_directions("energy", n_obj, pop_size)
        return NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    
    elif name == 'MOEAD':
        ref_dirs = get_reference_directions("energy", n_obj, pop_size)
        return MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=min(20, pop_size//2),
            prob_neighbor_mating=0.9
        )
    
    elif name == 'CTAEA':
        ref_dirs = get_reference_directions("energy", n_obj, pop_size)
        return CTAEA(ref_dirs=ref_dirs)
    
    else:
        raise ValueError(f"Unknown algorithm: {name}")


def calculate_metrics(pareto_front: np.ndarray, 
                     objective_type: str = 'traditional') -> Dict[str, float]:
    """
    Calculate performance metrics for a Pareto front
    
    Args:
        pareto_front: Pareto front solutions
        objective_type: Type of objectives
        
    Returns:
        Dictionary of metrics
    """
    if pareto_front is None or len(pareto_front) == 0:
        return {
            'hypervolume': 0.0,
            'n_solutions': 0,
            'spread': 0.0,
            'spacing': 0.0
        }
    
    # Calculate hypervolume
    ref_point = get_fixed_reference_point(objective_type)
    try:
        hv = HV(ref_point=ref_point)
        hypervolume = float(hv(pareto_front))
    except:
        hypervolume = 0.0
    
    # Calculate spread
    spread = np.std(pareto_front, axis=0).mean()
    
    # Calculate spacing
    if len(pareto_front) > 1:
        distances = []
        for i in range(len(pareto_front)):
            min_dist = float('inf')
            for j in range(len(pareto_front)):
                if i != j:
                    dist = np.linalg.norm(pareto_front[i] - pareto_front[j])
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)
        spacing = np.std(distances) if distances else 0.0
    else:
        spacing = 0.0
    
    return {
        'hypervolume': hypervolume,
        'n_solutions': len(pareto_front),
        'spread': float(spread),
        'spacing': float(spacing)
    }


def calculate_igd(pareto_front: np.ndarray, 
                  reference_front: np.ndarray) -> float:
    """
    Calculate Inverted Generational Distance
    
    Args:
        pareto_front: Approximation front
        reference_front: Reference Pareto front
        
    Returns:
        IGD value (lower is better)
    """
    if len(pareto_front) == 0 or len(reference_front) == 0:
        return float('inf')
    
    try:
        igd = IGD(reference_front)
        return float(igd(pareto_front))
    except:
        # Manual calculation if pymoo fails
        distances = []
        for ref_point in reference_front:
            min_dist = np.min(np.linalg.norm(pareto_front - ref_point, axis=1))
            distances.append(min_dist)
        return np.mean(distances)


def main():
    """Main entry point for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run parallel multi-objective optimization experiments'
    )
    parser.add_argument('--algorithms', nargs='+', 
                       default=['NSGA2', 'NSGA3', 'MOEAD', 'CTAEA'],
                       help='MO algorithms to test')
    parser.add_argument('--test-suite', default='basic',
                       help='Test program suite')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--runs', type=int, default=10,
                       help='Runs per configuration')
    parser.add_argument('--generations', type=int, default=50,
                       help='Generations per run')
    parser.add_argument('--pop-size', type=int, default=50,
                       help='Population size')
    parser.add_argument('--objective-type', default='conflicting',
                       choices=['traditional', 'conflicting'],
                       help='Objective type')
    
    args = parser.parse_args()
    
    # Create and run experiment
    runner = ParallelMOExperimentRunner(
        algorithms=args.algorithms,
        test_suite=args.test_suite,
        n_workers=args.workers,
        runs_per_config=args.runs,
        generations=args.generations,
        pop_size=args.pop_size,
        objective_type=args.objective_type
    )
    
    # Run experiments
    results = runner.run_experiments()
    
    # Analyze and save
    analysis = runner.analyze_and_compare()
    output_dir = runner.save_results()
    
    print(f"\nExperiment complete. Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())