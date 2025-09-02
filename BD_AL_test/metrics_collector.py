"""
Metrics Collector Module
Collects and manages performance metrics during optimization
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class RunMetrics:
    """Metrics for a single optimization run"""
    algorithm_name: str
    test_program: str
    run_id: int
    seed: int
    
    # Performance metrics
    best_fitness: float = float('inf')
    final_coverage: float = 0.0
    convergence_history: List[float] = field(default_factory=list)
    coverage_history: List[float] = field(default_factory=list)
    
    # Time metrics
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    execution_time: float = 0.0
    
    # Evaluation metrics
    total_evaluations: int = 0
    generations: int = 0
    
    # Solution details
    best_solution: Optional[np.ndarray] = None
    final_population: Optional[np.ndarray] = None
    
    # Tree coverage details
    walked_tree: List[str] = field(default_factory=list)
    whole_tree: List[str] = field(default_factory=list)
    branch_coverage: Dict[str, bool] = field(default_factory=dict)
    
    # Additional metrics
    memory_usage: float = 0.0
    convergence_generation: Optional[int] = None
    stagnation_counter: int = 0
    diversity_metrics: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'algorithm_name': self.algorithm_name,
            'test_program': self.test_program,
            'run_id': self.run_id,
            'seed': self.seed,
            'best_fitness': self.best_fitness,
            'final_coverage': self.final_coverage,
            'convergence_history': self.convergence_history,
            'coverage_history': self.coverage_history,
            'execution_time': self.execution_time,
            'total_evaluations': self.total_evaluations,
            'generations': self.generations,
            'best_solution': self.best_solution.tolist() if self.best_solution is not None else None,
            'walked_tree': self.walked_tree,
            'whole_tree': self.whole_tree,
            'branch_coverage': self.branch_coverage,
            'memory_usage': self.memory_usage,
            'convergence_generation': self.convergence_generation,
            'stagnation_counter': self.stagnation_counter,
            'diversity_metrics': self.diversity_metrics
        }


class MetricsCollector:
    """Collects and manages optimization metrics"""
    
    def __init__(self):
        """Initialize metrics collector"""
        self.runs: List[RunMetrics] = []
        self.current_run: Optional[RunMetrics] = None
        self.aggregated_metrics: Dict[str, Any] = {}
        
    def start_run(self, algorithm_name: str, test_program: str, 
                  run_id: int, seed: int) -> RunMetrics:
        """
        Start a new optimization run
        
        Args:
            algorithm_name: Name of the algorithm
            test_program: Name of the test program
            run_id: Run identifier
            seed: Random seed used
            
        Returns:
            RunMetrics instance for the new run
        """
        self.current_run = RunMetrics(
            algorithm_name=algorithm_name,
            test_program=test_program,
            run_id=run_id,
            seed=seed
        )
        self.runs.append(self.current_run)
        return self.current_run
    
    def end_run(self) -> None:
        """End the current run and calculate final metrics"""
        if self.current_run:
            self.current_run.end_time = time.time()
            self.current_run.execution_time = (
                self.current_run.end_time - self.current_run.start_time
            )
            self.current_run = None
    
    def update_generation(self, generation: int, best_fitness: float, 
                         population: Optional[np.ndarray] = None,
                         coverage: Optional[float] = None) -> None:
        """
        Update metrics for a generation
        
        Args:
            generation: Current generation number
            best_fitness: Best fitness in the generation
            population: Current population (optional)
            coverage: Current coverage (optional)
        """
        if not self.current_run:
            return
        
        self.current_run.generations = generation
        self.current_run.convergence_history.append(best_fitness)
        
        # Update best fitness if improved
        if best_fitness < self.current_run.best_fitness:
            self.current_run.best_fitness = best_fitness
            self.current_run.convergence_generation = generation
            self.current_run.stagnation_counter = 0
        else:
            self.current_run.stagnation_counter += 1
        
        # Update coverage if provided
        if coverage is not None:
            self.current_run.coverage_history.append(coverage)
            self.current_run.final_coverage = coverage
        
        # Calculate diversity if population provided
        if population is not None:
            diversity = self._calculate_diversity(population)
            self.current_run.diversity_metrics.append(diversity)
    
    def update_coverage(self, coverage: float, walked_tree: List[str], 
                       whole_tree: List[str]) -> None:
        """
        Update coverage metrics
        
        Args:
            coverage: Coverage percentage
            walked_tree: List of covered tree nodes
            whole_tree: List of all tree nodes
        """
        if not self.current_run:
            return
        
        self.current_run.final_coverage = coverage
        self.current_run.walked_tree = walked_tree
        self.current_run.whole_tree = whole_tree
        
        # Calculate branch coverage
        for node in whole_tree:
            self.current_run.branch_coverage[node] = node in walked_tree
    
    def update_solution(self, solution: np.ndarray, fitness: float) -> None:
        """
        Update best solution
        
        Args:
            solution: Solution vector
            fitness: Fitness value
        """
        if not self.current_run:
            return
        
        if fitness <= self.current_run.best_fitness:
            self.current_run.best_solution = solution.copy()
            self.current_run.best_fitness = fitness
    
    def increment_evaluations(self, count: int = 1) -> None:
        """Increment evaluation counter"""
        if self.current_run:
            self.current_run.total_evaluations += count
    
    def _calculate_diversity(self, population: np.ndarray) -> float:
        """
        Calculate population diversity
        
        Args:
            population: Population matrix
            
        Returns:
            Diversity metric
        """
        if len(population) < 2:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = np.linalg.norm(population[i] - population[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def get_run_summary(self, run: Optional[RunMetrics] = None) -> Dict[str, Any]:
        """
        Get summary of a run
        
        Args:
            run: RunMetrics instance (uses current if None)
            
        Returns:
            Summary dictionary
        """
        if run is None:
            run = self.current_run
        
        if not run:
            return {}
        
        return {
            'algorithm': run.algorithm_name,
            'test_program': run.test_program,
            'best_fitness': run.best_fitness,
            'coverage': run.final_coverage,
            'execution_time': run.execution_time,
            'generations': run.generations,
            'evaluations': run.total_evaluations,
            'convergence_generation': run.convergence_generation,
            'stagnation': run.stagnation_counter
        }
    
    def aggregate_metrics(self, algorithm_name: Optional[str] = None,
                         test_program: Optional[str] = None) -> Dict[str, Any]:
        """
        Aggregate metrics across runs
        
        Args:
            algorithm_name: Filter by algorithm (optional)
            test_program: Filter by test program (optional)
            
        Returns:
            Aggregated metrics dictionary
        """
        # Filter runs
        filtered_runs = self.runs
        if algorithm_name:
            filtered_runs = [r for r in filtered_runs if r.algorithm_name == algorithm_name]
        if test_program:
            filtered_runs = [r for r in filtered_runs if r.test_program == test_program]
        
        if not filtered_runs:
            return {}
        
        # Extract metrics
        best_fitnesses = [r.best_fitness for r in filtered_runs]
        coverages = [r.final_coverage for r in filtered_runs]
        times = [r.execution_time for r in filtered_runs]
        evaluations = [r.total_evaluations for r in filtered_runs]
        generations = [r.generations for r in filtered_runs]
        
        return {
            'count': len(filtered_runs),
            'best_fitness': {
                'mean': np.mean(best_fitnesses),
                'std': np.std(best_fitnesses),
                'min': np.min(best_fitnesses),
                'max': np.max(best_fitnesses),
                'median': np.median(best_fitnesses)
            },
            'coverage': {
                'mean': np.mean(coverages),
                'std': np.std(coverages),
                'min': np.min(coverages),
                'max': np.max(coverages),
                'median': np.median(coverages)
            },
            'execution_time': {
                'mean': np.mean(times),
                'std': np.std(times),
                'total': np.sum(times)
            },
            'evaluations': {
                'mean': np.mean(evaluations),
                'std': np.std(evaluations)
            },
            'generations': {
                'mean': np.mean(generations),
                'std': np.std(generations)
            }
        }
    
    def get_convergence_data(self, algorithm_name: Optional[str] = None,
                           test_program: Optional[str] = None) -> Dict[str, List[List[float]]]:
        """
        Get convergence data for plotting
        
        Args:
            algorithm_name: Filter by algorithm (optional)
            test_program: Filter by test program (optional)
            
        Returns:
            Dictionary with convergence histories
        """
        # Filter runs
        filtered_runs = self.runs
        if algorithm_name:
            filtered_runs = [r for r in filtered_runs if r.algorithm_name == algorithm_name]
        if test_program:
            filtered_runs = [r for r in filtered_runs if r.test_program == test_program]
        
        convergence_data = {}
        for run in filtered_runs:
            key = f"{run.algorithm_name}_{run.test_program}_run{run.run_id}"
            convergence_data[key] = run.convergence_history
        
        return convergence_data
    
    def get_comparison_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Get comparison matrix of algorithms vs test programs
        
        Returns:
            Nested dictionary with mean best fitness values
        """
        matrix = {}
        
        # Get unique algorithms and programs
        algorithms = list(set(r.algorithm_name for r in self.runs))
        programs = list(set(r.test_program for r in self.runs))
        
        for algo in algorithms:
            matrix[algo] = {}
            for prog in programs:
                runs = [r for r in self.runs 
                       if r.algorithm_name == algo and r.test_program == prog]
                if runs:
                    matrix[algo][prog] = np.mean([r.best_fitness for r in runs])
                else:
                    matrix[algo][prog] = None
        
        return matrix
    
    def export_to_csv(self, filename: str) -> None:
        """
        Export metrics to CSV file
        
        Args:
            filename: Output CSV filename
        """
        import pandas as pd
        
        data = []
        for run in self.runs:
            data.append({
                'Algorithm': run.algorithm_name,
                'Test_Program': run.test_program,
                'Run_ID': run.run_id,
                'Seed': run.seed,
                'Best_Fitness': run.best_fitness,
                'Coverage': run.final_coverage,
                'Execution_Time': run.execution_time,
                'Generations': run.generations,
                'Evaluations': run.total_evaluations,
                'Convergence_Gen': run.convergence_generation,
                'Stagnation': run.stagnation_counter
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
    
    def export_to_json(self, filename: str) -> None:
        """
        Export metrics to JSON file
        
        Args:
            filename: Output JSON filename
        """
        data = {
            'runs': [run.to_dict() for run in self.runs],
            'aggregated': self.aggregate_metrics(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear(self) -> None:
        """Clear all collected metrics"""
        self.runs.clear()
        self.current_run = None
        self.aggregated_metrics.clear()
    
    def get_best_run(self, algorithm_name: Optional[str] = None,
                     test_program: Optional[str] = None) -> Optional[RunMetrics]:
        """
        Get the best run based on fitness
        
        Args:
            algorithm_name: Filter by algorithm (optional)
            test_program: Filter by test program (optional)
            
        Returns:
            Best RunMetrics instance or None
        """
        # Filter runs
        filtered_runs = self.runs
        if algorithm_name:
            filtered_runs = [r for r in filtered_runs if r.algorithm_name == algorithm_name]
        if test_program:
            filtered_runs = [r for r in filtered_runs if r.test_program == test_program]
        
        if not filtered_runs:
            return None
        
        return min(filtered_runs, key=lambda r: r.best_fitness)
    
    def get_statistics(self) -> str:
        """
        Get formatted statistics summary
        
        Returns:
            Formatted string with statistics
        """
        if not self.runs:
            return "No runs collected yet."
        
        stats = []
        stats.append("=" * 60)
        stats.append("METRICS SUMMARY")
        stats.append("=" * 60)
        
        # Overall statistics
        stats.append(f"\nTotal runs: {len(self.runs)}")
        
        # Get unique algorithms and programs
        algorithms = list(set(r.algorithm_name for r in self.runs))
        programs = list(set(r.test_program for r in self.runs))
        
        stats.append(f"Algorithms tested: {', '.join(algorithms)}")
        stats.append(f"Programs tested: {', '.join(programs)}")
        
        # Best performances
        stats.append("\nBest Performances:")
        for prog in programs:
            prog_runs = [r for r in self.runs if r.test_program == prog]
            if prog_runs:
                best_run = min(prog_runs, key=lambda r: r.best_fitness)
                stats.append(f"  {prog}: {best_run.algorithm_name} "
                           f"(fitness={best_run.best_fitness:.6f}, "
                           f"coverage={best_run.final_coverage:.2%})")
        
        # Algorithm summaries
        stats.append("\nAlgorithm Performance (mean ± std):")
        for algo in algorithms:
            agg = self.aggregate_metrics(algorithm_name=algo)
            if agg:
                stats.append(f"  {algo}:")
                stats.append(f"    Fitness: {agg['best_fitness']['mean']:.6f} ± "
                           f"{agg['best_fitness']['std']:.6f}")
                stats.append(f"    Coverage: {agg['coverage']['mean']:.2%} ± "
                           f"{agg['coverage']['std']:.2%}")
                stats.append(f"    Time: {agg['execution_time']['mean']:.2f}s ± "
                           f"{agg['execution_time']['std']:.2f}s")
        
        stats.append("\n" + "=" * 60)
        return "\n".join(stats)