#!/usr/bin/env python3
"""
Multi-Objective Algorithm Testing and Analysis Script
Comprehensive analysis tool for multi-objective optimization algorithms
Focuses on Pareto front quality, trade-offs, and comparative performance
"""

import argparse
import sys
import time
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import scipy.stats as stats
from scipy.spatial.distance import cdist

# Framework imports
from algorithm_factory import AlgorithmFactory
from config_loader import ConfigLoader
from multi_objective_fitness import (
    MultiObjectiveFitness,
    MultiObjectiveProblem,
    MOFitnessFactory,
    calculate_adaptive_reference_point,
    get_fixed_reference_point
)

# Existing project imports
from tree_converter import TreeVisitor
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD


class MOMetricsCalculator:
    """Calculate multi-objective performance metrics"""
    
    def __init__(self, ref_point: Optional[np.ndarray] = None, adaptive: bool = True):
        """
        Initialize metrics calculator
        
        Args:
            ref_point: Reference point for hypervolume calculation
            adaptive: Whether to use adaptive reference point
        """
        self.ref_point = ref_point
        self.adaptive = adaptive
        self.default_ref_point = np.array([1000, 0])  # Fallback reference point
        
    def calculate_hypervolume(self, pareto_front: np.ndarray, 
                            ref_point: Optional[np.ndarray] = None,
                            objective_type: str = 'traditional') -> float:
        """Calculate hypervolume indicator"""
        if len(pareto_front) == 0:
            return 0.0
            
        # Use provided ref_point or get fixed reference point
        if ref_point is not None:
            use_ref_point = ref_point
        else:
            # Use fixed reference point based on objective type
            use_ref_point = get_fixed_reference_point(objective_type)
            
        try:
            hv = HV(ref_point=use_ref_point)
            return hv(pareto_front)
        except:
            return 0.0
    
    def calculate_igd(self, pareto_front: np.ndarray, reference_front: np.ndarray) -> float:
        """Calculate Inverted Generational Distance"""
        if len(pareto_front) == 0 or len(reference_front) == 0:
            return float('inf')
        try:
            igd = IGD(reference_front)
            return igd(pareto_front)
        except:
            return float('inf')
    
    def calculate_spread(self, pareto_front: np.ndarray) -> float:
        """
        Calculate spread (diversity) metric
        
        Args:
            pareto_front: Pareto front solutions
            
        Returns:
            Spread value (0 = perfect spread)
        """
        if len(pareto_front) < 2:
            return 0.0
        
        # Sort by first objective
        sorted_front = pareto_front[pareto_front[:, 0].argsort()]
        
        # Calculate distances between consecutive solutions
        distances = []
        for i in range(len(sorted_front) - 1):
            dist = np.linalg.norm(sorted_front[i] - sorted_front[i+1])
            distances.append(dist)
        
        # Calculate spread
        if len(distances) > 0:
            mean_dist = np.mean(distances)
            spread = np.std(distances) / mean_dist if mean_dist > 0 else 0
            return spread
        return 0.0
    
    def calculate_spacing(self, pareto_front: np.ndarray) -> float:
        """
        Calculate spacing (uniformity) metric
        
        Args:
            pareto_front: Pareto front solutions
            
        Returns:
            Spacing value (0 = perfectly uniform)
        """
        if len(pareto_front) < 2:
            return 0.0
        
        # Calculate minimum distance for each point
        min_distances = []
        for i in range(len(pareto_front)):
            distances = []
            for j in range(len(pareto_front)):
                if i != j:
                    dist = np.linalg.norm(pareto_front[i] - pareto_front[j])
                    distances.append(dist)
            if distances:
                min_distances.append(min(distances))
        
        if min_distances:
            mean_dist = np.mean(min_distances)
            spacing = np.sqrt(np.sum((min_distances - mean_dist) ** 2) / len(min_distances))
            return spacing
        return 0.0
    
    def calculate_maximum_spread(self, pareto_front: np.ndarray) -> float:
        """Calculate maximum spread across objectives"""
        if len(pareto_front) < 2:
            return 0.0
        
        spreads = []
        for obj_idx in range(pareto_front.shape[1]):
            obj_values = pareto_front[:, obj_idx]
            spread = np.max(obj_values) - np.min(obj_values)
            spreads.append(spread)
        
        return np.mean(spreads)
    
    def find_knee_point(self, pareto_front: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Find knee point (best trade-off solution)
        
        Args:
            pareto_front: Pareto front solutions
            
        Returns:
            Tuple of (index, solution)
        """
        if len(pareto_front) == 1:
            return 0, pareto_front[0]
        
        # Normalize objectives
        f_min = pareto_front.min(axis=0)
        f_max = pareto_front.max(axis=0)
        f_range = f_max - f_min
        f_range[f_range == 0] = 1.0  # Avoid division by zero
        f_norm = (pareto_front - f_min) / f_range
        
        # Find point closest to ideal point (0, 0)
        distances = np.sqrt(np.sum(f_norm ** 2, axis=1))
        knee_idx = np.argmin(distances)
        
        return knee_idx, pareto_front[knee_idx]


class MultiObjectiveAnalyzer:
    """Comprehensive multi-objective algorithm analyzer"""
    
    def __init__(self, algorithm_name: str, n_objectives: int = 2,
                 num_runs: int = 5, max_generations: int = 100,
                 pop_size: int = 100, verbose: bool = True,
                 objective_type: str = 'traditional'):
        """
        Initialize the analyzer
        
        Args:
            algorithm_name: Name of MO algorithm
            n_objectives: Number of objectives
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
        
        # Initialize components
        self.config_loader = ConfigLoader()
        self.factory = AlgorithmFactory(self.config_loader)
        self.metrics_calc = MOMetricsCalculator(adaptive=True)
        
        # Load configurations
        self.config_loader.load_all()
        self.test_programs = self.config_loader.test_programs_config['test_programs']
        
        # Results storage
        self.all_results = {}
        self.analysis_results = {}
        self.reference_fronts = {}
        
    def convert_tree(self, path: str) -> TreeVisitor:
        """Convert Python file to tree structure"""
        with open(path, 'r') as f:
            lines = f.readlines()
            tree = ast.parse(''.join(lines))
        
        visitor = TreeVisitor()
        visitor.visit(tree)
        return visitor
    
    def create_algorithm(self, n_obj: int) -> Any:
        """Create multi-objective algorithm instance"""
        algo_factory = self.factory.create_algorithm(
            self.algorithm_name,
            custom_params={'pop_size': self.pop_size}
        )
        
        if callable(algo_factory) and not hasattr(algo_factory, 'pop'):
            return algo_factory(n_obj)
        
        return algo_factory
    
    def run_single_test(self, test_program_name: str, test_config: Dict,
                        run_id: int, seed: int) -> Dict[str, Any]:
        """Run MO algorithm on single test program"""
        np.random.seed(seed)
        start_time = time.time()
        
        try:
            # Convert tree and create problem
            visitor = self.convert_tree(test_config['path'])
            
            if self.n_objectives == 2:
                problem = MOFitnessFactory.create_dual_objective(
                    visitor, test_config['dimensions'],
                    objective_type=self.objective_type
                )
            else:
                problem = MOFitnessFactory.create_three_objective(
                    visitor, test_config['dimensions']
                )
            
            # Create and run algorithm
            algorithm = self.create_algorithm(self.n_objectives)
            
            # Track convergence
            hypervolume_history = []
            
            def callback(algorithm):
                F = algorithm.pop.get("F")
                hv = self.metrics_calc.calculate_hypervolume(F)
                hypervolume_history.append(hv)
            
            result = minimize(
                problem,
                algorithm,
                ('n_gen', self.max_generations),
                callback=callback,
                verbose=False
            )
            
            # Extract and analyze results
            pareto_front = result.F
            pareto_set = result.X
            
            # Calculate all metrics
            metrics = self._calculate_all_metrics(pareto_front)
            
            # Find special solutions based on objective type
            if self.objective_type == 'conflicting':
                # Obj 1: negative coverage, Obj 2: complexity
                coverage_values = -pareto_front[:, 0]
                complexity_values = pareto_front[:, 1]
                # Use complexity as "fitness" for compatibility
                fitness_values = complexity_values
            else:
                # Traditional: Obj 1: fitness, Obj 2: negative coverage
                fitness_values = pareto_front[:, 0]
                coverage_values = -pareto_front[:, 1]
            
            best_coverage_idx = np.argmax(coverage_values)
            best_fitness_idx = np.argmin(fitness_values)
            knee_idx, knee_point = self.metrics_calc.find_knee_point(pareto_front)
            
            end_time = time.time()
            
            return {
                'run_id': run_id,
                'seed': seed,
                'execution_time': end_time - start_time,
                'pareto_front': pareto_front,
                'pareto_set': pareto_set,
                'metrics': metrics,
                'hypervolume_history': hypervolume_history,
                'best_coverage': {
                    'coverage': float(coverage_values[best_coverage_idx]),
                    'fitness': float(fitness_values[best_fitness_idx]),
                    'idx': best_coverage_idx
                },
                'best_fitness': {
                    'coverage': float(coverage_values[best_fitness_idx]),
                    'fitness': float(fitness_values[best_fitness_idx]),
                    'idx': best_fitness_idx
                },
                'knee_point': {
                    'coverage': float(-knee_point[1]),
                    'fitness': float(knee_point[0]),
                    'idx': knee_idx
                }
            }
            
        except Exception as e:
            if self.verbose:
                print(f"    Error in run {run_id}: {e}")
            return {
                'run_id': run_id,
                'seed': seed,
                'error': str(e),
                'execution_time': 0
            }
    
    def _calculate_all_metrics(self, pareto_front: np.ndarray) -> Dict[str, float]:
        """Calculate all quality metrics for a Pareto front"""
        metrics = {
            'n_solutions': len(pareto_front),
            'hypervolume': self.metrics_calc.calculate_hypervolume(
                pareto_front, objective_type=self.objective_type
            ),
            'spread': self.metrics_calc.calculate_spread(pareto_front),
            'spacing': self.metrics_calc.calculate_spacing(pareto_front),
            'max_spread': self.metrics_calc.calculate_maximum_spread(pareto_front)
        }
        
        # Objective-specific analysis
        if len(pareto_front) > 0:
            if self.objective_type == 'conflicting':
                # Coverage from first objective (negated)
                coverage_values = -pareto_front[:, 0]
                # Complexity from second objective
                complexity_values = pareto_front[:, 1]
                
                metrics['max_coverage'] = float(np.max(coverage_values))
                metrics['mean_coverage'] = float(np.mean(coverage_values))
                metrics['coverage_std'] = float(np.std(coverage_values))
                metrics['full_coverage_count'] = int(np.sum(coverage_values >= 1.0))
                
                metrics['min_complexity'] = float(np.min(complexity_values))
                metrics['mean_complexity'] = float(np.mean(complexity_values))
                metrics['complexity_std'] = float(np.std(complexity_values))
                
                # Use complexity as "fitness" for compatibility
                metrics['min_fitness'] = metrics['min_complexity']
                metrics['mean_fitness'] = metrics['mean_complexity']
                metrics['fitness_std'] = metrics['complexity_std']
            else:
                # Traditional objectives
                fitness_values = pareto_front[:, 0]
                coverage_values = -pareto_front[:, 1]
                
                metrics['max_coverage'] = float(np.max(coverage_values))
                metrics['mean_coverage'] = float(np.mean(coverage_values))
                metrics['coverage_std'] = float(np.std(coverage_values))
                metrics['full_coverage_count'] = int(np.sum(coverage_values >= 1.0))
                
                metrics['min_fitness'] = float(np.min(fitness_values))
                metrics['mean_fitness'] = float(np.mean(fitness_values))
                metrics['fitness_std'] = float(np.std(fitness_values))
        
        return metrics
    
    def run_comprehensive_test(self) -> Dict[str, List[Dict]]:
        """Run algorithm on all test programs"""
        print("\n" + "=" * 80)
        print(f"COMPREHENSIVE MO ANALYSIS: {self.algorithm_name}")
        print(f"Objectives: {self.n_objectives} | Runs: {self.num_runs} | " +
              f"Generations: {self.max_generations}")
        print("=" * 80)
        
        for idx, (prog_name, prog_config) in enumerate(self.test_programs.items(), 1):
            print(f"\n[{idx}/{len(self.test_programs)}] Testing: {prog_name}")
            
            prog_results = []
            all_fronts = []
            
            for run_id in range(self.num_runs):
                seed = 42 + run_id * 100
                
                if self.verbose:
                    print(f"  Run {run_id + 1}/{self.num_runs}...", end="")
                
                results = self.run_single_test(prog_name, prog_config, run_id + 1, seed)
                prog_results.append(results)
                
                if 'pareto_front' in results:
                    all_fronts.append(results['pareto_front'])
                    if self.verbose:
                        metrics = results['metrics']
                        print(f" HV={metrics['hypervolume']:.3f}, " +
                              f"Solutions={metrics['n_solutions']}")
                elif self.verbose:
                    print(" Failed")
            
            self.all_results[prog_name] = prog_results
            
            # Create reference front (union of all fronts)
            if all_fronts:
                combined_front = np.vstack(all_fronts)
                nds = NonDominatedSorting()
                fronts = nds.do(combined_front)
                if fronts and len(fronts[0]) > 0:
                    self.reference_fronts[prog_name] = combined_front[fronts[0]]
        
        return self.all_results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Perform comprehensive analysis of results"""
        print("\n" + "=" * 80)
        print("ANALYZING RESULTS")
        print("=" * 80)
        
        analysis = {}
        
        for prog_name, runs in self.all_results.items():
            valid_runs = [r for r in runs if 'metrics' in r]
            
            if not valid_runs:
                continue
            
            # Aggregate metrics
            metrics_df = pd.DataFrame([r['metrics'] for r in valid_runs])
            
            prog_analysis = {
                'n_runs': len(valid_runs),
                'metrics_mean': metrics_df.mean().to_dict(),
                'metrics_std': metrics_df.std().to_dict(),
                'metrics_min': metrics_df.min().to_dict(),
                'metrics_max': metrics_df.max().to_dict(),
                'success_rate': len(valid_runs) / len(runs),
                'avg_execution_time': np.mean([r['execution_time'] for r in valid_runs])
            }
            
            # Convergence analysis
            if valid_runs[0].get('hypervolume_history'):
                hv_histories = [r['hypervolume_history'] for r in valid_runs]
                prog_analysis['convergence'] = {
                    'final_hv': np.mean([h[-1] for h in hv_histories]),
                    'convergence_speed': self._analyze_convergence_speed(hv_histories)
                }
            
            # Trade-off analysis
            all_fronts = [r['pareto_front'] for r in valid_runs if 'pareto_front' in r]
            if all_fronts:
                prog_analysis['trade_off'] = self._analyze_trade_offs(all_fronts)
            
            analysis[prog_name] = prog_analysis
        
        self.analysis_results = analysis
        return analysis
    
    def _analyze_convergence_speed(self, hv_histories: List[List[float]]) -> Dict[str, float]:
        """Analyze convergence speed from hypervolume histories"""
        convergence_metrics = {}
        
        # Find generation where 90% of final HV is reached
        for history in hv_histories:
            if len(history) > 0:
                final_hv = history[-1]
                target_hv = 0.9 * final_hv
                
                for gen, hv in enumerate(history):
                    if hv >= target_hv:
                        convergence_metrics.setdefault('gens_to_90pct', []).append(gen)
                        break
        
        if 'gens_to_90pct' in convergence_metrics:
            convergence_metrics['avg_gens_to_90pct'] = np.mean(convergence_metrics['gens_to_90pct'])
        
        return convergence_metrics
    
    def _analyze_trade_offs(self, all_fronts: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze trade-off characteristics"""
        trade_off_analysis = {}
        
        # Combine all fronts
        combined = np.vstack(all_fronts)
        
        # Calculate correlation between objectives
        if len(combined) > 2:
            correlation = np.corrcoef(combined[:, 0], combined[:, 1])[0, 1]
            trade_off_analysis['objective_correlation'] = float(correlation)
            
            # Classify trade-off severity
            if abs(correlation) < 0.3:
                trade_off_analysis['conflict_level'] = 'low'
            elif abs(correlation) < 0.7:
                trade_off_analysis['conflict_level'] = 'medium'
            else:
                trade_off_analysis['conflict_level'] = 'high'
        
        # Analyze front shape
        for front in all_fronts:
            if len(front) > 2:
                # Sort by first objective
                sorted_front = front[front[:, 0].argsort()]
                
                # Calculate curvature (simplified)
                if len(sorted_front) >= 3:
                    mid_idx = len(sorted_front) // 2
                    p1 = sorted_front[0]
                    p2 = sorted_front[mid_idx]
                    p3 = sorted_front[-1]
                    
                    # Calculate area of triangle
                    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                                    (p3[0] - p1[0]) * (p2[1] - p1[1]))
                    
                    trade_off_analysis.setdefault('front_curvatures', []).append(area)
        
        if 'front_curvatures' in trade_off_analysis:
            trade_off_analysis['avg_curvature'] = float(np.mean(trade_off_analysis['front_curvatures']))
            del trade_off_analysis['front_curvatures']  # Clean up detailed data
        
        return trade_off_analysis
    
    def classify_test_difficulty(self) -> Dict[str, str]:
        """Classify test programs by multi-objective difficulty"""
        difficulty_scores = {}
        
        for prog_name, analysis in self.analysis_results.items():
            score = 0
            
            # Factor 1: Number of Pareto solutions (more = harder)
            n_solutions = analysis['metrics_mean'].get('n_solutions', 0)
            if n_solutions > 50:
                score += 3
            elif n_solutions > 20:
                score += 2
            elif n_solutions > 10:
                score += 1
            
            # Factor 2: Trade-off conflict level
            conflict = analysis.get('trade_off', {}).get('conflict_level', 'low')
            if conflict == 'high':
                score += 3
            elif conflict == 'medium':
                score += 2
            elif conflict == 'low':
                score += 1
            
            # Factor 3: Coverage achievement
            max_coverage = analysis['metrics_mean'].get('max_coverage', 0)
            if max_coverage < 0.5:
                score += 3
            elif max_coverage < 0.8:
                score += 2
            elif max_coverage < 1.0:
                score += 1
            
            # Classify based on score
            if score >= 7:
                difficulty = 'Hard'
            elif score >= 4:
                difficulty = 'Medium'
            else:
                difficulty = 'Easy'
            
            difficulty_scores[prog_name] = difficulty
        
        return difficulty_scores
    
    def create_visualization_dashboard(self, save_path: Optional[str] = None):
        """Create comprehensive visualization dashboard"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Pareto Front Overlay (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_pareto_overlay(ax1)
        
        # 2. Hypervolume Evolution (top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_hypervolume_evolution(ax2)
        
        # 3. Coverage Distribution (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_coverage_distribution(ax3)
        
        # 4. Trade-off Heatmap (middle-left)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_trade_off_heatmap(ax4)
        
        # 5. Performance Radar (middle-middle)
        ax5 = fig.add_subplot(gs[1, 1], projection='polar')
        self._plot_performance_radar(ax5)
        
        # 6. Difficulty Classification (middle-right)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_difficulty_classification(ax6)
        
        # 7. Metrics Comparison (bottom-left)
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_metrics_comparison(ax7)
        
        # 8. Solution Distribution (bottom-middle)
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_solution_distribution(ax8)
        
        # 9. Statistical Summary (bottom-right)
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_statistical_summary(ax9)
        
        plt.suptitle(f'{self.algorithm_name} - Multi-Objective Analysis Dashboard', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")
        
        plt.show()
    
    def _plot_pareto_overlay(self, ax):
        """Plot overlaid Pareto fronts for all test programs"""
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.all_results)))
        
        for idx, (prog_name, runs) in enumerate(self.all_results.items()):
            for run in runs:
                if 'pareto_front' in run:
                    front = run['pareto_front']
                    if self.objective_type == 'conflicting':
                        # Coverage vs Complexity
                        x_values = front[:, 1]  # Complexity
                        y_values = -front[:, 0] * 100  # Coverage
                        xlabel = 'Test Complexity (minimize)'
                        ylabel = 'Coverage % (maximize)'
                    else:
                        # Traditional
                        x_values = front[:, 0]  # Fitness
                        y_values = -front[:, 1] * 100  # Coverage
                        xlabel = 'Fitness (minimize)'
                        ylabel = 'Coverage % (maximize)'
                    
                    ax.scatter(x_values, y_values, alpha=0.3, s=10, 
                             color=colors[idx], label=prog_name if run == runs[0] else "")
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title('Pareto Fronts - All Programs')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    
    def _plot_hypervolume_evolution(self, ax):
        """Plot hypervolume evolution over generations"""
        for prog_name, runs in self.all_results.items():
            valid_runs = [r for r in runs if 'hypervolume_history' in r]
            
            if valid_runs:
                histories = [r['hypervolume_history'] for r in valid_runs]
                mean_history = np.mean(histories, axis=0)
                ax.plot(mean_history, label=prog_name[:15], alpha=0.7)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Hypervolume')
        ax.set_title('Convergence Analysis')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    
    def _plot_coverage_distribution(self, ax):
        """Plot coverage distribution as violin plot"""
        data = []
        labels = []
        
        for prog_name, runs in self.all_results.items():
            coverages = []
            for run in runs:
                if 'pareto_front' in run:
                    front_coverage = -run['pareto_front'][:, 1]
                    coverages.extend(front_coverage)
            
            if coverages:
                data.append(coverages)
                labels.append(prog_name[:10])
        
        if data:
            parts = ax.violinplot(data, showmeans=True, showmedians=True)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Coverage')
            ax.set_title('Coverage Distribution')
            ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_trade_off_heatmap(self, ax):
        """Plot trade-off characteristics heatmap"""
        programs = list(self.analysis_results.keys())[:10]  # Limit to 10 programs
        metrics = ['HV', 'Spread', 'Coverage', 'Solutions']
        
        data = []
        for prog in programs:
            if prog in self.analysis_results:
                analysis = self.analysis_results[prog]
                row = [
                    analysis['metrics_mean'].get('hypervolume', 0),
                    analysis['metrics_mean'].get('spread', 0),
                    analysis['metrics_mean'].get('max_coverage', 0),
                    analysis['metrics_mean'].get('n_solutions', 0) / 100  # Normalize
                ]
                data.append(row)
        
        if data:
            im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(metrics)
            ax.set_yticks(range(len(programs)))
            ax.set_yticklabels([p[:15] for p in programs])
            ax.set_title('Performance Heatmap')
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    def _plot_performance_radar(self, ax):
        """Plot performance radar chart"""
        categories = ['Hypervolume', 'Coverage', 'Diversity', 'Speed', 'Stability']
        
        # Calculate aggregate metrics
        all_metrics = []
        for analysis in self.analysis_results.values():
            all_metrics.append({
                'hv': analysis['metrics_mean'].get('hypervolume', 0),
                'cov': analysis['metrics_mean'].get('max_coverage', 0),
                'div': 1 - analysis['metrics_mean'].get('spacing', 1),  # Invert for "better"
                'speed': 1 / (analysis.get('avg_execution_time', 1) + 1),  # Invert time
                'stability': 1 / (analysis['metrics_std'].get('hypervolume', 1) + 0.1)
            })
        
        if all_metrics:
            # Average across all programs
            avg_metrics = {
                key: np.mean([m[key] for m in all_metrics])
                for key in all_metrics[0].keys()
            }
            
            # Normalize to 0-1
            values = list(avg_metrics.values())
            max_val = max(values) if max(values) > 0 else 1
            values = [v / max_val for v in values]
            
            # Plot
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            values = values + values[:1]  # Complete the circle
            angles = np.concatenate([angles, [angles[0]]])
            
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title('Overall Performance')
            ax.grid(True)
    
    def _plot_difficulty_classification(self, ax):
        """Plot test program difficulty classification"""
        difficulty = self.classify_test_difficulty()
        
        easy = sum(1 for d in difficulty.values() if d == 'Easy')
        medium = sum(1 for d in difficulty.values() if d == 'Medium')
        hard = sum(1 for d in difficulty.values() if d == 'Hard')
        
        labels = ['Easy', 'Medium', 'Hard']
        sizes = [easy, medium, hard]
        colors = ['green', 'orange', 'red']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.0f%%', startangle=90)
        ax.set_title('Test Difficulty Distribution')
    
    def _plot_metrics_comparison(self, ax):
        """Plot metrics comparison across programs"""
        programs = list(self.analysis_results.keys())[:5]  # Top 5 programs
        
        hypervolumes = []
        spreads = []
        
        for prog in programs:
            if prog in self.analysis_results:
                analysis = self.analysis_results[prog]
                hypervolumes.append(analysis['metrics_mean'].get('hypervolume', 0))
                spreads.append(analysis['metrics_mean'].get('spread', 0))
        
        if hypervolumes:
            x = np.arange(len(programs))
            width = 0.35
            
            ax.bar(x - width/2, hypervolumes, width, label='Hypervolume', color='blue', alpha=0.7)
            ax.bar(x + width/2, spreads, width, label='Spread', color='red', alpha=0.7)
            
            ax.set_xlabel('Test Program')
            ax.set_ylabel('Metric Value')
            ax.set_title('Key Metrics Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels([p[:10] for p in programs], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_solution_distribution(self, ax):
        """Plot distribution of Pareto solutions"""
        n_solutions = []
        labels = []
        
        for prog_name, analysis in self.analysis_results.items():
            n_sol = analysis['metrics_mean'].get('n_solutions', 0)
            if n_sol > 0:
                n_solutions.append(n_sol)
                labels.append(prog_name[:10])
        
        if n_solutions:
            ax.barh(range(len(n_solutions)), n_solutions, color='teal', alpha=0.7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xlabel('Number of Pareto Solutions')
            ax.set_title('Solution Set Sizes')
            ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_statistical_summary(self, ax):
        """Plot statistical summary table"""
        ax.axis('off')
        
        # Prepare summary data
        total_runs = sum(len(runs) for runs in self.all_results.values())
        successful_runs = sum(
            sum(1 for r in runs if 'pareto_front' in r)
            for runs in self.all_results.values()
        )
        
        avg_hv = np.mean([
            analysis['metrics_mean'].get('hypervolume', 0)
            for analysis in self.analysis_results.values()
        ])
        
        avg_coverage = np.mean([
            analysis['metrics_mean'].get('max_coverage', 0)
            for analysis in self.analysis_results.values()
        ])
        
        if self.objective_type == 'conflicting':
            obj_desc = "Coverage vs Complexity"
            fitness_label = "Lowest Complexity"
            avg_fitness = np.mean([
                analysis['metrics_mean'].get('mean_complexity', 0)
                for analysis in self.analysis_results.values()
            ])
            best_fitness = min([
                analysis['metrics_mean'].get('min_complexity', float('inf'))
                for analysis in self.analysis_results.values()
            ])
        else:
            obj_desc = "Fitness vs Coverage"
            fitness_label = "Best Fitness"
            avg_fitness = np.mean([
                analysis['metrics_mean'].get('mean_fitness', 0)
                for analysis in self.analysis_results.values()
            ])
            best_fitness = min([
                analysis['metrics_mean'].get('min_fitness', float('inf'))
                for analysis in self.analysis_results.values()
            ])
            
        summary_text = f"""
        Algorithm: {self.algorithm_name}
        Objectives: {obj_desc}
        Total Runs: {total_runs}
        Successful: {successful_runs} ({100*successful_runs/total_runs:.1f}%)
        
        Average Metrics:
        • Hypervolume: {avg_hv:.3f}
        • Max Coverage: {avg_coverage:.1%}
        • Avg Fitness/Complexity: {avg_fitness:.3f}
        • Programs Tested: {len(self.all_results)}
        
        Best Performance:
        • Highest HV: {max(a['metrics_mean'].get('hypervolume', 0) 
                          for a in self.analysis_results.values()):.3f}
        • Best Coverage: {max(a['metrics_mean'].get('max_coverage', 0)
                             for a in self.analysis_results.values()):.1%}
        • {fitness_label}: {best_fitness:.3f}
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
        ax.set_title('Statistical Summary')
    
    def generate_report(self, output_dir: str = "mo_analysis_results"):
        """Generate comprehensive analysis report"""
        output_path = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = output_path / f"{self.algorithm_name}_analysis_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        results_file = report_dir / "raw_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for prog, runs in self.all_results.items():
                json_runs = []
                for run in runs:
                    json_run = {}
                    for key, value in run.items():
                        if isinstance(value, np.ndarray):
                            json_run[key] = value.tolist()
                        elif isinstance(value, (np.integer, np.int64)):
                            json_run[key] = int(value)
                        elif isinstance(value, (np.floating, np.float64)):
                            json_run[key] = float(value)
                        elif isinstance(value, dict):
                            # Recursively convert dict values
                            json_run[key] = {
                                k: (v.tolist() if isinstance(v, np.ndarray) else
                                    int(v) if isinstance(v, (np.integer, np.int64)) else
                                    float(v) if isinstance(v, (np.floating, np.float64)) else v)
                                for k, v in value.items()
                            }
                        else:
                            json_run[key] = value
                    json_runs.append(json_run)
                json_results[prog] = json_runs
            json.dump(json_results, f, indent=2)
        
        # Save analysis results
        analysis_file = report_dir / "analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        
        # Generate summary CSV
        summary_data = []
        for prog_name, analysis in self.analysis_results.items():
            summary_data.append({
                'Program': prog_name,
                'Runs': analysis['n_runs'],
                'Success Rate': analysis['success_rate'],
                'Avg HV': analysis['metrics_mean'].get('hypervolume', 0),
                'Avg Solutions': analysis['metrics_mean'].get('n_solutions', 0),
                'Max Coverage': analysis['metrics_mean'].get('max_coverage', 0),
                'Avg Time': analysis['avg_execution_time']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(report_dir / "summary.csv", index=False)
        
        # Generate visualizations
        viz_path = report_dir / "dashboard.png"
        self.create_visualization_dashboard(str(viz_path))
        
        # Generate text report
        report_file = report_dir / "report.txt"
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"MULTI-OBJECTIVE ANALYSIS REPORT\n")
            f.write(f"Algorithm: {self.algorithm_name}\n")
            f.write(f"Objective Type: {self.objective_type.title()}\n")
            if self.objective_type == 'conflicting':
                f.write("Objectives: Maximize Coverage vs Minimize Complexity\n")
            else:
                f.write("Objectives: Minimize Fitness vs Maximize Coverage\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            total_programs = len(self.all_results)
            successful_programs = len(self.analysis_results)
            
            f.write(f"Programs Tested: {total_programs}\n")
            f.write(f"Successful: {successful_programs}\n")
            f.write(f"Total Runs: {self.num_runs * total_programs}\n\n")
            
            # Best performers
            if self.analysis_results:
                best_hv_prog = max(self.analysis_results.items(),
                                  key=lambda x: x[1]['metrics_mean'].get('hypervolume', 0))
                best_cov_prog = max(self.analysis_results.items(),
                                   key=lambda x: x[1]['metrics_mean'].get('max_coverage', 0))
                
                f.write("BEST PERFORMERS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Highest Hypervolume: {best_hv_prog[0]} "
                       f"({best_hv_prog[1]['metrics_mean']['hypervolume']:.3f})\n")
                f.write(f"Best Coverage: {best_cov_prog[0]} "
                       f"({best_cov_prog[1]['metrics_mean']['max_coverage']:.1%})\n\n")
            
            # Difficulty classification
            difficulty = self.classify_test_difficulty()
            f.write("DIFFICULTY CLASSIFICATION\n")
            f.write("-" * 40 + "\n")
            for level in ['Easy', 'Medium', 'Hard']:
                programs = [p for p, d in difficulty.items() if d == level]
                f.write(f"{level}: {len(programs)} programs\n")
                for prog in programs[:3]:  # Show first 3
                    f.write(f"  - {prog}\n")
                if len(programs) > 3:
                    f.write(f"  ... and {len(programs) - 3} more\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"\nReport generated in: {report_dir}")
        print(f"  - Summary: summary.csv")
        print(f"  - Visualization: dashboard.png")
        print(f"  - Full report: report.txt")
        print(f"  - Raw data: raw_results.json")
        
        return str(report_dir)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Comprehensive testing and analysis for multi-objective algorithms'
    )
    
    parser.add_argument('--algorithm', '-a', type=str, default='NSGA2',
                       choices=['NSGA2', 'NSGA3', 'MOEAD', 'CTAEA'],
                       help='Multi-objective algorithm (default: NSGA2)')
    parser.add_argument('--objectives', type=int, default=2, choices=[2, 3],
                       help='Number of objectives (default: 2)')
    parser.add_argument('--objective-type', type=str, default='conflicting',
                       choices=['traditional', 'conflicting'],
                       help='Objective type: traditional (fitness vs coverage) or conflicting (coverage vs complexity) (default: conflicting)')
    parser.add_argument('--runs', '-r', type=int, default=5,
                       help='Number of runs per test program (default: 5)')
    parser.add_argument('--generations', '-g', type=int, default=100,
                       help='Maximum generations (default: 100)')
    parser.add_argument('--pop-size', type=int, default=100,
                       help='Population size (default: 100)')
    parser.add_argument('--output', '-o', type=str, default='mo_analysis_results',
                       help='Output directory (default: mo_analysis_results)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    try:
        # Create analyzer
        analyzer = MultiObjectiveAnalyzer(
            algorithm_name=args.algorithm,
            n_objectives=args.objectives,
            num_runs=args.runs,
            max_generations=args.generations,
            pop_size=args.pop_size,
            verbose=not args.quiet,
            objective_type=args.objective_type
        )
        
        # Run comprehensive test
        results = analyzer.run_comprehensive_test()
        
        # Analyze results
        analysis = analyzer.analyze_results()
        
        # Generate report and visualizations
        report_dir = analyzer.generate_report(args.output)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())