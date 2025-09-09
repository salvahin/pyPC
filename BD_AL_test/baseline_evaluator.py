#!/usr/bin/env python3
"""
Baseline Evaluation Framework
Evaluates non-metaheuristic baselines and compares with MO algorithms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import ast
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

from baseline_test_generators import BaselineTestGenerator, HybridBaselines, TestSuite
from tree_converter import TreeVisitor
from test_fitness import Fitness
from multi_objective_fitness import MultiObjectiveFitness


@dataclass 
class BaselineResult:
    """Results from baseline evaluation"""
    method: str
    program: str
    test_suite: TestSuite
    coverage: float
    branches_covered: int
    total_branches: int
    unique_paths: int
    generation_time: float
    evaluation_time: float
    total_time: float
    test_suite_size: int
    coverage_per_test: float
    diversity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaselineEvaluator:
    """Evaluates baseline test generation methods"""
    
    def __init__(self, test_programs: Dict[str, Dict[str, Any]], verbose: bool = True):
        """
        Initialize evaluator
        
        Args:
            test_programs: Dictionary of program configurations
            verbose: Print progress messages
        """
        self.test_programs = test_programs
        self.verbose = verbose
        self.results = []
        
    def evaluate_test_suite(self, test_suite: TestSuite, 
                           visitor: TreeVisitor,
                           program_name: str) -> BaselineResult:
        """
        Evaluate a test suite on a program
        
        Args:
            test_suite: Generated test suite
            visitor: TreeVisitor for the program
            program_name: Name of the program
            
        Returns:
            BaselineResult with evaluation metrics
        """
        eval_start = time.time()
        
        # Create fitness evaluator
        fitness = Fitness(visitor)
        
        # Evaluate each test case
        covered_branches = set()
        unique_paths = set()
        
        for test in test_suite.test_cases:
            # Reset coverage tracking
            fitness.current_walked_tree = []
            fitness.coverage = 0
            
            # Evaluate test
            try:
                # Run fitness function (this executes the test)
                fitness.fitness_function(test.reshape(1, -1))
                
                # Track covered branches
                for branch in fitness.current_walked_tree:
                    covered_branches.add(branch)
                
                # Track unique path
                path_signature = tuple(fitness.current_walked_tree)
                unique_paths.add(path_signature)
            except:
                # Handle evaluation errors
                pass
        
        # Calculate coverage
        total_branches = len(fitness.whole_tree) if fitness.whole_tree else 1
        branches_covered = len(covered_branches)
        coverage = branches_covered / total_branches if total_branches > 0 else 0
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity(test_suite.test_cases)
        
        eval_time = time.time() - eval_start
        
        return BaselineResult(
            method=test_suite.method,
            program=program_name,
            test_suite=test_suite,
            coverage=coverage,
            branches_covered=branches_covered,
            total_branches=total_branches,
            unique_paths=len(unique_paths),
            generation_time=test_suite.generation_time,
            evaluation_time=eval_time,
            total_time=test_suite.generation_time + eval_time,
            test_suite_size=len(test_suite.test_cases),
            coverage_per_test=coverage / len(test_suite.test_cases) if len(test_suite.test_cases) > 0 else 0,
            diversity_score=diversity_score,
            metadata=test_suite.metadata
        )
    
    def _calculate_diversity(self, test_cases: np.ndarray) -> float:
        """Calculate diversity score for test suite"""
        if len(test_cases) < 2:
            return 0.0
        
        # Calculate average pairwise distance
        distances = []
        for i in range(min(len(test_cases), 50)):  # Sample for efficiency
            for j in range(i+1, min(len(test_cases), 50)):
                dist = np.linalg.norm(test_cases[i] - test_cases[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def evaluate_baseline(self, method_name: str, 
                         generator_func,
                         n_tests: int = 100) -> List[BaselineResult]:
        """
        Evaluate a baseline method on all test programs
        
        Args:
            method_name: Name of the method
            generator_func: Function to generate test suite
            n_tests: Number of tests to generate
            
        Returns:
            List of BaselineResults
        """
        results = []
        
        for prog_name, prog_config in self.test_programs.items():
            if self.verbose:
                print(f"Evaluating {method_name} on {prog_name}...")
            
            # Load and parse program
            with open(prog_config['path'], 'r') as f:
                tree = ast.parse(f.read())
            
            visitor = TreeVisitor()
            visitor.visit(tree)
            
            # Generate test suite
            test_suite = generator_func(visitor, prog_config['dimensions'], n_tests)
            
            # Evaluate
            result = self.evaluate_test_suite(test_suite, visitor, prog_name)
            results.append(result)
            
            if self.verbose:
                print(f"  Coverage: {result.coverage:.2%}, "
                      f"Time: {result.total_time:.3f}s")
        
        return results
    
    def run_all_baselines(self, n_tests: int = 100) -> pd.DataFrame:
        """
        Run all baseline methods on all programs
        
        Args:
            n_tests: Number of tests per method
            
        Returns:
            DataFrame with all results
        """
        all_results = []
        
        for prog_name, prog_config in self.test_programs.items():
            if self.verbose:
                print(f"\nEvaluating baselines on {prog_name}")
                print("-" * 40)
            
            # Load program
            with open(prog_config['path'], 'r') as f:
                tree = ast.parse(f.read())
            
            visitor = TreeVisitor()
            visitor.visit(tree)
            
            # Create generators
            generator = BaselineTestGenerator(visitor, prog_config['dimensions'])
            hybrid = HybridBaselines(visitor, prog_config['dimensions'])
            
            # Evaluate each method
            methods = {
                'Random': lambda: generator.random_testing(n_tests, seed=42),
                'ART': lambda: generator.adaptive_random_testing(n_tests, seed=42),
                'Sobol': lambda: generator.quasi_random_testing(n_tests, 'sobol', seed=42),
                'Halton': lambda: generator.quasi_random_testing(n_tests, 'halton', seed=42),
                'Latin Hypercube': lambda: generator.quasi_random_testing(n_tests, 'latin_hypercube', seed=42),
                'Grid Search': lambda: generator.grid_search(min(5, int(np.power(n_tests, 1/prog_config['dimensions'])) + 1)),
                'BVA': lambda: generator.boundary_value_analysis(1),
                'Hill Climbing': lambda: generator.hill_climbing(n_restarts=10),
                'Greedy Coverage': lambda: generator.greedy_coverage(n_tests),
                'Pattern Search': lambda: generator.pattern_search(n_tests=10),
                'Directed Random': lambda: hybrid.directed_random_testing(n_tests),
                'Adaptive Sampling': lambda: hybrid.adaptive_sampling()
            }
            
            # Add combinatorial testing only for low dimensions
            if prog_config['dimensions'] <= 4:
                methods['Pairwise'] = lambda: generator.combinatorial_testing(2, 3)
            
            for method_name, method_func in methods.items():
                if self.verbose:
                    print(f"  {method_name}...", end=" ")
                
                try:
                    # Generate test suite
                    test_suite = method_func()
                    
                    # Evaluate
                    result = self.evaluate_test_suite(test_suite, visitor, prog_name)
                    all_results.append(result)
                    
                    if self.verbose:
                        print(f"Coverage: {result.coverage:.2%}, Tests: {result.test_suite_size}")
                except Exception as e:
                    if self.verbose:
                        print(f"Failed: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'Method': r.method,
                'Program': r.program,
                'Coverage': r.coverage,
                'Branches_Covered': r.branches_covered,
                'Total_Branches': r.total_branches,
                'Unique_Paths': r.unique_paths,
                'Test_Suite_Size': r.test_suite_size,
                'Coverage_Per_Test': r.coverage_per_test,
                'Generation_Time': r.generation_time,
                'Evaluation_Time': r.evaluation_time,
                'Total_Time': r.total_time,
                'Diversity_Score': r.diversity_score
            }
            for r in all_results
        ])
        
        return df
    
    def compare_with_metaheuristics(self, baseline_df: pd.DataFrame,
                                   mo_results_path: str) -> pd.DataFrame:
        """
        Compare baseline results with metaheuristic results
        
        Args:
            baseline_df: DataFrame with baseline results
            mo_results_path: Path to MO results file
            
        Returns:
            Combined comparison DataFrame
        """
        # Load MO results
        mo_df = pd.read_csv(mo_results_path)
        
        # Aggregate baseline results by method
        baseline_summary = baseline_df.groupby('Method').agg({
            'Coverage': ['mean', 'std'],
            'Test_Suite_Size': 'mean',
            'Total_Time': 'mean',
            'Coverage_Per_Test': 'mean',
            'Diversity_Score': 'mean'
        }).round(4)
        
        # Aggregate MO results
        mo_summary = mo_df.groupby('Algorithm').agg({
            'coverage_mean': 'mean',
            'Solutions_Mean': 'mean',
            'Time_Mean': 'mean'
        }).round(4) if 'coverage_mean' in mo_df.columns else mo_df.groupby('Algorithm').agg({
            'HV_Mean': 'mean',
            'Solutions_Mean': 'mean', 
            'Time_Mean': 'mean'
        }).round(4)
        
        # Add method type
        baseline_summary['Type'] = 'Baseline'
        mo_summary['Type'] = 'Metaheuristic'
        
        # Combine
        comparison = pd.concat([baseline_summary, mo_summary])
        
        return comparison
    
    def visualize_results(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create visualization of baseline results
        
        Args:
            df: Results DataFrame
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Coverage comparison
        ax = axes[0, 0]
        coverage_by_method = df.groupby('Method')['Coverage'].mean().sort_values()
        coverage_by_method.plot(kind='barh', ax=ax)
        ax.set_xlabel('Coverage')
        ax.set_title('Average Coverage by Method')
        ax.axvline(x=coverage_by_method.median(), color='r', linestyle='--', alpha=0.5)
        
        # Test suite size
        ax = axes[0, 1]
        size_by_method = df.groupby('Method')['Test_Suite_Size'].mean().sort_values()
        size_by_method.plot(kind='barh', ax=ax)
        ax.set_xlabel('Test Suite Size')
        ax.set_title('Average Test Suite Size')
        
        # Time efficiency
        ax = axes[0, 2]
        time_by_method = df.groupby('Method')['Total_Time'].mean().sort_values()
        time_by_method.plot(kind='barh', ax=ax)
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Average Execution Time')
        
        # Coverage per test (efficiency)
        ax = axes[1, 0]
        eff_by_method = df.groupby('Method')['Coverage_Per_Test'].mean().sort_values()
        eff_by_method.plot(kind='barh', ax=ax)
        ax.set_xlabel('Coverage per Test')
        ax.set_title('Test Efficiency')
        
        # Program difficulty
        ax = axes[1, 1]
        coverage_by_program = df.groupby('Program')['Coverage'].mean().sort_values()
        coverage_by_program.plot(kind='barh', ax=ax)
        ax.set_xlabel('Coverage')
        ax.set_title('Average Coverage by Program')
        
        # Method performance heatmap
        ax = axes[1, 2]
        pivot = df.pivot_table(values='Coverage', index='Method', columns='Program')
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title('Coverage Heatmap')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            if self.verbose:
                print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, df: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive comparison report
        
        Args:
            df: Results DataFrame
            save_path: Optional path to save report
            
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 80)
        report.append("BASELINE TEST GENERATION METHODS EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS")
        report.append("-" * 40)
        
        summary = df.groupby('Method').agg({
            'Coverage': ['mean', 'std'],
            'Test_Suite_Size': 'mean',
            'Total_Time': 'mean',
            'Coverage_Per_Test': 'mean',
            'Diversity_Score': 'mean'
        }).round(4)
        
        report.append(summary.to_string())
        report.append("")
        
        # Best methods
        report.append("BEST PERFORMING METHODS")
        report.append("-" * 40)
        
        best_coverage = df.groupby('Method')['Coverage'].mean().idxmax()
        best_efficiency = df.groupby('Method')['Coverage_Per_Test'].mean().idxmax()
        fastest = df.groupby('Method')['Total_Time'].mean().idxmin()
        most_diverse = df.groupby('Method')['Diversity_Score'].mean().idxmax()
        
        report.append(f"• Best Coverage: {best_coverage} ({df[df['Method']==best_coverage]['Coverage'].mean():.2%})")
        report.append(f"• Most Efficient: {best_efficiency} ({df[df['Method']==best_efficiency]['Coverage_Per_Test'].mean():.4f} coverage/test)")
        report.append(f"• Fastest: {fastest} ({df[df['Method']==fastest]['Total_Time'].mean():.3f}s)")
        report.append(f"• Most Diverse: {most_diverse} ({df[df['Method']==most_diverse]['Diversity_Score'].mean():.2f})")
        report.append("")
        
        # Program-specific analysis
        report.append("PROGRAM-SPECIFIC ANALYSIS")
        report.append("-" * 40)
        
        for program in df['Program'].unique():
            prog_data = df[df['Program'] == program]
            report.append(f"\n{program}:")
            
            best = prog_data.loc[prog_data['Coverage'].idxmax()]
            report.append(f"  Best Method: {best['Method']} (Coverage: {best['Coverage']:.2%})")
            
            avg_coverage = prog_data['Coverage'].mean()
            report.append(f"  Average Coverage: {avg_coverage:.2%}")
            
            if avg_coverage > 0.8:
                report.append("  Difficulty: Easy")
            elif avg_coverage > 0.5:
                report.append("  Difficulty: Medium")
            else:
                report.append("  Difficulty: Hard")
        
        report.append("")
        
        # Method categories analysis
        report.append("METHOD CATEGORY ANALYSIS")
        report.append("-" * 40)
        
        # Categorize methods
        random_methods = ['Random', 'ART', 'Directed Random']
        quasi_random = ['Sobol', 'Halton', 'Latin Hypercube']
        systematic = ['Grid Search', 'BVA', 'Pairwise']
        search_based = ['Hill Climbing', 'Greedy Coverage', 'Pattern Search']
        
        categories = {
            'Random': random_methods,
            'Quasi-Random': quasi_random,
            'Systematic': systematic,
            'Search-Based': search_based
        }
        
        for cat_name, methods in categories.items():
            cat_data = df[df['Method'].isin(methods)]
            if not cat_data.empty:
                avg_cov = cat_data['Coverage'].mean()
                avg_time = cat_data['Total_Time'].mean()
                report.append(f"\n{cat_name}:")
                report.append(f"  Average Coverage: {avg_cov:.2%}")
                report.append(f"  Average Time: {avg_time:.3f}s")
        
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("• For simple programs: Use Grid Search or BVA for guaranteed coverage")
        report.append("• For complex programs: Use quasi-random methods (Sobol/Halton) for better distribution")
        report.append("• For quick testing: Use pure Random (fastest)")
        report.append("• For balanced performance: Use ART or Latin Hypercube")
        report.append("• When test budget is limited: Use Greedy Coverage or search-based methods")
        
        full_report = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(full_report)
            if self.verbose:
                print(f"Report saved to {save_path}")
        
        return full_report


if __name__ == "__main__":
    # Example usage
    import yaml
    
    print("Baseline Evaluation Framework")
    print("=" * 60)
    
    # Load test programs configuration
    with open('config/test_programs.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Select a few programs for testing
    test_programs = {
        'minimum': config['test_programs']['minimum'],
        'bubble_sort': config['test_programs']['bubble_sort'],
        'three_number_sort': config['test_programs']['three_number_sort']
    }
    
    # Create evaluator
    evaluator = BaselineEvaluator(test_programs)
    
    # Run evaluation
    print("\nRunning baseline evaluation...")
    df = evaluator.run_all_baselines(n_tests=50)
    
    # Generate report
    print("\nGenerating report...")
    report = evaluator.generate_report(df, "baseline_evaluation_report.txt")
    
    # Show summary
    print("\nSummary by Method:")
    print(df.groupby('Method')['Coverage'].agg(['mean', 'std']).round(3))
    
    print("\nEvaluation complete!")