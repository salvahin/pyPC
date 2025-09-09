#!/usr/bin/env python3
"""
Non-Metaheuristic Baseline Test Generators
Provides various baseline approaches for comparison with metaheuristic algorithms
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from scipy.stats import qmc
from itertools import product, combinations
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TestSuite:
    """Container for test suite with metadata"""
    test_cases: np.ndarray
    method: str
    generation_time: float
    metadata: Dict[str, Any] = None


class BaselineTestGenerator:
    """Base class for non-metaheuristic test generation approaches"""
    
    def __init__(self, visitor, dimensions: int, 
                 bounds: Tuple[float, float] = (-999999, 999999)):
        """
        Initialize baseline generator
        
        Args:
            visitor: TreeVisitor instance
            dimensions: Number of input parameters
            bounds: (lower, upper) bounds for inputs
        """
        self.visitor = visitor
        self.dimensions = dimensions
        self.lower_bound, self.upper_bound = bounds
        self.generation_times = {}
        
    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalize values to bounds"""
        return values * (self.upper_bound - self.lower_bound) + self.lower_bound
    
    # ============= Random Testing Baselines =============
    
    def random_testing(self, n_tests: int, seed: Optional[int] = None) -> TestSuite:
        """
        Pure random test generation
        
        Args:
            n_tests: Number of test cases to generate
            seed: Random seed for reproducibility
            
        Returns:
            TestSuite with random test cases
        """
        start_time = time.time()
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate uniform random test cases
        tests = np.random.uniform(
            self.lower_bound, 
            self.upper_bound, 
            size=(n_tests, self.dimensions)
        )
        
        return TestSuite(
            test_cases=tests,
            method="Random Testing",
            generation_time=time.time() - start_time,
            metadata={"seed": seed}
        )
    
    def adaptive_random_testing(self, n_tests: int, 
                               candidate_pool_size: int = 10,
                               seed: Optional[int] = None) -> TestSuite:
        """
        Adaptive Random Testing (ART)
        Selects test cases that maximize distance from existing tests
        
        Args:
            n_tests: Number of test cases to generate
            candidate_pool_size: Size of candidate pool for each selection
            seed: Random seed
            
        Returns:
            TestSuite with ART-generated tests
        """
        start_time = time.time()
        
        if seed is not None:
            np.random.seed(seed)
        
        tests = []
        
        # First test is random
        first_test = np.random.uniform(
            self.lower_bound, self.upper_bound, self.dimensions
        )
        tests.append(first_test)
        
        # Generate remaining tests
        for _ in range(1, n_tests):
            # Generate candidate pool
            candidates = np.random.uniform(
                self.lower_bound, self.upper_bound,
                size=(candidate_pool_size, self.dimensions)
            )
            
            # Calculate minimum distance to existing tests for each candidate
            min_distances = []
            for candidate in candidates:
                distances = [np.linalg.norm(candidate - test) for test in tests]
                min_distances.append(min(distances))
            
            # Select candidate with maximum minimum distance
            best_idx = np.argmax(min_distances)
            tests.append(candidates[best_idx])
        
        return TestSuite(
            test_cases=np.array(tests),
            method="Adaptive Random Testing",
            generation_time=time.time() - start_time,
            metadata={
                "seed": seed,
                "candidate_pool_size": candidate_pool_size
            }
        )
    
    def quasi_random_testing(self, n_tests: int, 
                            sequence: str = 'sobol',
                            seed: Optional[int] = None) -> TestSuite:
        """
        Quasi-random testing using low-discrepancy sequences
        
        Args:
            n_tests: Number of test cases
            sequence: Type of sequence ('sobol', 'halton', 'latin_hypercube')
            seed: Random seed
            
        Returns:
            TestSuite with quasi-random tests
        """
        start_time = time.time()
        
        if sequence == 'sobol':
            sampler = qmc.Sobol(d=self.dimensions, seed=seed)
        elif sequence == 'halton':
            sampler = qmc.Halton(d=self.dimensions, seed=seed)
        elif sequence == 'latin_hypercube':
            sampler = qmc.LatinHypercube(d=self.dimensions, seed=seed)
        else:
            raise ValueError(f"Unknown sequence type: {sequence}")
        
        # Generate samples in [0, 1]
        samples = sampler.random(n=n_tests)
        
        # Scale to bounds
        tests = self._normalize(samples)
        
        return TestSuite(
            test_cases=tests,
            method=f"Quasi-Random ({sequence})",
            generation_time=time.time() - start_time,
            metadata={"sequence": sequence, "seed": seed}
        )
    
    # ============= Systematic Approaches =============
    
    def grid_search(self, resolution: int) -> TestSuite:
        """
        Grid search - systematic exploration of input space
        
        Args:
            resolution: Number of points per dimension
            
        Returns:
            TestSuite with grid points
        """
        start_time = time.time()
        
        # Create grid for each dimension
        grid_points = []
        for _ in range(self.dimensions):
            points = np.linspace(self.lower_bound, self.upper_bound, resolution)
            grid_points.append(points)
        
        # Generate all combinations
        tests = list(product(*grid_points))
        tests = np.array(tests)
        
        return TestSuite(
            test_cases=tests,
            method="Grid Search",
            generation_time=time.time() - start_time,
            metadata={"resolution": resolution, "total_points": len(tests)}
        )
    
    def boundary_value_analysis(self, robustness: int = 1) -> TestSuite:
        """
        Boundary Value Analysis
        Tests boundary values and near-boundary values
        
        Args:
            robustness: 1 for basic (4n+1), 2 for robust (6n+1), 3 for worst-case
            
        Returns:
            TestSuite with boundary test cases
        """
        start_time = time.time()
        
        tests = []
        
        # Define boundary values based on robustness level
        if robustness == 1:
            # Basic: min, min+δ, typical, max-δ, max
            values = [
                self.lower_bound,
                self.lower_bound + 1,
                (self.lower_bound + self.upper_bound) / 2,
                self.upper_bound - 1,
                self.upper_bound
            ]
        elif robustness == 2:
            # Robust: adds min-δ and max+δ
            values = [
                self.lower_bound - 1,
                self.lower_bound,
                self.lower_bound + 1,
                (self.lower_bound + self.upper_bound) / 2,
                self.upper_bound - 1,
                self.upper_bound,
                self.upper_bound + 1
            ]
        else:
            # Worst-case: all combinations
            values = [
                self.lower_bound,
                self.lower_bound + 1,
                (self.lower_bound + self.upper_bound) / 2,
                self.upper_bound - 1,
                self.upper_bound
            ]
            tests = list(product(values, repeat=self.dimensions))
            return TestSuite(
                test_cases=np.array(tests),
                method="Boundary Value Analysis (Worst-Case)",
                generation_time=time.time() - start_time,
                metadata={"robustness": robustness}
            )
        
        # For basic and robust: vary one dimension at a time
        typical = (self.lower_bound + self.upper_bound) / 2
        base_test = [typical] * self.dimensions
        
        # Add base test
        tests.append(base_test.copy())
        
        # Vary each dimension
        for dim in range(self.dimensions):
            for value in values:
                if value != typical:
                    test = base_test.copy()
                    test[dim] = value
                    tests.append(test)
        
        return TestSuite(
            test_cases=np.array(tests),
            method=f"Boundary Value Analysis (Robustness={robustness})",
            generation_time=time.time() - start_time,
            metadata={"robustness": robustness}
        )
    
    def combinatorial_testing(self, strength: int = 2, 
                            values_per_param: int = 3) -> TestSuite:
        """
        Combinatorial testing (t-way testing)
        
        Args:
            strength: t value for t-way testing (typically 2 for pairwise)
            values_per_param: Number of values to test per parameter
            
        Returns:
            TestSuite with combinatorial test cases
        """
        start_time = time.time()
        
        # Define value levels for each parameter
        param_values = []
        for _ in range(self.dimensions):
            values = np.linspace(self.lower_bound, self.upper_bound, values_per_param)
            param_values.append(values)
        
        if strength >= self.dimensions:
            # Full combinatorial
            tests = list(product(*param_values))
        else:
            # Generate covering array (simplified version)
            tests = self._generate_covering_array(param_values, strength)
        
        return TestSuite(
            test_cases=np.array(tests),
            method=f"Combinatorial Testing ({strength}-way)",
            generation_time=time.time() - start_time,
            metadata={
                "strength": strength,
                "values_per_param": values_per_param
            }
        )
    
    def _generate_covering_array(self, param_values: List[np.ndarray], 
                                strength: int) -> List[List[float]]:
        """
        Generate a covering array for t-way testing
        Simplified implementation - not optimal but functional
        """
        tests = []
        n_params = len(param_values)
        
        # Get all t-way combinations of parameters
        param_combos = list(combinations(range(n_params), strength))
        
        # For each combination, ensure all value combinations are covered
        for param_indices in param_combos:
            # Get values for these parameters
            values_to_combine = [param_values[i] for i in param_indices]
            
            # Generate all combinations for these t parameters
            for value_combo in product(*values_to_combine):
                # Create a test case
                test = []
                value_idx = 0
                for i in range(n_params):
                    if i in param_indices:
                        test.append(value_combo[value_idx])
                        value_idx += 1
                    else:
                        # Random value for non-covered parameters
                        test.append(np.random.choice(param_values[i]))
                tests.append(test)
        
        # Remove duplicates
        tests = list(set(tuple(t) for t in tests))
        tests = [list(t) for t in tests]
        
        return tests
    
    # ============= Search-Based Non-Metaheuristic =============
    
    def hill_climbing(self, n_restarts: int = 10, 
                      max_iterations: int = 100,
                      neighborhood_size: float = 0.1,
                      fitness_function = None) -> TestSuite:
        """
        Hill climbing with random restarts
        
        Args:
            n_restarts: Number of random restarts
            max_iterations: Maximum iterations per restart
            neighborhood_size: Size of neighborhood (fraction of range)
            fitness_function: Fitness function to optimize
            
        Returns:
            TestSuite with best solutions from each restart
        """
        start_time = time.time()
        
        if fitness_function is None:
            # Use a simple coverage-based fitness if not provided
            def fitness_function(x):
                # This would need actual implementation with visitor
                return np.random.random()
        
        tests = []
        step_size = neighborhood_size * (self.upper_bound - self.lower_bound)
        
        for _ in range(n_restarts):
            # Random starting point
            current = np.random.uniform(
                self.lower_bound, self.upper_bound, self.dimensions
            )
            current_fitness = fitness_function(current.reshape(1, -1))
            
            for _ in range(max_iterations):
                # Generate neighbor
                neighbor = current + np.random.normal(0, step_size, self.dimensions)
                neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
                
                neighbor_fitness = fitness_function(neighbor.reshape(1, -1))
                
                # Move if better
                if neighbor_fitness < current_fitness:
                    current = neighbor
                    current_fitness = neighbor_fitness
                else:
                    break  # Local optimum reached
            
            tests.append(current)
        
        return TestSuite(
            test_cases=np.array(tests),
            method="Hill Climbing",
            generation_time=time.time() - start_time,
            metadata={
                "n_restarts": n_restarts,
                "max_iterations": max_iterations,
                "neighborhood_size": neighborhood_size
            }
        )
    
    def greedy_coverage(self, n_tests: int, 
                       coverage_function = None,
                       candidate_pool_size: int = 100) -> TestSuite:
        """
        Greedy coverage-driven test generation
        
        Args:
            n_tests: Number of tests to generate
            coverage_function: Function to calculate coverage
            candidate_pool_size: Candidates to evaluate per iteration
            
        Returns:
            TestSuite with greedy-selected tests
        """
        start_time = time.time()
        
        if coverage_function is None:
            # Dummy coverage function
            def coverage_function(x):
                return np.random.random()
        
        tests = []
        covered = set()
        
        for _ in range(n_tests):
            # Generate candidate pool
            candidates = np.random.uniform(
                self.lower_bound, self.upper_bound,
                size=(candidate_pool_size, self.dimensions)
            )
            
            best_candidate = None
            best_new_coverage = 0
            
            # Evaluate each candidate
            for candidate in candidates:
                # Calculate new coverage (simplified)
                new_coverage = coverage_function(candidate.reshape(1, -1))
                
                if new_coverage > best_new_coverage:
                    best_new_coverage = new_coverage
                    best_candidate = candidate
            
            if best_candidate is not None:
                tests.append(best_candidate)
        
        return TestSuite(
            test_cases=np.array(tests),
            method="Greedy Coverage",
            generation_time=time.time() - start_time,
            metadata={
                "candidate_pool_size": candidate_pool_size
            }
        )
    
    def pattern_search(self, n_tests: int = 10, 
                      initial_step: float = 1.0,
                      fitness_function = None) -> TestSuite:
        """
        Pattern search (Hooke-Jeeves) method
        
        Args:
            n_tests: Number of starting points
            initial_step: Initial step size
            fitness_function: Function to optimize
            
        Returns:
            TestSuite with pattern search results
        """
        start_time = time.time()
        
        if fitness_function is None:
            def fitness_function(x):
                return np.random.random()
        
        tests = []
        
        for _ in range(n_tests):
            # Random starting point
            x = np.random.uniform(
                self.lower_bound, self.upper_bound, self.dimensions
            )
            
            step = initial_step
            
            while step > 1e-6:
                improved = False
                
                # Try moving in each dimension
                for dim in range(self.dimensions):
                    for direction in [-1, 1]:
                        x_new = x.copy()
                        x_new[dim] += direction * step
                        
                        # Check bounds
                        if self.lower_bound <= x_new[dim] <= self.upper_bound:
                            if fitness_function(x_new.reshape(1, -1)) < fitness_function(x.reshape(1, -1)):
                                x = x_new
                                improved = True
                                break
                    
                    if improved:
                        break
                
                if not improved:
                    step *= 0.5
            
            tests.append(x)
        
        return TestSuite(
            test_cases=np.array(tests),
            method="Pattern Search",
            generation_time=time.time() - start_time,
            metadata={"initial_step": initial_step}
        )


class HybridBaselines(BaselineTestGenerator):
    """Advanced hybrid baseline approaches"""
    
    def directed_random_testing(self, n_tests: int, 
                               target_branches: List[str] = None,
                               bias_strength: float = 0.5) -> TestSuite:
        """
        Directed random testing - bias towards uncovered code
        
        Args:
            n_tests: Number of tests
            target_branches: Branches to target
            bias_strength: Strength of bias (0=random, 1=fully directed)
            
        Returns:
            TestSuite with directed tests
        """
        start_time = time.time()
        
        tests = []
        
        for _ in range(n_tests):
            if np.random.random() < bias_strength and target_branches:
                # Generate biased test
                # This would need actual implementation with branch distance
                test = np.random.uniform(
                    self.lower_bound, self.upper_bound, self.dimensions
                )
                # Apply bias based on target branches
                # Simplified - actual implementation would use branch distance
                test += np.random.normal(0, 0.1, self.dimensions)
            else:
                # Random test
                test = np.random.uniform(
                    self.lower_bound, self.upper_bound, self.dimensions
                )
            
            test = np.clip(test, self.lower_bound, self.upper_bound)
            tests.append(test)
        
        return TestSuite(
            test_cases=np.array(tests),
            method="Directed Random Testing",
            generation_time=time.time() - start_time,
            metadata={"bias_strength": bias_strength}
        )
    
    def adaptive_sampling(self, initial_samples: int = 10,
                         refinement_iterations: int = 5,
                         refinement_factor: int = 2) -> TestSuite:
        """
        Adaptive sampling - refine in interesting regions
        
        Args:
            initial_samples: Initial coarse sampling
            refinement_iterations: Number of refinement steps
            refinement_factor: How many samples to add per iteration
            
        Returns:
            TestSuite with adaptively sampled tests
        """
        start_time = time.time()
        
        # Initial coarse sampling
        tests = list(np.random.uniform(
            self.lower_bound, self.upper_bound,
            size=(initial_samples, self.dimensions)
        ))
        
        for _ in range(refinement_iterations):
            # Identify interesting regions (simplified)
            # In practice, would use coverage or fitness information
            interesting_indices = np.random.choice(
                len(tests), 
                size=min(refinement_factor, len(tests)),
                replace=False
            )
            
            # Refine around interesting points
            for idx in interesting_indices:
                center = tests[idx]
                # Generate nearby samples
                radius = (self.upper_bound - self.lower_bound) / (10 * (refinement_iterations + 1))
                new_samples = center + np.random.normal(
                    0, radius, size=(2, self.dimensions)
                )
                new_samples = np.clip(new_samples, self.lower_bound, self.upper_bound)
                tests.extend(new_samples)
        
        return TestSuite(
            test_cases=np.array(tests),
            method="Adaptive Sampling",
            generation_time=time.time() - start_time,
            metadata={
                "initial_samples": initial_samples,
                "refinement_iterations": refinement_iterations
            }
        )


def compare_baseline_methods(visitor, dimensions: int, 
                            n_tests: int = 100) -> Dict[str, TestSuite]:
    """
    Generate test suites using all baseline methods for comparison
    
    Args:
        visitor: TreeVisitor instance
        dimensions: Number of input parameters
        n_tests: Target number of tests (where applicable)
        
    Returns:
        Dictionary of method_name -> TestSuite
    """
    generator = BaselineTestGenerator(visitor, dimensions)
    hybrid = HybridBaselines(visitor, dimensions)
    
    results = {}
    
    # Random-based methods
    results['Random'] = generator.random_testing(n_tests, seed=42)
    results['ART'] = generator.adaptive_random_testing(n_tests, seed=42)
    results['Sobol'] = generator.quasi_random_testing(n_tests, 'sobol', seed=42)
    results['Halton'] = generator.quasi_random_testing(n_tests, 'halton', seed=42)
    results['Latin Hypercube'] = generator.quasi_random_testing(n_tests, 'latin_hypercube', seed=42)
    
    # Systematic methods
    grid_resolution = int(np.power(n_tests, 1/dimensions)) + 1
    results['Grid Search'] = generator.grid_search(grid_resolution)
    results['BVA Basic'] = generator.boundary_value_analysis(robustness=1)
    results['BVA Robust'] = generator.boundary_value_analysis(robustness=2)
    
    # Combinatorial testing
    if dimensions <= 4:
        results['Pairwise'] = generator.combinatorial_testing(strength=2)
    
    # Search-based methods
    results['Hill Climbing'] = generator.hill_climbing(n_restarts=10)
    results['Greedy Coverage'] = generator.greedy_coverage(n_tests)
    results['Pattern Search'] = generator.pattern_search(n_tests=10)
    
    # Hybrid methods
    results['Directed Random'] = hybrid.directed_random_testing(n_tests)
    results['Adaptive Sampling'] = hybrid.adaptive_sampling()
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Baseline Test Generators")
    print("=" * 60)
    
    # Create dummy visitor for testing
    class DummyVisitor:
        def __init__(self):
            self.nodes = {'body': []}
    
    visitor = DummyVisitor()
    dimensions = 4
    
    # Generate baselines
    generator = BaselineTestGenerator(visitor, dimensions)
    
    # Test each method
    print("\nGenerating test suites with different methods:")
    print("-" * 60)
    
    methods = {
        "Random": generator.random_testing(100),
        "ART": generator.adaptive_random_testing(50),
        "Sobol": generator.quasi_random_testing(100, 'sobol'),
        "Grid": generator.grid_search(5),
        "BVA": generator.boundary_value_analysis(1),
        "Pairwise": generator.combinatorial_testing(2, 3)
    }
    
    for name, suite in methods.items():
        print(f"{name:20s}: {suite.test_cases.shape[0]:4d} tests, "
              f"Time: {suite.generation_time:.4f}s")
    
    print("\nBaseline generators ready for comparison with metaheuristics!")