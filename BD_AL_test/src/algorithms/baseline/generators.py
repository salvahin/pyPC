#!/usr/bin/env python3
"""
Unified Baseline Test Generation Algorithms

This module consolidates all baseline test generation methods into a single,
clean interface supporting both simple and complex synthetic test programs.
"""

import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from abc import ABC, abstractmethod
import itertools
import math
from scipy.stats import qmc


class BaseTestGenerator(ABC):
    """Abstract base class for all test generators"""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    @abstractmethod
    def generate(self, n_tests: int, **kwargs) -> List[Any]:
        """Generate n test cases"""
        pass


class RandomTestGenerator(BaseTestGenerator):
    """Pure random test generation"""
    
    def generate(self, n_tests: int, bounds: List[Tuple[float, float]] = None, **kwargs) -> List[List[float]]:
        if bounds is None:
            bounds = [(-1000, 1000)] * kwargs.get('dimensions', 2)
        
        tests = []
        for _ in range(n_tests):
            test = []
            for low, high in bounds:
                test.append(random.uniform(low, high))
            tests.append(test)
        return tests


class AdaptiveRandomTestGenerator(BaseTestGenerator):
    """Adaptive Random Testing with distance-based selection"""
    
    def __init__(self, seed: Optional[int] = None, distance_threshold: float = 0.1):
        super().__init__(seed)
        self.distance_threshold = distance_threshold
        self.generated_tests = []
    
    def generate(self, n_tests: int, bounds: List[Tuple[float, float]] = None, **kwargs) -> List[List[float]]:
        if bounds is None:
            bounds = [(-1000, 1000)] * kwargs.get('dimensions', 2)
        
        tests = []
        for _ in range(n_tests):
            candidates = []
            # Generate multiple candidates
            for _ in range(10):
                candidate = []
                for low, high in bounds:
                    candidate.append(random.uniform(low, high))
                candidates.append(candidate)
            
            # Select candidate with maximum distance from existing tests
            if self.generated_tests:
                best_candidate = max(candidates, key=lambda c: min(
                    self._euclidean_distance(c, existing) for existing in self.generated_tests
                ))
            else:
                best_candidate = candidates[0]
            
            tests.append(best_candidate)
            self.generated_tests.append(best_candidate)
        
        return tests
    
    def _euclidean_distance(self, a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class QuasiRandomTestGenerator(BaseTestGenerator):
    """Quasi-random testing using low-discrepancy sequences"""
    
    def generate(self, n_tests: int, bounds: List[Tuple[float, float]] = None, **kwargs) -> List[List[float]]:
        if bounds is None:
            bounds = [(-1000, 1000)] * kwargs.get('dimensions', 2)
        
        dimensions = len(bounds)
        sampler = qmc.Halton(d=dimensions, seed=self.seed)
        samples = sampler.random(n=n_tests)
        
        # Scale to bounds
        tests = []
        for sample in samples:
            test = []
            for i, (low, high) in enumerate(bounds):
                scaled_value = low + sample[i] * (high - low)
                test.append(scaled_value)
            tests.append(test)
        
        return tests


class GridSearchTestGenerator(BaseTestGenerator):
    """Systematic grid-based test generation"""
    
    def generate(self, n_tests: int, bounds: List[Tuple[float, float]] = None, **kwargs) -> List[List[float]]:
        if bounds is None:
            bounds = [(-1000, 1000)] * kwargs.get('dimensions', 2)
        
        dimensions = len(bounds)
        points_per_dim = max(2, int(n_tests ** (1/dimensions)))
        
        # Create grid points for each dimension
        grid_points = []
        for low, high in bounds:
            points = np.linspace(low, high, points_per_dim)
            grid_points.append(points)
        
        # Generate all combinations
        tests = []
        for combination in itertools.product(*grid_points):
            if len(tests) >= n_tests:
                break
            tests.append(list(combination))
        
        # If we need more tests, add random samples
        while len(tests) < n_tests:
            test = []
            for low, high in bounds:
                test.append(random.uniform(low, high))
            tests.append(test)
        
        return tests[:n_tests]


class BoundaryValueTestGenerator(BaseTestGenerator):
    """Boundary value analysis test generation"""
    
    def generate(self, n_tests: int, bounds: List[Tuple[float, float]] = None, **kwargs) -> List[List[float]]:
        if bounds is None:
            bounds = [(-1000, 1000)] * kwargs.get('dimensions', 2)
        
        tests = []
        dimensions = len(bounds)
        
        # Generate boundary tests: min, max, and middle values
        for dim in range(dimensions):
            for boundary_type in ['min', 'max', 'mid']:
                if len(tests) >= n_tests:
                    break
                    
                test = []
                for i, (low, high) in enumerate(bounds):
                    if i == dim:
                        if boundary_type == 'min':
                            test.append(low)
                        elif boundary_type == 'max':
                            test.append(high)
                        else:  # mid
                            test.append((low + high) / 2)
                    else:
                        test.append((low + high) / 2)  # Middle value for other dimensions
                tests.append(test)
        
        # Fill remaining with random tests
        while len(tests) < n_tests:
            test = []
            for low, high in bounds:
                test.append(random.uniform(low, high))
            tests.append(test)
        
        return tests[:n_tests]


class HillClimbingTestGenerator(BaseTestGenerator):
    """Hill climbing local search test generation"""
    
    def __init__(self, seed: Optional[int] = None, step_size: float = 0.1):
        super().__init__(seed)
        self.step_size = step_size
    
    def generate(self, n_tests: int, bounds: List[Tuple[float, float]] = None, **kwargs) -> List[List[float]]:
        if bounds is None:
            bounds = [(-1000, 1000)] * kwargs.get('dimensions', 2)
        
        tests = []
        
        # Start with random initial solution
        current = []
        for low, high in bounds:
            current.append(random.uniform(low, high))
        tests.append(current[:])
        
        # Generate remaining tests using hill climbing
        for _ in range(n_tests - 1):
            # Generate neighbor
            neighbor = current[:]
            dim = random.randint(0, len(bounds) - 1)
            low, high = bounds[dim]
            
            # Random step in chosen dimension
            step = random.uniform(-self.step_size * (high - low), self.step_size * (high - low))
            neighbor[dim] = max(low, min(high, neighbor[dim] + step))
            
            tests.append(neighbor)
            current = neighbor
        
        return tests


class UnifiedBaselineGenerator:
    """Unified interface for all baseline test generation methods"""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.generators = {
            'random': RandomTestGenerator(seed),
            'adaptive_random': AdaptiveRandomTestGenerator(seed),
            'quasi_random': QuasiRandomTestGenerator(seed),
            'grid_search': GridSearchTestGenerator(seed),
            'boundary_value': BoundaryValueTestGenerator(seed),
            'hill_climbing': HillClimbingTestGenerator(seed),
        }
    
    def generate_tests(self, method: str, n_tests: int, **kwargs) -> List[Any]:
        """Generate tests using specified method"""
        if method not in self.generators:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.generators.keys())}")
        
        return self.generators[method].generate(n_tests, **kwargs)
    
    def get_available_methods(self) -> List[str]:
        """Get list of available generation methods"""
        return list(self.generators.keys())
    
    def generate_comprehensive_suite(self, n_tests_per_method: int = 50, **kwargs) -> Dict[str, List[Any]]:
        """Generate comprehensive test suite using all methods"""
        suite = {}
        for method in self.generators.keys():
            try:
                suite[method] = self.generate_tests(method, n_tests_per_method, **kwargs)
            except Exception as e:
                print(f"Warning: Failed to generate tests with {method}: {e}")
                suite[method] = []
        
        return suite


# Main interface for backward compatibility
class BaselineTestGenerator:
    """Main baseline test generator class - simplified interface"""
    
    def __init__(self, seed: Optional[int] = None):
        self.generator = UnifiedBaselineGenerator(seed)
    
    def generate_random_tests(self, n_tests: int, **kwargs) -> List[Any]:
        return self.generator.generate_tests('random', n_tests, **kwargs)
    
    def generate_adaptive_random_tests(self, n_tests: int, **kwargs) -> List[Any]:
        return self.generator.generate_tests('adaptive_random', n_tests, **kwargs)
    
    def generate_boundary_tests(self, n_tests: int, **kwargs) -> List[Any]:
        return self.generator.generate_tests('boundary_value', n_tests, **kwargs)
    
    def generate_all_methods(self, n_tests_per_method: int = 50, **kwargs) -> Dict[str, List[Any]]:
        return self.generator.generate_comprehensive_suite(n_tests_per_method, **kwargs)