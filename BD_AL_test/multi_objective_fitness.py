"""
Multi-Objective Fitness Module
Implements multi-objective optimization for test generation
Objectives: Minimize fitness (branch distance) and Maximize coverage
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from src.algorithms.multi_objective.test_fitness import Fitness
from tree_converter import TreeVisitor
from pymoo.core.problem import Problem


class MultiObjectiveFitness(Fitness):
    """
    Extended fitness class for multi-objective optimization
    Supports multiple objective combinations:
    - Traditional: fitness vs coverage
    - Conflicting: coverage vs complexity
    - Three-objective: fitness vs coverage vs complexity
    """
    
    def __init__(self, visitor: TreeVisitor, use_three_objectives: bool = False,
                 objective_type: str = 'traditional', normalize_objectives: bool = True):
        """
        Initialize multi-objective fitness
        
        Args:
            visitor: TreeVisitor instance
            use_three_objectives: If True, includes test complexity as 3rd objective
            objective_type: 'traditional' (fitness vs coverage) or 
                          'conflicting' (coverage vs complexity)
            normalize_objectives: Whether to normalize objectives to [0, 1] range
        """
        super().__init__(visitor)
        self.use_three_objectives = use_three_objectives
        self.objective_type = objective_type
        self.normalize_objectives = normalize_objectives
        self.population_coverage = {}  # Track coverage for each solution
        self.population_fitness = {}   # Track fitness for each solution
        self.population_complexity = {}  # Track complexity for each solution
        self.particle_execution_times = {}  # Override parent's execution_times list with dict
        
        # Normalization bounds for each objective
        self.normalization_bounds = {
            'fitness': (0.0, 100.0),
            'coverage': (0.0, 1.0),
            'complexity': (0.0, 10.0)
        }
        
    def multi_objective_fitness(self, param: np.ndarray) -> np.ndarray:
        """
        Calculate multiple objectives for each solution
        
        Args:
            param: Population array with shape (n_particles, dimensions)
            
        Returns:
            Array of objectives with shape (n_particles, n_objectives)
        """
        n_particles = len(param)
        n_objectives = 3 if self.use_three_objectives else 2
        objectives = np.zeros((n_particles, n_objectives))
        
        for idx, particle in enumerate(param):
            # Calculate all base metrics
            fitness_value = self._calculate_single_fitness(particle)
            coverage = self._calculate_coverage(particle)
            complexity = self._calculate_complexity(particle)
            
            # Store for analysis
            self.population_fitness[idx] = fitness_value
            self.population_coverage[idx] = coverage
            self.population_complexity[idx] = complexity
            
            # Assign objectives based on type
            if self.objective_type == 'conflicting':
                # Conflicting objectives: coverage vs complexity
                obj1 = -coverage  # Maximize coverage (negate to minimize)
                obj2 = complexity  # Minimize complexity
                
                if self.normalize_objectives:
                    obj1 = self._normalize_objective(obj1, 'coverage', negate=True)
                    obj2 = self._normalize_objective(obj2, 'complexity')
                
                objectives[idx, 0] = obj1
                objectives[idx, 1] = obj2
                
                if self.use_three_objectives:
                    obj3 = fitness_value
                    if self.normalize_objectives:
                        obj3 = self._normalize_objective(obj3, 'fitness')
                    objectives[idx, 2] = obj3
                    
            else:  # traditional
                # Traditional: fitness vs coverage
                obj1 = fitness_value  # Minimize fitness
                obj2 = -coverage  # Maximize coverage (negate)
                
                if self.normalize_objectives:
                    obj1 = self._normalize_objective(obj1, 'fitness')
                    obj2 = self._normalize_objective(obj2, 'coverage', negate=True)
                
                objectives[idx, 0] = obj1
                objectives[idx, 1] = obj2
                
                if self.use_three_objectives:
                    obj3 = complexity  # Minimize complexity
                    if self.normalize_objectives:
                        obj3 = self._normalize_objective(obj3, 'complexity')
                    objectives[idx, 2] = obj3
        
        return objectives
    
    def _calculate_single_fitness(self, particle: np.ndarray) -> float:
        """
        Calculate traditional fitness for a single particle
        
        Args:
            particle: Single solution vector
            
        Returns:
            Fitness value
        """
        # Use parent class fitness calculation
        fitness_array = super().fitness_function(np.array([particle]))
        return fitness_array[0]
    
    def _calculate_coverage(self, particle: np.ndarray) -> float:
        """
        Calculate code coverage for a single particle
        
        Args:
            particle: Single solution vector
            
        Returns:
            Coverage percentage (0.0 to 1.0)
        """
        import time
        
        # Reset walked tree for this particle
        self.walked_tree = []
        self.current_walked_tree = []
        
        # Execute with particle to determine coverage
        particle_array = np.array([particle], np.float32)
        
        # Measure execution time
        start_time = time.perf_counter()
        
        # Resolve path to get coverage
        try:
            self.resolve_path(particle_array)
        except:
            # If execution fails, return 0 coverage
            return 0.0
        finally:
            # Record execution time in milliseconds
            execution_time = (time.perf_counter() - start_time) * 1000
            # Convert particle to hashable key
            particle_key = tuple(particle.tolist()) if hasattr(particle, 'tolist') else tuple(particle)
            self.particle_execution_times[particle_key] = execution_time
        
        # Calculate coverage
        if len(self.whole_tree) == 0:
            return 0.0
        
        unique_walked = set(self.walked_tree)
        coverage = len(unique_walked) / len(self.whole_tree)
        
        return coverage
    
    def _calculate_complexity(self, particle: np.ndarray) -> float:
        """
        Calculate test case complexity using entropy-based measure
        Higher complexity indicates more diverse/random inputs
        
        Args:
            particle: Single solution vector
            
        Returns:
            Complexity score (higher = more complex)
        """
        # Entropy-based complexity (main component)
        # Discretize values into bins for entropy calculation
        particle_normalized = (particle - particle.min()) / (particle.max() - particle.min() + 1e-10)
        hist, _ = np.histogram(particle_normalized, bins=10)
        hist = hist / (hist.sum() + 1e-10)  # Normalize to probabilities
        
        # Calculate Shannon entropy
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        # Additional complexity factors
        range_score = np.max(particle) - np.min(particle)  # Value range
        variance_score = np.var(particle)  # Variance
        unique_ratio = len(np.unique(np.round(particle, 2))) / len(particle)  # Uniqueness
        
        # Weighted combination (entropy is primary)
        complexity = entropy + 0.1 * np.log(1 + range_score) + 0.1 * np.log(1 + variance_score) + 0.2 * unique_ratio
        
        return complexity
    
    def _calculate_path_coverage(self, particle: np.ndarray) -> float:
        """
        Calculate path coverage for the test input
        
        Args:
            particle: Single solution vector
            
        Returns:
            Path coverage ratio
        """
        # Use the path coverage method from parent class if available
        if hasattr(super(), 'calculate_path_coverage'):
            return super().calculate_path_coverage(particle)
        
        # Fallback to regular coverage
        return self._calculate_coverage(particle)
    
    def _calculate_execution_time(self, particle: np.ndarray) -> float:
        """
        Get execution time for the test input
        
        Args:
            particle: Single solution vector
            
        Returns:
            Execution time in milliseconds
        """
        # Convert particle to hashable key
        particle_key = tuple(particle.tolist()) if hasattr(particle, 'tolist') else tuple(particle)
        
        # Return cached execution time if available
        if particle_key in self.particle_execution_times:
            return self.particle_execution_times[particle_key]
        
        # If not cached, run coverage calculation which measures time
        self._calculate_coverage(particle)
        
        # Return the measured time
        return self.particle_execution_times.get(particle_key, 0.0)
    
    def _normalize_objective(self, value: float, objective_name: str, negate: bool = False) -> float:
        """
        Normalize objective value to [0, 1] range
        
        Args:
            value: Raw objective value
            objective_name: Name of the objective for bounds lookup
            negate: Whether the value is negated (for maximization objectives)
            
        Returns:
            Normalized value in [0, 1] range
        """
        min_val, max_val = self.normalization_bounds.get(objective_name, (0.0, 1.0))
        
        # Handle negated values (e.g., -coverage)
        if negate:
            # Value is negative, convert back for normalization
            actual_value = -value
            normalized = (actual_value - min_val) / (max_val - min_val + 1e-10)
            # Return negated normalized value to maintain minimization
            return -min(max(normalized, 0.0), 1.0)
        else:
            # Regular normalization
            normalized = (value - min_val) / (max_val - min_val + 1e-10)
            return min(max(normalized, 0.0), 1.0)
    
    def get_pareto_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the current population
        
        Returns:
            Dictionary with population metrics
        """
        if not self.population_coverage:
            return {}
        
        coverages = list(self.population_coverage.values())
        fitnesses = list(self.population_fitness.values())
        
        return {
            'max_coverage': max(coverages) if coverages else 0,
            'mean_coverage': np.mean(coverages) if coverages else 0,
            'min_fitness': min(fitnesses) if fitnesses else float('inf'),
            'mean_fitness': np.mean(fitnesses) if fitnesses else float('inf'),
            'n_full_coverage': sum(1 for c in coverages if c >= 1.0)
        }


class MultiObjectiveProblem(Problem):
    """
    Multi-objective problem formulation for pymoo
    """
    
    def __init__(self, mo_fitness: MultiObjectiveFitness, dimensions: int,
                 n_objectives: int = 2, bounds: Optional[Tuple[float, float]] = None):
        """
        Initialize multi-objective problem
        
        Args:
            mo_fitness: MultiObjectiveFitness instance
            dimensions: Number of decision variables
            n_objectives: Number of objectives (2 or 3)
            bounds: Optional bounds tuple (lower, upper)
        """
        self.mo_fitness = mo_fitness
        
        # Set bounds
        if bounds:
            xl, xu = bounds
        else:
            xl = -999999
            xu = 999999
        
        super().__init__(
            n_var=dimensions,
            n_obj=n_objectives,
            n_ieq_constr=0,
            xl=xl,
            xu=xu
        )
    
    def _evaluate(self, x: np.ndarray, out: Dict[str, Any], *args, **kwargs):
        """
        Evaluate objectives for population
        
        Args:
            x: Population array
            out: Output dictionary for pymoo
        """
        # Calculate all objectives
        objectives = self.mo_fitness.multi_objective_fitness(x)
        out["F"] = objectives
    
    def get_name(self) -> str:
        """Get problem name"""
        return f"MultiObjectiveTestGeneration_{self.n_obj}obj"


class MOFitnessFactory:
    """Factory for creating multi-objective fitness problems"""
    
    @staticmethod
    def create_dual_objective(visitor: TreeVisitor, dimensions: int,
                             bounds: Optional[Tuple[float, float]] = None,
                             objective_type: str = 'traditional') -> MultiObjectiveProblem:
        """
        Create a dual-objective problem
        
        Args:
            visitor: TreeVisitor instance
            dimensions: Number of decision variables
            bounds: Optional bounds
            objective_type: 'traditional' or 'conflicting'
            
        Returns:
            MultiObjectiveProblem instance
        """
        mo_fitness = MultiObjectiveFitness(visitor, use_three_objectives=False, 
                                          objective_type=objective_type)
        return MultiObjectiveProblem(mo_fitness, dimensions, n_objectives=2, bounds=bounds)
    
    @staticmethod
    def create_three_objective(visitor: TreeVisitor, dimensions: int,
                              bounds: Optional[Tuple[float, float]] = None) -> MultiObjectiveProblem:
        """
        Create a three-objective problem (fitness + coverage + complexity)
        
        Args:
            visitor: TreeVisitor instance
            dimensions: Number of decision variables
            bounds: Optional bounds
            
        Returns:
            MultiObjectiveProblem instance
        """
        mo_fitness = MultiObjectiveFitness(visitor, use_three_objectives=True)
        return MultiObjectiveProblem(mo_fitness, dimensions, n_objectives=3, bounds=bounds)
    
    @staticmethod
    def create_from_config(visitor: TreeVisitor, config: Dict[str, Any]) -> MultiObjectiveProblem:
        """
        Create problem from configuration dictionary
        
        Args:
            visitor: TreeVisitor instance
            config: Configuration dictionary
            
        Returns:
            MultiObjectiveProblem instance
        """
        dimensions = config.get('dimensions', 4)
        n_objectives = config.get('n_objectives', 2)
        bounds = config.get('bounds', None)
        
        if n_objectives == 2:
            return MOFitnessFactory.create_dual_objective(visitor, dimensions, bounds)
        elif n_objectives == 3:
            return MOFitnessFactory.create_three_objective(visitor, dimensions, bounds)
        else:
            raise ValueError(f"Unsupported number of objectives: {n_objectives}")


# Fixed reference points for different objective types
REFERENCE_POINTS = {
    'traditional': np.array([100.0, 0.0]),  # High fitness, zero coverage (negated)
    'conflicting': np.array([0.0, 10.0]),   # Zero coverage (negated), high complexity
    'path-time': np.array([0.0, 1000.0]),   # Zero paths, 1000ms execution time
    'fault-size': np.array([0.0, 100.0]),   # Zero faults detected, 100 test inputs
}

def get_fixed_reference_point(objective_type: str) -> np.ndarray:
    """
    Get fixed reference point for given objective type
    
    Args:
        objective_type: Type of objectives being used
        
    Returns:
        Fixed reference point
    """
    return REFERENCE_POINTS.get(objective_type, REFERENCE_POINTS['traditional'])


def calculate_adaptive_reference_point(objectives: np.ndarray, percentile: float = 95) -> np.ndarray:
    """
    Calculate adaptive reference point based on objective values
    NOTE: This method is deprecated in favor of fixed reference points
    
    Args:
        objectives: Array of objective values from initial population
        percentile: Percentile to use for reference point (default 95)
        
    Returns:
        Adaptive reference point
    """
    # Calculate percentile for each objective
    ref_point = np.percentile(objectives, percentile, axis=0)
    
    # Add margin (10% buffer)
    ref_point = ref_point * 1.1
    
    # Ensure minimum values for stability
    ref_point = np.maximum(ref_point, np.array([1.0, 1.0] + [1.0] * (objectives.shape[1] - 2)))
    
    return ref_point


def calculate_hypervolume(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Calculate hypervolume indicator for Pareto front
    
    Args:
        pareto_front: Array of Pareto optimal solutions
        reference_point: Reference point for hypervolume calculation
        
    Returns:
        Hypervolume value
    """
    from pymoo.indicators.hv import HV
    
    indicator = HV(ref_point=reference_point)
    return indicator(pareto_front)


def calculate_igd(pareto_front: np.ndarray, true_pareto: np.ndarray) -> float:
    """
    Calculate Inverted Generational Distance
    
    Args:
        pareto_front: Obtained Pareto front
        true_pareto: True/reference Pareto front
        
    Returns:
        IGD value
    """
    from pymoo.indicators.igd import IGD
    
    indicator = IGD(true_pareto)
    return indicator(pareto_front)