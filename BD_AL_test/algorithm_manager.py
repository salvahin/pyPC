"""
Algorithm Manager Module
Provides a unified interface for managing and configuring pymoo algorithms
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import all pymoo algorithms
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.sres import SRES
from pymoo.algorithms.soo.nonconvex.isres import ISRES
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA

# Multi-objective algorithms (for future extension)
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.ctaea import CTAEA

from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UX
from pymoo.operators.crossover.hux import HUX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.selection.rnd import RandomSelection


class AlgorithmType(Enum):
    """Enumeration of algorithm types"""
    EVOLUTIONARY = "evolutionary"
    SWARM = "swarm"
    GRADIENT_FREE = "gradient_free"
    MULTI_OBJECTIVE = "multi_objective"
    HYBRID = "hybrid"


@dataclass
class AlgorithmConfig:
    """Configuration for an algorithm"""
    name: str
    type: AlgorithmType
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    reference: str = ""
    min_dimensions: int = 1
    max_dimensions: Optional[int] = None
    supports_constraints: bool = False
    supports_multi_objective: bool = False


class AlgorithmManager:
    """Manages and configures optimization algorithms"""
    
    def __init__(self):
        self._algorithms = self._initialize_algorithms()
        self._custom_configs = {}
        
    def _initialize_algorithms(self) -> Dict[str, AlgorithmConfig]:
        """Initialize all available algorithms with their default configurations"""
        algorithms = {
            # Swarm Intelligence
            "PSO": AlgorithmConfig(
                name="Particle Swarm Optimization",
                type=AlgorithmType.SWARM,
                params={"pop_size": 100, "w": 0.7, "c1": 2.0, "c2": 2.0},
                description="Classic swarm intelligence algorithm",
                reference="Kennedy & Eberhart, 1995"
            ),
            
            # Evolutionary Algorithms
            "GA": AlgorithmConfig(
                name="Genetic Algorithm",
                type=AlgorithmType.EVOLUTIONARY,
                params={"pop_size": 100, "eliminate_duplicates": True},
                description="Standard genetic algorithm with real-valued encoding",
                reference="Holland, 1975"
            ),
            
            "DE": AlgorithmConfig(
                name="Differential Evolution",
                type=AlgorithmType.EVOLUTIONARY,
                params={
                    "pop_size": 100,
                    "variant": "DE/rand/1/bin",
                    "CR": 0.7,
                    "F": 0.5,
                    "dither": "vector",
                    "jitter": False
                },
                description="Differential evolution for continuous optimization",
                reference="Storn & Price, 1997"
            ),
            
            "ES": AlgorithmConfig(
                name="Evolution Strategy",
                type=AlgorithmType.EVOLUTIONARY,
                params={"n_offsprings": 200, "rule": 1.0/7.0},
                description="(μ+λ) Evolution Strategy",
                reference="Rechenberg, 1973"
            ),
            
            "G3PCX": AlgorithmConfig(
                name="Generalized Generation Gap with PCX",
                type=AlgorithmType.EVOLUTIONARY,
                params={"pop_size": 100, "n_offsprings": 2},
                description="Real-coded GA with parent-centric crossover",
                reference="Deb et al., 2002"
            ),
            
            "BRKGA": AlgorithmConfig(
                name="Biased Random Key Genetic Algorithm",
                type=AlgorithmType.EVOLUTIONARY,
                params={
                    "n_elites": 20,
                    "n_offsprings": 70,
                    "n_mutants": 10,
                    "bias": 0.7
                },
                description="GA variant using random keys representation",
                reference="Gonçalves & Resende, 2011"
            ),
            
            # Gradient-Free Optimization
            "NelderMead": AlgorithmConfig(
                name="Nelder-Mead Simplex",
                type=AlgorithmType.GRADIENT_FREE,
                params={
                    "alpha": 1.0,
                    "beta": 2.0,
                    "gamma": 0.5,
                    "delta": 0.05
                },
                description="Direct search method using simplex",
                reference="Nelder & Mead, 1965"
            ),
            
            "PatternSearch": AlgorithmConfig(
                name="Pattern Search",
                type=AlgorithmType.GRADIENT_FREE,
                params={
                    "rho": 0.5,
                    "delta": 0.25,
                    "explr_delta": 0.25
                },
                description="Direct search using pattern exploration",
                reference="Hooke & Jeeves, 1961"
            ),
            
            "CMAES": AlgorithmConfig(
                name="Covariance Matrix Adaptation ES",
                type=AlgorithmType.EVOLUTIONARY,
                params={"sigma": 0.1, "restarts": 0},
                description="Advanced ES with covariance matrix adaptation",
                reference="Hansen & Ostermeier, 2001",
                min_dimensions=2  # CMAES requires at least 2 dimensions
            ),
            
            # Constrained Optimization
            "SRES": AlgorithmConfig(
                name="Stochastic Ranking ES",
                type=AlgorithmType.EVOLUTIONARY,
                params={
                    "n_offsprings": 200,
                    "rule": 1.0/7.0,
                    "gamma": 0.85,
                    "alpha": 0.2
                },
                description="ES with stochastic ranking for constraints",
                reference="Runarsson & Yao, 2000",
                supports_constraints=True
            ),
            
            "ISRES": AlgorithmConfig(
                name="Improved SRES",
                type=AlgorithmType.EVOLUTIONARY,
                params={
                    "n_offsprings": 200,
                    "rule": 1.0/7.0,
                    "gamma": 0.85,
                    "alpha": 0.2
                },
                description="Improved stochastic ranking ES",
                reference="Runarsson & Yao, 2005",
                supports_constraints=True
            ),
            
            # Multi-Objective Algorithms
            "NSGA2": AlgorithmConfig(
                name="Non-dominated Sorting GA II",
                type=AlgorithmType.MULTI_OBJECTIVE,
                params={
                    "pop_size": 100,
                    "eliminate_duplicates": True,
                    "n_offsprings": None  # Will be set to pop_size if None
                },
                description="Popular multi-objective evolutionary algorithm using non-dominated sorting",
                reference="Deb et al., 2002",
                supports_multi_objective=True
            ),
            
            "NSGA3": AlgorithmConfig(
                name="Non-dominated Sorting GA III",
                type=AlgorithmType.MULTI_OBJECTIVE,
                params={
                    "pop_size": 100,
                    "eliminate_duplicates": True,
                    "n_offsprings": None,
                    "ref_dirs": None  # Will be auto-generated based on n_obj
                },
                description="Many-objective optimization algorithm with reference directions",
                reference="Deb & Jain, 2014",
                supports_multi_objective=True
            ),
            
            "MOEAD": AlgorithmConfig(
                name="Multi-Objective EA with Decomposition",
                type=AlgorithmType.MULTI_OBJECTIVE,
                params={
                    "n_neighbors": 20,
                    "decomposition": "auto",  # auto, tchebi, pbi, weighted
                    "prob_neighbor_mating": 0.9,
                    "ref_dirs": None  # Will be auto-generated
                },
                description="Decomposition-based multi-objective algorithm",
                reference="Zhang & Li, 2007",
                supports_multi_objective=True
            ),
            
            "CTAEA": AlgorithmConfig(
                name="Constrained Two-Archive EA",
                type=AlgorithmType.MULTI_OBJECTIVE,
                params={
                    "pop_size": 100,
                    "ref_dirs": None,  # Will be auto-generated
                    "eliminate_duplicates": True
                },
                description="Two-archive algorithm for constrained multi-objective optimization",
                reference="Li et al., 2019",
                supports_multi_objective=True,
                supports_constraints=True
            ),
        }
        return algorithms
    
    def get_algorithm(self, name: str, custom_params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get an algorithm instance by name
        
        Args:
            name: Algorithm identifier (e.g., 'PSO', 'GA', 'DE')
            custom_params: Optional custom parameters to override defaults
            
        Returns:
            Configured algorithm instance
        """
        if name not in self._algorithms:
            raise ValueError(f"Algorithm '{name}' not found. Available: {list(self._algorithms.keys())}")
        
        config = self._algorithms[name]
        params = config.params.copy()
        
        # Override with custom parameters if provided
        if custom_params:
            params.update(custom_params)
        
        # Create algorithm instance based on name
        algorithm_map = {
            "PSO": PSO,
            "GA": GA,
            "DE": DE,
            "ES": ES,
            "G3PCX": G3PCX,
            "BRKGA": BRKGA,
            "NelderMead": NelderMead,
            "PatternSearch": PatternSearch,
            "CMAES": CMAES,
            "SRES": SRES,
            "ISRES": ISRES,
            "NSGA2": NSGA2,
            "NSGA3": NSGA3,
            "MOEAD": MOEAD,
            "CTAEA": CTAEA,
        }
        
        algorithm_class = algorithm_map.get(name)
        if not algorithm_class:
            raise ValueError(f"Algorithm class for '{name}' not implemented")
        
        # Special handling for CMAES which needs x0
        if name == "CMAES" and "x0" not in params:
            # x0 will need to be set based on problem dimensions
            return lambda dimensions: CMAES(x0=np.random.random(dimensions), **{k:v for k,v in params.items() if k != "x0"})
        
        # Special handling for DE variants
        if name == "DE":
            if "sampling" not in params:
                params["sampling"] = LHS()
        
        # Special handling for multi-objective algorithms with reference directions
        if name in ["NSGA3", "MOEAD", "CTAEA"]:
            if params.get("ref_dirs") is None:
                # Auto-generate reference directions based on number of objectives
                # This will be set when the problem is defined
                # For now, create a lambda that accepts n_obj
                filtered_params = {k: v for k, v in params.items() if k != "ref_dirs"}
                return lambda n_obj: self._create_mo_algorithm_with_ref_dirs(
                    algorithm_class, n_obj, filtered_params
                )
        
        return algorithm_class(**params)
    
    def _create_mo_algorithm_with_ref_dirs(self, algorithm_class: Any, 
                                          n_obj: int, params: Dict[str, Any]) -> Any:
        """
        Create multi-objective algorithm with auto-generated reference directions
        
        Args:
            algorithm_class: The algorithm class
            n_obj: Number of objectives
            params: Algorithm parameters
            
        Returns:
            Algorithm instance with reference directions
        """
        from pymoo.util.ref_dirs import get_reference_directions
        
        # Generate reference directions based on number of objectives
        if n_obj == 2:
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=99)
        elif n_obj == 3:
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
        else:
            # For many objectives, use fewer partitions
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=4)
        
        params["ref_dirs"] = ref_dirs
        return algorithm_class(**params)
    
    def get_algorithm_info(self, name: str) -> AlgorithmConfig:
        """Get configuration info for an algorithm"""
        if name not in self._algorithms:
            raise ValueError(f"Algorithm '{name}' not found")
        return self._algorithms[name]
    
    def list_algorithms(self, type_filter: Optional[AlgorithmType] = None) -> List[str]:
        """
        List available algorithms
        
        Args:
            type_filter: Optional filter by algorithm type
            
        Returns:
            List of algorithm names
        """
        if type_filter:
            return [name for name, config in self._algorithms.items() 
                   if config.type == type_filter]
        return list(self._algorithms.keys())
    
    def get_compatible_algorithms(self, dimensions: int, 
                                 has_constraints: bool = False,
                                 multi_objective: bool = False) -> List[str]:
        """
        Get algorithms compatible with problem characteristics
        
        Args:
            dimensions: Number of decision variables
            has_constraints: Whether problem has constraints
            multi_objective: Whether problem is multi-objective
            
        Returns:
            List of compatible algorithm names
        """
        compatible = []
        for name, config in self._algorithms.items():
            # Check dimension compatibility
            if dimensions < config.min_dimensions:
                continue
            if config.max_dimensions and dimensions > config.max_dimensions:
                continue
            
            # Check constraint compatibility
            if has_constraints and not config.supports_constraints:
                # Skip algorithms that don't handle constraints well
                if name not in ["GA", "DE", "PSO"]:  # These can handle soft constraints
                    continue
            
            # Check multi-objective compatibility
            if multi_objective and not config.supports_multi_objective:
                continue
            
            compatible.append(name)
        
        return compatible
    
    def create_batch(self, algorithm_names: List[str], 
                    common_params: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Any]]:
        """
        Create a batch of algorithms for comparison
        
        Args:
            algorithm_names: List of algorithm names
            common_params: Common parameters to apply to all
            
        Returns:
            List of (name, algorithm_instance) tuples
        """
        batch = []
        for name in algorithm_names:
            try:
                algo = self.get_algorithm(name, common_params)
                batch.append((name, algo))
            except Exception as e:
                print(f"Warning: Could not create {name}: {e}")
        return batch
    
    def register_custom_algorithm(self, name: str, config: AlgorithmConfig, 
                                algorithm_class: Any):
        """
        Register a custom algorithm
        
        Args:
            name: Unique identifier for the algorithm
            config: Algorithm configuration
            algorithm_class: The algorithm class or factory function
        """
        if name in self._algorithms:
            raise ValueError(f"Algorithm '{name}' already exists")
        
        self._algorithms[name] = config
        self._custom_configs[name] = algorithm_class
    
    def run_algorithm(self, algorithm_name: str, target_program: str,
                     generations: int = 100, population_size: int = 50) -> Dict[str, Any]:
        """
        Run a multi-objective algorithm on a target program
        
        Args:
            algorithm_name: Name of the algorithm to run
            target_program: Name of the target test program
            generations: Number of generations
            population_size: Population size
            
        Returns:
            Dictionary with algorithm execution results
        """
        import time
        import random
        from pymoo.optimize import minimize
        from pymoo.core.problem import Problem
        
        start_time = time.time()
        
        try:
            # Get the algorithm instance
            algorithm = self.get_algorithm(algorithm_name, {
                'pop_size': population_size
            })
            
            # Handle algorithms that need special initialization
            if algorithm_name in ['NSGA3', 'MOEAD', 'CTAEA'] and callable(algorithm):
                algorithm = algorithm(2)  # 2 objectives
            elif algorithm_name == 'CMAES' and callable(algorithm):
                algorithm = algorithm(2)  # 2 dimensions
            
            # Get program dimensions from evaluator
            from src.evaluation.evaluator import UnifiedTestEvaluator
            evaluator = UnifiedTestEvaluator()
            
            if target_program not in evaluator.test_programs:
                raise ValueError(f"Program '{target_program}' not found")
            
            program_config = evaluator.test_programs[target_program]
            dimensions = program_config.get('dimensions', 2)
            
            # Define a simple test optimization problem for smoke test
            class SimpleTestProblem(Problem):
                def __init__(self):
                    super().__init__(n_var=dimensions, n_obj=2, xl=-1000, xu=1000)
                
                def _evaluate(self, X, out, *args, **kwargs):
                    # Simple mock objectives for testing algorithm interface
                    n_particles = X.shape[0]
                    
                    # Objective 1: Minimize distance from origin (fitness-like)
                    obj1 = np.sqrt(np.sum(X**2, axis=1))
                    
                    # Objective 2: Maximize coverage (negative for minimization)
                    obj2 = -np.random.random(n_particles)  # Mock coverage
                    
                    out["F"] = np.column_stack([obj1, obj2])
            
            problem = SimpleTestProblem()
            
            # Run optimization
            result = minimize(
                problem=problem,
                algorithm=algorithm,
                termination=('n_gen', generations),
                verbose=False
            )
            
            execution_time = time.time() - start_time
            
            # Calculate metrics
            if result.F is not None and len(result.F) > 0:
                # Calculate hypervolume (simplified)
                ref_point = np.array([100.0, 0.0])  # Reference point for hypervolume
                try:
                    from pymoo.indicators.hv import HV
                    hv = HV(ref_point=ref_point)
                    hypervolume = hv(result.F)
                except:
                    hypervolume = 0.0
                
                # Calculate IGD (simplified - distance to ideal point)
                ideal_point = np.array([0.0, -1.0])
                distances = np.linalg.norm(result.F - ideal_point, axis=1)
                igd = np.mean(distances)
                
                best_fitness = float(np.min(result.F[:, 0])) if len(result.F) > 0 else float('inf')
                best_coverage = float(np.max(-result.F[:, 1])) if len(result.F) > 0 else 0.0
            else:
                hypervolume = 0.0
                igd = float('inf')
                best_fitness = float('inf')
                best_coverage = 0.0
            
            return {
                'success': True,
                'algorithm_name': algorithm_name,
                'target_program': target_program,
                'hypervolume': hypervolume,
                'igd': igd,
                'best_fitness': best_fitness,
                'best_coverage': best_coverage,
                'execution_time': execution_time,
                'generations': generations,
                'population_size': population_size,
                'final_population_size': len(result.F) if result.F is not None else 0
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'success': False,
                'algorithm_name': algorithm_name,
                'target_program': target_program,
                'error_message': str(e),
                'hypervolume': 0.0,
                'igd': float('inf'),
                'best_fitness': float('inf'),
                'best_coverage': 0.0,
                'execution_time': execution_time,
                'generations': generations,
                'population_size': population_size
            }
    
    def get_algorithm_summary(self) -> str:
        """Get a summary of all available algorithms"""
        summary = []
        summary.append("=" * 80)
        summary.append("AVAILABLE ALGORITHMS")
        summary.append("=" * 80)
        
        for type_ in AlgorithmType:
            algos = self.list_algorithms(type_)
            if algos:
                summary.append(f"\n{type_.value.upper().replace('_', ' ')}:")
                summary.append("-" * 40)
                for name in algos:
                    config = self._algorithms[name]
                    summary.append(f"  • {name}: {config.name}")
                    summary.append(f"    {config.description}")
                    if config.reference:
                        summary.append(f"    Reference: {config.reference}")
        
        summary.append("\n" + "=" * 80)
        return "\n".join(summary)