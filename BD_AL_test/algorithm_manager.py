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
            
            # Multi-Objective (for future use)
            "NSGA2": AlgorithmConfig(
                name="Non-dominated Sorting GA II",
                type=AlgorithmType.MULTI_OBJECTIVE,
                params={"pop_size": 100, "eliminate_duplicates": True},
                description="Popular multi-objective evolutionary algorithm",
                reference="Deb et al., 2002",
                supports_multi_objective=True
            ),
            
            "NSGA3": AlgorithmConfig(
                name="Non-dominated Sorting GA III",
                type=AlgorithmType.MULTI_OBJECTIVE,
                params={"pop_size": 100, "eliminate_duplicates": True},
                description="Many-objective optimization algorithm",
                reference="Deb & Jain, 2014",
                supports_multi_objective=True
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