"""
Algorithm Factory Module
Creates algorithm instances from configuration
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from algorithm_manager import AlgorithmManager
from config_loader import ConfigLoader


class AlgorithmFactory:
    """Factory for creating algorithm instances from configuration"""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        """
        Initialize algorithm factory
        
        Args:
            config_loader: Optional config loader instance
        """
        self.config_loader = config_loader or ConfigLoader()
        self.algorithm_manager = AlgorithmManager()
        self._cache = {}  # Cache for algorithm instances
        
    def create_algorithm(self, name: str, 
                        variant: Optional[str] = None,
                        custom_params: Optional[Dict[str, Any]] = None,
                        dimensions: Optional[int] = None) -> Any:
        """
        Create an algorithm instance
        
        Args:
            name: Algorithm name from configuration
            variant: Optional variant name
            custom_params: Optional custom parameters
            dimensions: Number of dimensions (required for some algorithms)
            
        Returns:
            Algorithm instance
        """
        # Get parameters from configuration
        config_params = self.config_loader.get_algorithm_params(name, variant)
        
        # Merge with custom parameters if provided
        params = config_params.copy()
        if custom_params:
            params.update(custom_params)
        
        # Handle special cases
        if name == "CMAES":
            if dimensions is None:
                raise ValueError("CMAES requires dimensions parameter")
            # CMAES needs x0 parameter
            algo_factory = self.algorithm_manager.get_algorithm(name, params)
            return algo_factory(dimensions)  # Call the lambda function with dimensions
        
        # Create algorithm instance
        return self.algorithm_manager.get_algorithm(name, params)
    
    def create_from_experiment(self, experiment_name: str, 
                              dimensions: Optional[int] = None) -> List[Tuple[str, Any]]:
        """
        Create algorithms from experiment configuration
        
        Args:
            experiment_name: Name of the experiment
            dimensions: Number of dimensions for the problem
            
        Returns:
            List of (name, algorithm) tuples
        """
        exp_config = self.config_loader.get_experiment_config(experiment_name)
        algorithms_config = exp_config.get('algorithms', [])
        
        algorithms = []
        
        # Handle different algorithm configuration formats
        if isinstance(algorithms_config, str):
            if algorithms_config.startswith("preset:"):
                # Use preset
                preset_name = algorithms_config.replace("preset:", "")
                algo_names = self.config_loader.get_preset_algorithms(preset_name)
                for name in algo_names:
                    try:
                        algo = self.create_algorithm(name, dimensions=dimensions)
                        algorithms.append((name, algo))
                    except Exception as e:
                        print(f"Warning: Could not create {name}: {e}")
                        
            elif algorithms_config == "all":
                # Use all enabled algorithms
                algo_names = self.config_loader.get_enabled_algorithms()
                for name in algo_names:
                    try:
                        algo = self.create_algorithm(name, dimensions=dimensions)
                        algorithms.append((name, algo))
                    except Exception as e:
                        print(f"Warning: Could not create {name}: {e}")
                        
        elif isinstance(algorithms_config, list):
            # List of algorithm names
            for name in algorithms_config:
                try:
                    algo = self.create_algorithm(name, dimensions=dimensions)
                    algorithms.append((name, algo))
                except Exception as e:
                    print(f"Warning: Could not create {name}: {e}")
                    
        elif isinstance(algorithms_config, dict):
            # Dictionary with algorithm configurations
            for name, configs in algorithms_config.items():
                if isinstance(configs, list):
                    # Multiple configurations for the same algorithm
                    for i, config in enumerate(configs):
                        try:
                            algo = self.create_algorithm(name, custom_params=config, 
                                                       dimensions=dimensions)
                            algo_name = f"{name}_config{i+1}"
                            algorithms.append((algo_name, algo))
                        except Exception as e:
                            print(f"Warning: Could not create {name} with config {i+1}: {e}")
                else:
                    # Single configuration
                    try:
                        algo = self.create_algorithm(name, custom_params=configs, 
                                                   dimensions=dimensions)
                        algorithms.append((name, algo))
                    except Exception as e:
                        print(f"Warning: Could not create {name}: {e}")
        
        return algorithms
    
    def create_from_preset(self, preset_name: str, 
                          dimensions: Optional[int] = None) -> List[Tuple[str, Any]]:
        """
        Create algorithms from a preset
        
        Args:
            preset_name: Name of the preset
            dimensions: Number of dimensions for the problem
            
        Returns:
            List of (name, algorithm) tuples
        """
        algo_names = self.config_loader.get_preset_algorithms(preset_name)
        algorithms = []
        
        for name in algo_names:
            try:
                algo = self.create_algorithm(name, dimensions=dimensions)
                algorithms.append((name, algo))
            except Exception as e:
                print(f"Warning: Could not create {name}: {e}")
        
        return algorithms
    
    def create_with_variants(self, algorithm_name: str,
                           dimensions: Optional[int] = None) -> List[Tuple[str, Any]]:
        """
        Create all variants of an algorithm
        
        Args:
            algorithm_name: Name of the algorithm
            dimensions: Number of dimensions
            
        Returns:
            List of (variant_name, algorithm) tuples
        """
        algorithms = []
        algo_config = self.config_loader.get_algorithm_config(algorithm_name)
        
        # Create default version
        try:
            default_algo = self.create_algorithm(algorithm_name, dimensions=dimensions)
            algorithms.append((f"{algorithm_name}_default", default_algo))
        except Exception as e:
            print(f"Warning: Could not create default {algorithm_name}: {e}")
        
        # Create variants if available
        if 'variants' in algo_config:
            for variant_name in algo_config['variants']:
                try:
                    variant_algo = self.create_algorithm(algorithm_name, 
                                                        variant=variant_name,
                                                        dimensions=dimensions)
                    algorithms.append((f"{algorithm_name}_{variant_name}", variant_algo))
                except Exception as e:
                    print(f"Warning: Could not create {algorithm_name} variant {variant_name}: {e}")
        
        return algorithms
    
    def create_population_scaled(self, algorithm_name: str,
                                population_sizes: List[int],
                                dimensions: Optional[int] = None) -> List[Tuple[str, Any]]:
        """
        Create algorithms with different population sizes
        
        Args:
            algorithm_name: Name of the algorithm
            population_sizes: List of population sizes to test
            dimensions: Number of dimensions
            
        Returns:
            List of (name_with_pop_size, algorithm) tuples
        """
        algorithms = []
        
        for pop_size in population_sizes:
            try:
                algo = self.create_algorithm(
                    algorithm_name,
                    custom_params={'pop_size': pop_size},
                    dimensions=dimensions
                )
                algorithms.append((f"{algorithm_name}_pop{pop_size}", algo))
            except Exception as e:
                print(f"Warning: Could not create {algorithm_name} with pop_size {pop_size}: {e}")
        
        return algorithms
    
    def get_compatible_algorithms(self, test_program_name: str) -> List[str]:
        """
        Get algorithms compatible with a test program
        
        Args:
            test_program_name: Name of the test program
            
        Returns:
            List of compatible algorithm names
        """
        prog_config = self.config_loader.get_test_program_config(test_program_name)
        dimensions = prog_config['dimensions']
        
        # Check for special requirements
        has_constraints = prog_config.get('has_constraints', False)
        
        return self.algorithm_manager.get_compatible_algorithms(
            dimensions=dimensions,
            has_constraints=has_constraints
        )
    
    def validate_algorithm_config(self, name: str, 
                                 params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate algorithm configuration
        
        Args:
            name: Algorithm name
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            algo_config = self.config_loader.get_algorithm_config(name)
        except ValueError as e:
            errors.append(str(e))
            return False, errors
        
        # Check required parameters
        default_params = algo_config.get('default_params', {})
        for param_name in default_params:
            if param_name in params:
                # Type checking could be added here
                param_value = params[param_name]
                default_value = default_params[param_name]
                
                # Basic type validation
                if default_value is not None and param_value is not None:
                    if not isinstance(param_value, type(default_value)):
                        # Allow int/float conversion
                        if not (isinstance(default_value, (int, float)) and 
                               isinstance(param_value, (int, float))):
                            errors.append(
                                f"Parameter '{param_name}' type mismatch: "
                                f"expected {type(default_value).__name__}, "
                                f"got {type(param_value).__name__}"
                            )
        
        # Check dimension requirements
        if 'requirements' in algo_config:
            min_dims = algo_config['requirements'].get('min_dimensions')
            if min_dims and 'dimensions' in params:
                if params['dimensions'] < min_dims:
                    errors.append(
                        f"Algorithm {name} requires at least {min_dims} dimensions"
                    )
        
        return len(errors) == 0, errors
    
    def batch_create(self, configurations: List[Dict[str, Any]]) -> List[Tuple[str, Any]]:
        """
        Create multiple algorithms from a list of configurations
        
        Args:
            configurations: List of configuration dictionaries
                Each should have 'name' and optionally 'variant', 'params', 'dimensions'
                
        Returns:
            List of (name, algorithm) tuples
        """
        algorithms = []
        
        for config in configurations:
            name = config.get('name')
            if not name:
                print("Warning: Configuration missing 'name' field")
                continue
            
            try:
                algo = self.create_algorithm(
                    name=name,
                    variant=config.get('variant'),
                    custom_params=config.get('params'),
                    dimensions=config.get('dimensions')
                )
                
                # Create descriptive name
                algo_name = name
                if config.get('variant'):
                    algo_name += f"_{config['variant']}"
                if config.get('label'):
                    algo_name = config['label']
                    
                algorithms.append((algo_name, algo))
            except Exception as e:
                print(f"Warning: Could not create algorithm from config: {e}")
        
        return algorithms
    
    def get_algorithm_info(self, name: str) -> str:
        """
        Get information about an algorithm
        
        Args:
            name: Algorithm name
            
        Returns:
            Information string
        """
        algo_config = self.config_loader.get_algorithm_config(name)
        algo_info = self.algorithm_manager.get_algorithm_info(name)
        
        info_lines = []
        info_lines.append(f"Algorithm: {algo_info.name}")
        info_lines.append(f"Type: {algo_info.type.value}")
        info_lines.append(f"Description: {algo_info.description}")
        
        if algo_info.reference:
            info_lines.append(f"Reference: {algo_info.reference}")
        
        info_lines.append(f"Default Parameters: {algo_config['default_params']}")
        
        if 'variants' in algo_config:
            info_lines.append(f"Available Variants: {list(algo_config['variants'].keys())}")
        
        if 'notes' in algo_config:
            info_lines.append(f"Notes: {algo_config['notes']}")
        
        return "\n".join(info_lines)