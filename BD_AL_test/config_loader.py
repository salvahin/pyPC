"""
Configuration Loader Module
Loads and validates YAML configuration files for the testing framework
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path


class ConfigLoader:
    """Loads and manages configuration from YAML files"""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration loader
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.algorithms_config = None
        self.experiments_config = None
        self.test_programs_config = None
        self._loaded = False
        
    def load_all(self) -> None:
        """Load all configuration files"""
        self.algorithms_config = self.load_algorithms()
        self.experiments_config = self.load_experiments()
        self.test_programs_config = self.load_test_programs()
        self._loaded = True
        
    def load_algorithms(self) -> Dict[str, Any]:
        """Load algorithm configurations"""
        config_path = self.config_dir / "algorithms.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Algorithm config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self._validate_algorithms_config(config)
        return config
    
    def load_experiments(self) -> Dict[str, Any]:
        """Load experiment configurations"""
        config_path = self.config_dir / "experiments.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self._validate_experiments_config(config)
        return config
    
    def load_test_programs(self) -> Dict[str, Any]:
        """Load test program configurations"""
        config_path = self.config_dir / "test_programs.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Test programs config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self._validate_test_programs_config(config)
        return config
    
    def get_algorithm_config(self, name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific algorithm
        
        Args:
            name: Algorithm name
            
        Returns:
            Algorithm configuration dictionary
        """
        if not self.algorithms_config:
            self.algorithms_config = self.load_algorithms()
        
        algorithms = self.algorithms_config.get('algorithms', {})
        if name not in algorithms:
            raise ValueError(f"Algorithm '{name}' not found in configuration")
        
        return algorithms[name]
    
    def get_algorithm_params(self, name: str, variant: Optional[str] = None) -> Dict[str, Any]:
        """
        Get parameters for an algorithm
        
        Args:
            name: Algorithm name
            variant: Optional variant name
            
        Returns:
            Parameter dictionary
        """
        algo_config = self.get_algorithm_config(name)
        
        if variant and 'variants' in algo_config:
            if variant in algo_config['variants']:
                # Merge default params with variant params
                params = algo_config.get('default_params', {}).copy()
                params.update(algo_config['variants'][variant])
                return params
            else:
                raise ValueError(f"Variant '{variant}' not found for algorithm '{name}'")
        
        return algo_config.get('default_params', {})
    
    def get_preset_algorithms(self, preset_name: str) -> List[str]:
        """
        Get list of algorithms from a preset
        
        Args:
            preset_name: Name of the preset
            
        Returns:
            List of algorithm names
        """
        if not self.algorithms_config:
            self.algorithms_config = self.load_algorithms()
        
        presets = self.algorithms_config.get('presets', {})
        if preset_name not in presets:
            raise ValueError(f"Preset '{preset_name}' not found")
        
        return presets[preset_name].get('algorithms', [])
    
    def get_experiment_config(self, name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific experiment
        
        Args:
            name: Experiment name
            
        Returns:
            Experiment configuration dictionary
        """
        if not self.experiments_config:
            self.experiments_config = self.load_experiments()
        
        experiments = self.experiments_config.get('experiments', {})
        if name not in experiments:
            raise ValueError(f"Experiment '{name}' not found in configuration")
        
        return experiments[name]
    
    def get_test_program_config(self, name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific test program
        
        Args:
            name: Test program name
            
        Returns:
            Test program configuration dictionary
        """
        if not self.test_programs_config:
            self.test_programs_config = self.load_test_programs()
        
        programs = self.test_programs_config.get('test_programs', {})
        if name not in programs:
            raise ValueError(f"Test program '{name}' not found in configuration")
        
        return programs[name]
    
    def get_test_suite_programs(self, suite_name: str) -> List[str]:
        """
        Get list of programs in a test suite
        
        Args:
            suite_name: Name of the test suite
            
        Returns:
            List of program names
        """
        if not self.test_programs_config:
            self.test_programs_config = self.load_test_programs()
        
        suites = self.test_programs_config.get('test_suites', {})
        if suite_name not in suites:
            raise ValueError(f"Test suite '{suite_name}' not found")
        
        return suites[suite_name].get('programs', [])
    
    def get_enabled_algorithms(self) -> List[str]:
        """Get list of enabled algorithms"""
        if not self.algorithms_config:
            self.algorithms_config = self.load_algorithms()
        
        algorithms = self.algorithms_config.get('algorithms', {})
        return [name for name, config in algorithms.items() 
                if config.get('enabled', True)]
    
    def get_global_settings(self) -> Dict[str, Any]:
        """Get global experiment settings"""
        if not self.experiments_config:
            self.experiments_config = self.load_experiments()
        
        return self.experiments_config.get('global_settings', {})
    
    def get_termination_criteria(self) -> Dict[str, Any]:
        """Get termination criteria settings"""
        if not self.experiments_config:
            self.experiments_config = self.load_experiments()
        
        return self.experiments_config.get('termination', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        if not self.experiments_config:
            self.experiments_config = self.load_experiments()
        
        return self.experiments_config.get('output', {})
    
    def _validate_algorithms_config(self, config: Dict[str, Any]) -> None:
        """Validate algorithm configuration structure"""
        if 'algorithms' not in config:
            raise ValueError("Missing 'algorithms' section in algorithms.yaml")
        
        for name, algo_config in config['algorithms'].items():
            required_fields = ['name', 'type', 'default_params']
            for field in required_fields:
                if field not in algo_config:
                    raise ValueError(f"Algorithm '{name}' missing required field '{field}'")
    
    def _validate_experiments_config(self, config: Dict[str, Any]) -> None:
        """Validate experiment configuration structure"""
        if 'experiments' not in config:
            raise ValueError("Missing 'experiments' section in experiments.yaml")
        
        for name, exp_config in config['experiments'].items():
            required_fields = ['name', 'algorithms', 'test_programs']
            for field in required_fields:
                if field not in exp_config:
                    raise ValueError(f"Experiment '{name}' missing required field '{field}'")
    
    def _validate_test_programs_config(self, config: Dict[str, Any]) -> None:
        """Validate test program configuration structure"""
        if 'test_programs' not in config:
            raise ValueError("Missing 'test_programs' section in test_programs.yaml")
        
        for name, prog_config in config['test_programs'].items():
            required_fields = ['path', 'dimensions', 'category']
            for field in required_fields:
                if field not in prog_config:
                    raise ValueError(f"Test program '{name}' missing required field '{field}'")
    
    def save_config(self, config_type: str, config: Dict[str, Any]) -> None:
        """
        Save configuration back to file
        
        Args:
            config_type: Type of config ('algorithms', 'experiments', 'test_programs')
            config: Configuration dictionary to save
        """
        config_path = self.config_dir / f"{config_type}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def get_config_summary(self) -> str:
        """Get a summary of loaded configurations"""
        if not self._loaded:
            self.load_all()
        
        summary = []
        summary.append("=" * 60)
        summary.append("CONFIGURATION SUMMARY")
        summary.append("=" * 60)
        
        # Algorithms
        algorithms = self.algorithms_config.get('algorithms', {})
        enabled_algos = self.get_enabled_algorithms()
        summary.append(f"\nAlgorithms: {len(algorithms)} defined, {len(enabled_algos)} enabled")
        summary.append(f"Enabled: {', '.join(enabled_algos)}")
        
        # Presets
        presets = self.algorithms_config.get('presets', {})
        summary.append(f"\nPresets: {len(presets)} available")
        for name, preset in presets.items():
            summary.append(f"  • {name}: {preset['description']}")
        
        # Experiments
        experiments = self.experiments_config.get('experiments', {})
        summary.append(f"\nExperiments: {len(experiments)} defined")
        for name, exp in experiments.items():
            summary.append(f"  • {name}: {exp.get('description', 'No description')}")
        
        # Test Programs
        programs = self.test_programs_config.get('test_programs', {})
        suites = self.test_programs_config.get('test_suites', {})
        summary.append(f"\nTest Programs: {len(programs)} available")
        summary.append(f"Test Suites: {len(suites)} defined")
        
        # Categories
        categories = {}
        for prog in programs.values():
            cat = prog.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        summary.append("\nPrograms by category:")
        for cat, count in categories.items():
            summary.append(f"  • {cat}: {count}")
        
        summary.append("\n" + "=" * 60)
        return "\n".join(summary)