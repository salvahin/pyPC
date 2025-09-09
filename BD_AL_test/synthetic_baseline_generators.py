#!/usr/bin/env python3
"""
Enhanced Baseline Test Generators for Synthetic Dataset
Specialized generators for complex parameter types and structures
"""

import numpy as np
import time
import json
import string
import random
from typing import List, Tuple, Optional, Dict, Any, Union
from scipy.stats import qmc
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from baseline_test_generators import TestSuite, BaselineTestGenerator


class SyntheticInputGenerator:
    """Generates complex input structures for synthetic functions"""
    
    def __init__(self, param_ranges: List[Dict[str, Any]], seed: Optional[int] = None):
        self.param_ranges = param_ranges
        self.rng = np.random.RandomState(seed) if seed else np.random
        
    def generate_test_inputs(self, n_tests: int) -> List[Tuple[Any, ...]]:
        """Generate test inputs based on parameter specifications"""
        test_inputs = []
        
        for _ in range(n_tests):
            inputs = []
            for param_spec in self.param_ranges:
                param_value = self._generate_parameter(param_spec)
                inputs.append(param_value)
            test_inputs.append(tuple(inputs))
            
        return test_inputs
    
    def _generate_parameter(self, param_spec: Dict[str, Any]) -> Any:
        """Generate a single parameter value based on specification"""
        param_type = param_spec.get('type', 'dict')
        
        if param_type == 'dict':
            return self._generate_dict(param_spec)
        elif param_type == 'list':
            return self._generate_list(param_spec)
        elif param_type == 'matrix':
            return self._generate_matrix(param_spec)
        elif param_type == 'string':
            return self._generate_string(param_spec)
        elif param_type == 'bytes':
            return self._generate_bytes(param_spec)
        elif param_type == 'int':
            return self._generate_int(param_spec)
        elif param_type == 'float':
            return self._generate_float(param_spec)
        else:
            return {}
    
    def _generate_dict(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dictionary parameter"""
        result = {}
        
        # Add required keys
        required_keys = spec.get('required_keys', [])
        for key in required_keys:
            if key in spec:
                if isinstance(spec[key], list) and len(spec[key]) == 2:
                    # Range specification
                    result[key] = self.rng.randint(spec[key][0], spec[key][1] + 1)
                else:
                    result[key] = spec[key]
            else:
                result[key] = self._generate_default_value(key)
        
        # Add optional keys randomly
        optional_keys = ['timeout', 'max_depth', 'threshold', 'iterations']
        for key in optional_keys:
            if self.rng.random() < 0.3:  # 30% chance
                result[key] = self._generate_default_value(key)
        
        return result
    
    def _generate_list(self, spec: Dict[str, Any]) -> List[Any]:
        """Generate list parameter"""
        min_length = spec.get('min_length', 0)
        max_length = spec.get('max_length', 10)
        element_type = spec.get('element_type', 'int')
        
        length = self.rng.randint(min_length, max_length + 1)
        
        if element_type == 'int':
            range_spec = spec.get('range', [-100, 100])
            return [self.rng.randint(range_spec[0], range_spec[1]) for _ in range(length)]
        elif element_type == 'float':
            range_spec = spec.get('range', [-100.0, 100.0])
            return [self.rng.uniform(range_spec[0], range_spec[1]) for _ in range(length)]
        elif element_type == 'string':
            return [self._generate_random_string() for _ in range(length)]
        elif element_type == 'dict':
            operation_types = spec.get('operation_types', ['insert', 'delete', 'search'])
            return [{'operation': self.rng.choice(operation_types), 
                    'value': self.rng.randint(0, 1000)} for _ in range(length)]
        else:
            return [0] * length
    
    def _generate_matrix(self, spec: Dict[str, Any]) -> List[List[float]]:
        """Generate matrix parameter"""
        dimensions = spec.get('dimensions', [2, 5])
        element_range = spec.get('element_range', [-10, 10])
        
        if isinstance(dimensions, list) and len(dimensions) == 2:
            rows = self.rng.randint(dimensions[0], dimensions[1] + 1)
            cols = self.rng.randint(dimensions[0], dimensions[1] + 1)
        else:
            rows = cols = dimensions
        
        return [[self.rng.uniform(element_range[0], element_range[1]) 
                for _ in range(cols)] for _ in range(rows)]
    
    def _generate_string(self, spec: Dict[str, Any]) -> str:
        """Generate string parameter"""
        max_length = spec.get('max_length', 100)
        patterns = spec.get('patterns', ['normal'])
        
        if 'json' in patterns:
            return self._generate_json_string()
        elif 'malformed' in patterns:
            return self._generate_malformed_string()
        else:
            length = self.rng.randint(0, max_length)
            return ''.join(self.rng.choice(string.ascii_letters + string.digits) 
                          for _ in range(length))
    
    def _generate_bytes(self, spec: Dict[str, Any]) -> bytes:
        """Generate bytes parameter"""
        min_length = spec.get('min_length', 0)
        max_length = spec.get('max_length', 100)
        
        length = self.rng.randint(min_length, max_length + 1)
        return bytes(self.rng.randint(0, 256, length))
    
    def _generate_int(self, spec: Dict[str, Any]) -> int:
        """Generate integer parameter"""
        range_spec = spec.get('range', [0, 100])
        return self.rng.randint(range_spec[0], range_spec[1] + 1)
    
    def _generate_float(self, spec: Dict[str, Any]) -> float:
        """Generate float parameter"""
        range_spec = spec.get('range', [0.0, 100.0])
        return self.rng.uniform(range_spec[0], range_spec[1])
    
    def _generate_default_value(self, key: str) -> Any:
        """Generate default value based on key name"""
        if 'timeout' in key:
            return self.rng.randint(1, 100)
        elif 'depth' in key or 'level' in key:
            return self.rng.randint(1, 10)
        elif 'count' in key or 'size' in key:
            return self.rng.randint(1, 100)
        elif 'rate' in key or 'ratio' in key:
            return self.rng.uniform(0.0, 1.0)
        else:
            return self.rng.randint(0, 100)
    
    def _generate_json_string(self) -> str:
        """Generate valid JSON string"""
        data = {
            'id': self.rng.randint(1, 1000),
            'name': f'item_{self.rng.randint(1, 100)}',
            'values': [self.rng.randint(0, 100) for _ in range(self.rng.randint(1, 5))],
            'config': {'enabled': self.rng.choice([True, False])}
        }
        return json.dumps(data)
    
    def _generate_malformed_string(self) -> str:
        """Generate malformed string for testing validation"""
        templates = [
            '{"key": value}',  # Missing quotes
            '{"key": "value",}',  # Trailing comma
            '{key: "value"}',  # Unquoted key
            '{"key": "value"',  # Missing closing brace
            '<script>alert("xss")</script>',  # XSS attempt
            '{"key": "' + 'A' * 1000 + '"}',  # Very long value
        ]
        return self.rng.choice(templates)
    
    def _generate_random_string(self) -> str:
        """Generate random string of moderate length"""
        length = self.rng.randint(3, 20)
        return ''.join(self.rng.choice(string.ascii_letters) for _ in range(length))


class AdvancedBaselineGenerators:
    """Advanced baseline generators for synthetic dataset"""
    
    def __init__(self, param_ranges: List[Dict[str, Any]], seed: Optional[int] = None):
        self.param_ranges = param_ranges
        self.input_generator = SyntheticInputGenerator(param_ranges, seed)
        self.rng = np.random.RandomState(seed) if seed else np.random
    
    def stratified_random(self, n_tests: int) -> TestSuite:
        """
        Stratified random sampling ensuring coverage of parameter space regions
        """
        start_time = time.time()
        
        # Divide parameter space into strata
        n_strata = min(8, n_tests)  # Maximum 8 strata
        tests_per_stratum = n_tests // n_strata
        remaining_tests = n_tests % n_strata
        
        test_inputs = []
        
        for stratum in range(n_strata):
            # Define stratum boundaries
            stratum_tests = tests_per_stratum + (1 if stratum < remaining_tests else 0)
            
            # Generate tests within this stratum
            for _ in range(stratum_tests):
                # Modify random seed for stratification
                old_state = self.rng.get_state()
                self.rng.seed((stratum * 1000) + self.rng.randint(0, 1000))
                
                inputs = self.input_generator.generate_test_inputs(1)[0]
                test_inputs.append(inputs)
                
                # Restore random state
                self.rng.set_state(old_state)
        
        return TestSuite(
            test_cases=np.array(test_inputs, dtype=object),
            method="Stratified Random",
            generation_time=time.time() - start_time,
            metadata={'n_strata': n_strata}
        )
    
    def importance_sampling(self, n_tests: int) -> TestSuite:
        """
        Importance sampling focusing on complex parameter combinations
        """
        start_time = time.time()
        
        test_inputs = []
        
        # 50% focus on complex parameters, 50% on simple ones
        complex_tests = n_tests // 2
        simple_tests = n_tests - complex_tests
        
        # Generate complex parameter combinations
        for _ in range(complex_tests):
            inputs = []
            for param_spec in self.param_ranges:
                param_value = self._generate_complex_parameter(param_spec)
                inputs.append(param_value)
            test_inputs.append(tuple(inputs))
        
        # Generate simple parameter combinations
        for _ in range(simple_tests):
            inputs = []
            for param_spec in self.param_ranges:
                param_value = self._generate_simple_parameter(param_spec)
                inputs.append(param_value)
            test_inputs.append(tuple(inputs))
        
        return TestSuite(
            test_cases=np.array(test_inputs, dtype=object),
            method="Importance Sampling",
            generation_time=time.time() - start_time,
            metadata={'complex_ratio': 0.5}
        )
    
    def coverage_guided_random(self, n_tests: int, coverage_targets: List[str] = None) -> TestSuite:
        """
        Random generation guided by coverage targets
        """
        start_time = time.time()
        
        if coverage_targets is None:
            coverage_targets = ['error_paths', 'boundary_conditions', 'edge_cases']
        
        test_inputs = []
        tests_per_target = n_tests // len(coverage_targets)
        
        for target in coverage_targets:
            for _ in range(tests_per_target):
                inputs = self._generate_coverage_targeted_inputs(target)
                test_inputs.append(inputs)
        
        # Fill remaining tests with random
        remaining = n_tests - len(test_inputs)
        for _ in range(remaining):
            inputs = self.input_generator.generate_test_inputs(1)[0]
            test_inputs.append(inputs)
        
        return TestSuite(
            test_cases=np.array(test_inputs, dtype=object),
            method="Coverage-Guided Random",
            generation_time=time.time() - start_time,
            metadata={'targets': coverage_targets}
        )
    
    def multi_stage_hybrid(self, n_tests: int) -> TestSuite:
        """
        Multi-stage generation combining different strategies
        """
        start_time = time.time()
        
        # Stage 1: Boundary value analysis (20%)
        boundary_tests = max(1, n_tests // 5)
        boundary_inputs = self._generate_boundary_tests(boundary_tests)
        
        # Stage 2: Quasi-random sampling (30%)
        quasi_tests = max(1, (n_tests * 3) // 10)
        quasi_inputs = self._generate_quasi_random_tests(quasi_tests)
        
        # Stage 3: Error-focused generation (25%)
        error_tests = max(1, n_tests // 4)
        error_inputs = self._generate_error_focused_tests(error_tests)
        
        # Stage 4: Random fill (remaining)
        remaining = n_tests - boundary_tests - quasi_tests - error_tests
        random_inputs = self.input_generator.generate_test_inputs(remaining)
        
        # Combine all stages
        all_inputs = boundary_inputs + quasi_inputs + error_inputs + random_inputs
        
        # Shuffle to avoid bias
        self.rng.shuffle(all_inputs)
        
        return TestSuite(
            test_cases=np.array(all_inputs, dtype=object),
            method="Multi-Stage Hybrid",
            generation_time=time.time() - start_time,
            metadata={
                'boundary_tests': boundary_tests,
                'quasi_tests': quasi_tests,
                'error_tests': error_tests,
                'random_tests': remaining
            }
        )
    
    def adaptive_parameter_space(self, n_tests: int, adaptation_rounds: int = 3) -> TestSuite:
        """
        Adaptive exploration of parameter space based on feedback
        """
        start_time = time.time()
        
        all_inputs = []
        tests_per_round = n_tests // adaptation_rounds
        
        for round_num in range(adaptation_rounds):
            if round_num == 0:
                # Initial random exploration
                round_inputs = self.input_generator.generate_test_inputs(tests_per_round)
            else:
                # Adaptive exploration based on previous rounds
                round_inputs = self._adaptive_round_generation(tests_per_round, all_inputs)
            
            all_inputs.extend(round_inputs)
        
        # Fill remaining tests
        remaining = n_tests - len(all_inputs)
        if remaining > 0:
            all_inputs.extend(self.input_generator.generate_test_inputs(remaining))
        
        return TestSuite(
            test_cases=np.array(all_inputs, dtype=object),
            method="Adaptive Parameter Space",
            generation_time=time.time() - start_time,
            metadata={'adaptation_rounds': adaptation_rounds}
        )
    
    # Helper methods for advanced generation strategies
    
    def _generate_complex_parameter(self, param_spec: Dict[str, Any]) -> Any:
        """Generate complex parameter values"""
        param_type = param_spec.get('type', 'dict')
        
        if param_type == 'dict':
            # Create complex dictionary with many keys
            result = self.input_generator._generate_dict(param_spec)
            # Add extra complexity
            result.update({
                'nested': {'deep': {'value': self.rng.randint(0, 1000)}},
                'list_values': [self.rng.randint(0, 100) for _ in range(10)],
                'complex_config': {
                    'algorithm': self.rng.choice(['advanced', 'complex', 'hybrid']),
                    'parameters': {f'param_{i}': self.rng.uniform(0, 1) for i in range(5)}
                }
            })
            return result
        elif param_type == 'list':
            # Generate larger, more complex lists
            max_length = min(param_spec.get('max_length', 10) * 2, 100)
            return self.input_generator._generate_list({**param_spec, 'max_length': max_length})
        else:
            return self.input_generator._generate_parameter(param_spec)
    
    def _generate_simple_parameter(self, param_spec: Dict[str, Any]) -> Any:
        """Generate simple parameter values"""
        param_type = param_spec.get('type', 'dict')
        
        if param_type == 'dict':
            # Create minimal dictionary
            required_keys = param_spec.get('required_keys', [])
            result = {}
            for key in required_keys[:2]:  # Only first 2 required keys
                result[key] = self.input_generator._generate_default_value(key)
            return result
        elif param_type == 'list':
            # Generate shorter lists
            max_length = max(1, param_spec.get('min_length', 1))
            return self.input_generator._generate_list({**param_spec, 'max_length': max_length})
        else:
            return self.input_generator._generate_parameter(param_spec)
    
    def _generate_coverage_targeted_inputs(self, target: str) -> Tuple[Any, ...]:
        """Generate inputs targeting specific coverage goals"""
        if target == 'error_paths':
            # Generate inputs likely to trigger error conditions
            inputs = []
            for param_spec in self.param_ranges:
                if param_spec.get('type') == 'dict':
                    # Missing required keys
                    inputs.append({})
                elif param_spec.get('type') == 'list':
                    # Empty or oversized lists
                    if self.rng.random() < 0.5:
                        inputs.append([])
                    else:
                        inputs.append([0] * 1000)  # Very large list
                else:
                    inputs.append(None)  # Invalid value
            return tuple(inputs)
        
        elif target == 'boundary_conditions':
            # Generate boundary values
            inputs = []
            for param_spec in self.param_ranges:
                param_type = param_spec.get('type', 'dict')
                if param_type == 'int':
                    range_spec = param_spec.get('range', [0, 100])
                    inputs.append(self.rng.choice([range_spec[0], range_spec[1]]))
                elif param_type == 'list':
                    min_len = param_spec.get('min_length', 0)
                    max_len = param_spec.get('max_length', 10)
                    length = self.rng.choice([min_len, max_len])
                    inputs.append([0] * length)
                else:
                    inputs.append(self.input_generator._generate_parameter(param_spec))
            return tuple(inputs)
        
        else:
            # Default to random
            return self.input_generator.generate_test_inputs(1)[0]
    
    def _generate_boundary_tests(self, n_tests: int) -> List[Tuple[Any, ...]]:
        """Generate boundary value tests"""
        return [self._generate_coverage_targeted_inputs('boundary_conditions') 
                for _ in range(n_tests)]
    
    def _generate_quasi_random_tests(self, n_tests: int) -> List[Tuple[Any, ...]]:
        """Generate quasi-random tests using Sobol sequence"""
        # For complex parameter types, use structured quasi-random generation
        tests = []
        sobol = qmc.Sobol(d=min(8, len(self.param_ranges)), seed=42)
        points = sobol.random(n_tests)
        
        for point in points:
            inputs = []
            for i, param_spec in enumerate(self.param_ranges):
                # Use quasi-random point to guide parameter generation
                qr_value = point[i % len(point)]
                param_value = self._quasi_random_parameter(param_spec, qr_value)
                inputs.append(param_value)
            tests.append(tuple(inputs))
        
        return tests
    
    def _quasi_random_parameter(self, param_spec: Dict[str, Any], qr_value: float) -> Any:
        """Generate parameter using quasi-random value"""
        param_type = param_spec.get('type', 'dict')
        
        if param_type == 'int':
            range_spec = param_spec.get('range', [0, 100])
            return int(range_spec[0] + qr_value * (range_spec[1] - range_spec[0]))
        elif param_type == 'float':
            range_spec = param_spec.get('range', [0.0, 100.0])
            return range_spec[0] + qr_value * (range_spec[1] - range_spec[0])
        elif param_type == 'list':
            min_len = param_spec.get('min_length', 0)
            max_len = param_spec.get('max_length', 10)
            length = int(min_len + qr_value * (max_len - min_len))
            return [int(qr_value * 100) for _ in range(length)]
        else:
            return self.input_generator._generate_parameter(param_spec)
    
    def _generate_error_focused_tests(self, n_tests: int) -> List[Tuple[Any, ...]]:
        """Generate tests focused on triggering errors"""
        return [self._generate_coverage_targeted_inputs('error_paths') 
                for _ in range(n_tests)]
    
    def _adaptive_round_generation(self, n_tests: int, previous_inputs: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
        """Generate tests adaptively based on previous rounds"""
        # Analyze previous inputs to find underexplored regions
        # For simplicity, generate tests that are different from previous ones
        
        tests = []
        for _ in range(n_tests):
            # Generate new test that's different from previous ones
            attempts = 0
            max_attempts = 50
            
            while attempts < max_attempts:
                candidate = self.input_generator.generate_test_inputs(1)[0]
                
                # Check if sufficiently different from previous tests
                if self._is_sufficiently_different(candidate, previous_inputs):
                    tests.append(candidate)
                    break
                
                attempts += 1
            
            # If no sufficiently different test found, use random
            if attempts >= max_attempts:
                tests.append(self.input_generator.generate_test_inputs(1)[0])
        
        return tests
    
    def _is_sufficiently_different(self, candidate: Tuple[Any, ...], 
                                 previous_inputs: List[Tuple[Any, ...]]) -> bool:
        """Check if candidate is sufficiently different from previous inputs"""
        if not previous_inputs:
            return True
        
        # Simple difference check - in practice, this would be more sophisticated
        # For now, use string representation comparison
        candidate_str = str(candidate)
        
        for prev_input in previous_inputs[-20:]:  # Check against last 20 inputs
            if str(prev_input) == candidate_str:
                return False
        
        return True


class SyntheticDatasetBaselineGenerator(BaselineTestGenerator):
    """Extended baseline generator specifically for synthetic dataset"""
    
    def __init__(self, param_ranges: List[Dict[str, Any]], 
                 bounds: Tuple[float, float] = (-999999, 999999),
                 seed: Optional[int] = None):
        # Initialize parent with minimal dimensions (will be overridden)
        super().__init__(None, 1, bounds)  
        self.param_ranges = param_ranges
        self.advanced_generators = AdvancedBaselineGenerators(param_ranges, seed)
        self.seed = seed
    
    def generate_synthetic_tests(self, method: str, n_tests: int) -> TestSuite:
        """
        Generate tests using specified method for synthetic functions
        """
        if method == "stratified_random":
            return self.advanced_generators.stratified_random(n_tests)
        elif method == "importance_sampling":
            return self.advanced_generators.importance_sampling(n_tests)
        elif method == "coverage_guided":
            return self.advanced_generators.coverage_guided_random(n_tests)
        elif method == "multi_stage_hybrid":
            return self.advanced_generators.multi_stage_hybrid(n_tests)
        elif method == "adaptive_space":
            return self.advanced_generators.adaptive_parameter_space(n_tests)
        elif method == "pure_random":
            return TestSuite(
                test_cases=np.array(self.advanced_generators.input_generator.generate_test_inputs(n_tests), 
                                  dtype=object),
                method="Pure Random (Synthetic)",
                generation_time=0.0,
                metadata={'seed': self.seed}
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_all_methods(self) -> List[str]:
        """Get all available synthetic dataset methods"""
        return [
            "pure_random",
            "stratified_random", 
            "importance_sampling",
            "coverage_guided",
            "multi_stage_hybrid",
            "adaptive_space"
        ]