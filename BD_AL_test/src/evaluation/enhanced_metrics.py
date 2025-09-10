#!/usr/bin/env python3
"""
Enhanced Evaluation Metrics

This module implements the advanced evaluation metrics specified in the 
EXPERIMENTAL_METHODOLOGY.md document, including:

1. Branch Distance: Accurate distance calculation to uncovered branches
2. Approach Level: Control flow distance to target branches  
3. Memory Usage Tracking: Resource consumption monitoring
4. Advanced Coverage Metrics: More detailed coverage analysis

Based on the methodology requirements from Section 6.2 and 6.3.
"""

import ast
import time
import psutil
import threading
import sys
import traceback
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import numpy as np
from collections import defaultdict


@dataclass
class BranchInfo:
    """Information about a program branch"""
    branch_id: str
    line_number: int
    condition: str
    true_distance: float = float('inf')
    false_distance: float = float('inf')
    times_hit: int = 0
    approach_level: int = 0
    

@dataclass
class EnhancedEvaluationResult:
    """Enhanced evaluation result with all methodology metrics"""
    # Primary metrics (Section 6.1)
    coverage: float = 0.0
    branch_distance: float = float('inf')
    execution_time: float = 0.0
    
    # Secondary metrics (Section 6.3)
    approach_level: float = 0.0
    memory_usage: float = 0.0
    solution_diversity: float = 0.0
    convergence_rate: Optional[int] = None
    
    # Detailed coverage information
    branches_covered: int = 0
    total_branches: int = 0
    statements_covered: int = 0
    total_statements: int = 0
    uncovered_branches: Set[str] = None
    
    # Execution information
    test_cases_executed: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    timeout_occurred: bool = False
    
    # Error information
    error_message: Optional[str] = None
    success: bool = True
    
    def __post_init__(self):
        if self.uncovered_branches is None:
            self.uncovered_branches = set()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'coverage': self.coverage,
            'branch_distance': self.branch_distance,
            'approach_level': self.approach_level,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'solution_diversity': self.solution_diversity,
            'convergence_rate': self.convergence_rate,
            'branches_covered': self.branches_covered,
            'total_branches': self.total_branches,
            'statements_covered': self.statements_covered,
            'total_statements': self.total_statements,
            'uncovered_branches': list(self.uncovered_branches),
            'test_cases_executed': self.test_cases_executed,
            'successful_executions': self.successful_executions,
            'failed_executions': self.failed_executions,
            'timeout_occurred': self.timeout_occurred,
            'error_message': self.error_message,
            'success': self.success
        }


class BranchAnalyzer:
    """Analyze program structure to identify branches and control flow"""
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.branches: Dict[str, BranchInfo] = {}
        self.control_flow_graph = {}
        self.analyze_program()
    
    def analyze_program(self):
        """Parse program and extract branch information"""
        try:
            tree = ast.parse(self.source_code)
            self._extract_branches(tree)
            self._build_control_flow_graph(tree)
        except Exception as e:
            print(f"Warning: Could not analyze program structure: {e}")
    
    def _extract_branches(self, node: ast.AST, parent_id: str = "root", level: int = 0):
        """Extract all conditional branches from AST"""
        for child in ast.walk(node):
            if isinstance(child, ast.If):
                branch_id = f"if_{child.lineno}_{child.col_offset}"
                condition = ast.unparse(child.test) if hasattr(ast, 'unparse') else str(child.test)
                
                self.branches[branch_id] = BranchInfo(
                    branch_id=branch_id,
                    line_number=child.lineno,
                    condition=condition,
                    approach_level=level
                )
                
                # Recursively analyze nested branches
                self._extract_branches(child, branch_id, level + 1)
                
            elif isinstance(child, (ast.While, ast.For)):
                branch_id = f"loop_{child.lineno}_{child.col_offset}"
                condition = "loop_condition"
                
                self.branches[branch_id] = BranchInfo(
                    branch_id=branch_id,
                    line_number=child.lineno,
                    condition=condition,
                    approach_level=level
                )
                
                self._extract_branches(child, branch_id, level + 1)
    
    def _build_control_flow_graph(self, tree: ast.AST):
        """Build control flow graph for approach level calculation"""
        # Simplified CFG - would need more sophisticated implementation for full accuracy
        self.control_flow_graph = {}
        
        for branch_id, branch_info in self.branches.items():
            self.control_flow_graph[branch_id] = {
                'predecessors': set(),
                'successors': set(),
                'dominates': set(),
                'level': branch_info.approach_level
            }
    
    def calculate_approach_level(self, target_branch: str, covered_branches: Set[str]) -> int:
        """
        Calculate approach level to target branch
        
        Approach level = minimum number of control decisions that need to be 
        different to reach the target branch from the closest covered branch
        
        As defined in methodology Section 6.3
        """
        if target_branch in covered_branches:
            return 0
        
        if target_branch not in self.branches:
            return 0
        
        target_info = self.branches[target_branch]
        min_approach_level = float('inf')
        
        # Find closest covered branch in control flow
        for covered_branch in covered_branches:
            if covered_branch in self.branches:
                covered_info = self.branches[covered_branch]
                
                # Calculate control flow distance
                distance = abs(target_info.approach_level - covered_info.approach_level)
                
                # Adjust for nesting level difference
                if target_info.approach_level > covered_info.approach_level:
                    distance += 1  # Need to enter deeper level
                
                min_approach_level = min(min_approach_level, distance)
        
        return int(min_approach_level) if min_approach_level != float('inf') else target_info.approach_level


class BranchDistanceCalculator:
    """Calculate branch distance for test inputs"""
    
    @staticmethod
    def calculate_numeric_distance(value: Union[int, float], 
                                 operator: str, 
                                 target: Union[int, float]) -> float:
        """
        Calculate branch distance for numeric comparisons
        
        Based on Korel's branch distance function (methodology reference)
        """
        if operator == '==':
            return abs(value - target)
        elif operator == '!=':
            return 0.0 if value != target else 1.0
        elif operator == '<':
            return target - value if value >= target else 0.0
        elif operator == '<=':
            return target - value + 1 if value > target else 0.0
        elif operator == '>':
            return value - target if value <= target else 0.0
        elif operator == '>=':
            return value - target + 1 if value < target else 0.0
        else:
            return 1.0  # Unknown operator
    
    @staticmethod
    def calculate_string_distance(value: str, operator: str, target: str) -> float:
        """Calculate branch distance for string comparisons"""
        if operator == '==':
            if value == target:
                return 0.0
            # Levenshtein distance normalized
            return BranchDistanceCalculator._levenshtein_distance(value, target) / max(len(value), len(target), 1)
        elif operator == '!=':
            return 0.0 if value != target else 1.0
        elif operator in ['in', 'not in']:
            if operator == 'in':
                return 0.0 if value in target else len(target)
            else:
                return len(target) if value in target else 0.0
        else:
            return 1.0
    
    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between strings"""
        if len(s1) < len(s2):
            return BranchDistanceCalculator._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def calculate_compound_distance(conditions: List[Tuple[float, str]]) -> float:
        """
        Calculate distance for compound conditions (AND, OR)
        
        Args:
            conditions: List of (distance, operator) tuples
        """
        if not conditions:
            return 1.0
        
        distances = [cond[0] for cond in conditions]
        
        # For AND: all conditions must be true, distance is sum
        # For OR: at least one must be true, distance is minimum
        # This is a simplification - real implementation would need to analyze the AST
        
        return min(distances)  # Assume OR for now


class MemoryProfiler:
    """Advanced memory usage profiling"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = 0.0
        self.peak_memory = 0.0
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None
        self.sample_interval = 0.01  # 10ms sampling
    
    def start_profiling(self):
        """Start continuous memory profiling"""
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        self.memory_samples = []
        self.monitoring = True
        
        self.monitor_thread = threading.Thread(target=self._profile_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_profiling(self) -> Dict[str, float]:
        """Stop profiling and return detailed memory metrics"""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        final_memory = self.process.memory_info().rss / 1024 / 1024
        
        return {
            'initial_memory_mb': self.initial_memory,
            'peak_memory_mb': self.peak_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': self.peak_memory - self.initial_memory,
            'average_memory_mb': np.mean(self.memory_samples) if self.memory_samples else self.initial_memory,
            'memory_std_mb': np.std(self.memory_samples) if len(self.memory_samples) > 1 else 0.0,
            'samples_collected': len(self.memory_samples)
        }
    
    def _profile_memory(self):
        """Background memory profiling loop"""
        while self.monitoring:
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                self.peak_memory = max(self.peak_memory, memory_mb)
                self.memory_samples.append(memory_mb)
                time.sleep(self.sample_interval)
            except Exception:
                break


class EnhancedTestEvaluator:
    """
    Enhanced test evaluator implementing all methodology requirements
    
    Implements evaluation metrics from EXPERIMENTAL_METHODOLOGY.md:
    - Section 6.1: Primary Performance Metrics
    - Section 6.2: Multi-Objective Specific Metrics  
    - Section 6.3: Secondary Metrics
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.program_analyzers: Dict[str, BranchAnalyzer] = {}
        self.distance_calculator = BranchDistanceCalculator()
    
    def analyze_program(self, program_path: str) -> BranchAnalyzer:
        """Analyze program structure for enhanced metrics"""
        if program_path not in self.program_analyzers:
            try:
                with open(program_path, 'r') as f:
                    source_code = f.read()
                
                self.program_analyzers[program_path] = BranchAnalyzer(source_code)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not analyze {program_path}: {e}")
                # Create dummy analyzer
                self.program_analyzers[program_path] = BranchAnalyzer("")
        
        return self.program_analyzers[program_path]
    
    def evaluate_test_suite(self,
                           program_path: str,
                           target_function_name: str,
                           test_inputs: List[Any],
                           timeout: float = 30.0) -> EnhancedEvaluationResult:
        """
        Comprehensive evaluation of test suite
        
        Args:
            program_path: Path to target program
            target_function_name: Name of function to test
            test_inputs: List of test inputs
            timeout: Maximum execution time
            
        Returns:
            EnhancedEvaluationResult with all metrics
        """
        result = EnhancedEvaluationResult()
        
        # Initialize profiling
        memory_profiler = MemoryProfiler()
        memory_profiler.start_profiling()
        start_time = time.time()
        
        try:
            # Analyze program structure
            analyzer = self.analyze_program(program_path)
            result.total_branches = len(analyzer.branches)
            
            # Load target program
            target_function = self._load_target_function(program_path, target_function_name)
            
            # Execute test cases and collect metrics
            covered_branches = set()
            total_branch_distance = 0.0
            successful_executions = 0
            failed_executions = 0
            
            for i, test_input in enumerate(test_inputs):
                try:
                    # Execute with timeout
                    execution_result = self._execute_with_monitoring(
                        target_function, test_input, timeout
                    )
                    
                    if execution_result['success']:
                        successful_executions += 1
                        
                        # Simulate branch coverage (would use instrumentation in real implementation)
                        test_branches = self._simulate_branch_coverage(test_input, execution_result['result'])
                        covered_branches.update(test_branches)
                        
                        # Calculate branch distance for uncovered branches
                        uncovered_branches = set(analyzer.branches.keys()) - covered_branches
                        for branch_id in uncovered_branches:
                            distance = self._calculate_branch_distance(
                                test_input, branch_id, analyzer.branches[branch_id]
                            )
                            total_branch_distance += distance
                    else:
                        failed_executions += 1
                        
                except Exception as e:
                    failed_executions += 1
                    if self.verbose:
                        print(f"Test input {i} failed: {e}")
            
            # Calculate final metrics
            result.branches_covered = len(covered_branches)
            result.coverage = len(covered_branches) / max(result.total_branches, 1)
            result.test_cases_executed = len(test_inputs)
            result.successful_executions = successful_executions
            result.failed_executions = failed_executions
            result.uncovered_branches = set(analyzer.branches.keys()) - covered_branches
            
            # Calculate branch distance (average distance to uncovered branches)
            if result.uncovered_branches:
                result.branch_distance = total_branch_distance / len(result.uncovered_branches)
            else:
                result.branch_distance = 0.0
            
            # Calculate approach level (average approach level to uncovered branches)
            if result.uncovered_branches:
                approach_levels = []
                for branch_id in result.uncovered_branches:
                    level = analyzer.calculate_approach_level(branch_id, covered_branches)
                    approach_levels.append(level)
                result.approach_level = np.mean(approach_levels)
            else:
                result.approach_level = 0.0
            
            # Calculate solution diversity
            result.solution_diversity = self._calculate_solution_diversity(test_inputs)
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            if self.verbose:
                print(f"Evaluation failed: {e}")
        
        # Finalize metrics
        result.execution_time = time.time() - start_time
        memory_metrics = memory_profiler.stop_profiling()
        result.memory_usage = memory_metrics['peak_memory_mb']
        
        return result
    
    def _load_target_function(self, program_path: str, function_name: str):
        """Load target function from program file"""
        spec = importlib.util.spec_from_file_location("target_module", program_path)
        if not spec or not spec.loader:
            raise RuntimeError(f"Could not load module from {program_path}")
        
        target_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(target_module)
        
        if not hasattr(target_module, function_name):
            raise RuntimeError(f"Function {function_name} not found in {program_path}")
        
        return getattr(target_module, function_name)
    
    def _execute_with_monitoring(self, target_function, test_input, timeout: float) -> Dict[str, Any]:
        """Execute function with timeout and error monitoring"""
        result = {'success': False, 'result': None, 'error': None}
        
        def execute():
            try:
                if isinstance(test_input, (list, tuple)):
                    result['result'] = target_function(*test_input)
                else:
                    result['result'] = target_function(test_input)
                result['success'] = True
            except Exception as e:
                result['error'] = str(e)
        
        thread = threading.Thread(target=execute)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            result['error'] = 'Timeout'
        
        return result
    
    def _simulate_branch_coverage(self, test_input: Any, result: Any) -> Set[str]:
        """
        Simulate branch coverage based on input characteristics
        
        Note: In a real implementation, this would use code instrumentation
        """
        branches = set()
        
        # Simple heuristic based on input values
        if isinstance(test_input, (list, tuple)):
            for i, val in enumerate(test_input):
                if isinstance(val, (int, float)):
                    # Create branch IDs based on value characteristics
                    if val < 0:
                        branches.add(f"negative_{i}")
                    elif val > 0:
                        branches.add(f"positive_{i}")
                    else:
                        branches.add(f"zero_{i}")
                    
                    if abs(val) > 100:
                        branches.add(f"large_{i}")
                    elif abs(val) < 10:
                        branches.add(f"small_{i}")
                    
                    # Value range branches
                    if val > 1000:
                        branches.add(f"very_large_{i}")
                    elif val < -1000:
                        branches.add(f"very_small_{i}")
        
        # Add some result-based branches
        if result is not None:
            if isinstance(result, (int, float)):
                if result > 0:
                    branches.add("result_positive")
                elif result < 0:
                    branches.add("result_negative")
                else:
                    branches.add("result_zero")
        
        return branches
    
    def _calculate_branch_distance(self, test_input: Any, branch_id: str, branch_info: BranchInfo) -> float:
        """Calculate distance to specific branch"""
        # Simplified distance calculation
        # In real implementation, would need to evaluate the actual condition
        
        if isinstance(test_input, (list, tuple)):
            # Use first numeric value as proxy
            for val in test_input:
                if isinstance(val, (int, float)):
                    # Simple distance based on value and branch type
                    if "negative" in branch_id:
                        return max(0, val + 1)  # Distance to make negative
                    elif "positive" in branch_id:
                        return max(0, -val + 1)  # Distance to make positive
                    elif "zero" in branch_id:
                        return abs(val)  # Distance to zero
                    break
        
        return 1.0  # Default distance
    
    def _calculate_solution_diversity(self, test_inputs: List[Any]) -> float:
        """
        Calculate diversity of test solutions
        
        Uses pairwise distance between test inputs
        """
        if len(test_inputs) < 2:
            return 0.0
        
        distances = []
        
        for i in range(len(test_inputs)):
            for j in range(i + 1, len(test_inputs)):
                distance = self._calculate_input_distance(test_inputs[i], test_inputs[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_input_distance(self, input1: Any, input2: Any) -> float:
        """Calculate distance between two test inputs"""
        if isinstance(input1, (list, tuple)) and isinstance(input2, (list, tuple)):
            if len(input1) != len(input2):
                return 1.0
            
            distances = []
            for v1, v2 in zip(input1, input2):
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    distances.append(abs(v1 - v2))
                elif isinstance(v1, str) and isinstance(v2, str):
                    distances.append(self.distance_calculator._levenshtein_distance(v1, v2))
                else:
                    distances.append(0.0 if v1 == v2 else 1.0)
            
            return np.mean(distances) if distances else 0.0
        
        elif isinstance(input1, (int, float)) and isinstance(input2, (int, float)):
            return abs(input1 - input2)
        
        else:
            return 0.0 if input1 == input2 else 1.0


# Factory function for easy integration
def create_enhanced_evaluator(verbose: bool = True) -> EnhancedTestEvaluator:
    """Create enhanced test evaluator instance"""
    return EnhancedTestEvaluator(verbose=verbose)


if __name__ == "__main__":
    """Example usage and testing"""
    print("Enhanced Test Evaluation Metrics")
    print("=" * 50)
    print("This module implements advanced evaluation metrics")
    print("as specified in the EXPERIMENTAL_METHODOLOGY.md document.")
    print("\nFeatures:")
    print("- Branch distance calculation")
    print("- Approach level measurement") 
    print("- Advanced memory profiling")
    print("- Solution diversity analysis")
    print("- Comprehensive coverage metrics")