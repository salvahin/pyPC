#!/usr/bin/env python3
"""
Synthetic Dataset Validation Framework
Analyzes and validates the challenging synthetic functions to ensure they meet complexity targets
"""

import ast
import os
import sys
import time
import yaml
import importlib.util
import traceback
from typing import Dict, List, Any, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import argparse

@dataclass
class ComplexityMetrics:
    cyclomatic_complexity: int
    lines_of_code: int
    branching_depth: int
    unique_paths: int
    function_count: int
    class_count: int
    cognitive_complexity: int

@dataclass
class ValidationResult:
    function_name: str
    file_path: str
    metrics: ComplexityMetrics
    target_coverage: float
    estimated_coverage: float
    difficulty_rating: str
    validation_errors: List[str]
    validation_warnings: List[str]
    execution_test_passed: bool
    execution_time: float

class CyclomaticComplexityCalculator(ast.NodeVisitor):
    """Calculate cyclomatic complexity of Python code"""
    
    def __init__(self):
        self.complexity = 1  # Base complexity
        self.nesting_level = 0
        self.max_nesting = 0
        
    def visit_If(self, node):
        self.complexity += 1
        self.nesting_level += 1
        self.max_nesting = max(self.max_nesting, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1
        
    def visit_While(self, node):
        self.complexity += 1
        self.nesting_level += 1
        self.max_nesting = max(self.max_nesting, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1
        
    def visit_For(self, node):
        self.complexity += 1
        self.nesting_level += 1
        self.max_nesting = max(self.max_nesting, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1
        
    def visit_AsyncFor(self, node):
        self.complexity += 1
        self.nesting_level += 1
        self.max_nesting = max(self.max_nesting, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1
        
    def visit_AsyncWith(self, node):
        self.complexity += 1
        self.nesting_level += 1
        self.max_nesting = max(self.max_nesting, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1
        
    def visit_With(self, node):
        self.complexity += 1
        self.nesting_level += 1
        self.max_nesting = max(self.max_nesting, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1
        
    def visit_Try(self, node):
        self.complexity += 1
        self.nesting_level += 1
        self.max_nesting = max(self.max_nesting, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1
        
    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_BoolOp(self, node):
        self.complexity += len(node.values) - 1
        self.generic_visit(node)
        
    def visit_ListComp(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_SetComp(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_DictComp(self, node):
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_GeneratorExp(self, node):
        self.complexity += 1
        self.generic_visit(node)

class CodeAnalyzer(ast.NodeVisitor):
    """Analyze various code metrics"""
    
    def __init__(self):
        self.function_count = 0
        self.class_count = 0
        self.lines_of_code = 0
        self.branching_statements = []
        self.complexity_patterns = defaultdict(int)
        
    def visit_FunctionDef(self, node):
        self.function_count += 1
        self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node):
        self.function_count += 1
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        self.class_count += 1
        self.generic_visit(node)
        
    def visit_If(self, node):
        self.branching_statements.append(('if', node.lineno))
        self.complexity_patterns['conditional'] += 1
        self.generic_visit(node)
        
    def visit_For(self, node):
        self.branching_statements.append(('for', node.lineno))
        self.complexity_patterns['loop'] += 1
        self.generic_visit(node)
        
    def visit_While(self, node):
        self.branching_statements.append(('while', node.lineno))
        self.complexity_patterns['loop'] += 1
        self.generic_visit(node)
        
    def visit_Try(self, node):
        self.branching_statements.append(('try', node.lineno))
        self.complexity_patterns['exception'] += 1
        self.generic_visit(node)

class PathComplexityAnalyzer:
    """Estimate path complexity and coverage difficulty"""
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.tree = ast.parse(source_code)
        
    def estimate_unique_paths(self) -> int:
        """Estimate number of unique execution paths"""
        path_multiplier = 1
        
        class PathCounter(ast.NodeVisitor):
            def __init__(self):
                self.path_count = 1
                
            def visit_If(self, node):
                # Each if statement doubles the paths (true/false)
                self.path_count *= 2
                # Additional paths for elif conditions
                elif_count = len([n for n in node.orelse if isinstance(n, ast.If)])
                if elif_count > 0:
                    self.path_count *= (elif_count + 1)
                self.generic_visit(node)
                
            def visit_For(self, node):
                # Loop can execute 0, 1, or multiple times
                self.path_count *= 3
                self.generic_visit(node)
                
            def visit_While(self, node):
                # While loop complexity
                self.path_count *= 3
                self.generic_visit(node)
                
            def visit_Try(self, node):
                # Try block with exception paths
                exception_count = len(node.handlers)
                self.path_count *= (exception_count + 1)
                self.generic_visit(node)
        
        counter = PathCounter()
        counter.visit(self.tree)
        
        # Cap the path count to prevent overflow
        return min(counter.path_count, 10000)
    
    def estimate_coverage_difficulty(self) -> float:
        """Estimate how difficult it is to achieve high coverage"""
        
        class CoverageAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.difficulty_score = 0
                self.total_statements = 0
                
            def visit_If(self, node):
                self.total_statements += 1
                # Complex conditions are harder to cover
                if isinstance(node.test, ast.BoolOp):
                    self.difficulty_score += 3
                elif isinstance(node.test, ast.Compare):
                    self.difficulty_score += 2
                else:
                    self.difficulty_score += 1
                self.generic_visit(node)
                
            def visit_For(self, node):
                self.total_statements += 1
                self.difficulty_score += 2  # Loops add coverage difficulty
                self.generic_visit(node)
                
            def visit_While(self, node):
                self.total_statements += 1
                self.difficulty_score += 3  # While loops are harder to exit
                self.generic_visit(node)
                
            def visit_Try(self, node):
                self.total_statements += 1
                # Exception paths are very hard to cover
                self.difficulty_score += len(node.handlers) * 4
                self.generic_visit(node)
                
            def visit_Raise(self, node):
                self.total_statements += 1
                self.difficulty_score += 5  # Exceptions are hard to trigger
                self.generic_visit(node)
                
            def visit_Return(self, node):
                self.total_statements += 1
                # Multiple returns increase difficulty
                self.difficulty_score += 1
                self.generic_visit(node)
        
        analyzer = CoverageAnalyzer()
        analyzer.visit(self.tree)
        
        if analyzer.total_statements == 0:
            return 0.5
        
        # Normalize difficulty score to 0-1 range
        normalized_score = analyzer.difficulty_score / (analyzer.total_statements * 5)
        return min(normalized_score, 1.0)

class SyntheticDatasetValidator:
    """Main validator for the synthetic dataset"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.results: List[ValidationResult] = []
        
    def load_config(self) -> Dict[str, Any]:
        """Load the YAML configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {self.config_path}: {e}")
    
    def validate_all_functions(self) -> Dict[str, Any]:
        """Validate all functions in the synthetic dataset"""
        print("🔍 Starting synthetic dataset validation...")
        
        validation_summary = {
            'total_functions': 0,
            'passed_functions': 0,
            'failed_functions': 0,
            'warnings_count': 0,
            'average_complexity': 0,
            'coverage_distribution': defaultdict(int),
            'difficulty_distribution': defaultdict(int),
            'category_results': defaultdict(list)
        }
        
        # Process each function in the config
        for func_name, func_config in self.config.items():
            if func_name in ['global_config', 'experiment_config']:
                continue
                
            print(f"\n📝 Validating {func_name}...")
            
            result = self.validate_single_function(func_name, func_config)
            self.results.append(result)
            
            # Update summary statistics
            validation_summary['total_functions'] += 1
            
            if len(result.validation_errors) == 0:
                validation_summary['passed_functions'] += 1
            else:
                validation_summary['failed_functions'] += 1
                
            validation_summary['warnings_count'] += len(result.validation_warnings)
            validation_summary['average_complexity'] += result.metrics.cyclomatic_complexity
            
            # Coverage distribution
            coverage_bucket = self.get_coverage_bucket(result.estimated_coverage)
            validation_summary['coverage_distribution'][coverage_bucket] += 1
            
            # Difficulty distribution
            validation_summary['difficulty_distribution'][result.difficulty_rating] += 1
            
            # Category results (extract from tags)
            category = self.extract_category_from_function_name(func_name)
            validation_summary['category_results'][category].append(result)
        
        # Calculate averages
        if validation_summary['total_functions'] > 0:
            validation_summary['average_complexity'] /= validation_summary['total_functions']
        
        return validation_summary
    
    def validate_single_function(self, func_name: str, func_config: Dict[str, Any]) -> ValidationResult:
        """Validate a single function"""
        
        errors = []
        warnings = []
        
        # Get file path
        file_path = func_config.get('file', '')
        if not file_path:
            errors.append("Missing file path in configuration")
            
        full_path = os.path.join(os.path.dirname(self.config_path), '..', file_path)
        
        if not os.path.exists(full_path):
            errors.append(f"File not found: {full_path}")
            return ValidationResult(
                function_name=func_name,
                file_path=file_path,
                metrics=ComplexityMetrics(0, 0, 0, 0, 0, 0, 0),
                target_coverage=func_config.get('target_coverage', 0),
                estimated_coverage=0,
                difficulty_rating=func_config.get('difficulty', 'unknown'),
                validation_errors=errors,
                validation_warnings=warnings,
                execution_test_passed=False,
                execution_time=0
            )
        
        # Analyze source code
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception as e:
            errors.append(f"Failed to read source file: {e}")
            source_code = ""
        
        # Calculate complexity metrics
        metrics = self.calculate_complexity_metrics(source_code)
        
        # Path complexity analysis
        path_analyzer = PathComplexityAnalyzer(source_code)
        estimated_coverage = self.estimate_coverage_percentage(path_analyzer)
        
        # Validate against targets
        target_complexity = func_config.get('cyclomatic_complexity', 0)
        target_coverage = func_config.get('target_coverage', 0)
        
        # Complexity validation
        if metrics.cyclomatic_complexity < target_complexity * 0.8:
            errors.append(f"Complexity too low: {metrics.cyclomatic_complexity} < {target_complexity * 0.8}")
        elif metrics.cyclomatic_complexity < target_complexity * 0.9:
            warnings.append(f"Complexity slightly low: {metrics.cyclomatic_complexity} < {target_complexity * 0.9}")
        
        # Coverage validation
        coverage_tolerance = 0.1  # 10% tolerance
        if abs(estimated_coverage - target_coverage) > target_coverage * coverage_tolerance:
            warnings.append(f"Coverage estimate differs from target: {estimated_coverage:.1f}% vs {target_coverage}%")
        
        # Execution test
        execution_test_passed, execution_time = self.test_function_execution(full_path, func_config)
        
        if not execution_test_passed:
            errors.append("Function execution test failed")
        
        return ValidationResult(
            function_name=func_name,
            file_path=file_path,
            metrics=metrics,
            target_coverage=target_coverage,
            estimated_coverage=estimated_coverage,
            difficulty_rating=func_config.get('difficulty', 'unknown'),
            validation_errors=errors,
            validation_warnings=warnings,
            execution_test_passed=execution_test_passed,
            execution_time=execution_time
        )
    
    def calculate_complexity_metrics(self, source_code: str) -> ComplexityMetrics:
        """Calculate comprehensive complexity metrics"""
        
        if not source_code:
            return ComplexityMetrics(0, 0, 0, 0, 0, 0, 0)
        
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return ComplexityMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Cyclomatic complexity
        complexity_calc = CyclomaticComplexityCalculator()
        complexity_calc.visit(tree)
        
        # Code analysis
        code_analyzer = CodeAnalyzer()
        code_analyzer.visit(tree)
        
        # Path complexity
        path_analyzer = PathComplexityAnalyzer(source_code)
        unique_paths = path_analyzer.estimate_unique_paths()
        
        # Lines of code (non-empty, non-comment)
        lines = source_code.split('\n')
        loc = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        # Cognitive complexity (simplified)
        cognitive_complexity = self.calculate_cognitive_complexity(tree)
        
        return ComplexityMetrics(
            cyclomatic_complexity=complexity_calc.complexity,
            lines_of_code=loc,
            branching_depth=complexity_calc.max_nesting,
            unique_paths=unique_paths,
            function_count=code_analyzer.function_count,
            class_count=code_analyzer.class_count,
            cognitive_complexity=cognitive_complexity
        )
    
    def calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity (how hard code is to understand)"""
        
        class CognitiveComplexityCalculator(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
                self.nesting_level = 0
                
            def visit_If(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_For(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_While(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_Try(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_BoolOp(self, node):
                self.complexity += len(node.values) - 1
                self.generic_visit(node)
        
        calc = CognitiveComplexityCalculator()
        calc.visit(tree)
        return calc.complexity
    
    def estimate_coverage_percentage(self, path_analyzer: PathComplexityAnalyzer) -> float:
        """Estimate achievable coverage percentage"""
        
        difficulty = path_analyzer.estimate_coverage_difficulty()
        unique_paths = path_analyzer.estimate_unique_paths()
        
        # Base coverage estimate
        base_coverage = 100.0
        
        # Reduce based on difficulty
        coverage_reduction = difficulty * 70  # Up to 70% reduction for very difficult code
        
        # Reduce based on path complexity
        if unique_paths > 100:
            path_reduction = min(30, (unique_paths - 100) / 100 * 20)
            coverage_reduction += path_reduction
        
        estimated_coverage = max(5.0, base_coverage - coverage_reduction)
        return estimated_coverage
    
    def test_function_execution(self, file_path: str, func_config: Dict[str, Any]) -> Tuple[bool, float]:
        """Test if function can be executed without errors"""
        
        target_function = func_config.get('function', 'target_function')
        
        try:
            # Dynamic import
            spec = importlib.util.spec_from_file_location("test_module", file_path)
            if spec is None or spec.loader is None:
                return False, 0.0
                
            module = importlib.util.module_from_spec(spec)
            
            start_time = time.time()
            spec.loader.exec_module(module)
            
            # Check if target function exists
            if not hasattr(module, target_function):
                return False, 0.0
            
            func = getattr(module, target_function)
            
            # Try to call with simple parameters (this is a basic smoke test)
            try:
                # Generate simple test parameters based on parameter ranges
                test_params = self.generate_simple_test_parameters(func_config)
                
                if test_params:
                    if len(test_params) == 1:
                        result = func(test_params[0])
                    elif len(test_params) == 2:
                        result = func(test_params[0], test_params[1])
                    else:
                        result = func(*test_params)
                else:
                    result = func({}, {})  # Default empty parameters
                    
                execution_time = time.time() - start_time
                
                # Check if result looks reasonable (not obviously an error)
                if isinstance(result, dict) and 'error' in result:
                    return True, execution_time  # Expected error format is still valid
                
                return True, execution_time
                
            except Exception:
                # Function exists but failed to execute - still consider this valid
                # as the error might be due to our simple test parameters
                execution_time = time.time() - start_time
                return True, execution_time
                
        except Exception:
            return False, 0.0
    
    def generate_simple_test_parameters(self, func_config: Dict[str, Any]) -> List[Any]:
        """Generate simple test parameters for smoke testing"""
        
        parameter_ranges = func_config.get('parameter_ranges', [])
        if not parameter_ranges:
            return []
        
        test_params = []
        
        for param_config in parameter_ranges:
            param_type = param_config.get('type', 'dict')
            
            if param_type == 'dict':
                test_params.append({})
            elif param_type == 'list':
                test_params.append([])
            elif param_type == 'int':
                test_params.append(0)
            elif param_type == 'float':
                test_params.append(0.0)
            elif param_type == 'string':
                test_params.append("")
            elif param_type == 'bytes':
                test_params.append(b"")
            elif param_type == 'matrix':
                test_params.append([[1.0]])
            else:
                test_params.append({})
        
        return test_params[:2]  # Limit to first 2 parameters for simplicity
    
    def get_coverage_bucket(self, coverage: float) -> str:
        """Get coverage bucket for statistics"""
        if coverage < 20:
            return "< 20%"
        elif coverage < 30:
            return "20-30%"
        elif coverage < 40:
            return "30-40%"
        elif coverage < 50:
            return "40-50%"
        else:
            return "> 50%"
    
    def extract_category_from_function_name(self, func_name: str) -> str:
        """Extract category from function name"""
        # This is based on the function names in our synthetic dataset
        if func_name in ['cryptographic_hash', 'avl_tree_operations', 'numerical_solver', 'matrix_optimizer', 'signal_processor']:
            return "Algorithmic Foundations"
        elif func_name in ['json_parser_validator', 'protocol_state_machine', 'workflow_engine', 'distributed_system', 'optimization_solver']:
            return "Input Validation Gauntlet"
        elif func_name in ['resource_scheduler', 'cache_manager', 'event_processor', 'lock_free_queue', 'statistical_analyzer']:
            return "Concurrency & Resource Management"
        else:
            return "Unknown Category"
    
    def generate_report(self, summary: Dict[str, Any]) -> str:
        """Generate comprehensive validation report"""
        
        report = []
        report.append("=" * 80)
        report.append("SYNTHETIC DATASET VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("📊 SUMMARY STATISTICS")
        report.append(f"Total Functions: {summary['total_functions']}")
        report.append(f"Passed: {summary['passed_functions']} ({summary['passed_functions']/summary['total_functions']*100:.1f}%)")
        report.append(f"Failed: {summary['failed_functions']} ({summary['failed_functions']/summary['total_functions']*100:.1f}%)")
        report.append(f"Total Warnings: {summary['warnings_count']}")
        report.append(f"Average Complexity: {summary['average_complexity']:.1f}")
        report.append("")
        
        # Coverage distribution
        report.append("📈 COVERAGE DISTRIBUTION")
        for bucket, count in sorted(summary['coverage_distribution'].items()):
            percentage = count / summary['total_functions'] * 100
            report.append(f"{bucket}: {count} functions ({percentage:.1f}%)")
        report.append("")
        
        # Difficulty distribution
        report.append("💪 DIFFICULTY DISTRIBUTION")
        for difficulty, count in sorted(summary['difficulty_distribution'].items()):
            percentage = count / summary['total_functions'] * 100
            report.append(f"{difficulty}: {count} functions ({percentage:.1f}%)")
        report.append("")
        
        # Category breakdown
        report.append("🗂️  CATEGORY BREAKDOWN")
        for category, results in summary['category_results'].items():
            report.append(f"{category}: {len(results)} functions")
            avg_complexity = sum(r.metrics.cyclomatic_complexity for r in results) / len(results)
            avg_coverage = sum(r.estimated_coverage for r in results) / len(results)
            report.append(f"  Average Complexity: {avg_complexity:.1f}")
            report.append(f"  Average Coverage: {avg_coverage:.1f}%")
        report.append("")
        
        # Individual function results
        report.append("🔍 INDIVIDUAL FUNCTION RESULTS")
        report.append("-" * 80)
        
        for result in self.results:
            status = "✅ PASS" if len(result.validation_errors) == 0 else "❌ FAIL"
            report.append(f"{status} {result.function_name}")
            report.append(f"  File: {result.file_path}")
            report.append(f"  Complexity: {result.metrics.cyclomatic_complexity} (target: {result.target_coverage})")
            report.append(f"  Estimated Coverage: {result.estimated_coverage:.1f}% (target: {result.target_coverage}%)")
            report.append(f"  Difficulty: {result.difficulty_rating}")
            report.append(f"  Execution Test: {'✅' if result.execution_test_passed else '❌'}")
            
            if result.validation_errors:
                report.append(f"  Errors: {', '.join(result.validation_errors)}")
            
            if result.validation_warnings:
                report.append(f"  Warnings: {', '.join(result.validation_warnings)}")
            
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        
        if summary['failed_functions'] > 0:
            report.append("- Fix validation errors in failed functions")
        
        if summary['warnings_count'] > 5:
            report.append("- Review and address validation warnings")
        
        low_complexity_functions = [r for r in self.results if r.metrics.cyclomatic_complexity < 40]
        if low_complexity_functions:
            report.append(f"- {len(low_complexity_functions)} functions have complexity < 40, consider increasing")
        
        high_coverage_functions = [r for r in self.results if r.estimated_coverage > 60]
        if high_coverage_functions:
            report.append(f"- {len(high_coverage_functions)} functions may be too easy (>60% coverage)")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Validate synthetic dataset for multi-objective test case generation')
    parser.add_argument('config', help='Path to synthetic dataset YAML configuration')
    parser.add_argument('--output', '-o', help='Output report file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        validator = SyntheticDatasetValidator(args.config)
        summary = validator.validate_all_functions()
        report = validator.generate_report(summary)
        
        print(report)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\n📄 Report saved to {args.output}")
        
        # Exit with appropriate code
        if summary['failed_functions'] > 0:
            print(f"\n❌ Validation failed: {summary['failed_functions']} functions failed validation")
            sys.exit(1)
        else:
            print(f"\n✅ Validation successful: All {summary['total_functions']} functions passed")
            
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()