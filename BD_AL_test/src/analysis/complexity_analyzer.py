#!/usr/bin/env python3
"""
Cyclomatic Complexity Analyzer

This module provides comprehensive program complexity analysis including cyclomatic
complexity measurement, AST analysis, and automatic program categorization for the
experimental methodology framework.

Features:
- Cyclomatic complexity calculation using multiple methods
- AST-based control flow analysis
- Automatic complexity categorization
- Expected coverage estimation
- Program metadata generation
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import yaml
import radon.complexity as cc
from radon.visitors import ComplexityVisitor
from radon.metrics import mi_visit, h_visit
import argparse


@dataclass
class FunctionComplexity:
    """Individual function complexity metrics"""
    name: str
    cyclomatic_complexity: int
    cognitive_complexity: int
    lines_of_code: int
    maintainability_index: float
    halstead_metrics: Dict[str, float]
    parameters: int
    returns: int
    nested_depth: int
    branch_count: int
    loop_count: int


@dataclass
class ProgramComplexity:
    """Complete program complexity analysis"""
    file_path: str
    program_name: str
    total_cyclomatic_complexity: int
    average_cyclomatic_complexity: float
    max_cyclomatic_complexity: int
    functions: List[FunctionComplexity]
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    maintainability_index: float
    complexity_category: str
    expected_coverage_range: Tuple[float, float]
    difficulty_score: float
    domain_classification: str
    parameter_dimensions: int
    suggested_bounds: List[Tuple[float, float]]
    timeout_recommendation: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


class ComplexityAnalyzer:
    """Advanced program complexity analyzer"""
    
    def __init__(self):
        self.complexity_thresholds = {
            'simple': (1, 8),
            'medium': (9, 20),
            'complex': (21, 40),
            'very_complex': (41, 60),
            'extreme': (61, float('inf'))
        }
        
        self.coverage_predictions = {
            'simple': (0.85, 1.0),
            'medium': (0.65, 0.85),
            'complex': (0.35, 0.65),
            'very_complex': (0.15, 0.35),
            'extreme': (0.05, 0.15)
        }
        
        self.domain_keywords = {
            'mathematical': ['calc', 'math', 'trig', 'numerical', 'solver', 'optimizer'],
            'sorting': ['sort', 'merge', 'bubble', 'quick', 'heap'],
            'data_structure': ['tree', 'bst', 'avl', 'queue', 'stack', 'list', 'hash'],
            'algorithm': ['search', 'graph', 'path', 'traversal', 'pattern', 'constraint'],
            'system': ['state', 'machine', 'cache', 'scheduler', 'processor', 'manager'],
            'game': ['game', 'player', 'move', 'score', 'guess', 'choice'],
            'enterprise': ['distributed', 'protocol', 'workflow', 'json', 'parser']
        }
    
    def analyze_program(self, file_path: str) -> ProgramComplexity:
        """Analyze a single program file comprehensively"""
        
        file_path = Path(file_path)
        program_name = file_path.stem
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception as e:
            raise ValueError(f"Cannot read file {file_path}: {e}")
        
        # Parse AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in {file_path}: {e}")
        
        # Calculate various complexity metrics
        radon_results = cc.cc_visit(source_code)
        mi_score = mi_visit(source_code, multi=True)
        halstead = h_visit(source_code)
        
        # Extract function complexities
        functions = []
        total_cc = 0
        max_cc = 0
        
        for item in radon_results:
            if hasattr(item, 'complexity'):
                func_complexity = self._analyze_function(item, tree, source_code)
                functions.append(func_complexity)
                total_cc += item.complexity
                max_cc = max(max_cc, item.complexity)
        
        # If no functions found with radon, analyze AST directly
        if not functions:
            functions, total_cc, max_cc = self._analyze_ast_functions(tree, source_code)
        
        # Calculate overall metrics
        avg_cc = total_cc / len(functions) if functions else 0
        
        # Line counting
        lines_info = self._count_lines(source_code)
        
        # Determine complexity category
        complexity_category = self._categorize_complexity(max_cc or total_cc)
        
        # Predict expected coverage
        coverage_range = self.coverage_predictions[complexity_category]
        
        # Calculate difficulty score
        difficulty_score = self._calculate_difficulty_score(total_cc, max_cc, lines_info['code_lines'])
        
        # Domain classification
        domain = self._classify_domain(program_name.lower())
        
        # Parameter analysis
        param_info = self._analyze_parameters(tree)
        
        # Generate bounds and timeout recommendations
        suggested_bounds = self._suggest_parameter_bounds(domain, param_info['dimensions'])
        timeout_rec = self._recommend_timeout(complexity_category, lines_info['code_lines'])
        
        return ProgramComplexity(
            file_path=str(file_path),
            program_name=program_name,
            total_cyclomatic_complexity=total_cc,
            average_cyclomatic_complexity=avg_cc,
            max_cyclomatic_complexity=max_cc,
            functions=functions,
            total_lines=lines_info['total_lines'],
            code_lines=lines_info['code_lines'],
            comment_lines=lines_info['comment_lines'],
            blank_lines=lines_info['blank_lines'],
            maintainability_index=mi_score,
            complexity_category=complexity_category,
            expected_coverage_range=coverage_range,
            difficulty_score=difficulty_score,
            domain_classification=domain,
            parameter_dimensions=param_info['dimensions'],
            suggested_bounds=suggested_bounds,
            timeout_recommendation=timeout_rec
        )
    
    def _analyze_function(self, radon_item, tree: ast.AST, source_code: str) -> FunctionComplexity:
        """Analyze individual function complexity"""
        
        # Find corresponding AST node
        func_node = self._find_function_node(tree, radon_item.name)
        
        # Calculate additional metrics
        nested_depth = self._calculate_nesting_depth(func_node) if func_node else 0
        branch_count = self._count_branches(func_node) if func_node else 0
        loop_count = self._count_loops(func_node) if func_node else 0
        param_count = len(func_node.args.args) if func_node and hasattr(func_node, 'args') else 0
        return_count = self._count_returns(func_node) if func_node else 0
        
        # Halstead metrics for function
        try:
            func_halstead = h_visit(self._extract_function_source(func_node, source_code))
            halstead_dict = {
                'volume': getattr(func_halstead, 'volume', 0),
                'difficulty': getattr(func_halstead, 'difficulty', 0),
                'effort': getattr(func_halstead, 'effort', 0),
                'time': getattr(func_halstead, 'time', 0),
                'bugs': getattr(func_halstead, 'bugs', 0)
            } if func_halstead else {}
        except Exception:
            halstead_dict = {}
        
        return FunctionComplexity(
            name=radon_item.name,
            cyclomatic_complexity=radon_item.complexity,
            cognitive_complexity=radon_item.complexity,  # Approximation
            lines_of_code=radon_item.endline - radon_item.lineno + 1,
            maintainability_index=0.0,  # Would need more detailed calculation
            halstead_metrics=halstead_dict,
            parameters=param_count,
            returns=return_count,
            nested_depth=nested_depth,
            branch_count=branch_count,
            loop_count=loop_count
        )
    
    def _analyze_ast_functions(self, tree: ast.AST, source_code: str) -> Tuple[List[FunctionComplexity], int, int]:
        """Analyze functions directly from AST when radon fails"""
        functions = []
        total_cc = 0
        max_cc = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                cc_score = self._calculate_cyclomatic_complexity(node)
                func_complexity = FunctionComplexity(
                    name=node.name,
                    cyclomatic_complexity=cc_score,
                    cognitive_complexity=cc_score,
                    lines_of_code=getattr(node, 'end_lineno', node.lineno) - node.lineno + 1,
                    maintainability_index=0.0,
                    halstead_metrics={},
                    parameters=len(node.args.args),
                    returns=self._count_returns(node),
                    nested_depth=self._calculate_nesting_depth(node),
                    branch_count=self._count_branches(node),
                    loop_count=self._count_loops(node)
                )
                functions.append(func_complexity)
                total_cc += cc_score
                max_cc = max(max_cc, cc_score)
        
        # If still no functions, assume main level code
        if not functions:
            main_cc = self._calculate_cyclomatic_complexity(tree)
            main_func = FunctionComplexity(
                name="<main>",
                cyclomatic_complexity=main_cc,
                cognitive_complexity=main_cc,
                lines_of_code=len(source_code.splitlines()),
                maintainability_index=0.0,
                halstead_metrics={},
                parameters=0,
                returns=1,
                nested_depth=self._calculate_nesting_depth(tree),
                branch_count=self._count_branches(tree),
                loop_count=self._count_loops(tree)
            )
            functions = [main_func]
            total_cc = main_cc
            max_cc = main_cc
        
        return functions, total_cc, max_cc
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity from AST node"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With, ast.AsyncWith):
                complexity += 1
            elif isinstance(child, ast.Assert):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # For boolean operations (and/or), add one for each additional condition
                complexity += len(child.values) - 1
        
        return complexity
    
    def _count_lines(self, source_code: str) -> Dict[str, int]:
        """Count different types of lines in source code"""
        lines = source_code.splitlines()
        total_lines = len(lines)
        blank_lines = 0
        comment_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('#'):
                comment_lines += 1
        
        code_lines = total_lines - blank_lines - comment_lines
        
        return {
            'total_lines': total_lines,
            'code_lines': code_lines,
            'comment_lines': comment_lines,
            'blank_lines': blank_lines
        }
    
    def _categorize_complexity(self, max_complexity: int) -> str:
        """Categorize complexity level based on cyclomatic complexity"""
        for category, (min_val, max_val) in self.complexity_thresholds.items():
            if min_val <= max_complexity <= max_val:
                return category
        return 'extreme'
    
    def _calculate_difficulty_score(self, total_cc: int, max_cc: int, code_lines: int) -> float:
        """Calculate normalized difficulty score (0-100)"""
        # Weighted combination of complexity factors
        cc_factor = min(max_cc / 50, 1.0) * 40  # Max 40 points for complexity
        size_factor = min(code_lines / 200, 1.0) * 30  # Max 30 points for size
        distribution_factor = (total_cc / max(max_cc, 1)) * 30  # Max 30 points for complexity distribution
        
        return cc_factor + size_factor + distribution_factor
    
    def _classify_domain(self, program_name: str) -> str:
        """Classify program domain based on name and keywords"""
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in program_name for keyword in keywords):
                return domain
        return 'general'
    
    def _analyze_parameters(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze parameter structure of main functions"""
        max_params = 0
        total_params = 0
        func_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                param_count = len(node.args.args)
                max_params = max(max_params, param_count)
                total_params += param_count
                func_count += 1
        
        # Default to 4 dimensions if no clear pattern
        dimensions = max_params if max_params > 0 else 4
        
        return {
            'dimensions': dimensions,
            'max_parameters': max_params,
            'average_parameters': total_params / max(func_count, 1)
        }
    
    def _suggest_parameter_bounds(self, domain: str, dimensions: int) -> List[Tuple[float, float]]:
        """Suggest parameter bounds based on domain and complexity"""
        domain_bounds = {
            'mathematical': (-1000, 1000),
            'sorting': (-999999, 999999),
            'data_structure': (-100, 100),
            'algorithm': (-50, 50),
            'system': (-20, 20),
            'game': (-10, 10),
            'general': (-1000, 1000)
        }
        
        bounds = domain_bounds.get(domain, (-1000, 1000))
        return [bounds] * dimensions
    
    def _recommend_timeout(self, complexity_category: str, code_lines: int) -> float:
        """Recommend execution timeout based on complexity"""
        base_timeouts = {
            'simple': 15.0,
            'medium': 30.0,
            'complex': 45.0,
            'very_complex': 60.0,
            'extreme': 120.0
        }
        
        base_timeout = base_timeouts.get(complexity_category, 30.0)
        
        # Adjust based on code size
        size_factor = 1.0 + (code_lines / 100) * 0.1
        
        return base_timeout * size_factor
    
    def _find_function_node(self, tree: ast.AST, func_name: str) -> Optional[ast.FunctionDef]:
        """Find AST node for named function"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return node
        return None
    
    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        if not node:
            return 0
            
        max_depth = 0
        current_depth = 0
        
        def visit_node(n, depth):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            if isinstance(n, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                depth += 1
            
            for child in ast.iter_child_nodes(n):
                visit_node(child, depth)
        
        visit_node(node, 0)
        return max_depth
    
    def _count_branches(self, node: ast.AST) -> int:
        """Count conditional branches"""
        if not node:
            return 0
            
        branches = 0
        for child in ast.walk(node):
            if isinstance(child, ast.If):
                branches += 1
                if child.orelse:
                    branches += 1
        return branches
    
    def _count_loops(self, node: ast.AST) -> int:
        """Count loop constructs"""
        if not node:
            return 0
            
        loops = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
                loops += 1
        return loops
    
    def _count_returns(self, node: ast.AST) -> int:
        """Count return statements"""
        if not node:
            return 0
            
        returns = 0
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                returns += 1
        return returns
    
    def _extract_function_source(self, node: ast.AST, full_source: str) -> str:
        """Extract source code for specific function (approximation)"""
        if not node:
            return ""
        
        lines = full_source.splitlines()
        start_line = node.lineno - 1
        end_line = getattr(node, 'end_lineno', len(lines)) - 1
        
        return '\n'.join(lines[start_line:end_line + 1])


class ProgramMetadataGenerator:
    """Generate comprehensive program metadata for experimental methodology"""
    
    def __init__(self, analyzer: Optional[ComplexityAnalyzer] = None):
        self.analyzer = analyzer or ComplexityAnalyzer()
        self.metadata_cache = {}
    
    def analyze_test_suite(self, test_programs_dir: str = "test_programs") -> Dict[str, Dict[str, Any]]:
        """Analyze entire test program suite"""
        
        test_dir = Path(test_programs_dir)
        if not test_dir.exists():
            raise FileNotFoundError(f"Test programs directory not found: {test_dir}")
        
        suite_metadata = {}
        
        # Get all Python files
        python_files = list(test_dir.glob("*.py"))
        python_files.extend(test_dir.rglob("*.py"))  # Include subdirectories
        
        print(f"Analyzing {len(python_files)} test programs...")
        
        for py_file in python_files:
            if py_file.name.startswith('__'):
                continue
                
            try:
                print(f"Analyzing: {py_file.name}")
                complexity_info = self.analyzer.analyze_program(py_file)
                
                # Convert to configuration format
                program_config = self._to_config_format(complexity_info)
                suite_metadata[complexity_info.program_name] = program_config
                
            except Exception as e:
                print(f"Warning: Failed to analyze {py_file}: {e}")
                # Create minimal entry
                suite_metadata[py_file.stem] = {
                    'path': str(py_file.relative_to(Path.cwd())),
                    'function': 'target_function',
                    'dimensions': 4,
                    'category': 'unknown',
                    'complexity': 'medium',
                    'cyclomatic_complexity': 10,
                    'expected_coverage': 0.5,
                    'bounds': [-1000, 1000],
                    'timeout': 30.0,
                    'error': str(e)
                }
        
        return suite_metadata
    
    def _to_config_format(self, complexity_info: ProgramComplexity) -> Dict[str, Any]:
        """Convert complexity analysis to configuration format"""
        
        try:
            file_path = Path(complexity_info.file_path)
            if file_path.is_absolute():
                # Try to make relative to current directory
                try:
                    relative_path = file_path.relative_to(Path.cwd())
                except ValueError:
                    # If can't make relative, just use the name
                    relative_path = file_path.name
            else:
                relative_path = file_path
        except Exception:
            relative_path = Path(complexity_info.file_path).name
        
        return {
            'path': str(relative_path),
            'function': 'target_function',  # Will need standardization
            'dimensions': complexity_info.parameter_dimensions,
            'category': complexity_info.domain_classification,
            'complexity': complexity_info.complexity_category,
            'cyclomatic_complexity': complexity_info.max_cyclomatic_complexity,
            'expected_coverage': float(complexity_info.expected_coverage_range[1]),  # Use upper bound as target
            'coverage_range': [float(complexity_info.expected_coverage_range[0]), float(complexity_info.expected_coverage_range[1])],
            'bounds': list(complexity_info.suggested_bounds[0]) if complexity_info.suggested_bounds else [-1000, 1000],
            'timeout': float(complexity_info.timeout_recommendation),
            'difficulty_score': float(complexity_info.difficulty_score),
            'total_lines': complexity_info.code_lines,
            'function_count': len(complexity_info.functions),
            'maintainability_index': float(complexity_info.maintainability_index),
            'description': f"{complexity_info.complexity_category.title()} complexity {complexity_info.domain_classification} program with {complexity_info.max_cyclomatic_complexity} cyclomatic complexity"
        }
    
    def export_to_yaml(self, metadata: Dict[str, Any], output_path: str = "program_metadata.yaml"):
        """Export metadata to YAML configuration format"""
        
        # Create unified config structure
        config_structure = {
            'test_programs': metadata,
            'metadata_info': {
                'generated_by': 'ComplexityAnalyzer',
                'generation_timestamp': str(Path(__file__).stat().st_mtime),
                'total_programs': len(metadata),
                'complexity_distribution': self._calculate_complexity_distribution(metadata)
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_structure, f, default_flow_style=False, sort_keys=True, indent=2)
        
        print(f"Program metadata exported to: {output_path}")
    
    def _calculate_complexity_distribution(self, metadata: Dict[str, Any]) -> Dict[str, int]:
        """Calculate distribution of complexity categories"""
        distribution = {}
        
        for program_info in metadata.values():
            category = program_info.get('complexity', 'unknown')
            distribution[category] = distribution.get(category, 0) + 1
        
        return distribution
    
    def generate_summary_report(self, metadata: Dict[str, Any]) -> str:
        """Generate human-readable summary report"""
        
        total_programs = len(metadata)
        complexity_dist = self._calculate_complexity_distribution(metadata)
        
        report = [
            "=" * 60,
            "TEST PROGRAM SUITE COMPLEXITY ANALYSIS",
            "=" * 60,
            f"Total Programs Analyzed: {total_programs}",
            "",
            "Complexity Distribution:",
        ]
        
        for category, count in sorted(complexity_dist.items()):
            percentage = (count / total_programs) * 100
            report.append(f"  {category.title()}: {count} programs ({percentage:.1f}%)")
        
        report.extend([
            "",
            "Program Details:",
            "-" * 40
        ])
        
        # Sort programs by complexity for reporting
        sorted_programs = sorted(
            metadata.items(),
            key=lambda x: x[1].get('cyclomatic_complexity', 0),
            reverse=True
        )
        
        for prog_name, prog_info in sorted_programs[:10]:  # Top 10 most complex
            cc = prog_info.get('cyclomatic_complexity', 0)
            category = prog_info.get('complexity', 'unknown').title()
            coverage = prog_info.get('expected_coverage', 0.0)
            
            report.append(f"{prog_name:25} | CC: {cc:2d} | {category:12} | Coverage: {coverage:.1%}")
        
        if len(sorted_programs) > 10:
            report.append(f"... and {len(sorted_programs) - 10} more programs")
        
        return '\n'.join(report)


def main():
    """Command line interface for complexity analysis"""
    parser = argparse.ArgumentParser(
        description="Analyze test program complexity for experimental methodology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 src/analysis/complexity_analyzer.py                    # Analyze all programs
  python3 src/analysis/complexity_analyzer.py --program minimum.py  # Single program  
  python3 src/analysis/complexity_analyzer.py --output metadata.yaml # Custom output
        """
    )
    
    parser.add_argument('--program', '-p', 
                       help='Analyze single program file')
    parser.add_argument('--directory', '-d', default='test_programs',
                       help='Test programs directory (default: test_programs)')
    parser.add_argument('--output', '-o', default='program_metadata.yaml',
                       help='Output file for metadata (default: program_metadata.yaml)')
    parser.add_argument('--format', choices=['yaml', 'json'], default='yaml',
                       help='Output format (default: yaml)')
    parser.add_argument('--report', '-r', action='store_true',
                       help='Generate summary report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        analyzer = ComplexityAnalyzer()
        generator = ProgramMetadataGenerator(analyzer)
        
        if args.program:
            # Analyze single program
            result = analyzer.analyze_program(args.program)
            print(f"Analysis of {args.program}:")
            print(f"Cyclomatic Complexity: {result.max_cyclomatic_complexity}")
            print(f"Category: {result.complexity_category}")
            print(f"Expected Coverage: {result.expected_coverage_range[0]:.1%} - {result.expected_coverage_range[1]:.1%}")
            print(f"Domain: {result.domain_classification}")
            
            if args.verbose:
                print("\nDetailed Analysis:")
                for func in result.functions:
                    print(f"  Function {func.name}: CC={func.cyclomatic_complexity}, Lines={func.lines_of_code}")
        
        else:
            # Analyze entire suite
            metadata = generator.analyze_test_suite(args.directory)
            
            # Export results
            if args.format == 'yaml':
                generator.export_to_yaml(metadata, args.output)
            else:
                with open(args.output, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                print(f"Program metadata exported to: {args.output}")
            
            # Generate report if requested
            if args.report:
                report = generator.generate_summary_report(metadata)
                print("\n" + report)
                
                # Save report to file
                report_file = Path(args.output).with_suffix('.txt')
                with open(report_file, 'w') as f:
                    f.write(report)
                print(f"\nDetailed report saved to: {report_file}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())