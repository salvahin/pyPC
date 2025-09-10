#!/usr/bin/env python3
"""
Test Program Interface Standardizer

This tool standardizes test program interfaces to follow the experimental methodology
requirements. It analyzes existing functions and creates standardized wrappers that
follow the target_function(a, b, c, d=None) interface pattern.

Features:
- Automatic detection of main functions in test programs
- Generation of standardized wrapper functions
- Parameter type inference and bounds checking
- Backup creation before modifications
- Validation of standardized interfaces
"""

import ast
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import argparse
import yaml
from dataclasses import dataclass
import inspect


@dataclass
class FunctionInfo:
    """Information about a function in a test program"""
    name: str
    parameters: List[str]
    parameter_count: int
    has_defaults: bool
    has_varargs: bool
    has_kwargs: bool
    docstring: Optional[str]
    return_annotation: Optional[str]
    is_main_function: bool
    line_start: int
    line_end: int


class ProgramStandardizer:
    """Standardizes test program interfaces"""
    
    def __init__(self):
        self.backup_dir = Path("backups/original_programs")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Load program metadata if available
        self.program_metadata = self._load_program_metadata()
        
        # Standard interface template
        self.standard_template = '''
def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original {original_function} to provide
    a consistent 4-parameter interface for automated test generation.
    
    Args:
        a: First parameter (required)
        b: Second parameter (optional, defaults to suitable value)
        c: Third parameter (optional, defaults to suitable value) 
        d: Fourth parameter (optional, defaults to suitable value)
    
    Returns:
        Result from original function
    """
    # Handle parameter defaults based on original function requirements
    {parameter_handling}
    
    # Call original function with appropriate parameters
    try:
        {function_call}
    except Exception as e:
        # Handle potential errors gracefully for test generation
        if "too many" in str(e).lower():
            # Try with fewer parameters
            {fallback_calls}
        raise e
'''
    
    def _load_program_metadata(self) -> Dict[str, Any]:
        """Load existing program metadata if available"""
        metadata_path = Path("config/program_metadata.yaml")
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    data = yaml.safe_load(f)
                return data.get('test_programs', {})
            except Exception as e:
                print(f"Warning: Could not load program metadata: {e}")
        return {}
    
    def analyze_program(self, file_path: str) -> List[FunctionInfo]:
        """Analyze a program file to identify functions"""
        
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception as e:
            raise ValueError(f"Cannot read file {file_path}: {e}")
        
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in {file_path}: {e}")
        
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = self._analyze_function_node(node, source_code)
                functions.append(func_info)
        
        # Identify the most likely main function
        main_function = self._identify_main_function(functions, file_path.stem)
        if main_function:
            main_function.is_main_function = True
        
        return functions
    
    def _analyze_function_node(self, node: ast.FunctionDef, source_code: str) -> FunctionInfo:
        """Analyze a function AST node"""
        
        # Get parameter information
        parameters = []
        has_defaults = len(node.args.defaults) > 0
        has_varargs = node.args.vararg is not None
        has_kwargs = node.args.kwarg is not None
        
        for arg in node.args.args:
            parameters.append(arg.arg)
        
        # Get docstring
        docstring = None
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
        
        # Get line numbers
        line_start = node.lineno
        line_end = getattr(node, 'end_lineno', line_start)
        
        return FunctionInfo(
            name=node.name,
            parameters=parameters,
            parameter_count=len(parameters),
            has_defaults=has_defaults,
            has_varargs=has_varargs,
            has_kwargs=has_kwargs,
            docstring=docstring,
            return_annotation=None,
            is_main_function=False,
            line_start=line_start,
            line_end=line_end
        )
    
    def _identify_main_function(self, functions: List[FunctionInfo], program_name: str) -> Optional[FunctionInfo]:
        """Identify the most likely main function to standardize"""
        
        if not functions:
            return None
        
        # Priority order for function selection
        priorities = [
            # Exact program name match
            lambda f: f.name == program_name,
            # Common main function names
            lambda f: f.name in ['main', 'target_function', 'run', 'execute'],
            # Functions with reasonable parameter counts (2-6)
            lambda f: 2 <= f.parameter_count <= 6,
            # Functions without special arguments (varargs/kwargs)
            lambda f: not f.has_varargs and not f.has_kwargs,
            # Non-private functions (don't start with _)
            lambda f: not f.name.startswith('_'),
            # Functions with parameters
            lambda f: f.parameter_count > 0
        ]
        
        # Apply priority filters
        candidates = functions[:]
        
        for priority_filter in priorities:
            filtered = [f for f in candidates if priority_filter(f)]
            if filtered:
                candidates = filtered
            if len(candidates) == 1:
                break
        
        # Return the first candidate (or the one with most parameters if tie)
        if candidates:
            return max(candidates, key=lambda f: f.parameter_count)
        
        return functions[0] if functions else None
    
    def standardize_program(self, file_path: str, dry_run: bool = False) -> bool:
        """Standardize a single program file"""
        
        file_path = Path(file_path)
        program_name = file_path.stem
        
        print(f"Standardizing: {file_path.name}")
        
        # Create backup
        if not dry_run:
            backup_path = self.backup_dir / file_path.name
            shutil.copy2(file_path, backup_path)
            print(f"  Backup created: {backup_path}")
        
        # Analyze the program
        try:
            functions = self.analyze_program(file_path)
        except Exception as e:
            print(f"  Error: Failed to analyze program: {e}")
            return False
        
        if not functions:
            print(f"  Warning: No functions found in {file_path.name}")
            return False
        
        # Find main function
        main_function = next((f for f in functions if f.is_main_function), None)
        if not main_function:
            print(f"  Warning: No suitable main function found")
            return False
        
        print(f"  Main function identified: {main_function.name}({', '.join(main_function.parameters)})")
        
        # Check if already standardized
        if main_function.name == 'target_function' and main_function.parameter_count == 4:
            print(f"  Already standardized")
            return True
        
        # Generate standardized wrapper
        wrapper_code = self._generate_wrapper(main_function, program_name)
        
        if dry_run:
            print("  Generated wrapper:")
            print("  " + "\n  ".join(wrapper_code.split('\n')[:10]))  # Show first 10 lines
            print("  ...")
            return True
        
        # Read original file
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Add wrapper to the end of the file
        modified_content = original_content.rstrip() + '\n\n' + wrapper_code
        
        # Write modified file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"  ✓ Standardized successfully")
        return True
    
    def _generate_wrapper(self, main_function: FunctionInfo, program_name: str) -> str:
        """Generate standardized wrapper code for a function"""
        
        # Get metadata for parameter suggestions
        metadata = self.program_metadata.get(program_name, {})
        bounds = metadata.get('bounds', [-1000, 1000])
        
        # Generate parameter handling logic
        param_handling = self._generate_parameter_handling(main_function, bounds)
        
        # Generate function call
        function_call = self._generate_function_call(main_function)
        
        # Generate fallback calls for error handling
        fallback_calls = self._generate_fallback_calls(main_function)
        
        wrapper = f'''
def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original {main_function.name} to provide
    a consistent 4-parameter interface for automated test generation.
    
    Original function: {main_function.name}({', '.join(main_function.parameters)})
    
    Args:
        a: First parameter (required)
        b: Second parameter (optional)
        c: Third parameter (optional) 
        d: Fourth parameter (optional)
    
    Returns:
        Result from {main_function.name}
    """
{param_handling}
    
    # Call original function with appropriate parameters
    try:
{function_call}
    except Exception as e:
        # Handle potential errors gracefully for test generation
        if "too many" in str(e).lower() or "unexpected keyword" in str(e).lower():
            # Try with fewer parameters
{fallback_calls}
        raise e
'''
        return wrapper
    
    def _generate_parameter_handling(self, func_info: FunctionInfo, bounds: List[float]) -> str:
        """Generate parameter handling logic"""
        
        min_val, max_val = bounds
        
        # Default values based on parameter position and bounds
        defaults = [
            f"0",  # a is always provided
            f"{min_val}",  # b default
            f"0",  # c default (middle value)
            f"{max_val}"  # d default
        ]
        
        handling = []
        handling.append("    # Set defaults for optional parameters based on function requirements")
        
        # Handle parameter count mismatch
        if func_info.parameter_count < 4:
            for i in range(func_info.parameter_count, 4):
                param_name = ['a', 'b', 'c', 'd'][i]
                handling.append(f"    if {param_name} is None:")
                handling.append(f"        {param_name} = {defaults[i]}")
        
        # Set defaults for None values
        for i, param_name in enumerate(['a', 'b', 'c', 'd']):
            if i > 0:  # Skip 'a' as it's required
                handling.append(f"    if {param_name} is None:")
                handling.append(f"        {param_name} = {defaults[i]}")
        
        # Add parameter validation
        handling.append(f"    ")
        handling.append(f"    # Validate parameter ranges")
        handling.append(f"    for param, name in [(a, 'a'), (b, 'b'), (c, 'c'), (d, 'd')]:")
        handling.append(f"        if param is not None and isinstance(param, (int, float)):")
        handling.append(f"            if not ({min_val} <= param <= {max_val}):")
        handling.append(f"                param = max({min_val}, min({max_val}, param))  # Clamp to bounds")
        
        return '\n'.join(handling)
    
    def _generate_function_call(self, func_info: FunctionInfo) -> str:
        """Generate the appropriate function call"""
        
        param_names = ['a', 'b', 'c', 'd']
        
        if func_info.parameter_count <= 4:
            # Use exactly the number of parameters the function expects
            call_params = param_names[:func_info.parameter_count]
            call = f"        return {func_info.name}({', '.join(call_params)})"
        else:
            # Function has more than 4 parameters - use first 4 and add defaults
            call_params = param_names[:4]
            call = f"        return {func_info.name}({', '.join(call_params)})"
        
        return call
    
    def _generate_fallback_calls(self, func_info: FunctionInfo) -> str:
        """Generate fallback function calls with fewer parameters"""
        
        fallbacks = []
        param_names = ['a', 'b', 'c', 'd']
        
        # Try with decreasing number of parameters
        for i in range(min(func_info.parameter_count - 1, 3), 0, -1):
            call_params = param_names[:i]
            fallbacks.append(f"            try:")
            fallbacks.append(f"                return {func_info.name}({', '.join(call_params)})")
            fallbacks.append(f"            except:")
            fallbacks.append(f"                pass")
        
        # Final fallback with just 'a'
        if func_info.parameter_count >= 1:
            fallbacks.append(f"            return {func_info.name}(a)")
        
        return '\n'.join(fallbacks) if fallbacks else "            pass"
    
    def standardize_test_suite(self, test_dir: str = "test_programs", 
                             dry_run: bool = False, 
                             programs: Optional[List[str]] = None) -> Dict[str, bool]:
        """Standardize entire test program suite"""
        
        test_dir = Path(test_dir)
        if not test_dir.exists():
            raise FileNotFoundError(f"Test programs directory not found: {test_dir}")
        
        # Get list of Python files
        if programs:
            python_files = [test_dir / f"{prog}.py" for prog in programs]
            python_files = [f for f in python_files if f.exists()]
        else:
            python_files = [f for f in test_dir.glob("*.py") if not f.name.startswith('__')]
        
        print(f"{'Dry run: ' if dry_run else ''}Standardizing {len(python_files)} programs...")
        print("=" * 60)
        
        results = {}
        success_count = 0
        
        for py_file in sorted(python_files):
            try:
                success = self.standardize_program(py_file, dry_run=dry_run)
                results[py_file.stem] = success
                if success:
                    success_count += 1
            except Exception as e:
                print(f"  Error processing {py_file.name}: {e}")
                results[py_file.stem] = False
        
        print("=" * 60)
        print(f"Standardization complete: {success_count}/{len(python_files)} programs successful")
        
        if not dry_run:
            print(f"Backups saved to: {self.backup_dir}")
        
        return results
    
    def validate_standardization(self, test_dir: str = "test_programs") -> Dict[str, bool]:
        """Validate that programs have been properly standardized"""
        
        test_dir = Path(test_dir)
        python_files = [f for f in test_dir.glob("*.py") if not f.name.startswith('__')]
        
        print(f"Validating {len(python_files)} standardized programs...")
        print("=" * 60)
        
        validation_results = {}
        
        for py_file in sorted(python_files):
            try:
                functions = self.analyze_program(py_file)
                has_target_function = any(f.name == 'target_function' for f in functions)
                
                if has_target_function:
                    target_func = next(f for f in functions if f.name == 'target_function')
                    is_valid = target_func.parameter_count >= 4
                    
                    print(f"{py_file.name:30} | {'✓' if is_valid else '✗'} | target_function({target_func.parameter_count} params)")
                    validation_results[py_file.stem] = is_valid
                else:
                    print(f"{py_file.name:30} | ✗ | No target_function found")
                    validation_results[py_file.stem] = False
                    
            except Exception as e:
                print(f"{py_file.name:30} | ✗ | Error: {e}")
                validation_results[py_file.stem] = False
        
        valid_count = sum(validation_results.values())
        print("=" * 60)
        print(f"Validation complete: {valid_count}/{len(python_files)} programs properly standardized")
        
        return validation_results


def main():
    """Command line interface for program standardization"""
    
    parser = argparse.ArgumentParser(
        description="Standardize test program interfaces for experimental methodology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tools/program_standardizer.py                           # Standardize all programs
  python3 tools/program_standardizer.py --dry-run                 # Preview changes only
  python3 tools/program_standardizer.py --programs minimum bubble_sort  # Specific programs
  python3 tools/program_standardizer.py --validate                # Validate standardization
        """
    )
    
    parser.add_argument('--programs', '-p', nargs='+',
                       help='Specific programs to standardize')
    parser.add_argument('--directory', '-d', default='test_programs',
                       help='Test programs directory (default: test_programs)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Preview changes without modifying files')
    parser.add_argument('--validate', '-v', action='store_true',
                       help='Validate existing standardization')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        standardizer = ProgramStandardizer()
        
        if args.validate:
            # Validate existing standardization
            results = standardizer.validate_standardization(args.directory)
            
            failed_programs = [prog for prog, success in results.items() if not success]
            if failed_programs:
                print(f"\\nPrograms needing standardization: {', '.join(failed_programs)}")
                return 1
            else:
                print(f"\\n✓ All programs properly standardized!")
                return 0
        
        else:
            # Standardize programs
            results = standardizer.standardize_test_suite(
                test_dir=args.directory,
                dry_run=args.dry_run,
                programs=args.programs
            )
            
            failed_programs = [prog for prog, success in results.items() if not success]
            if failed_programs:
                print(f"\\nFailed to standardize: {', '.join(failed_programs)}")
                return 1
            else:
                print(f"\\n✓ All programs standardized successfully!")
                return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())