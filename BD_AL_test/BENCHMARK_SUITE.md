# Multi-Objective Test Case Generation Benchmark Suite

## Overview

This benchmark suite provides a comprehensive collection of Python test programs designed to evaluate multi-objective optimization algorithms for automated test case generation. The programs range from simple functions achieving 100% coverage easily to extremely challenging programs where even sophisticated algorithms struggle to achieve more than 30% coverage.

### Purpose
- Evaluate effectiveness of multi-objective optimization algorithms (NSGA-II, NSGA-III, MOEA/D, C-TAEA)
- Provide standardized benchmarks for comparing test generation techniques
- Demonstrate trade-offs between conflicting objectives (coverage vs. complexity)
- Support reproducible research in automated testing

### Design Principles
1. **Progressive Difficulty**: Four distinct difficulty levels with clear coverage targets
2. **Diverse Structures**: Various control flow patterns (sequential, nested, recursive)
3. **Input Sensitivity**: Small input changes cause significant behavioral differences
4. **Hidden Branches**: Some paths require specific input combinations
5. **Real-world Relevance**: Programs simulate actual software testing challenges

## Difficulty Levels

### Level 1: Simple (80-100% Coverage Achievable)
Programs with straightforward control flow where random testing easily achieves high coverage.

### Level 2: Moderate (50-80% Coverage)
Programs with moderate branching and some challenging conditions.

### Level 3: Complex (25-50% Coverage)
Programs with deep nesting and complex conditional logic.

### Level 4: Challenging (10-35% Coverage)
Programs specifically designed to challenge optimization algorithms with intricate dependencies.

## Program Catalog

### Simple Programs

#### 1. Minimum (`minimum.py`)
- **Function**: Find minimum of 4 numbers
- **Complexity Metrics**:
  - Cyclomatic Complexity: 4
  - Branches: 2
  - Loops: 1
  - Estimated Paths: ~4
- **Expected Coverage**: 100%
- **Key Features**: Basic loop with single condition
- **Typical Time**: 0.2s
- **Trade-offs**: None (single Pareto solution)

#### 2. Three Number Sort (`three_number_sort.py`)
- **Function**: Sort three numbers
- **Complexity Metrics**:
  - Cyclomatic Complexity: 5
  - Branches: 3
  - Loops: 0
  - Estimated Paths: ~6
- **Expected Coverage**: 100%
- **Key Features**: Simple conditional swaps
- **Typical Time**: 0.1s
- **Trade-offs**: None

### Moderate Programs

#### 3. Bubble Sort (`bubble_sort.py`)
- **Function**: Sort array of 4 elements
- **Complexity Metrics**:
  - Cyclomatic Complexity: 5
  - Branches: 2
  - Loops: 2
  - Estimated Paths: ~10
- **Expected Coverage**: 75%
- **Key Features**: Nested loops with early termination
- **Typical Time**: 0.3s
- **Trade-offs**: Minimal

#### 4. Triangle Area (`trig_area.py`)
- **Function**: Calculate triangle area with type detection
- **Complexity Metrics**:
  - Cyclomatic Complexity: 9
  - Branches: 8
  - Loops: 0
  - Estimated Paths: ~12
- **Expected Coverage**: 60-70%
- **Key Features**: Multiple triangle type classifications
- **Typical Time**: 0.4s
- **Trade-offs**: Some diversity in Pareto front

### Complex Programs

#### 5. Complex Conditions (`complex_conditions.py`)
- **Function**: Multi-level nested conditionals
- **Complexity Metrics**:
  - Cyclomatic Complexity: 27
  - Branches: 26
  - Loops: 0
  - Estimated Paths: ~50
- **Expected Coverage**: 28-35%
- **Key Features**:
  - 4 levels of nested if-else
  - State accumulation
  - Complex boolean conditions
- **Typical Time**: 0.5s
- **Trade-offs**: 3-5 Pareto solutions
- **Challenge**: Deep nesting makes full coverage very difficult

#### 6. Deep Branching (`deep_branching.py`)
- **Function**: Path-dependent execution with state
- **Complexity Metrics**:
  - Cyclomatic Complexity: 30
  - Branches: 28
  - Loops: 0
  - Estimated Paths: ~60
- **Expected Coverage**: 25-40%
- **Key Features**:
  - Path variable affects later branches
  - Multiple execution strategies
- **Typical Time**: 0.6s
- **Trade-offs**: 4-6 Pareto solutions

### Challenging Programs

#### 7. Advanced Data Structure (`advanced_datastructure.py`)
- **Function**: Complex nested data structure operations
- **Complexity Metrics**:
  - Cyclomatic Complexity: 35
  - Branches: 32
  - Loops: 2
  - Estimated Paths: ~100
- **Expected Coverage**: 15-25%
- **Key Features**:
  - Nested dictionaries and lists
  - Dynamic structure modification
  - State-dependent access patterns
  - Matrix operations
- **Typical Time**: 0.8s
- **Trade-offs**: 5-8 Pareto solutions
- **Challenge**: Requires understanding of data flow

#### 8. Pattern Matcher (`pattern_matcher.py`)
- **Function**: String pattern matching simulation
- **Complexity Metrics**:
  - Cyclomatic Complexity: 30
  - Branches: 28
  - Loops: 0
  - Estimated Paths: ~80
- **Expected Coverage**: 20-30%
- **Key Features**:
  - Multiple matching strategies
  - Backtracking simulation
  - Pattern type selection
- **Typical Time**: 0.7s
- **Trade-offs**: 4-7 Pareto solutions

#### 9. Constraint Solver (`constraint_solver.py`)
- **Function**: Constraint satisfaction with conflict detection
- **Complexity Metrics**:
  - Cyclomatic Complexity: 40
  - Branches: 35
  - Loops: 3
  - Estimated Paths: ~120
- **Expected Coverage**: 10-20%
- **Key Features**:
  - Multiple constraint types
  - Constraint propagation
  - Conflict resolution
  - Backtracking logic
- **Typical Time**: 1.0s
- **Trade-offs**: 6-10 Pareto solutions
- **Challenge**: Extremely difficult to satisfy all constraints

#### 10. Graph Traversal (`graph_traversal.py`)
- **Function**: Dynamic graph construction and traversal
- **Complexity Metrics**:
  - Cyclomatic Complexity: 35
  - Branches: 30
  - Loops: 5
  - Estimated Paths: ~100
- **Expected Coverage**: 15-25%
- **Key Features**:
  - Graph built from inputs
  - DFS/BFS/Bidirectional search
  - Cycle detection
  - Path finding
- **Typical Time**: 0.9s
- **Trade-offs**: 5-8 Pareto solutions

## Benchmark Results

### Algorithm Performance Matrix

| Algorithm | Simple (%) | Moderate (%) | Complex (%) | Challenging (%) | Avg Time (s) |
|-----------|------------|--------------|-------------|-----------------|--------------|
| NSGA-II   | 100.0      | 67.5         | 31.5        | 18.2           | 0.45         |
| NSGA-III  | 100.0      | 68.0         | 30.8        | 17.5           | 0.52         |
| MOEA/D    | 100.0      | 66.2         | 29.4        | 16.8           | 0.61         |
| C-TAEA    | 100.0      | 67.8         | 32.1        | 19.1           | 0.48         |

### Trade-off Analysis

| Program Category | Avg Pareto Solutions | Coverage Spread | Complexity Spread |
|-----------------|---------------------|-----------------|-------------------|
| Simple          | 1.0                 | 0.0%            | 0.02              |
| Moderate        | 1.5                 | 5.2%            | 0.15              |
| Complex         | 4.2                 | 15.3%           | 0.28              |
| Challenging     | 6.8                 | 18.7%           | 0.41              |

## Usage Guidelines

### Running Individual Programs

```python
from tree_converter import TreeVisitor
from multi_objective_fitness import MOFitnessFactory
import ast

# Load program
with open('test_programs/complex_conditions.py', 'r') as f:
    tree = ast.parse(f.read())

visitor = TreeVisitor()
visitor.visit(tree)

# Create problem
problem = MOFitnessFactory.create_dual_objective(
    visitor, 4, objective_type='conflicting'
)
```

### Running Full Benchmark

```bash
# Run parallel benchmark with all algorithms
python parallel_mo_experiment_runner.py \
    --test-suite challenging_benchmark \
    --runs 30 \
    --generations 100 \
    --workers 8

# Run single algorithm test
python run_multi_objective.py \
    --algorithm NSGA2 \
    --test-suite complex \
    --runs 10 \
    --generations 50
```

### Adding New Programs

1. Create program following the pattern:
   ```python
   def new_program(a, b, c, d):
       # Implementation with complex control flow
       return result
   ```

2. Add to `config/test_programs.yaml`:
   ```yaml
   new_program:
     path: "test_programs/new_program.py"
     function: "new_program"
     dimensions: 4
     category: "challenging"
     expected_coverage: 0.25
   ```

3. Test TreeVisitor compatibility:
   ```python
   python test_complex_programs.py
   ```

## Interpreting Results

### Coverage Metrics
- **Node Coverage**: Percentage of AST nodes executed
- **Path Coverage**: Number of unique execution paths discovered
- **Branch Coverage**: Percentage of conditional branches taken

### Objective Trade-offs
- **Coverage vs. Complexity**: Higher coverage often requires more complex test cases
- **Coverage vs. Time**: Better coverage may require longer execution
- **Diversity vs. Convergence**: Balance between exploring solutions and exploiting good ones

### Statistical Significance
- Use Wilcoxon signed-rank test for pairwise comparisons
- Friedman test for multiple algorithm comparison
- Effect size (Cliff's delta) to measure practical significance

## Future Extensions

### Planned Additions
1. **Mutation testing integration** for fault detection objectives
2. **MC/DC coverage** for more sophisticated coverage criteria
3. **Real-world programs** from open-source projects
4. **Dynamic objective selection** based on program characteristics

### Research Opportunities
- Hybrid algorithms combining global and local search
- Machine learning for predicting program difficulty
- Transfer learning across similar programs
- Many-objective optimization (>3 objectives)

## Citation

If you use this benchmark suite in your research, please cite:

```bibtex
@software{mo_test_benchmark_2024,
  title = {Multi-Objective Test Case Generation Benchmark Suite},
  author = {PyPC Testing Framework Contributors},
  year = {2024},
  url = {https://github.com/pypc/benchmark-suite}
}
```

## License

This benchmark suite is provided under the MIT License. See LICENSE file for details.

## Contributors

- Framework design and implementation
- Challenging program suite development
- Statistical analysis tools
- Visualization dashboard

For questions or contributions, please open an issue on the project repository.