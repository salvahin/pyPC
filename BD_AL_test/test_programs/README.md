# Test Programs Suite

This directory contains a comprehensive collection of test programs designed to evaluate and compare different test generation approaches. The programs span various complexity levels and problem domains.

## Program Categories

### Basic Programs (Low Complexity)
These programs have simple control flow and are good for initial testing:

- **`minimum.py`** - Find minimum of 4 numbers
  - Dimensions: 4, Complexity: Low, Expected Coverage: 100%
  - Simple comparison logic, easy to achieve full coverage

- **`three_number_sort.py`** - Sort three numbers
  - Dimensions: 3, Complexity: Low, Expected Coverage: 100%  
  - Basic sorting with few conditional branches

- **`test.py`** & **`test2.py`** - Simple test functions
  - Basic test cases for framework validation

### Intermediate Programs (Medium Complexity)

- **`bubble_sort.py`** - Bubble sort implementation
  - Dimensions: 4, Complexity: Medium, Expected Coverage: 100%
  - Classic sorting algorithm with nested loops

- **`trig_area.py`** - Triangle area calculation
  - Dimensions: 3, Complexity: Medium, Expected Coverage: 80%
  - Geometric calculations with validity checks

### Complex Programs (High Complexity)

- **`complex_conditions.py`** - Deep nested conditionals
  - Dimensions: 4, Complexity: Hard, Expected Coverage: 30%
  - Multiple nested if-else statements creating complex paths

- **`deep_branching.py`** - Multi-level branching
  - Dimensions: 4, Complexity: Hard, Expected Coverage: 40%
  - Deep decision trees with path-dependent execution

- **`nested_loops.py`** - Nested loops with branches
  - Dimensions: 4, Complexity: Hard, Expected Coverage: 35%
  - Combines iteration with conditional logic

### Advanced Data Structure Programs

- **`binary_search_tree.py`** - BST operations
  - Complex tree operations with multiple methods
  - State-dependent behavior and invariant maintenance

- **`avl_tree_operations.py`** - Self-balancing AVL tree
  - Advanced tree operations with rotation logic
  - Complex invariants and balancing conditions

- **`advanced_datastructure.py`** - Complex nested data structures
  - Dimensions: 4, Complexity: Very Hard, Expected Coverage: 20%
  - Nested dictionaries with state-dependent access

### Algorithm Implementation Programs

- **`pattern_matcher.py`** - Pattern matching with backtracking
  - Dimensions: 4, Complexity: Very Hard, Expected Coverage: 25%
  - Multiple matching strategies and backtracking logic

- **`constraint_solver.py`** - Constraint satisfaction
  - Dimensions: 4, Complexity: Very Hard, Expected Coverage: 15%
  - Constraint propagation and conflict detection

- **`graph_traversal.py`** - Dynamic graph operations
  - Dimensions: 4, Complexity: Very Hard, Expected Coverage: 20%
  - DFS/BFS/Bidirectional search with cycle detection

- **`path_finder.py`** - Pathfinding algorithms
  - Grid-based navigation with obstacle avoidance

### Computational Programs

- **`recursive_calc.py`** - Recursive mathematical calculations
  - Dimensions: 3, Complexity: Hard, Expected Coverage: 25%
  - Recursive algorithms with complex base cases

- **`numerical_solver.py`** - Numerical computation methods
  - Complex mathematical algorithms
  - Iterative solvers and convergence criteria

- **`cryptographic_hash.py`** - Hash function implementations
  - Cryptographic operations with bit manipulation
  - Complex mathematical transformations

### System-Level Programs

- **`state_machine.py`** - State machine implementation
  - State transitions with complex conditions
  - Event-driven behavior patterns

- **`cache_manager.py`** - Cache management system
  - Memory management with eviction policies
  - Performance optimization logic

- **`lock_free_queue.py`** - Concurrent data structure
  - Thread-safe operations without locks
  - Complex synchronization patterns

- **`event_processor.py`** - Event processing system
  - Asynchronous event handling
  - Complex workflow management

- **`resource_scheduler.py`** - Resource allocation scheduler
  - Complex scheduling algorithms
  - Priority-based resource management

### Enterprise-Level Programs

- **`protocol_state_machine.py`** - Network protocol implementation
  - Complex protocol state management
  - Error handling and recovery mechanisms

- **`distributed_system.py`** - Distributed system simulation
  - Node communication and consensus
  - Fault tolerance and recovery

- **`workflow_engine.py`** - Business workflow engine
  - Complex process orchestration
  - Conditional workflows and branching

- **`optimization_solver.py`** - Mathematical optimization
  - Multiple optimization algorithms
  - Constraint handling and convergence

- **`json_parser_validator.py`** - JSON parsing and validation
  - Complex parsing logic with validation
  - Error recovery and format compliance

- **`matrix_optimizer.py`** - Matrix optimization operations
  - Linear algebra optimizations
  - Complex mathematical transformations

- **`signal_processor.py`** - Digital signal processing
  - Signal filtering and transformation
  - Complex mathematical operations

- **`statistical_analyzer.py`** - Statistical analysis toolkit
  - Multiple statistical methods
  - Data analysis and hypothesis testing

## Game Programs

Located in `test_game_programs/function_only_testings/`:

- **Rock-Paper-Scissors components**
  - `rock_paper_scissor_player_choice.py`
  - `rock_paper_scissor_number_to_name.py`

- **Number guessing game**
  - `guess_the_number_input_guess.py`

- **Tic-tac-toe**
  - `jogo_da_velha_python_actualizar_jogadas.py`

- **Bounce game**
  - `bounce_draw.py`

- **RPG character system**
  - `TRPG_character_create_character.py`

## Complexity Analysis

### Cyclomatic Complexity Distribution

- **Low (1-5)**: Basic programs, linear flow
- **Medium (6-15)**: Moderate branching, some loops
- **High (16-30)**: Complex control flow, nested structures
- **Very High (31-50)**: Deep nesting, multiple decision points
- **Extreme (50+)**: Enterprise-level complexity

### Expected Coverage Ranges

- **90-100%**: Simple programs with straightforward logic
- **70-90%**: Moderate complexity with some hard-to-reach paths
- **40-70%**: Complex programs with deep branching
- **20-40%**: Very complex programs with intricate conditions
- **10-20%**: Extremely complex enterprise-level programs

## Usage Guidelines

### For Baseline Method Testing
Start with simple programs to validate your implementation:
```python
# Good starting programs
programs = ["minimum", "three_number_sort", "bubble_sort"]
```

### For Multi-Objective Algorithm Evaluation
Use programs with clear trade-offs between coverage and other objectives:
```python
# Programs showing MO trade-offs
programs = ["complex_conditions", "deep_branching", "constraint_solver"]
```

### For Comprehensive Comparison
Include a mix across all complexity levels:
```python
# Balanced test suite
programs = [
    "minimum",                    # Simple
    "bubble_sort",               # Medium
    "complex_conditions",        # Complex
    "advanced_datastructure",    # Very Complex
    "distributed_system"         # Extreme
]
```

## Adding New Test Programs

To add a new test program:

1. Create the Python file with a `target_function` that accepts numeric parameters
2. Add configuration to `config/unified_config.yaml`:

```yaml
test_programs:
  my_program:
    path: "test_programs/my_program.py"
    function: "target_function"
    dimensions: 3
    category: "custom"
    complexity: "medium"
    cyclomatic_complexity: 12
    expected_coverage: 0.8
    bounds: [-100, 100]
    timeout: 30.0
```

3. Test your program:
```bash
python main.py baseline --method random --programs my_program
```

## Program Selection Strategies

### Progressive Testing
1. Start with low complexity: `minimum`, `three_number_sort`
2. Add medium complexity: `bubble_sort`, `trig_area`
3. Include high complexity: `complex_conditions`, `deep_branching`
4. Challenge with very high: `advanced_datastructure`, `constraint_solver`

### Domain-Specific Testing
- **Sorting**: `bubble_sort`, `three_number_sort`
- **Mathematical**: `minimum`, `trig_area`, `numerical_solver`
- **Data Structures**: `binary_search_tree`, `avl_tree_operations`
- **Algorithms**: `pattern_matcher`, `graph_traversal`, `constraint_solver`
- **Systems**: `state_machine`, `cache_manager`, `distributed_system`

### Research Benchmarking
For academic comparisons, use programs with known characteristics:
- Well-defined complexity metrics
- Established expected coverage ranges
- Diverse algorithmic challenges
- Reproducible results

---

Each program is designed to test different aspects of automated test generation, from simple boundary conditions to complex algorithmic behaviors. The diversity ensures comprehensive evaluation of different test generation approaches.