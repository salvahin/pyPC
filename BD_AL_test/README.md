# Unified Test Generation Framework

A comprehensive framework for comparing baseline test generation methods against multi-objective optimization algorithms for automated test case generation.

## Overview

This framework provides a unified interface for:
- **Baseline test generation** using traditional methods (random, adaptive random, quasi-random, etc.)
- **Multi-objective optimization** for test generation (NSGA-II, NSGA-III, MOEA/D, C-TAEA)
- **Statistical comparison** and analysis of different approaches
- **Comprehensive evaluation** across diverse test programs

## Quick Start

```bash
# Run a quick demo
python main.py demo

# Run baseline experiment with specific methods
python main.py baseline --method random adaptive_random --n-tests 50

# Run multi-objective experiment
python main.py multi-objective --algorithm NSGA2 NSGA3 --generations 100

# Compare baseline vs multi-objective results
python main.py compare --baseline results/baseline.json --mo results/mo.json

# List available methods
python main.py list --type baseline
python main.py list --type mo
```

## Project Structure

```
├── main.py                    # Unified entry point
├── config/
│   └── unified_config.yaml   # Consolidated configuration
├── src/
│   ├── algorithms/
│   │   └── baseline/
│   │       └── generators.py # Baseline test generators
│   ├── evaluation/
│   │   └── evaluator.py      # Test evaluation framework
│   └── analysis/
│       └── statistical.py    # Statistical analysis tools
├── test_programs/             # Test programs for evaluation
├── results/                   # Organized experiment results
│   ├── baseline/             # Baseline method results
│   ├── multi_objective/      # MO algorithm results
│   ├── comparisons/          # Comparative analyses
│   └── archived/             # Historical results
└── docs/                     # Documentation and reports
    ├── reports/              # Generated analysis reports
    ├── analysis/             # Statistical analysis outputs
    └── benchmarks/           # Benchmark configurations
```

## Algorithms

### Baseline Methods
- **Random Testing**: Pure random test case generation
- **Adaptive Random Testing**: Distance-based test case selection
- **Quasi-Random Testing**: Low-discrepancy sequence testing
- **Grid Search**: Systematic grid-based exploration
- **Boundary Value Analysis**: Focus on boundary conditions
- **Hill Climbing**: Local search optimization

### Multi-Objective Algorithms
- **NSGA-II**: Non-dominated Sorting Genetic Algorithm II
- **NSGA-III**: Many-objective optimization with reference directions
- **MOEA/D**: Multi-Objective Evolutionary Algorithm based on Decomposition
- **C-TAEA**: Constrained Two-Archive Evolutionary Algorithm

## Test Programs

The framework includes various test programs with different complexity levels:

### Simple Programs
- **minimum**: Find minimum of 4 numbers
- **three_number_sort**: Sort three numbers
- **triangle_area**: Calculate triangle area from sides

### Complex Programs
- **bubble_sort**: Bubble sort implementation
- **complex_conditions**: Deep nested conditionals
- **deep_branching**: Multi-level branching
- **nested_loops**: Nested loops with branches

## Usage Examples

### Basic Baseline Comparison

```bash
# Run standard baseline comparison
python main.py baseline \
  --method random adaptive_random quasi_random \
  --programs minimum bubble_sort triangle_area \
  --repetitions 10 \
  --output results/baseline_comparison.json
```

### Multi-Objective Experiment

```bash
# Run MO algorithms comparison
python main.py multi-objective \
  --algorithm NSGA2 NSGA3 MOEAD \
  --programs complex_conditions deep_branching \
  --generations 150 \
  --population 100 \
  --repetitions 5 \
  --output results/mo_comparison.json
```

### Statistical Analysis

```bash
# Perform comprehensive comparison
python main.py compare \
  --baseline results/baseline_comparison.json \
  --mo results/mo_comparison.json \
  --output results/statistical_analysis/

# Direct statistical analysis of results
python main.py analyze \
  --data results/experiment_results.json \
  --output docs/analysis/statistical_report
```

## Configuration

The framework uses a unified configuration file (`config/unified_config.yaml`) that consolidates:

- Algorithm parameters
- Test program definitions
- Experiment configurations
- Statistical analysis settings
- Output and reporting options

### Key Configuration Sections

```yaml
# Baseline methods configuration
baseline_methods:
  random:
    parameters:
      bounds: [[-1000, 1000], [-1000, 1000]]
      seed: null

# Multi-objective algorithms
multi_objective_algorithms:
  NSGA2:
    parameters:
      population_size: 100
      generations: 100

# Test suites
test_suites:
  quick:
    programs: ["minimum", "three_number_sort"]
    max_time_minutes: 5
    
  comprehensive:
    programs: ["minimum", "bubble_sort", "triangle_area", "complex_conditions"]
    max_time_minutes: 120
```

## Statistical Analysis

The framework provides comprehensive statistical analysis including:

- **Non-parametric tests**: Mann-Whitney U, Wilcoxon signed-rank, Friedman
- **Effect size calculations**: Cohen's d, Hedges' g, Cliff's delta
- **Multiple comparison corrections**: Bonferroni, FDR-BH
- **Bootstrap confidence intervals**
- **Critical difference analysis**

### Metrics Analyzed
- Coverage achieved
- Branch distance
- Approach level
- Execution time
- Memory usage
- Hypervolume (for MO algorithms)
- Inverted Generational Distance (IGD)

## Results and Reporting

The framework generates multiple output formats:

### Result Files
- **JSON**: Detailed experimental results
- **CSV**: Tabular data for analysis
- **TXT**: Human-readable reports

### Analysis Reports
- **Statistical summaries** with significance tests
- **Method performance rankings**
- **Effect size analysis**
- **Visualization plots** (boxplots, heatmaps, Pareto fronts)

### Report Locations
- `results/baseline/`: Baseline experiment results
- `results/multi_objective/`: MO algorithm results
- `results/comparisons/`: Comparative analysis reports
- `docs/reports/`: Generated analysis reports
- `docs/analysis/`: Statistical analysis outputs

## Advanced Features

### Parallel Execution
The framework supports parallel execution for faster experiments:

```bash
# Set number of parallel processes
export PYTHONPATH="${PYTHONPATH}:src"
python main.py multi-objective --workers 4
```

### Custom Test Programs
Add new test programs by updating `config/unified_config.yaml`:

```yaml
test_programs:
  my_program:
    path: "test_programs/my_program.py"
    function: "target_function"
    dimensions: 3
    bounds: [-100, 100]
    complexity: "medium"
```

### Experiment Presets
Use predefined experiment configurations:

```yaml
experiments:
  my_experiment:
    baseline_methods: ["random", "adaptive_random"]
    mo_algorithms: ["NSGA2", "NSGA3"]
    test_suite: "standard"
    repetitions: 20
```

## Performance Considerations

- **Memory Usage**: Large populations and generations can consume significant memory
- **Execution Time**: Complex test programs may require longer timeouts
- **Disk Space**: Results can accumulate quickly; use `results/archived/` for storage
- **Parallel Processing**: Adjust worker count based on available CPU cores

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes the `src` directory
2. **Memory Issues**: Reduce population size or number of parallel processes
3. **Timeout Errors**: Increase timeout values in configuration
4. **Missing Dependencies**: Install required packages:
   ```bash
   pip install numpy scipy pandas matplotlib seaborn statsmodels
   ```

### Debug Mode
Enable verbose logging for debugging:

```bash
python main.py --verbose demo
```

## Contributing

1. **Adding Baseline Methods**: Extend `src/algorithms/baseline/generators.py`
2. **Adding Test Programs**: Create programs in `test_programs/` and update config
3. **Extending Analysis**: Modify `src/analysis/statistical.py`
4. **Documentation**: Update relevant sections in `docs/`

## Research Applications

This framework is designed for academic research in:

- **Search-Based Software Testing** (SBST)
- **Multi-Objective Optimization** for test generation
- **Automated Test Case Generation**
- **Software Testing Tool Evaluation**
- **Metaheuristic Algorithm Comparison**

## References

- Deb, K., et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II
- Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition
- Li, K., et al. (2019). Two-archive evolutionary algorithm for constrained multiobjective optimization

## License

This framework is provided for academic and research use. Please cite appropriately if used in publications.

---

For detailed API documentation, see `docs/` directory.
For usage examples, see experiment configurations in `config/unified_config.yaml`.
For questions or issues, please refer to the troubleshooting section above.