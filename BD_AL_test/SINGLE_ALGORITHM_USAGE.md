# Single Algorithm Runner - Usage Guide

## Overview
The `run_single_algorithm.py` script allows you to run a single metaheuristic algorithm on all test programs and observe detailed results.

## Basic Usage

### Run PSO on all test programs
```bash
python3 run_single_algorithm.py --algorithm PSO
```

### Run with custom parameters
```bash
python3 run_single_algorithm.py --algorithm GA --pop-size 200 --generations 150 --runs 5
```

### Run algorithm variant
```bash
python3 run_single_algorithm.py --algorithm PSO --variant aggressive
```

### Run DE algorithm quietly without plots
```bash
python3 run_single_algorithm.py --algorithm DE --quiet --no-plot
```

## Command-Line Options

- `--algorithm, -a` (required): Algorithm name (PSO, GA, DE, etc.)
- `--variant, -v`: Algorithm variant (e.g., aggressive, conservative)
- `--runs, -r`: Number of runs per test program (default: 1)
- `--generations, -g`: Maximum generations (default: 100)
- `--pop-size`: Population size (overrides default)
- `--output, -o`: Output directory (default: results)
- `--no-plot`: Disable plotting
- `--quiet, -q`: Minimal output
- `--list-algorithms`: List available algorithms and exit

## Available Algorithms

### Swarm Intelligence
- **PSO**: Particle Swarm Optimization
  - Variants: standard, aggressive, conservative

### Evolutionary Algorithms
- **GA**: Genetic Algorithm
  - Variants: standard, large_pop, small_pop
- **DE**: Differential Evolution
  - Variants: best1bin, rand2bin, currenttobest
- **ES**: Evolution Strategy
- **G3PCX**: Generalized Generation Gap with PCX
- **BRKGA**: Biased Random Key Genetic Algorithm
- **CMAES**: Covariance Matrix Adaptation ES (requires min 2 dimensions)
- **SRES**: Stochastic Ranking ES
- **ISRES**: Improved SRES

### Gradient-Free Optimization
- **NelderMead**: Nelder-Mead Simplex
- **PatternSearch**: Pattern Search

## Output Files

Results are saved in the specified output directory with timestamp:
- `results.csv`: Detailed metrics for each run
- `results.json`: Complete metrics in JSON format
- `convergence_data.json`: Convergence history for each run
- `config.json`: Configuration used for the experiment
- `{algorithm}_analysis.png`: Performance visualization plots (if enabled)

## Examples

### 1. Quick Test with PSO
```bash
python3 run_single_algorithm.py --algorithm PSO --runs 3 --generations 50
```

### 2. Compare PSO Variants
```bash
# Standard PSO
python3 run_single_algorithm.py --algorithm PSO --variant standard --output results/pso_standard

# Aggressive PSO
python3 run_single_algorithm.py --algorithm PSO --variant aggressive --output results/pso_aggressive

# Conservative PSO
python3 run_single_algorithm.py --algorithm PSO --variant conservative --output results/pso_conservative
```

### 3. High-Performance GA Run
```bash
python3 run_single_algorithm.py --algorithm GA --pop-size 200 --generations 200 --runs 10
```

### 4. Test Different Algorithms
```bash
# PSO
python3 run_single_algorithm.py --algorithm PSO --runs 5

# Genetic Algorithm
python3 run_single_algorithm.py --algorithm GA --runs 5

# Differential Evolution
python3 run_single_algorithm.py --algorithm DE --runs 5

# Nelder-Mead (no population)
python3 run_single_algorithm.py --algorithm NelderMead --runs 5
```

## Understanding Results

### Console Output
- **Per Test Program**: Shows dimensions, category, and per-run results
- **Summary Table**: Displays best/mean fitness, coverage, and execution time
- **Overall Statistics**: Aggregated metrics across all test programs

### Metrics Explained
- **Best Fitness**: Lowest fitness value achieved (lower is better)
- **Coverage**: Percentage of code branches covered (100% is ideal)
- **Execution Time**: Time taken for optimization
- **Convergence Generation**: Generation where best solution was found
- **Stagnation**: Number of generations without improvement

### Visualization (if enabled)
1. **Coverage Bar Chart**: Shows code coverage for each test program
2. **Convergence Curves**: Fitness improvement over generations
3. **Execution Time**: Time distribution across test programs
4. **Fitness Distribution**: Box plot or bar chart of fitness values

## Tips for Best Results

1. **Start Small**: Use fewer generations and runs for initial testing
2. **Increase Gradually**: Once working, increase generations and runs
3. **Use Variants**: Try different variants to find the best for your problem
4. **Multiple Runs**: Use multiple runs (5-30) for statistical validity
5. **Save Results**: Use different output directories for comparison

## Troubleshooting

- If an algorithm fails on certain test programs, it will continue with others
- Check the `results.json` file for detailed error information
- Some algorithms (like CMAES) have dimension requirements
- Adjust population size if getting poor results

## Academic Use

For academic papers, recommended settings:
```bash
python3 run_single_algorithm.py --algorithm PSO --runs 30 --generations 100 --output results/paper_data
```

This provides statistically significant results suitable for publication.