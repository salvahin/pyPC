# Revised Experimental Methodology: Implementation Guide for Test Generation Comparison Study

## CRITICAL IMPLEMENTATION NOTICE
**This document serves as the authoritative implementation guide for conducting rigorous experimental comparison of test generation algorithms. All experiments must use REAL DATA from actual algorithm executions. Mock data, synthetic results, or placeholder values are STRICTLY PROHIBITED. Every data point must come from actual algorithm runs on real test programs.**

## 1. Executive Summary

This guide provides complete specifications for implementing a comparative study of baseline test generation methods versus multi-objective evolutionary algorithms. The methodology has been revised to address critical design issues including statistical power, computational fairness, and proper randomization.

### 1.1 Core Requirements
- **Real Data Only**: Every result must come from actual algorithm execution
- **Reproducibility**: All experiments must be fully reproducible with provided seeds
- **Statistical Rigor**: Proper power analysis, multiple comparison correction, and effect size reporting
- **Fair Comparison**: Normalized computational budgets across all algorithms

## 2. Experimental Design Corrections

### 2.1 A Priori Power Analysis
```python
# IMPLEMENTATION REQUIREMENT: Calculate required sample size BEFORE experiments
required_sample_size = {
    "target_power": 0.80,  # 80% power to detect effects
    "alpha": 0.05,         # Type I error rate
    "expected_effect_size": 0.5,  # Medium effect from literature
    "calculation_method": "two-tailed Mann-Whitney U test",
    "minimum_repetitions": 30,  # Increased from 10 based on power analysis
    "adjustment_for_multiple_comparisons": True
}
```

### 2.2 Computational Budget Normalization
```python
# CRITICAL: Equal computational budgets for fair comparison
computational_budgets = {
    "baseline_methods": {
        "fitness_evaluations": 5000,  # Matched to MO algorithms
        "implementation": "Generate 5000 test cases for random methods",
        "rationale": "Equal computational effort across all algorithms"
    },
    "multi_objective_algorithms": {
        "fitness_evaluations": 5000,  # 50 population × 100 generations
        "implementation": "Standard evolutionary parameters",
        "tracking": "Count actual fitness evaluations, not generations"
    }
}
```

### 2.3 Hierarchical Testing Strategy
```python
testing_hierarchy = {
    "level_1": "Test algorithm categories (baseline vs. multi-objective)",
    "level_2": "Test within-category differences only if level_1 significant",
    "level_3": "Individual algorithm comparisons with stricter alpha",
    "alpha_adjustment": {
        "level_1": 0.05,
        "level_2": 0.025,
        "level_3": 0.01
    }
}
```

## 3. Complete Algorithm Specifications

### 3.1 Baseline Algorithms with Normalized Budgets
```python
baseline_algorithms = {
    "random_testing": {
        "test_cases": 5000,
        "implementation": "Uniform random sampling",
        "parameters": {"seed": "experiment_seed + repetition"}
    },
    "adaptive_random": {
        "test_cases": 5000,
        "candidates_per_selection": 10,
        "distance_threshold": 0.1
    },
    "quasi_random": {
        "test_cases": 5000,
        "sequence": "Halton",
        "scrambling": True
    },
    "grid_search": {
        "total_points": 5000,
        "distribution": "Uniform across dimensions"
    },
    "boundary_value": {
        "boundary_tests": "3^n where n = parameters",
        "random_fill": "Remaining to reach 5000 total"
    },
    "hill_climbing": {
        "total_evaluations": 5000,
        "restarts": "As many as fit in budget",
        "step_size": 0.1
    }
}
```

### 3.2 Multi-Objective Algorithms
```python
mo_algorithms = {
    "NSGA-II": {
        "population_size": 50,
        "generations": 100,
        "total_evaluations": 5000,
        "crossover_prob": 0.9,
        "mutation_prob": 1/n_variables
    },
    "NSGA-III": {
        "population_size": 52,  # Must match reference directions
        "generations": 96,       # To stay under 5000 evaluations
        "reference_directions": "Das-Dennis systematic"
    },
    "MOEA/D": {
        "population_size": 50,
        "generations": 100,
        "neighborhood_size": 20,
        "decomposition": "Tchebycheff"
    },
    "C-TAEA": {
        "population_size": 50,
        "generations": 100,
        "archive_size": 50
    }
}
```

### 3.3 Additional Baseline: Symbolic Execution (Optional)
```python
symbolic_execution = {
    "tool": "KLEE or equivalent",
    "timeout": 30,
    "max_paths": 5000,
    "note": "Include if feasible, otherwise document as limitation"
}
```

## 4. Test Program Suite Specifications

### 4.1 Program Selection with Real-World Validation
```python
program_categories = {
    "academic_benchmarks": {
        "count": 16,
        "source": "Traditional test generation benchmarks",
        "examples": ["triangle", "bubble_sort", "binary_search"]
    },
    "real_world_functions": {
        "count": 16,
        "source": "Open source projects (Apache Commons, GNU Coreutils)",
        "selection_criteria": "Most frequently called functions",
        "complexity_range": "Match academic benchmark distribution"
    },
    "stratification": {
        "simple": "33% (CC 4-8)",
        "medium": "33% (CC 9-20)", 
        "complex": "34% (CC 21+)"
    }
}
```

### 4.2 Program Metadata Collection
```python
# REQUIRED: Collect for EVERY program before experiments
program_metadata = {
    "program_name": str,
    "cyclomatic_complexity": int,  # Measured using radon or similar
    "lines_of_code": int,
    "number_of_branches": int,
    "number_of_parameters": int,
    "parameter_types": list,
    "parameter_bounds": list,
    "domain": str,  # e.g., "sorting", "mathematical", "string_processing"
    "source": str,  # "academic" or "project_name"
    "theoretical_max_coverage": float,  # Some branches may be unreachable
    "mutation_score_baseline": float  # If mutation testing included
}
```

## 5. Execution Protocol with Randomization

### 5.1 Block Randomization Design
```python
execution_schedule = {
    "randomization": "Latin square design",
    "blocks": "Time-based (morning, afternoon, evening)",
    "within_block": "Random permutation of algorithm-program pairs",
    "implementation": """
    import numpy as np
    from itertools import product
    
    # Generate all algorithm-program pairs
    pairs = list(product(algorithms, programs))
    
    # For each repetition
    for rep in range(30):
        # Shuffle pairs with repetition-specific seed
        np.random.seed(base_seed + rep)
        np.random.shuffle(pairs)
        
        # Execute in randomized order
        for algorithm, program in pairs:
            run_experiment(algorithm, program, rep)
    """
}
```

### 5.2 Convergence Criteria for Evolutionary Algorithms
```python
convergence_criteria = {
    "method": "Improvement threshold",
    "window": 10,  # generations
    "threshold": 0.001,  # relative improvement
    "implementation": """
    def has_converged(history, window=10, threshold=0.001):
        if len(history) < window:
            return False
        recent = history[-window:]
        improvement = (recent[-1] - recent[0]) / (recent[0] + 1e-10)
        return abs(improvement) < threshold
    """,
    "early_stopping": True,
    "minimum_generations": 20
}
```

## 6. Metrics and Data Collection

### 6.1 Primary Metrics (ALL REQUIRED)
```python
primary_metrics = {
    "branch_coverage": {
        "type": float,
        "range": [0.0, 1.0],
        "collection": "Actual execution with coverage tool",
        "tool": "coverage.py or equivalent"
    },
    "mutation_score": {
        "type": float,
        "range": [0.0, 1.0],
        "collection": "Apply mutation testing tool",
        "tool": "mutmut or equivalent",
        "optional": False  # Now required for quality assessment
    },
    "fitness_value": {
        "type": float,
        "range": [0.0, float('inf')],
        "collection": "Sum of branch distances"
    },
    "execution_time": {
        "type": float,
        "unit": "seconds",
        "collection": "Wall clock time excluding setup"
    },
    "fitness_evaluations": {
        "type": int,
        "collection": "Count every fitness calculation"
    }
}
```

### 6.2 Multi-Objective Specific Metrics
```python
mo_metrics = {
    "hypervolume": {
        "reference_point": "Nadir point of all solutions",
        "normalization": "Required before calculation"
    },
    "igd": {
        "reference_set": "Best known Pareto front or union of all runs"
    },
    "spacing": {
        "measure": "Distribution uniformity"
    },
    "convergence_generation": {
        "type": int,
        "collection": "Generation where convergence detected"
    }
}
```

### 6.3 Data Storage Format
```python
# CRITICAL: Every field must contain REAL measured values
experiment_result = {
    "metadata": {
        "timestamp": "ISO 8601 format",
        "machine_id": "Unique identifier for hardware",
        "software_versions": {
            "python": "3.x.x",
            "numpy": "x.x.x",
            "pymoo": "x.x.x"
        }
    },
    "experiment": {
        "algorithm": str,
        "program": str,
        "repetition": int,
        "random_seed": int,
        "execution_order": int  # Position in randomized schedule
    },
    "results": {
        "branch_coverage": float,  # REAL measurement
        "mutation_score": float,   # REAL measurement
        "fitness_value": float,    # REAL measurement
        "execution_time": float,   # REAL measurement
        "fitness_evaluations": int,  # REAL count
        "timeout": bool,
        "error": str or None
    },
    "test_suite": {
        "test_cases": list,  # Actual generated tests
        "coverage_per_test": list  # Individual test contributions
    },
    "mo_specific": {  # Only for MO algorithms
        "hypervolume": float,
        "igd": float,
        "pareto_front": list,
        "convergence_generation": int
    }
}
```

## 7. Statistical Analysis Pipeline

### 7.1 Analysis Workflow
```python
statistical_pipeline = {
    "step_1": "Data validation and cleaning",
    "step_2": "Descriptive statistics calculation",
    "step_3": "Normality testing (Shapiro-Wilk)",
    "step_4": "Hierarchical hypothesis testing",
    "step_5": "Effect size calculation with CIs",
    "step_6": "Cross-validation analysis",
    "step_7": "Failure mode analysis",
    "step_8": "Interaction effect analysis"
}
```

### 7.2 Mixed Effects Model
```python
# Superior to multiple pairwise comparisons
mixed_model_spec = {
    "model": "coverage ~ algorithm_type + complexity + (1|program)",
    "fixed_effects": ["algorithm_type", "complexity", "interaction"],
    "random_effects": ["program"],
    "implementation": """
    import statsmodels.formula.api as smf
    
    model = smf.mixedlm(
        "coverage ~ algorithm_category * complexity_category", 
        data=results_df,
        groups=results_df["program"],
        re_formula="1"
    )
    """
}
```

### 7.3 Cross-Validation Implementation
```python
cross_validation = {
    "method": "Leave-one-program-out",
    "implementation": """
    for held_out_program in programs:
        train_programs = [p for p in programs if p != held_out_program]
        
        # Train: Optimize algorithm parameters on training programs
        best_params = optimize_parameters(algorithm, train_programs)
        
        # Test: Evaluate on held-out program
        test_performance = evaluate(algorithm, held_out_program, best_params)
        
        cv_results.append({
            'held_out': held_out_program,
            'train_performance': train_perf,
            'test_performance': test_performance
        })
    """
}
```

## 8. Tables and Visualizations Specifications

### 8.1 Table 1: Test Program Characteristics
```python
table_1_spec = {
    "format": "LaTeX/Markdown table",
    "columns": [
        "Category",
        "Program",
        "CC",
        "LOC", 
        "Branches",
        "Parameters",
        "Domain",
        "Source"
    ],
    "grouping": "By complexity category",
    "sorting": "Within category by CC",
    "data_source": "program_metadata collected in Section 4.2",
    "note": "MUST be actual measured values, not estimates"
}
```

### 8.2 Figure 1: Primary Performance Comparison
```python
figure_1_spec = {
    "type": "Multi-panel box plot with violin overlay",
    "layout": "2x3 grid",
    "panels": [
        "Coverage-Simple", "Coverage-Medium", "Coverage-Complex",
        "MutationScore-Simple", "MutationScore-Medium", "MutationScore-Complex"
    ],
    "implementation": """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, metric in enumerate(['coverage', 'mutation_score']):
        for j, complexity in enumerate(['simple', 'medium', 'complex']):
            ax = axes[i, j]
            data = results_df[
                (results_df['complexity_category'] == complexity)
            ]
            
            # Violin plot
            parts = ax.violinplot(
                [data[data['algorithm'] == alg][metric].values 
                 for alg in algorithms],
                positions=range(len(algorithms)),
                showmeans=True
            )
            
            # Overlay box plot
            ax.boxplot(
                [data[data['algorithm'] == alg][metric].values 
                 for alg in algorithms],
                positions=range(len(algorithms))
            )
            
            # Add significance indicators
            add_significance_bars(ax, statistical_results)
    """,
    "data_requirements": "Actual experimental results from 30 repetitions",
    "statistical_overlay": "Significance bars with p-values"
}
```

### 8.3 Table 2: Statistical Comparison Matrix
```python
table_2_spec = {
    "format": "Heat map table",
    "structure": """
    | Algorithm Pair    | Coverage         | Mutation Score   | Time            |
    |                  | p-val | g [CI]   | p-val | g [CI]   | p-val | g [CI]  |
    |------------------|-------|----------|-------|----------|-------|---------|
    | NSGA-II vs Random| 0.001 | 0.82     | 0.003 | 0.65     | 0.234 | -0.12   |
    |                  |       |[0.5,1.1] |       |[0.3,0.9] |       |[-0.4,0.2]|
    """,
    "color_coding": {
        "p_values": "Gradient from green (significant) to red (not significant)",
        "effect_sizes": "Blue (negative) to white (zero) to orange (positive)"
    },
    "data_source": "Statistical analysis results from Section 7",
    "confidence_intervals": "Bootstrap with 10,000 resamples"
}
```

### 8.4 Figure 2: Complexity Scaling Analysis
```python
figure_2_spec = {
    "type": "Scatter plot with LOESS regression",
    "axes": {
        "x": "Cyclomatic Complexity (log scale)",
        "y": "Coverage Achieved"
    },
    "implementation": """
    import numpy as np
    from scipy.interpolate import UnivariateSpline
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algorithm in algorithms:
        alg_data = results_df[results_df['algorithm'] == algorithm]
        
        # Aggregate by program
        program_means = alg_data.groupby(['program', 'complexity']).agg({
            'coverage': 'mean',
            'execution_time': 'mean'
        }).reset_index()
        
        # Scatter plot with size proportional to time
        ax.scatter(
            program_means['complexity'],
            program_means['coverage'],
            s=program_means['execution_time'] * 10,
            alpha=0.6,
            label=algorithm
        )
        
        # LOESS smoothing
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(
            program_means['coverage'],
            np.log(program_means['complexity']),
            frac=0.3
        )
        ax.plot(np.exp(smoothed[:, 0]), smoothed[:, 1], linewidth=2)
    
    ax.set_xscale('log')
    """,
    "annotations": "Mark complexity category boundaries"
}
```

### 8.5 Figure 3: Pareto Front Analysis
```python
figure_3_spec = {
    "type": "2D scatter with Pareto fronts",
    "objectives": {
        "x": "1 - Coverage (minimization)",
        "y": "Execution Time (log scale)"
    },
    "implementation": """
    def compute_pareto_front(points):
        # points: array of [1-coverage, time]
        sorted_points = points[points[:, 0].argsort()]
        pareto_front = [sorted_points[0]]
        
        for point in sorted_points[1:]:
            if point[1] < pareto_front[-1][1]:
                pareto_front.append(point)
        
        return np.array(pareto_front)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for algorithm in algorithms:
        alg_data = results_df[results_df['algorithm'] == algorithm]
        points = np.column_stack([
            1 - alg_data['coverage'].values,
            alg_data['execution_time'].values
        ])
        
        # Plot all points
        ax.scatter(points[:, 0], points[:, 1], alpha=0.3)
        
        # Compute and plot Pareto front
        pareto = compute_pareto_front(points)
        ax.plot(pareto[:, 0], pareto[:, 1], linewidth=2, marker='o')
    
    ax.set_yscale('log')
    """,
    "shading": "Dominated regions with transparency"
}
```

### 8.6 Table 3: Algorithm Rankings with Critical Difference
```python
table_3_spec = {
    "format": "Ranked table with CD indicators",
    "implementation": """
    from scipy.stats import friedmanchisquare, rankdata
    import scikit_posthocs as sp
    
    # Prepare data matrix: algorithms x programs
    data_matrix = []
    for algorithm in algorithms:
        algorithm_scores = []
        for program in programs:
            score = results_df[
                (results_df['algorithm'] == algorithm) & 
                (results_df['program'] == program)
            ]['coverage'].mean()
            algorithm_scores.append(score)
        data_matrix.append(algorithm_scores)
    
    # Friedman test
    stat, p_value = friedmanchisquare(*data_matrix)
    
    # Post-hoc Nemenyi test
    if p_value < 0.05:
        nemenyi_results = sp.posthoc_nemenyi_friedman(
            np.array(data_matrix).T
        )
    
    # Generate ranking table with significance groups
    """,
    "visualization": "Include CD diagram below table"
}
```

### 8.7 Figure 4: Convergence Analysis
```python
figure_4_spec = {
    "type": "Multi-line plot with confidence bands",
    "data": "MO algorithms only",
    "implementation": """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, complexity in enumerate(['simple', 'medium', 'complex']):
        ax = axes[idx]
        
        for algorithm in mo_algorithms:
            alg_data = convergence_data[
                (convergence_data['algorithm'] == algorithm) &
                (convergence_data['complexity'] == complexity)
            ]
            
            # Calculate mean and CI across repetitions
            mean_hv = alg_data.groupby('generation')['hypervolume'].mean()
            std_hv = alg_data.groupby('generation')['hypervolume'].std()
            ci = 1.96 * std_hv / np.sqrt(30)  # 30 repetitions
            
            generations = mean_hv.index
            ax.plot(generations, mean_hv, label=algorithm, linewidth=2)
            ax.fill_between(
                generations,
                mean_hv - ci,
                mean_hv + ci,
                alpha=0.2
            )
            
            # Mark average convergence point
            conv_gen = alg_data['convergence_generation'].mean()
            ax.axvline(conv_gen, linestyle='--', alpha=0.5)
    """,
    "annotations": "Convergence points for each algorithm"
}
```

### 8.8 Supplementary Figure S1: Distribution Details
```python
figure_s1_spec = {
    "type": "Ridge plot (joy plot)",
    "implementation": """
    import joypy
    
    fig, axes = joypy.joyplot(
        results_df,
        by='algorithm',
        column='coverage',
        figsize=(10, 8),
        colormap=plt.cm.viridis,
        alpha=0.7
    )
    """,
    "purpose": "Show full distribution shapes"
}
```

### 8.9 Figure S2: Failure Analysis
```python
figure_s2_spec = {
    "type": "Stacked bar chart",
    "categories": ["Success", "Timeout", "Invalid Tests", "Error"],
    "implementation": """
    failure_data = results_df.groupby(['algorithm', 'complexity_category']).agg({
        'timeout': 'sum',
        'error': lambda x: (x != None).sum(),
        'coverage': lambda x: (x > 0).sum()  # Success
    })
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # ... stacking logic ...
    """,
    "data_requirement": "Actual failure counts from experiments"
}
```

### 8.10 Interactive Dashboard Specification
```python
dashboard_spec = {
    "framework": "Plotly Dash or Streamlit",
    "components": [
        {
            "name": "Algorithm Selector",
            "type": "Multi-select dropdown",
            "options": "All algorithms"
        },
        {
            "name": "Metric Selector", 
            "type": "Radio buttons",
            "options": ["Coverage", "Mutation Score", "Time", "Efficiency"]
        },
        {
            "name": "Program Filter",
            "type": "Range slider",
            "parameter": "Cyclomatic complexity"
        },
        {
            "name": "Statistical Comparison",
            "type": "Dynamic table",
            "updates": "Based on selections"
        }
    ],
    "export_options": ["PNG", "SVG", "CSV", "JSON"]
}
```

## 9. Quality Assurance Checklist

### 9.1 Pre-Execution Validation
```python
pre_execution_checks = [
    "Power analysis completed and sample size justified",
    "All algorithms implemented and tested on toy examples",
    "Computational budgets verified as equal",
    "Randomization schedule generated and saved",
    "Hardware specifications documented",
    "Software versions frozen and documented"
]
```

### 9.2 During Execution Monitoring
```python
runtime_checks = [
    "Memory usage within bounds",
    "No systematic failures for specific algorithm-program pairs",
    "Execution times reasonable (no infinite loops)",
    "Coverage values in valid range [0, 1]",
    "Fitness evaluations match expected budgets",
    "Random seeds properly logged"
]
```

### 9.3 Post-Execution Validation
```python
post_execution_checks = [
    "All expected data points present (no missing values beyond timeouts)",
    "Statistical assumptions tested and documented",
    "Effect sizes have confidence intervals",
    "Multiple comparison corrections applied",
    "Cross-validation completed",
    "Reproducibility verified on subset"
]
```

## 10. Common Pitfalls to Avoid

### 10.1 Data Collection Errors
- **NEVER** use placeholder or synthetic data
- **NEVER** interpolate missing values without documentation
- **ALWAYS** record actual execution times, not estimates
- **ALWAYS** count real fitness evaluations

### 10.2 Statistical Errors
- **DON'T** use post-hoc power analysis to justify sample size
- **DON'T** p-hack by testing multiple outcomes without correction
- **DON'T** report only significant results
- **DO** report all planned analyses regardless of outcome

### 10.3 Visualization Errors
- **DON'T** use 3D plots for 2D data
- **DON'T** truncate y-axes to exaggerate differences
- **DO** show confidence intervals or error bars
- **DO** use colorblind-friendly palettes

## 11. Implementation Timeline

### 11.1 Development Phase (Weeks 1-2)
- Implement all algorithms with equal computational budgets
- Validate on subset of programs
- Ensure data collection pipeline works correctly

### 11.2 Pilot Study (Week 3)
- Run 5 repetitions on 8 programs (2 per category)
- Verify statistical pipeline
- Estimate execution time for full study

### 11.3 Full Execution (Weeks 4-6)
- Run complete experiment (30 reps × 32 programs × 10 algorithms)
- Monitor progress and catch failures early
- Backup data incrementally

### 11.4 Analysis Phase (Weeks 7-8)
- Statistical analysis with hierarchical testing
- Generate all tables and figures
- Write results section

## 12. Code Implementation Structure

```python
project_structure = """
test_generation_study/
├── src/
│   ├── algorithms/
│   │   ├── baseline/
│   │   │   ├── random_testing.py
│   │   │   ├── adaptive_random.py
│   │   │   ├── quasi_random.py
│   │   │   ├── grid_search.py
│   │   │   ├── boundary_value.py
│   │   │   └── hill_climbing.py
│   │   └── multi_objective/
│   │       ├── nsga2.py
│   │       ├── nsga3.py
│   │       ├── moead.py
│   │       └── ctaea.py
│   ├── evaluation/
│   │   ├── fitness_evaluator.py
│   │   ├── coverage_analyzer.py
│   │   └── mutation_testing.py
│   ├── analysis/
│   │   ├── statistical_tests.py
│   │   ├── effect_sizes.py
│   │   ├── cross_validation.py
│   │   └── visualization.py
│   └── utils/
│       ├── data_validator.py
│       ├── randomization.py
│       └── progress_tracker.py
├── test_programs/
│   ├── academic/
│   └── real_world/
├── experiments/
│   ├── run_pilot.py
│   ├── run_full_experiment.py
│   └── run_analysis.py
├── results/
│   ├── raw_data/
│   ├── processed_data/
│   ├── figures/
│   └── tables/
├── config/
│   ├── algorithm_params.yaml
│   ├── experiment_config.yaml
│   └── analysis_config.yaml
└── requirements.txt
"""
```

## 13. Final Implementation Notes

1. **Every data point must be real** - No interpolation, no synthetic data, no mockups
2. **Document everything** - Hardware, software versions, random seeds, failures
3. **Validate continuously** - Check data quality at each step
4. **Be transparent** - Report all results, not just favorable ones
5. **Ensure reproducibility** - Anyone should be able to replicate your results

## 14. Deliverables Checklist

- [ ] Raw experimental data (JSON/CSV format)
- [ ] Statistical analysis results with all tests
- [ ] Tables 1-3 as specified
- [ ] Figures 1-4 as specified
- [ ] Supplementary figures S1-S2
- [ ] Interactive dashboard (if applicable)
- [ ] Complete source code with documentation
- [ ] Reproducibility package with seeds and configs
- [ ] Technical report with methodology details
- [ ] Manuscript draft with results section

---

**CRITICAL REMINDER**: This document specifies a rigorous experimental methodology. Cutting corners or using mock data will invalidate the entire study. Every result must come from actual algorithm executions on real programs with proper statistical analysis.
