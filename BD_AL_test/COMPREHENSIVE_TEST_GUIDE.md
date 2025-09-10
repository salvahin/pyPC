# Enhanced Comprehensive Test Runner Guide

## Overview

The updated `run_comprehensive_tests.py` script provides a complete implementation of the experimental methodology from EXPERIMENTAL_METHODOLOGY.md, including enhanced evaluation metrics, statistical analysis, visualization, and publication tools.

## Quick Start

```bash
# Standard smoke test with enhanced features
python3 run_comprehensive_tests.py --mode smoke --enhanced

# Full comprehensive test with publication outputs
python3 run_comprehensive_tests.py --enhanced --publication

# Quick validation with specific programs
python3 run_comprehensive_tests.py --quick --programs minimum bubble_sort --enhanced

# Baseline methods only with enhanced metrics
python3 run_comprehensive_tests.py --mode baseline --enhanced
```

## New Enhanced Features

### 🚀 **Enhanced Framework Integration**
- **Enhanced Evaluation Metrics**: Branch distance, approach level, memory tracking
- **Advanced Statistical Analysis**: Mann-Whitney U, FDR correction, effect sizes
- **Interactive Visualizations**: Critical difference diagrams, interactive dashboards
- **Publication Tools**: LaTeX tables, citation management, replication packages

### 🔧 **New Command Line Options**

```bash
--enhanced              # Use enhanced evaluation metrics and analysis
--publication          # Generate publication-ready outputs
--no-analysis          # Skip statistical analysis (experiments only)
--no-visualization     # Skip visualization generation
```

### ⚙️ **Enhanced Configuration**

The runner now uses three parameter sets optimized for different scenarios:

#### **Full Parameters** (Production Research)
```yaml
baseline:
  n_tests: 100
  repetitions: 10
  enhanced_metrics: true
  timeout: 30.0

mo:
  generations: 100
  population_size: 50
  enhanced_metrics: true
  objective_type: traditional

analysis:
  statistical_tests: true
  effect_sizes: true
  power_analysis: true
  bootstrap_samples: 10000
```

#### **Quick Parameters** (Validation)
```yaml
baseline:
  n_tests: 50
  repetitions: 5
  enhanced_metrics: true

mo:
  generations: 50
  population_size: 100
  enhanced_metrics: true

analysis:
  effect_sizes: true
  bootstrap_samples: 5000
```

#### **Smoke Parameters** (Testing)
```yaml
baseline:
  n_tests: 20
  repetitions: 3
  enhanced_metrics: false

mo:
  generations: 20
  population_size: 20
  enhanced_metrics: false

analysis:
  statistical_tests: true
  # Minimal analysis for speed
```

## Enhanced Output Structure

The runner creates a comprehensive directory structure:

```
results/comprehensive_test_MODE_TIMESTAMP/
├── experiment_config.json              # Complete experiment configuration
├── baseline_results/
│   └── unified_baseline_results.json   # Enhanced baseline metrics
├── mo_results/
│   └── unified_mo_results.json         # Enhanced MO metrics
├── statistical_analysis/
│   ├── comprehensive_statistical_analysis.json
│   └── statistical_report.html         # Interactive statistical report
├── visualizations/
│   ├── index.html                       # Visualization index
│   ├── interactive_dashboard.html       # Interactive dashboard
│   └── *.png                           # Generated plots
├── publication/
│   ├── results_table.tex               # LaTeX tables
│   ├── references.bib                  # Bibliography
│   └── replication_package/            # Complete replication package
└── logs/                               # Execution logs
```

## Usage Examples

### 1. **Academic Research (Full Pipeline)**
```bash
python3 run_comprehensive_tests.py \
    --enhanced \
    --publication \
    --programs minimum bubble_sort trig_area complex_conditions
```
- ✅ Enhanced evaluation metrics
- ✅ Complete statistical analysis with effect sizes
- ✅ Interactive visualizations and critical difference diagrams
- ✅ Publication-ready LaTeX tables and citations
- ✅ Replication package for reproducibility

### 2. **Quick Development Testing**
```bash
python3 run_comprehensive_tests.py \
    --mode smoke \
    --quick \
    --enhanced \
    --no-visualization
```
- ✅ Fast execution (~1-2 minutes)
- ✅ Enhanced metrics validation
- ✅ Basic statistical analysis
- ❌ Skip time-consuming visualizations

### 3. **Baseline Methods Focus**
```bash
python3 run_comprehensive_tests.py \
    --mode baseline \
    --enhanced \
    --baseline-methods random adaptive_random quasi_random
```
- ✅ Focus on baseline method comparison
- ✅ Enhanced evaluation metrics
- ✅ Complete analysis pipeline

### 4. **Multi-Objective Focus**
```bash
python3 run_comprehensive_tests.py \
    --mode mo \
    --enhanced \
    --mo-algorithms NSGA2 NSGA3 \
    --publication
```
- ✅ Focus on MO algorithm comparison
- ✅ Enhanced multi-objective metrics (hypervolume, IGD)
- ✅ Publication outputs

## Enhanced Methodology Implementation

### **Statistical Analysis Framework**
- **Non-parametric Tests**: Mann-Whitney U, Kruskal-Wallis
- **Multiple Comparison Correction**: FDR (Benjamini-Hochberg)
- **Effect Sizes**: Hedges' g, Cliff's delta, Vargha-Delaney A
- **Bootstrap Confidence Intervals**: BCa method (10,000 iterations)
- **Power Analysis**: Post-hoc and prospective power calculation

### **Advanced Visualization**
- **Critical Difference Diagrams**: Demšar's method implementation
- **Interactive Dashboards**: Multi-panel Plotly dashboards
- **Pareto Front Analysis**: MO optimization visualization
- **Statistical Heatmaps**: Significance matrices
- **Effect Size Plots**: Visual effect magnitude assessment

### **Publication Tools**
- **LaTeX Table Generation**: Publication-ready booktabs format
- **Automated Citations**: Complete bibliography generation
- **Paper Templates**: IEEE/ACM/Springer formats
- **Replication Packages**: Docker containerization support

## Configuration Integration

The runner automatically loads:
- **`config/unified_config.yaml`**: Framework configuration
- **`config/program_metadata.yaml`**: Program complexity database

## Performance Benchmarks

| Mode | Duration | Features | Use Case |
|------|----------|----------|-----------|
| Smoke | 1-2 min | Basic validation | Development testing |
| Quick | 10-30 min | Enhanced metrics | Validation runs |
| Full | 2-8 hours | Complete pipeline | Academic research |

## Error Handling and Diagnostics

The enhanced runner provides:
- **Graceful Degradation**: Falls back to legacy methods if enhanced framework unavailable
- **Detailed Error Reporting**: Component-specific failure isolation
- **Progress Monitoring**: Real-time status updates
- **Diagnostic Information**: Detailed configuration and feature reporting

## Integration with Methodology

This enhanced runner implements **100% of the requirements** from EXPERIMENTAL_METHODOLOGY.md:

- ✅ **Section 4**: All 6 baseline methods + 4 MO algorithms
- ✅ **Section 5**: Complete parameter configurations
- ✅ **Section 6**: Enhanced evaluation metrics (branch distance, approach level)
- ✅ **Section 7**: Advanced statistical analysis framework
- ✅ **Section 8**: Robust experimental procedures
- ✅ **Section 10**: Complete visualization suite
- ✅ **Section 11**: Publication tools and replication packages
- ✅ **Section 13**: Academic publication pipeline

## Next Steps

After running experiments:

1. **Review Statistical Analysis**: Open `statistical_analysis/statistical_report.html`
2. **Explore Visualizations**: Open `visualizations/interactive_dashboard.html`
3. **Check Publication Outputs**: Review `publication/` directory
4. **Use Replication Package**: Deploy `publication/replication_package/`
5. **Submit for Publication**: Use generated LaTeX tables and citations

The enhanced framework provides a complete, production-ready implementation of the experimental methodology for academic software engineering research.