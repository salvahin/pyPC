# Comprehensive Smoke Test Guide

## Overview

The `smoke_test_comprehensive.py` script provides complete validation of the Unified Test Generation Framework, testing all core components, analysis tools, and reporting capabilities as specified in the EXPERIMENTAL_METHODOLOGY.md.

## Quick Start

```bash
# Run standard smoke test (recommended)
python3 smoke_test_comprehensive.py

# Run quick test mode (faster, reduced parameters)
python3 smoke_test_comprehensive.py --quick

# Run with minimal output
python3 smoke_test_comprehensive.py --quiet

# Save detailed JSON report
python3 smoke_test_comprehensive.py --save-report smoke_test_results.json
```

## Test Coverage

### 🔧 Core Framework Components
- ✅ **Configuration Loading**: YAML config files validation
- ✅ **Program Metadata System**: Complexity analyzer and program database
- ✅ **Baseline Methods**: All 6 baseline test generators
- ✅ **Multi-Objective Algorithms**: All 4 MO algorithms (NSGA-II, NSGA-III, MOEA/D, C-TAEA)
- ✅ **Enhanced Evaluation Metrics**: Branch distance, approach level, memory tracking

### 📊 Analysis and Reporting Tools
- ✅ **Statistical Analysis Framework**: Non-parametric tests, effect sizes, power analysis
- ✅ **Advanced Visualization Suite**: Interactive dashboards, critical difference diagrams
- ✅ **Publication Tools**: LaTeX generation, citation system, replication packages
- ✅ **Report Generation**: JSON/CSV export, summary reports, meta-analysis

### 🔄 Integration Testing
- ✅ **End-to-End Mini Experiment**: Complete workflow validation

## Execution Time

- **Standard Mode**: ~15-30 minutes
- **Quick Mode**: ~5-15 minutes
- **Minimal Parameters**: Optimized for speed while maintaining coverage

## Test Parameters

### Standard Mode
```yaml
baseline:
  n_tests: 15
  repetitions: 3
  timeout: 10.0

multi_objective:
  generations: 10
  population_size: 20
  repetitions: 3
  timeout: 15.0
```

### Quick Mode
```yaml
baseline:
  n_tests: 10
  repetitions: 2
  timeout: 5.0

multi_objective:
  generations: 5
  population_size: 10
  repetitions: 2
  timeout: 10.0
```

## Test Programs Used

The smoke test uses a representative subset of 6 programs across complexity levels:

1. **minimum** (Simple, CC=3)
2. **three_number_sort** (Simple, CC=4)
3. **bubble_sort** (Simple, CC=5)
4. **trig_area** (Medium, CC=14)
5. **complex_conditions** (Complex, CC=32)
6. **deep_branching** (Complex, CC=31)

## Output and Results

### Console Output
- 🔍 Real-time test progress
- ✅/❌ Pass/fail status for each component
- ⚡ Performance timing information
- 📊 Comprehensive summary report

### Generated Files
- **Temporary working directory**: `/tmp/smoke_test_*/`
- **Test visualizations**: Boxplots, critical difference diagrams
- **Sample reports**: JSON, CSV, summary text files
- **LaTeX tables**: Publication-ready table examples

### Success Criteria
- **90%+ success rate**: Framework ready for production
- **75-89% success rate**: Good, address failed components
- **<75% success rate**: Framework needs attention

## Command Line Options

```bash
python3 smoke_test_comprehensive.py [OPTIONS]

Options:
  -v, --verbose     Enable verbose output (default)
  -q, --quiet       Minimal output mode
  --quick          Quick test mode (reduced parameters)
  --save-report    Save detailed JSON report to file
  --no-cleanup     Do not clean up temporary files
  -h, --help       Show help message
```

## Example Usage Scenarios

### 1. Development Validation
```bash
# Quick validation during development
python3 smoke_test_comprehensive.py --quick --quiet
```

### 2. Pre-Production Testing
```bash
# Full validation before production deployment
python3 smoke_test_comprehensive.py --save-report production_test.json
```

### 3. CI/CD Integration
```bash
# Automated testing in CI/CD pipeline
python3 smoke_test_comprehensive.py --quiet
if [ $? -eq 0 ]; then
    echo "All tests passed"
else
    echo "Tests failed"
    exit 1
fi
```

### 4. Debugging Mode
```bash
# Keep temporary files for debugging
python3 smoke_test_comprehensive.py --verbose --no-cleanup
```

## Interpreting Results

### Success Indicators
- ✅ All core components load successfully
- ✅ Baseline methods generate test cases
- ✅ MO algorithms create instances without errors
- ✅ Evaluation metrics compute correctly
- ✅ Statistical analysis functions work
- ✅ Visualization files are generated
- ✅ Publication tools produce output

### Warning Signs
- ⚠️ Import errors for framework modules
- ⚠️ Configuration files missing or malformed
- ⚠️ Test program evaluation failures
- ⚠️ Statistical analysis errors
- ⚠️ Visualization generation failures

### Critical Failures
- ❌ Core modules cannot be imported
- ❌ No test programs found
- ❌ Algorithm creation consistently fails
- ❌ Evaluation system non-functional

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   Solution: Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

2. **Missing Test Programs**
   ```
   Solution: Verify test_programs/ directory exists with .py files
   ```

3. **Configuration Errors**
   ```
   Solution: Check config/ directory for valid YAML files
   ```

4. **Memory Issues**
   ```
   Solution: Use --quick mode or increase available memory
   ```

5. **Timeout Errors**
   ```
   Solution: Increase timeout parameters in the script
   ```

### Debug Information

Use verbose mode to see detailed error traces:
```bash
python3 smoke_test_comprehensive.py --verbose --no-cleanup
```

Check temporary directory for intermediate files:
```bash
ls /tmp/smoke_test_*/results/
```

## Integration with Main Framework

The smoke test integrates seamlessly with the main framework:

- Uses same configuration files (`config/unified_config.yaml`)
- Tests actual framework modules (not mocks)
- Validates real program metadata
- Exercises complete analysis pipeline

## Performance Benchmarks

Expected performance on standard hardware:

| Component | Time Range | Notes |
|-----------|------------|--------|
| Configuration | <1s | YAML loading |
| Baseline Methods | 5-15s | Test generation |
| MO Algorithms | 10-30s | Instance creation |
| Evaluation | 5-10s | Metrics calculation |
| Statistical Analysis | 2-5s | Sample data analysis |
| Visualization | 5-15s | Plot generation |
| Publication Tools | 3-8s | LaTeX/citation generation |

## Continuous Integration

For CI/CD pipelines, use exit codes:
- **0**: All tests passed (≥75% success rate)
- **1**: Tests failed (<75% success rate)
- **130**: Interrupted by user
- **Other**: System error

## Extending the Smoke Test

To add new test components:

1. Add test method to `ComprehensiveSmokeTest` class
2. Call it in `run_all_tests()` method
3. Update this documentation

Example:
```python
def test_new_component(self) -> Dict[str, Any]:
    """Test new framework component"""
    try:
        # Test implementation
        return {'success': True, 'details': 'Component working'}
    except Exception as e:
        return {'success': False, 'error': str(e)}
```

## Support

For issues with the smoke test:
1. Check this guide for common solutions
2. Run with `--verbose` for detailed diagnostics
3. Review framework logs and error messages
4. Verify system requirements and dependencies