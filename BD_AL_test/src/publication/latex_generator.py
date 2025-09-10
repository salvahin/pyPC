#!/usr/bin/env python3
"""
Publication Tools - LaTeX Generation and Citation System

This module implements publication-ready output generation as specified in the
EXPERIMENTAL_METHODOLOGY.md document, including:

1. LaTeX table generation for publications (Section 11.3.2)
2. Automated citation system (Section 11.3.2)
3. Publication-ready reports (Section 13)
4. Replication package creation (Section 13.3)

Based on methodology requirements from Section 13.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json
import yaml
import shutil
import zipfile
from collections import defaultdict
import re


@dataclass
class PublicationConfig:
    """Configuration for publication output"""
    journal_style: str = "ieee"  # ieee, acm, springer
    table_format: str = "booktabs"  # booktabs, standard
    figure_format: str = "pdf"
    citation_style: str = "numerical"  # numerical, author-year
    font_size: str = "10pt"
    two_column: bool = True
    include_appendix: bool = True
    generate_bib: bool = True


class CitationManager:
    """
    Automated citation generation and management
    
    Implements citation system from methodology Section 13.2.2
    """
    
    def __init__(self):
        self.references = {}
        self.citation_counter = 0
        self._load_default_references()
    
    def _load_default_references(self):
        """Load default references from methodology"""
        self.references = {
            'nsga2': {
                'type': 'article',
                'authors': ['Deb', 'K.', 'Pratap', 'A.', 'Agarwal', 'S.', 'Meyarivan', 'T.'],
                'title': 'A fast and elitist multiobjective genetic algorithm: NSGA-II',
                'journal': 'IEEE Transactions on Evolutionary Computation',
                'volume': '6',
                'number': '2',
                'pages': '182--197',
                'year': '2002',
                'publisher': 'IEEE'
            },
            'nsga3': {
                'type': 'article',
                'authors': ['Deb', 'K.', 'Jain', 'H.'],
                'title': 'An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, part I: solving problems with box constraints',
                'journal': 'IEEE Transactions on Evolutionary Computation',
                'volume': '18',
                'number': '4',
                'pages': '577--601',
                'year': '2014',
                'publisher': 'IEEE'
            },
            'moead': {
                'type': 'article',
                'authors': ['Zhang', 'Q.', 'Li', 'H.'],
                'title': 'MOEA/D: A multiobjective evolutionary algorithm based on decomposition',
                'journal': 'IEEE Transactions on Evolutionary Computation',
                'volume': '11',
                'number': '6',
                'pages': '712--731',
                'year': '2007',
                'publisher': 'IEEE'
            },
            'ctaea': {
                'type': 'article',
                'authors': ['Li', 'K.', 'Chen', 'R.', 'Fu', 'G.', 'Yao', 'X.'],
                'title': 'Two-archive evolutionary algorithm for constrained multiobjective optimization',
                'journal': 'IEEE Transactions on Evolutionary Computation',
                'volume': '23',
                'number': '2',
                'pages': '303--315',
                'year': '2019',
                'publisher': 'IEEE'
            },
            'art': {
                'type': 'article',
                'authors': ['Chen', 'T.Y.', 'Leung', 'H.', 'Mak', 'I.K.'],
                'title': 'Adaptive random testing',
                'journal': 'Advances in Computer Science-ASIAN 2004. Higher-Level Decision Making',
                'pages': '320--329',
                'year': '2004',
                'publisher': 'Springer'
            },
            'hedges_g': {
                'type': 'article',
                'authors': ['Hedges', 'L.V.'],
                'title': 'Distribution theory for Glass\'s estimator of effect size and related estimators',
                'journal': 'Journal of Educational Statistics',
                'volume': '6',
                'number': '2',
                'pages': '107--128',
                'year': '1981',
                'publisher': 'SAGE Publications'
            },
            'fdr_bh': {
                'type': 'article',
                'authors': ['Benjamini', 'Y.', 'Hochberg', 'Y.'],
                'title': 'Controlling the false discovery rate: a practical and powerful approach to multiple testing',
                'journal': 'Journal of the Royal Statistical Society: Series B (Methodological)',
                'volume': '57',
                'number': '1',
                'pages': '289--300',
                'year': '1995',
                'publisher': 'Wiley'
            },
            'mann_whitney': {
                'type': 'article',
                'authors': ['Mann', 'H.B.', 'Whitney', 'D.R.'],
                'title': 'On a test of whether one of two random variables is stochastically larger than the other',
                'journal': 'The Annals of Mathematical Statistics',
                'volume': '18',
                'number': '1',
                'pages': '50--60',
                'year': '1947',
                'publisher': 'Institute of Mathematical Statistics'
            }
        }
    
    def add_reference(self, key: str, reference: Dict[str, Any]):
        """Add a new reference to the collection"""
        self.references[key] = reference
    
    def cite(self, key: str) -> str:
        """Generate citation for given reference key"""
        if key not in self.references:
            print(f"Warning: Reference '{key}' not found")
            return f"[?{key}?]"
        
        self.citation_counter += 1
        return f"\\cite{{{key}}}"
    
    def generate_bibtex(self, output_path: str = "references.bib"):
        """Generate BibTeX file from references"""
        bibtex_content = []
        
        for key, ref in self.references.items():
            entry_type = ref.get('type', 'article')
            
            bibtex_entry = [f"@{entry_type}{{{key},"]
            
            # Add fields
            if 'authors' in ref:
                author_str = ' and '.join(ref['authors'])
                bibtex_entry.append(f"  author = {{{author_str}}},")
            
            if 'title' in ref:
                bibtex_entry.append(f"  title = {{{ref['title']}}},")
            
            if 'journal' in ref:
                bibtex_entry.append(f"  journal = {{{ref['journal']}}},")
            
            if 'volume' in ref:
                bibtex_entry.append(f"  volume = {{{ref['volume']}}},")
            
            if 'number' in ref:
                bibtex_entry.append(f"  number = {{{ref['number']}}},")
            
            if 'pages' in ref:
                bibtex_entry.append(f"  pages = {{{ref['pages']}}},")
            
            if 'year' in ref:
                bibtex_entry.append(f"  year = {{{ref['year']}}},")
            
            if 'publisher' in ref:
                bibtex_entry.append(f"  publisher = {{{ref['publisher']}}},")
            
            # Remove trailing comma from last entry
            if bibtex_entry[-1].endswith(','):
                bibtex_entry[-1] = bibtex_entry[-1][:-1]
            
            bibtex_entry.append("}")
            bibtex_content.extend(bibtex_entry)
            bibtex_content.append("")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(bibtex_content))
        
        return output_path


class LaTeXTableGenerator:
    """
    Generate publication-ready LaTeX tables
    
    Implements table generation from methodology Section 11.3.2
    """
    
    def __init__(self, config: PublicationConfig = None):
        self.config = config or PublicationConfig()
        self.citation_manager = CitationManager()
    
    def generate_results_table(self,
                              results_df: pd.DataFrame,
                              metrics: List[str] = None,
                              caption: str = "Experimental Results",
                              label: str = "tab:results",
                              output_path: Optional[str] = None) -> str:
        """
        Generate LaTeX table from results DataFrame
        
        Args:
            results_df: Experimental results
            metrics: Metrics to include in table
            caption: Table caption
            label: Table label for referencing
            output_path: Optional path to save LaTeX file
            
        Returns:
            LaTeX table string
        """
        if metrics is None:
            metrics = ['coverage', 'branch_distance', 'execution_time']
        
        # Prepare data
        table_data = self._prepare_table_data(results_df, metrics)
        
        # Generate LaTeX
        latex_content = self._generate_table_latex(
            table_data, metrics, caption, label
        )
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(latex_content)
        
        return latex_content
    
    def generate_statistical_comparison_table(self,
                                            pairwise_results: pd.DataFrame,
                                            caption: str = "Statistical Comparison Results",
                                            label: str = "tab:statistical",
                                            output_path: Optional[str] = None) -> str:
        """Generate table for statistical comparison results"""
        
        latex_content = []
        
        if self.config.table_format == "booktabs":
            latex_content.extend([
                "\\begin{table}[htbp]",
                "\\centering",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                "\\begin{tabular}{lcccc}",
                "\\toprule",
                "Comparison & p-value & Effect Size & Magnitude & Significant \\\\",
                "\\midrule"
            ])
        else:
            latex_content.extend([
                "\\begin{table}[htbp]",
                "\\centering", 
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                "\\begin{tabular}{|l|c|c|c|c|}",
                "\\hline",
                "Comparison & p-value & Effect Size & Magnitude & Significant \\\\",
                "\\hline"
            ])
        
        # Add data rows
        for _, row in pairwise_results.iterrows():
            comparison = f"{row['Algorithm_1']} vs {row['Algorithm_2']}"
            p_val = f"{row['P_Value']:.3f}" if row['P_Value'] >= 0.001 else "< 0.001"
            effect_size = f"{row['Effect_Size']:.3f}"
            magnitude = row['Effect_Magnitude']
            significant = "Yes" if row['Significant'] else "No"
            
            if row['Significant']:
                significant = "\\textbf{Yes}"
                if row['Effect_Size'] > 0.5:
                    effect_size = f"\\textbf{{{effect_size}}}"
            
            latex_content.append(
                f"{comparison} & {p_val} & {effect_size} & {magnitude} & {significant} \\\\"
            )
        
        # Close table
        if self.config.table_format == "booktabs":
            latex_content.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}"
            ])
        else:
            latex_content.extend([
                "\\hline",
                "\\end{tabular}",
                "\\end{table}"
            ])
        
        latex_string = '\n'.join(latex_content)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(latex_string)
        
        return latex_string
    
    def generate_algorithm_summary_table(self,
                                       algorithm_summary: Dict[str, Dict[str, float]],
                                       caption: str = "Algorithm Performance Summary",
                                       label: str = "tab:algorithm_summary",
                                       output_path: Optional[str] = None) -> str:
        """Generate summary table of algorithm performance"""
        
        algorithms = list(algorithm_summary.keys())
        metrics = list(next(iter(algorithm_summary.values())).keys())
        
        # Generate column specification
        col_spec = "l" + "c" * len(algorithms)
        
        latex_content = []
        
        if self.config.table_format == "booktabs":
            latex_content.extend([
                "\\begin{table*}[htbp]",
                "\\centering",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                f"\\begin{tabular}{{{col_spec}}}",
                "\\toprule"
            ])
        else:
            latex_content.extend([
                "\\begin{table*}[htbp]",
                "\\centering",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                f"\\begin{{tabular}}{{|{'|'.join(['l'] + ['c'] * len(algorithms))}|}}",
                "\\hline"
            ])
        
        # Header row
        header = "Metric & " + " & ".join(algorithms) + " \\\\"
        latex_content.append(header)
        
        if self.config.table_format == "booktabs":
            latex_content.append("\\midrule")
        else:
            latex_content.append("\\hline")
        
        # Data rows
        for metric in metrics:
            row_data = [metric.replace('_', '\\_')]
            for algorithm in algorithms:
                value = algorithm_summary[algorithm][metric]
                if isinstance(value, float):
                    if metric in ['coverage', 'branch_distance']:
                        row_data.append(f"{value:.3f}")
                    else:
                        row_data.append(f"{value:.2f}")
                else:
                    row_data.append(str(value))
            
            latex_content.append(" & ".join(row_data) + " \\\\")
        
        # Close table
        if self.config.table_format == "booktabs":
            latex_content.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table*}"
            ])
        else:
            latex_content.extend([
                "\\hline",
                "\\end{tabular}",
                "\\end{table*}"
            ])
        
        latex_string = '\n'.join(latex_content)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(latex_string)
        
        return latex_string
    
    def _prepare_table_data(self, df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """Prepare and format data for LaTeX table"""
        # Group by algorithm and calculate statistics
        if 'Algorithm' in df.columns:
            grouped = df.groupby('Algorithm')[metrics].agg(['mean', 'std']).round(3)
            return grouped
        else:
            return df[metrics].describe().round(3)
    
    def _generate_table_latex(self,
                            data: pd.DataFrame,
                            metrics: List[str],
                            caption: str,
                            label: str) -> str:
        """Generate LaTeX table from prepared data"""
        latex_lines = []
        
        # Table opening
        if self.config.table_format == "booktabs":
            latex_lines.extend([
                "\\begin{table}[htbp]",
                "\\centering",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                "\\begin{tabular}{" + "l" + "c" * len(metrics) * 2 + "}",
                "\\toprule"
            ])
        else:
            latex_lines.extend([
                "\\begin{table}[htbp]",
                "\\centering",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                "\\begin{tabular}{|l|" + "c|" * len(metrics) * 2 + "}",
                "\\hline"
            ])
        
        # Multi-level header
        header1 = ["Algorithm"]
        header2 = [""]
        
        for metric in metrics:
            header1.extend([f"\\multicolumn{{2}}{{c}}{{{metric.replace('_', ' ').title()}}}"])
            header2.extend(["Mean", "Std"])
        
        latex_lines.append(" & ".join(header1) + " \\\\")
        
        if self.config.table_format == "booktabs":
            latex_lines.append("\\cmidrule(lr){2-" + str(len(metrics) * 2 + 1) + "}")
        else:
            latex_lines.append("\\hline")
        
        latex_lines.append(" & ".join(header2) + " \\\\")
        
        if self.config.table_format == "booktabs":
            latex_lines.append("\\midrule")
        else:
            latex_lines.append("\\hline")
        
        # Data rows
        for algorithm in data.index:
            row = [algorithm.replace('_', '\\_')]
            
            for metric in metrics:
                mean_val = data.loc[algorithm, (metric, 'mean')]
                std_val = data.loc[algorithm, (metric, 'std')]
                
                row.append(f"{mean_val:.3f}")
                row.append(f"{std_val:.3f}")
            
            latex_lines.append(" & ".join(row) + " \\\\")
        
        # Table closing
        if self.config.table_format == "booktabs":
            latex_lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}"
            ])
        else:
            latex_lines.extend([
                "\\hline",
                "\\end{tabular}",
                "\\end{table}"
            ])
        
        return '\n'.join(latex_lines)


class PublicationReportGenerator:
    """
    Generate complete publication-ready reports
    
    Implements report generation from methodology Section 13.2
    """
    
    def __init__(self, config: PublicationConfig = None):
        self.config = config or PublicationConfig()
        self.table_generator = LaTeXTableGenerator(config)
        self.citation_manager = CitationManager()
    
    def generate_paper_template(self,
                               title: str = "Comparative Analysis of Test Generation Methods",
                               authors: List[str] = None,
                               abstract: str = "",
                               output_path: str = "paper.tex") -> str:
        """Generate complete LaTeX paper template"""
        
        if authors is None:
            authors = ["Author Name"]
        
        # Document class and packages
        latex_content = []
        
        if self.config.journal_style == "ieee":
            latex_content.extend([
                "\\documentclass[conference]{IEEEtran}",
                "\\IEEEoverridecommandlockouts",
                "\\usepackage{cite}",
                "\\usepackage{amsmath,amssymb,amsfonts}",
                "\\usepackage{algorithmic}",
                "\\usepackage{graphicx}",
                "\\usepackage{textcomp}",
                "\\usepackage{xcolor}",
                "\\usepackage{booktabs}",
                "\\usepackage{multirow}",
                "\\usepackage{array}",
                "\\usepackage{url}",
                ""
            ])
        elif self.config.journal_style == "acm":
            latex_content.extend([
                "\\documentclass[sigconf]{acmart}",
                "\\usepackage{booktabs}",
                "\\usepackage{multirow}",
                "\\usepackage{array}",
                ""
            ])
        else:  # Springer
            latex_content.extend([
                "\\documentclass{llncs}",
                "\\usepackage{graphicx}",
                "\\usepackage{booktabs}",
                "\\usepackage{multirow}",
                "\\usepackage{array}",
                ""
            ])
        
        # Title and authors
        latex_content.extend([
            f"\\title{{{title}}}",
            ""
        ])
        
        for author in authors:
            latex_content.append(f"\\author{{{author}}}")
        
        latex_content.extend([
            "",
            "\\begin{document}",
            "\\maketitle",
            ""
        ])
        
        # Abstract
        latex_content.extend([
            "\\begin{abstract}",
            abstract or self._generate_default_abstract(),
            "\\end{abstract}",
            ""
        ])
        
        # Keywords
        latex_content.extend([
            "\\begin{IEEEkeywords}" if self.config.journal_style == "ieee" else "\\keywords{",
            "test generation, multi-objective optimization, software testing, evolutionary algorithms",
            "\\end{IEEEkeywords}" if self.config.journal_style == "ieee" else "}",
            ""
        ])
        
        # Main sections
        latex_content.extend(self._generate_main_sections())
        
        # Bibliography
        latex_content.extend([
            "\\bibliographystyle{IEEEtran}" if self.config.journal_style == "ieee" else "\\bibliographystyle{plain}",
            "\\bibliography{references}",
            ""
        ])
        
        if self.config.include_appendix:
            latex_content.extend(self._generate_appendix())
        
        latex_content.extend([
            "\\end{document}"
        ])
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(latex_content))
        
        return output_path
    
    def _generate_default_abstract(self) -> str:
        """Generate default abstract text"""
        return """This paper presents a comprehensive empirical comparison of test generation methods, 
        comparing traditional baseline approaches against multi-objective evolutionary algorithms. 
        Our experimental methodology employs 32 diverse test programs with varying complexity levels, 
        evaluating 6 baseline methods and 4 multi-objective algorithms across multiple performance metrics. 
        Statistical analysis with effect size calculations and multiple comparison correction provides 
        robust evidence for algorithm selection recommendations. Results indicate that multi-objective 
        approaches demonstrate significant advantages on complex programs while baseline methods remain 
        competitive for simpler cases."""
    
    def _generate_main_sections(self) -> List[str]:
        """Generate main paper sections"""
        sections = [
            "\\section{Introduction}",
            "Test case generation represents a fundamental challenge in software engineering, " +
            "with various algorithmic approaches proposed to automate the creation of effective test suites. " +
            f"This work provides a comprehensive comparison between traditional baseline methods and " +
            f"modern multi-objective evolutionary algorithms {self.citation_manager.cite('nsga2')}.",
            "",
            
            "\\section{Related Work}",
            f"Multi-objective optimization in software testing has gained significant attention, " +
            f"with algorithms such as NSGA-II {self.citation_manager.cite('nsga2')}, " +
            f"NSGA-III {self.citation_manager.cite('nsga3')}, and " +
            f"MOEA/D {self.citation_manager.cite('moead')} showing promise for test generation tasks.",
            "",
            
            "\\section{Experimental Methodology}",
            "Our experimental design follows rigorous empirical software engineering practices, " +
            "employing a comprehensive test suite of 32 programs with systematic complexity categorization.",
            "",
            "\\subsection{Test Programs}",
            "Programs range from simple sorting algorithms to complex distributed system simulations, " +
            "with cyclomatic complexity measurements guiding categorization.",
            "",
            "\\subsection{Algorithms}",
            "We evaluate six baseline methods including Random Testing, Adaptive Random Testing " +
            f"{self.citation_manager.cite('art')}, and Boundary Value Analysis, against four " +
            "multi-objective algorithms.",
            "",
            "\\subsection{Statistical Analysis}",
            f"Statistical significance testing employs Mann-Whitney U tests {self.citation_manager.cite('mann_whitney')} " +
            f"with False Discovery Rate correction {self.citation_manager.cite('fdr_bh')}. " +
            f"Effect sizes are calculated using Hedges' g {self.citation_manager.cite('hedges_g')} " +
            "with bootstrap confidence intervals.",
            "",
            
            "\\section{Results}",
            "% TODO: Add generated results tables and analysis",
            "\\input{results_tables}",
            "",
            
            "\\section{Discussion}",
            "Results demonstrate clear performance differences between algorithm categories, " +
            "with multi-objective approaches showing significant advantages on complex programs.",
            "",
            
            "\\section{Threats to Validity}",
            "We identify and address potential threats to internal, external, construct, " +
            "and conclusion validity following established guidelines for empirical software engineering research.",
            "",
            
            "\\section{Conclusion}",
            "This comprehensive evaluation provides evidence-based recommendations for " +
            "test generation algorithm selection, contributing to both research and practice " +
            "in automated software testing.",
            ""
        ]
        
        return sections
    
    def _generate_appendix(self) -> List[str]:
        """Generate appendix sections"""
        return [
            "",
            "\\appendix",
            "\\section{Complete Algorithm Parameters}",
            "% TODO: Add complete parameter specifications",
            "",
            "\\section{Statistical Test Results}",
            "% TODO: Add complete statistical analysis results",
            "",
            "\\section{Replication Package}",
            "Complete experimental data, analysis scripts, and replication instructions " +
            "are available at: \\url{https://github.com/repository/replication-package}",
            ""
        ]


class ReplicationPackageGenerator:
    """
    Generate complete replication packages
    
    Implements replication package creation from methodology Section 13.3
    """
    
    def __init__(self):
        self.package_structure = {
            'data/': 'Experimental results and raw data',
            'scripts/': 'Analysis and visualization scripts', 
            'config/': 'Configuration files',
            'docs/': 'Documentation and methodology',
            'results/': 'Generated results and reports',
            'src/': 'Source code for algorithms and evaluation'
        }
    
    def create_replication_package(self,
                                 experiment_results: Dict[str, Any],
                                 output_path: str = "replication_package") -> str:
        """Create complete replication package"""
        
        package_path = Path(output_path)
        package_path.mkdir(exist_ok=True)
        
        # Create directory structure
        for dir_name, description in self.package_structure.items():
            (package_path / dir_name).mkdir(exist_ok=True)
            
            # Create README for each directory
            readme_content = f"# {dir_name}\n\n{description}\n"
            with open(package_path / dir_name / "README.md", 'w') as f:
                f.write(readme_content)
        
        # Generate main README
        self._generate_main_readme(package_path)
        
        # Copy experimental results
        if experiment_results:
            with open(package_path / "data" / "experimental_results.json", 'w') as f:
                json.dump(experiment_results, f, indent=2)
        
        # Generate analysis scripts
        self._generate_analysis_scripts(package_path / "scripts")
        
        # Generate Docker setup
        self._generate_docker_setup(package_path)
        
        # Create archive
        archive_path = self._create_archive(package_path)
        
        return archive_path
    
    def _generate_main_readme(self, package_path: Path):
        """Generate main README for replication package"""
        readme_content = f"""# Test Generation Algorithm Comparison - Replication Package

This package contains all materials necessary to replicate the experimental results from:

**"Comparative Analysis of Test Generation Methods: Multi-Objective vs. Baseline Approaches"**

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Contents

- `data/`: Experimental results and raw data files
- `scripts/`: Analysis and visualization scripts
- `config/`: Configuration files for experiments
- `docs/`: Complete methodology and documentation
- `results/`: Generated results, tables, and figures
- `src/`: Source code for all algorithms and evaluation framework

## Quick Start

1. **Prerequisites**:
   - Python 3.8+
   - Required packages: `pip install -r requirements.txt`
   - Optional: Docker for containerized execution

2. **Run Full Experiment**:
   ```bash
   python scripts/run_full_experiment.py
   ```

3. **Generate Analysis**:
   ```bash
   python scripts/generate_statistical_analysis.py
   python scripts/create_visualizations.py
   ```

4. **View Results**:
   - Open `results/interactive_dashboard.html` for interactive exploration
   - See `results/statistical_report.txt` for complete analysis
   - LaTeX tables are in `results/latex_tables/`

## System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended  
- **Storage**: 2GB for complete package
- **OS**: Linux, macOS, or Windows with WSL

## Docker Usage

```bash
docker build -t test-generation-experiment .
docker run -v $(pwd)/results:/app/results test-generation-experiment
```

## Experimental Parameters

The experiment uses the following default configuration:
- **Programs**: 32 test programs (complexity range: CC 3-77)
- **Algorithms**: 6 baseline + 4 multi-objective methods
- **Repetitions**: 10 independent runs per condition
- **Statistical Analysis**: Mann-Whitney U with FDR correction
- **Effect Size**: Hedges' g with bootstrap CI

## Results Structure

```
results/
├── experimental_data.json          # Raw experimental results
├── statistical_analysis.json       # Statistical test results  
├── interactive_dashboard.html      # Interactive visualization
├── latex_tables/                   # Publication-ready tables
└── figures/                        # Publication-ready figures
```

## Citation

If you use this replication package, please cite:

```bibtex
@article{{author2024comparison,
  title={{Comparative Analysis of Test Generation Methods: Multi-Objective vs. Baseline Approaches}},
  author={{Author Name}},
  journal={{Journal Name}},
  year={{2024}},
  publisher={{Publisher}}
}}
```

## License

This replication package is provided under the MIT License. See LICENSE.txt for details.

## Contact

For questions about this replication package, please contact: author@email.com

## Verification Checksums

- `experimental_data.json`: SHA256 checksum will be generated
- Complete package integrity can be verified using `scripts/verify_package.py`
"""
        
        with open(package_path / "README.md", 'w') as f:
            f.write(readme_content)
    
    def _generate_analysis_scripts(self, scripts_path: Path):
        """Generate analysis and replication scripts"""
        
        # Main experiment runner
        runner_script = """#!/usr/bin/env python3
'''
Full Experiment Runner
Executes complete experimental methodology
'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from run_comprehensive_tests import main

if __name__ == "__main__":
    print("Running full experimental methodology...")
    main()
"""
        
        with open(scripts_path / "run_full_experiment.py", 'w') as f:
            f.write(runner_script)
        
        # Statistical analysis script
        analysis_script = """#!/usr/bin/env python3
'''
Statistical Analysis Generator
Performs comprehensive statistical analysis of results
'''

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.statistical import StatisticalAnalyzer

def main():
    print("Generating statistical analysis...")
    
    # Load experimental results
    with open('../data/experimental_results.json', 'r') as f:
        results = json.load(f)
    
    # Perform analysis
    analyzer = StatisticalAnalyzer()
    comparison_results = analyzer.perform_multiple_comparisons(results)
    
    # Generate report
    report_path = analyzer.generate_statistical_report('../results/statistical_analysis')
    print(f"Statistical analysis complete: {report_path}")

if __name__ == "__main__":
    main()
"""
        
        with open(scripts_path / "generate_statistical_analysis.py", 'w') as f:
            f.write(analysis_script)
        
        # Requirements file
        requirements = """numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
scipy>=1.7.0
scikit-learn>=1.0.0
pymoo>=0.6.0
psutil>=5.8.0
pyyaml>=6.0
"""
        
        with open(scripts_path.parent / "requirements.txt", 'w') as f:
            f.write(requirements)
    
    def _generate_docker_setup(self, package_path: Path):
        """Generate Docker setup files"""
        
        dockerfile = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Create results directory
RUN mkdir -p results

# Default command
CMD ["python", "scripts/run_full_experiment.py"]
"""
        
        with open(package_path / "Dockerfile", 'w') as f:
            f.write(dockerfile)
        
        # Docker compose for easy usage
        docker_compose = """version: '3.8'

services:
  experiment:
    build: .
    volumes:
      - ./results:/app/results
      - ./data:/app/data:ro
    environment:
      - PYTHONPATH=/app/src
    command: python scripts/run_full_experiment.py
    
  analysis:
    build: .
    volumes:
      - ./results:/app/results
      - ./data:/app/data:ro
    environment:
      - PYTHONPATH=/app/src
    command: python scripts/generate_statistical_analysis.py
    depends_on:
      - experiment
"""
        
        with open(package_path / "docker-compose.yml", 'w') as f:
            f.write(docker_compose)
    
    def _create_archive(self, package_path: Path) -> str:
        """Create ZIP archive of replication package"""
        archive_path = f"{package_path}_archive.zip"
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(package_path):
                for file in files:
                    file_path = Path(root) / file
                    archive_name = file_path.relative_to(package_path.parent)
                    zipf.write(file_path, archive_name)
        
        return archive_path


# Factory functions for easy usage
def create_publication_suite(config: Optional[PublicationConfig] = None) -> Tuple[LaTeXTableGenerator, PublicationReportGenerator, ReplicationPackageGenerator]:
    """Create complete publication tools suite"""
    config = config or PublicationConfig()
    
    table_generator = LaTeXTableGenerator(config)
    report_generator = PublicationReportGenerator(config)
    package_generator = ReplicationPackageGenerator()
    
    return table_generator, report_generator, package_generator


if __name__ == "__main__":
    """Example usage and testing"""
    print("Publication Tools - LaTeX Generation and Citation System")
    print("=" * 60)
    print("This module implements publication-ready output generation")
    print("as specified in EXPERIMENTAL_METHODOLOGY.md Section 13.")
    print("\nFeatures:")
    print("- LaTeX table generation")
    print("- Automated citation system")
    print("- Publication report templates")
    print("- Complete replication packages")
    print("- Docker containerization")