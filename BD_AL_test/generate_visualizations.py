#!/usr/bin/env python3
"""
Statistical Report Visualization Generator

This script generates comprehensive visualizations from statistical analysis reports,
including comparison charts, effect size plots, method performance rankings, and more.

Usage:
    python3 generate_visualizations.py --input results/comparison_analysis/
    python3 generate_visualizations.py --input results/comparison_analysis/ --output custom_charts/
    python3 generate_visualizations.py --json results/detailed_results.json
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import warnings

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Configure matplotlib for better quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


class StatisticalVisualizationGenerator:
    """Generate comprehensive visualizations from statistical analysis results"""
    
    def __init__(self, output_dir: str = "visualizations", figsize: Tuple[int, int] = (12, 8)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.figsize = figsize
        self.logger = self._setup_logger()
        
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'neutral': '#85858A',
            'light': '#F5F5F5'
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('VisualizationGenerator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_statistical_results(self, input_path: str) -> Dict[str, Any]:
        """Load statistical results from JSON file or directory"""
        
        input_path = Path(input_path)
        
        if input_path.is_file() and input_path.suffix == '.json':
            # Direct JSON file
            with open(input_path, 'r') as f:
                return json.load(f)
                
        elif input_path.is_dir():
            # Look for detailed results file in directory
            json_files = list(input_path.glob("*detailed_results*.json"))
            if json_files:
                with open(json_files[0], 'r') as f:
                    return json.load(f)
            else:
                raise FileNotFoundError(f"No detailed results JSON found in {input_path}")
        else:
            raise FileNotFoundError(f"Invalid input path: {input_path}")
    
    def extract_comparison_data(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Extract comparison results into a structured DataFrame"""
        
        if 'comparison_results' not in results:
            raise ValueError("No comparison_results found in data")
        
        comparisons = results['comparison_results']
        
        rows = []
        for comp in comparisons:
            row = {
                'Method_1': comp['method1'],
                'Method_2': comp['method2'], 
                'Metric': comp['metric'],
                'Function': comp['function_name'],
                'P_Value': comp['statistical_test']['p_value'],
                'Corrected_P_Value': comp['statistical_test'].get('corrected_p_value', None),
                'Effect_Size': comp['statistical_test'].get('effect_size', 0.0),
                'Effect_Magnitude': comp['statistical_test'].get('interpretation', 'unknown'),
                'Significant': comp['statistical_test']['significant'],
                'Practically_Significant': comp.get('practical_significance', False),
                'Test_Name': comp['statistical_test']['test_name']
            }
            
            # Add descriptive statistics
            if 'descriptive_stats' in comp:
                for method in [comp['method1'], comp['method2']]:
                    if method in comp['descriptive_stats']:
                        stats = comp['descriptive_stats'][method]
                        row.update({
                            f'{method}_Mean': stats.get('mean', 0.0),
                            f'{method}_Std': stats.get('std', 0.0),
                            f'{method}_Median': stats.get('median', 0.0)
                        })
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_effect_size_distribution(self, df: pd.DataFrame, save: bool = True) -> None:
        """Plot effect size distribution across all comparisons"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Effect Size Analysis', fontsize=16, fontweight='bold')
        
        # 1. Overall effect size distribution
        ax1 = axes[0, 0]
        effect_sizes = df['Effect_Size'].dropna()
        ax1.hist(effect_sizes, bins=30, alpha=0.7, color=self.colors['primary'], edgecolor='black')
        ax1.axvline(effect_sizes.mean(), color=self.colors['accent'], linestyle='--', 
                   label=f'Mean: {effect_sizes.mean():.3f}')
        ax1.axvline(effect_sizes.median(), color=self.colors['secondary'], linestyle='--', 
                   label=f'Median: {effect_sizes.median():.3f}')
        ax1.set_xlabel('Effect Size')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Effect Size Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Effect size by metric
        ax2 = axes[0, 1]
        metrics = df['Metric'].unique()
        effect_by_metric = [df[df['Metric'] == metric]['Effect_Size'].dropna().values 
                           for metric in metrics]
        
        box_plot = ax2.boxplot(effect_by_metric, labels=metrics, patch_artist=True)
        colors = sns.color_palette("husl", len(metrics))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Metric')
        ax2.set_ylabel('Effect Size')
        ax2.set_title('Effect Size by Metric')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Effect magnitude counts
        ax3 = axes[1, 0]
        magnitude_counts = df['Effect_Magnitude'].value_counts()
        bars = ax3.bar(magnitude_counts.index, magnitude_counts.values, 
                      color=sns.color_palette("viridis", len(magnitude_counts)))
        ax3.set_xlabel('Effect Magnitude')
        ax3.set_ylabel('Count')
        ax3.set_title('Effect Magnitude Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 4. Significance vs Effect Size scatter
        ax4 = axes[1, 1]
        significant = df[df['Significant'] == True]
        not_significant = df[df['Significant'] == False]
        
        ax4.scatter(not_significant['Effect_Size'], not_significant['P_Value'], 
                   alpha=0.6, color=self.colors['neutral'], label='Not Significant', s=30)
        ax4.scatter(significant['Effect_Size'], significant['P_Value'], 
                   alpha=0.6, color=self.colors['success'], label='Significant', s=30)
        
        ax4.axhline(y=0.05, color=self.colors['accent'], linestyle='--', 
                   label='α = 0.05')
        ax4.set_xlabel('Effect Size')
        ax4.set_ylabel('P-Value')
        ax4.set_title('Significance vs Effect Size')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'effect_size_analysis.png', 
                       bbox_inches='tight', facecolor='white')
            self.logger.info(f"Saved effect size analysis to {self.output_dir / 'effect_size_analysis.png'}")
        
        plt.show()
    
    def plot_method_performance_ranking(self, df: pd.DataFrame, save: bool = True) -> None:
        """Plot method performance ranking across metrics"""
        
        metrics = df['Metric'].unique()
        methods = set(df['Method_1'].unique()) | set(df['Method_2'].unique())
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Method Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Win rate matrix
        ax1 = axes[0, 0]
        win_matrix = pd.DataFrame(0, index=methods, columns=methods)
        
        for _, row in df.iterrows():
            if row['Significant'] and row['Effect_Size'] > 0:
                win_matrix.loc[row['Method_1'], row['Method_2']] += 1
            elif row['Significant'] and row['Effect_Size'] < 0:
                win_matrix.loc[row['Method_2'], row['Method_1']] += 1
        
        sns.heatmap(win_matrix, annot=True, fmt='d', cmap='RdYlBu_r', 
                   ax=ax1, cbar_kws={'label': 'Significant Wins'})
        ax1.set_title('Significant Wins Matrix')
        ax1.set_xlabel('Loser')
        ax1.set_ylabel('Winner')
        
        # 2. Overall win rates
        ax2 = axes[0, 1]
        total_wins = win_matrix.sum(axis=1).sort_values(ascending=True)
        bars = ax2.barh(range(len(total_wins)), total_wins.values, 
                       color=sns.color_palette("viridis", len(total_wins)))
        ax2.set_yticks(range(len(total_wins)))
        ax2.set_yticklabels(total_wins.index)
        ax2.set_xlabel('Total Significant Wins')
        ax2.set_title('Method Performance Ranking')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, total_wins.values)):
            ax2.text(value + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{int(value)}', va='center')
        
        # 3. Performance by metric
        ax3 = axes[1, 0]
        metric_performance = {}
        
        for metric in metrics:
            metric_df = df[df['Metric'] == metric]
            metric_wins = {}
            
            for method in methods:
                wins = 0
                method1_wins = metric_df[(metric_df['Method_1'] == method) & 
                                       (metric_df['Significant']) & 
                                       (metric_df['Effect_Size'] > 0)]
                method2_wins = metric_df[(metric_df['Method_2'] == method) & 
                                       (metric_df['Significant']) & 
                                       (metric_df['Effect_Size'] < 0)]
                wins = len(method1_wins) + len(method2_wins)
                metric_wins[method] = wins
            
            metric_performance[metric] = metric_wins
        
        # Create stacked bar chart
        metric_df_plot = pd.DataFrame(metric_performance).fillna(0)
        metric_df_plot.plot(kind='bar', stacked=True, ax=ax3, 
                           color=sns.color_palette("Set2", len(metrics)))
        ax3.set_title('Wins by Method and Metric')
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Significant Wins')
        ax3.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Practical significance rate
        ax4 = axes[1, 1]
        practical_sig_rates = {}
        
        for method in methods:
            method_comparisons = df[(df['Method_1'] == method) | (df['Method_2'] == method)]
            if len(method_comparisons) > 0:
                practical_rate = (method_comparisons['Practically_Significant'].sum() / 
                                len(method_comparisons)) * 100
                practical_sig_rates[method] = practical_rate
        
        if practical_sig_rates:
            sorted_rates = dict(sorted(practical_sig_rates.items(), key=lambda x: x[1]))
            bars = ax4.bar(range(len(sorted_rates)), list(sorted_rates.values()),
                          color=sns.color_palette("plasma", len(sorted_rates)))
            ax4.set_xticks(range(len(sorted_rates)))
            ax4.set_xticklabels(list(sorted_rates.keys()), rotation=45)
            ax4.set_ylabel('Practical Significance Rate (%)')
            ax4.set_title('Practical Significance Rate by Method')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, sorted_rates.values()):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'method_performance_ranking.png', 
                       bbox_inches='tight', facecolor='white')
            self.logger.info(f"Saved method performance ranking to {self.output_dir / 'method_performance_ranking.png'}")
        
        plt.show()
    
    def plot_metric_comparison_heatmap(self, df: pd.DataFrame, save: bool = True) -> None:
        """Plot heatmap of significance across metrics and method pairs"""
        
        metrics = df['Metric'].unique()
        method_pairs = df[['Method_1', 'Method_2']].apply(
            lambda x: f"{x['Method_1']} vs {x['Method_2']}", axis=1
        ).unique()
        
        # Create significance matrix
        sig_matrix = pd.DataFrame(0, index=method_pairs, columns=metrics)
        effect_matrix = pd.DataFrame(0.0, index=method_pairs, columns=metrics)
        
        for _, row in df.iterrows():
            pair = f"{row['Method_1']} vs {row['Method_2']}"
            metric = row['Metric']
            sig_matrix.loc[pair, metric] = 1 if row['Significant'] else 0
            effect_matrix.loc[pair, metric] = abs(row['Effect_Size'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle('Metric Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Significance heatmap
        sns.heatmap(sig_matrix, annot=True, fmt='d', cmap='RdYlGn', 
                   ax=ax1, cbar_kws={'label': 'Significant (1) / Not Significant (0)'})
        ax1.set_title('Statistical Significance by Method Pair and Metric')
        ax1.set_xlabel('Metric')
        ax1.set_ylabel('Method Comparison')
        
        # Effect size heatmap
        sns.heatmap(effect_matrix, annot=True, fmt='.3f', cmap='viridis', 
                   ax=ax2, cbar_kws={'label': 'Absolute Effect Size'})
        ax2.set_title('Effect Size by Method Pair and Metric')
        ax2.set_xlabel('Metric')
        ax2.set_ylabel('Method Comparison')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'metric_comparison_heatmap.png', 
                       bbox_inches='tight', facecolor='white')
            self.logger.info(f"Saved metric comparison heatmap to {self.output_dir / 'metric_comparison_heatmap.png'}")
        
        plt.show()
    
    def plot_statistical_summary(self, results: Dict[str, Any], save: bool = True) -> None:
        """Plot summary statistics from the analysis"""
        
        if 'summary_statistics' not in results:
            self.logger.warning("No summary statistics found in results")
            return
        
        summary = results['summary_statistics']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Statistical Analysis Summary', fontsize=16, fontweight='bold')
        
        # 1. Significance rates
        ax1 = axes[0, 0]
        sig_rate = summary.get('significance_rate', 0) * 100
        practical_rate = summary.get('practical_significance_rate', 0) * 100
        
        rates = [sig_rate, practical_rate]
        labels = ['Statistical\nSignificance', 'Practical\nSignificance']
        colors = [self.colors['primary'], self.colors['secondary']]
        
        bars = ax1.bar(labels, rates, color=colors, alpha=0.7)
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Significance Rates')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars, rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Effect size distribution summary
        ax2 = axes[0, 1]
        effect_dist = summary.get('effect_size_distribution', {})
        
        if effect_dist:
            stats_names = ['Mean', 'Median', 'Std', 'Min', 'Max']
            stats_values = [
                effect_dist.get('mean', 0),
                effect_dist.get('median', 0), 
                effect_dist.get('std', 0),
                effect_dist.get('min', 0),
                effect_dist.get('max', 0)
            ]
            
            bars = ax2.bar(stats_names, stats_values, 
                          color=sns.color_palette("viridis", len(stats_names)))
            ax2.set_ylabel('Effect Size')
            ax2.set_title('Effect Size Distribution Summary')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, stats_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Method performance ranking
        ax3 = axes[1, 0]
        method_ranking = summary.get('method_performance_ranking', {})
        
        if method_ranking:
            methods = list(method_ranking.keys())[:10]  # Top 10
            wins = list(method_ranking.values())[:10]
            
            bars = ax3.barh(range(len(methods)), wins,
                           color=sns.color_palette("plasma", len(methods)))
            ax3.set_yticks(range(len(methods)))
            ax3.set_yticklabels(methods)
            ax3.set_xlabel('Significant Wins')
            ax3.set_title('Top Method Performance')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, wins):
                ax3.text(value + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{int(value)}', va='center')
        
        # 4. Analysis metadata
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create text summary
        total_comp = summary.get('total_comparisons', 0)
        sig_comp = summary.get('significant_comparisons', 0)
        practical_comp = summary.get('practically_significant', 0)
        correction_method = summary.get('multiple_correction_method', 'unknown')
        timestamp = summary.get('analysis_timestamp', 'unknown')
        
        summary_text = f"""
Analysis Summary:
──────────────────
Total Comparisons: {total_comp:,}
Significant: {sig_comp:,} ({sig_rate:.1f}%)
Practically Significant: {practical_comp:,} ({practical_rate:.1f}%)

Configuration:
──────────────────
Multiple Correction: {correction_method}
Effect Size Threshold: {summary.get('effect_size_threshold', 'N/A')}

Generated: {timestamp.split('T')[0] if 'T' in str(timestamp) else timestamp}
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['light']))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'statistical_summary.png', 
                       bbox_inches='tight', facecolor='white')
            self.logger.info(f"Saved statistical summary to {self.output_dir / 'statistical_summary.png'}")
        
        plt.show()
    
    def generate_all_visualizations(self, input_path: str) -> None:
        """Generate all available visualizations"""
        
        self.logger.info(f"Loading statistical results from {input_path}")
        results = self.load_statistical_results(input_path)
        
        # Extract comparison data
        df = self.extract_comparison_data(results)
        self.logger.info(f"Extracted {len(df)} comparison results")
        
        # Generate all plots
        self.logger.info("Generating effect size analysis...")
        self.plot_effect_size_distribution(df)
        
        self.logger.info("Generating method performance ranking...")
        self.plot_method_performance_ranking(df)
        
        self.logger.info("Generating metric comparison heatmap...")
        self.plot_metric_comparison_heatmap(df)
        
        self.logger.info("Generating statistical summary...")
        self.plot_statistical_summary(results)
        
        # Generate index HTML
        self._generate_html_index()
        
        self.logger.info(f"All visualizations generated in {self.output_dir}")
    
    def _generate_html_index(self) -> None:
        """Generate HTML index page for all visualizations"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Statistical Analysis Visualizations</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        .header {{ text-align: center; margin-bottom: 30px; color: #333; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
        .chart {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; }}
        .chart img {{ width: 100%; height: auto; border-radius: 4px; }}
        .chart h3 {{ margin-top: 0; color: #2E86AB; }}
        .timestamp {{ text-align: center; color: #666; margin-top: 30px; font-style: italic; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Statistical Analysis Visualizations</h1>
            <p>Comprehensive visual analysis of test generation method comparisons</p>
        </div>
        
        <div class="grid">
            <div class="chart">
                <h3>🎯 Effect Size Analysis</h3>
                <img src="effect_size_analysis.png" alt="Effect Size Analysis">
                <p>Distribution and analysis of effect sizes across all comparisons, including magnitude classifications and significance relationships.</p>
            </div>
            
            <div class="chart">
                <h3>🏆 Method Performance Ranking</h3>
                <img src="method_performance_ranking.png" alt="Method Performance Ranking">
                <p>Comprehensive ranking of test generation methods based on significant wins and practical significance rates.</p>
            </div>
            
            <div class="chart">
                <h3>🔥 Metric Comparison Heatmap</h3>
                <img src="metric_comparison_heatmap.png" alt="Metric Comparison Heatmap">
                <p>Heatmap visualization of statistical significance and effect sizes across different metrics and method pairs.</p>
            </div>
            
            <div class="chart">
                <h3>📈 Statistical Summary</h3>
                <img src="statistical_summary.png" alt="Statistical Summary">
                <p>High-level summary of the statistical analysis including significance rates, effect size distribution, and top performers.</p>
            </div>
        </div>
        
        <div class="timestamp">
            Generated on {timestamp}
        </div>
    </div>
</body>
</html>
        """
        
        index_path = self.output_dir / 'index.html'
        with open(index_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Generated HTML index at {index_path}")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Generate visualizations from statistical analysis reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 generate_visualizations.py --input results/comparison_analysis/
  python3 generate_visualizations.py --json results/detailed_results.json --output charts/
  python3 generate_visualizations.py --input results/comparison_analysis/ --no-show
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', help='Input directory containing statistical results')
    group.add_argument('--json', '-j', help='Direct path to JSON results file')
    
    parser.add_argument('--output', '-o', default='visualizations',
                       help='Output directory for visualizations (default: visualizations)')
    
    parser.add_argument('--no-show', action='store_true',
                       help='Save plots without displaying them')
    
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8],
                       help='Figure size for plots (width height)')
    
    args = parser.parse_args()
    
    # Configure matplotlib display
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    
    try:
        # Initialize generator
        generator = StatisticalVisualizationGenerator(
            output_dir=args.output,
            figsize=tuple(args.figsize)
        )
        
        # Determine input path
        input_path = args.input if args.input else args.json
        
        # Generate visualizations
        generator.generate_all_visualizations(input_path)
        
        print(f"\n✅ All visualizations generated successfully!")
        print(f"📁 Output directory: {Path(args.output).absolute()}")
        print(f"🌐 Open {Path(args.output) / 'index.html'} in your browser to view all charts")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())