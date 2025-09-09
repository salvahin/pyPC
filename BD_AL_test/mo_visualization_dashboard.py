#!/usr/bin/env python3
"""
Visualization Dashboard for Multi-Objective Optimization Results
Creates comprehensive visualizations for MO algorithm comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
from datetime import datetime

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


class MOVisualizationDashboard:
    """Create comprehensive visualizations for MO optimization results"""
    
    def __init__(self, results: Dict = None, figsize: Tuple[int, int] = (20, 12)):
        """
        Initialize visualization dashboard
        
        Args:
            results: Results dictionary from ParallelMOExperimentRunner
            figsize: Default figure size
        """
        self.results = results
        self.figsize = figsize
        # Create color palette manually
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))[:8]
        self.markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        
    def load_results(self, results_file: str):
        """Load results from pickle file"""
        with open(results_file, 'rb') as f:
            data = pickle.load(f)
            self.results = data.get('results', {})
            self.reference_fronts = data.get('reference_fronts', {})
            self.config = data.get('config', {})
    
    def create_comprehensive_dashboard(self, save_path: str = None) -> plt.Figure:
        """
        Create comprehensive dashboard with multiple visualizations
        
        Args:
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(24, 16))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Algorithm ranking heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_ranking_heatmap(ax1)
        
        # 2. Hypervolume comparison
        ax2 = fig.add_subplot(gs[0, 2:])
        self.plot_hypervolume_comparison(ax2)
        
        # 3. Convergence curves
        ax3 = fig.add_subplot(gs[1, :2])
        self.plot_convergence_curves(ax3)
        
        # 4. Pareto front comparison
        ax4 = fig.add_subplot(gs[1, 2:])
        self.plot_pareto_fronts_comparison(ax4)
        
        # 5. Performance profiles
        ax5 = fig.add_subplot(gs[2, :2])
        self.plot_performance_profiles(ax5)
        
        # 6. Statistical significance matrix
        ax6 = fig.add_subplot(gs[2, 2:])
        self.plot_significance_matrix(ax6)
        
        # Add title
        fig.suptitle('Multi-Objective Optimization Algorithm Comparison Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")
        
        return fig
    
    def plot_ranking_heatmap(self, ax: plt.Axes):
        """Plot algorithm ranking heatmap across programs"""
        # Create ranking matrix
        algorithms = list(self.results.keys())
        programs = list(next(iter(self.results.values())).keys())
        
        ranking_matrix = np.zeros((len(programs), len(algorithms)))
        
        for i, prog in enumerate(programs):
            # Calculate hypervolume for each algorithm
            hv_values = []
            for alg in algorithms:
                results = self.results[alg][prog]
                successful = [r for r in results if r.is_successful()]
                if successful:
                    hvs = [r.metrics.get('hypervolume', 0) for r in successful]
                    hv_values.append((alg, np.mean(hvs)))
                else:
                    hv_values.append((alg, 0))
            
            # Rank algorithms
            hv_values.sort(key=lambda x: x[1], reverse=True)
            for rank, (alg, _) in enumerate(hv_values, 1):
                j = algorithms.index(alg)
                ranking_matrix[i, j] = rank
        
        # Plot heatmap
        im = ax.imshow(ranking_matrix, cmap='RdYlGn_r', aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Rank')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(algorithms)))
        ax.set_yticks(np.arange(len(programs)))
        ax.set_xticklabels(algorithms)
        ax.set_yticklabels(programs)
        
        # Add text annotations
        for i in range(len(programs)):
            for j in range(len(algorithms)):
                text = ax.text(j, i, f'{ranking_matrix[i, j]:.0f}',
                             ha="center", va="center", color="black")
        ax.set_title('Algorithm Rankings by Test Program', fontweight='bold')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Test Program')
    
    def plot_hypervolume_comparison(self, ax: plt.Axes):
        """Plot hypervolume comparison box plots"""
        data_for_plot = []
        
        for alg_name, alg_results in self.results.items():
            for prog_name, prog_results in alg_results.items():
                for result in prog_results:
                    if result.is_successful():
                        data_for_plot.append({
                            'Algorithm': alg_name,
                            'Program': prog_name,
                            'Hypervolume': result.metrics.get('hypervolume', 0)
                        })
        
        if data_for_plot:
            df = pd.DataFrame(data_for_plot)
            # Create box plot manually
            algorithms = df['Algorithm'].unique()
            data_by_alg = [df[df['Algorithm'] == alg]['Hypervolume'].values 
                          for alg in algorithms]
            
            bp = ax.boxplot(data_by_alg, labels=algorithms, patch_artist=True)
            
            # Color the boxes
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(self.colors[i % len(self.colors)])
                box.set_alpha(0.7)
            ax.set_title('Hypervolume Distribution by Algorithm', fontweight='bold')
            ax.set_ylabel('Hypervolume')
            ax.grid(True, alpha=0.3)
    
    def plot_convergence_curves(self, ax: plt.Axes):
        """Plot average convergence curves"""
        algorithms = list(self.results.keys())
        
        for i, alg in enumerate(algorithms):
            all_histories = []
            
            # Collect all convergence histories
            for prog_results in self.results[alg].values():
                for result in prog_results:
                    if result.is_successful() and result.convergence_history:
                        all_histories.append(result.convergence_history)
            
            if all_histories:
                # Pad histories to same length
                max_len = max(len(h) for h in all_histories)
                padded = []
                for h in all_histories:
                    padded_h = h + [h[-1]] * (max_len - len(h)) if h else []
                    padded.append(padded_h)
                
                # Calculate mean and std
                mean_history = np.mean(padded, axis=0)
                std_history = np.std(padded, axis=0)
                
                generations = np.arange(len(mean_history)) * 10
                
                ax.plot(generations, mean_history, label=alg, 
                       color=self.colors[i], linewidth=2)
                ax.fill_between(generations, 
                               mean_history - std_history,
                               mean_history + std_history,
                               alpha=0.2, color=self.colors[i])
        
        ax.set_title('Average Convergence Curves', fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Objective Value')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    def plot_pareto_fronts_comparison(self, ax: plt.Axes):
        """Plot Pareto fronts for selected program"""
        # Select first program with results
        prog_name = None
        for prog in self.results[list(self.results.keys())[0]].keys():
            has_results = any(
                any(r.is_successful() for r in self.results[alg][prog])
                for alg in self.results.keys()
            )
            if has_results:
                prog_name = prog
                break
        
        if not prog_name:
            ax.text(0.5, 0.5, 'No successful results to plot',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        for i, (alg_name, alg_results) in enumerate(self.results.items()):
            # Get best run
            best_hv = 0
            best_front = None
            
            for result in alg_results[prog_name]:
                if result.is_successful():
                    hv = result.metrics.get('hypervolume', 0)
                    if hv > best_hv:
                        best_hv = hv
                        best_front = result.pareto_front
            
            if best_front is not None:
                # Plot based on objective type
                if len(best_front[0]) >= 2:
                    ax.scatter(best_front[:, 0], best_front[:, 1],
                             label=f'{alg_name} (HV={best_hv:.3f})',
                             color=self.colors[i], s=50, alpha=0.7,
                             marker=self.markers[i % len(self.markers)])
        
        ax.set_title(f'Pareto Fronts Comparison ({prog_name})', fontweight='bold')
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    def plot_performance_profiles(self, ax: plt.Axes):
        """Plot performance profiles"""
        # Calculate performance ratios
        algorithms = list(self.results.keys())
        programs = list(next(iter(self.results.values())).keys())
        
        performance_data = []
        
        for prog in programs:
            # Get best performance for this program
            best_hv = 0
            prog_hvs = {}
            
            for alg in algorithms:
                results = self.results[alg][prog]
                successful = [r for r in results if r.is_successful()]
                if successful:
                    hv = max(r.metrics.get('hypervolume', 0) for r in successful)
                    prog_hvs[alg] = hv
                    best_hv = max(best_hv, hv)
            
            # Calculate performance ratios
            if best_hv > 0:
                for alg in algorithms:
                    if alg in prog_hvs:
                        ratio = prog_hvs[alg] / best_hv
                        performance_data.append({
                            'Algorithm': alg,
                            'Ratio': ratio
                        })
        
        if performance_data:
            df = pd.DataFrame(performance_data)
            
            # Calculate performance profile
            tau_values = np.linspace(0.5, 1.0, 100)
            
            for i, alg in enumerate(algorithms):
                alg_data = df[df['Algorithm'] == alg]['Ratio'].values
                if len(alg_data) > 0:
                    profile = [np.mean(alg_data >= tau) for tau in tau_values]
                    ax.plot(tau_values, profile, label=alg, 
                           color=self.colors[i], linewidth=2)
        
        ax.set_title('Performance Profiles', fontweight='bold')
        ax.set_xlabel('Performance Ratio τ')
        ax.set_ylabel('P(r ≥ τ)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.5, 1.0])
        ax.set_ylim([0, 1.05])
    
    def plot_significance_matrix(self, ax: plt.Axes):
        """Plot statistical significance matrix"""
        from mo_statistical_analysis import MOStatisticalAnalyzer
        
        # Prepare data for analysis
        analysis_data = []
        for alg_name, alg_results in self.results.items():
            for prog_name, prog_results in alg_results.items():
                successful = [r for r in prog_results if r.is_successful()]
                if successful:
                    hvs = [r.metrics.get('hypervolume', 0) for r in successful]
                    analysis_data.append({
                        'Algorithm': alg_name,
                        'Program': prog_name,
                        'HV_Mean': np.mean(hvs)
                    })
        
        if not analysis_data:
            ax.text(0.5, 0.5, 'Insufficient data for significance analysis',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        df = pd.DataFrame(analysis_data)
        analyzer = MOStatisticalAnalyzer()
        
        algorithms = df['Algorithm'].unique()
        n_algs = len(algorithms)
        
        # Create significance matrix
        sig_matrix = np.zeros((n_algs, n_algs))
        
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i != j:
                    # Get paired data
                    programs = df['Program'].unique()
                    data1 = []
                    data2 = []
                    
                    for prog in programs:
                        val1 = df[(df['Algorithm'] == alg1) & 
                                 (df['Program'] == prog)]['HV_Mean'].values
                        val2 = df[(df['Algorithm'] == alg2) & 
                                 (df['Program'] == prog)]['HV_Mean'].values
                        
                        if len(val1) > 0 and len(val2) > 0:
                            data1.append(val1[0])
                            data2.append(val2[0])
                    
                    if len(data1) >= 5:
                        result = analyzer.wilcoxon_test(
                            np.array(data1), 
                            np.array(data2)
                        )
                        if result['significant']:
                            sig_matrix[i, j] = 1 if np.mean(data1) > np.mean(data2) else -1
        
        # Plot matrix
        im = ax.imshow(sig_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Significance')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(n_algs))
        ax.set_yticks(np.arange(n_algs))
        ax.set_xticklabels(algorithms)
        ax.set_yticklabels(algorithms)
        
        # Add text annotations
        for i in range(n_algs):
            for j in range(n_algs):
                if sig_matrix[i, j] != 0:
                    text = ax.text(j, i, f'{sig_matrix[i, j]:.0f}',
                                 ha="center", va="center", color="white" if abs(sig_matrix[i, j]) > 0.5 else "black")
        ax.set_title('Statistical Significance Matrix', fontweight='bold')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Algorithm')
    
    def create_parallel_coordinates(self, save_path: str = None) -> plt.Figure:
        """Create parallel coordinates plot for multi-metric comparison"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data
        metrics_data = []
        for alg_name, alg_results in self.results.items():
            alg_metrics = {
                'Algorithm': alg_name,
                'Hypervolume': [],
                'Solutions': [],
                'Time': [],
                'Success_Rate': []
            }
            
            for prog_results in alg_results.values():
                successful = [r for r in prog_results if r.is_successful()]
                total = len(prog_results)
                
                if successful:
                    alg_metrics['Hypervolume'].extend([
                        r.metrics.get('hypervolume', 0) for r in successful
                    ])
                    alg_metrics['Solutions'].extend([
                        r.metrics.get('n_solutions', 0) for r in successful
                    ])
                    alg_metrics['Time'].extend([
                        r.execution_time for r in successful
                    ])
                    alg_metrics['Success_Rate'].append(len(successful) / total)
            
            if alg_metrics['Hypervolume']:
                metrics_data.append({
                    'Algorithm': alg_name,
                    'Hypervolume': np.mean(alg_metrics['Hypervolume']),
                    'Solutions': np.mean(alg_metrics['Solutions']),
                    'Time': np.mean(alg_metrics['Time']),
                    'Success_Rate': np.mean(alg_metrics['Success_Rate'])
                })
        
        if not metrics_data:
            ax.text(0.5, 0.5, 'No data for parallel coordinates',
                   ha='center', va='center')
            return fig
        
        df = pd.DataFrame(metrics_data)
        
        # Normalize metrics
        metrics = ['Hypervolume', 'Solutions', 'Success_Rate']
        for metric in metrics:
            if metric in df.columns:
                df[f'{metric}_norm'] = (df[metric] - df[metric].min()) / \
                                       (df[metric].max() - df[metric].min() + 1e-10)
        
        # For time, invert normalization (lower is better)
        if 'Time' in df.columns:
            df['Time_norm'] = 1 - (df['Time'] - df['Time'].min()) / \
                                  (df['Time'].max() - df['Time'].min() + 1e-10)
        
        # Plot
        x = np.arange(4)
        for i, row in df.iterrows():
            values = [
                row.get('Hypervolume_norm', 0),
                row.get('Solutions_norm', 0),
                row.get('Time_norm', 0),
                row.get('Success_Rate_norm', 0)
            ]
            ax.plot(x, values, 'o-', label=row['Algorithm'],
                   color=self.colors[i % len(self.colors)],
                   linewidth=2, markersize=8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Hypervolume', 'Solutions', 'Time\n(inverted)', 'Success Rate'])
        ax.set_ylabel('Normalized Value')
        ax.set_title('Multi-Metric Parallel Coordinates', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_all_visualizations(self, output_dir: str = "visualizations"):
        """Save all visualizations to directory"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main dashboard
        self.create_comprehensive_dashboard(
            output_path / f"dashboard_{timestamp}.png"
        )
        
        # Parallel coordinates
        self.create_parallel_coordinates(
            output_path / f"parallel_coords_{timestamp}.png"
        )
        
        print(f"All visualizations saved to: {output_path}")


def create_comparison_report(results_file: str, output_dir: str = "reports"):
    """
    Create comprehensive comparison report from results file
    
    Args:
        results_file: Path to pickle results file
        output_dir: Directory for output
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    dashboard = MOVisualizationDashboard()
    dashboard.load_results(results_file)
    
    # Create visualizations
    dashboard.save_all_visualizations(output_path / "visualizations")
    
    # Perform statistical analysis
    from mo_statistical_analysis import perform_comprehensive_analysis, generate_statistical_report
    
    # Prepare data for analysis
    analysis_data = []
    for alg_name, alg_results in dashboard.results.items():
        for prog_name, prog_results in alg_results.items():
            successful = [r for r in prog_results if r.is_successful()]
            if successful:
                hvs = [r.metrics.get('hypervolume', 0) for r in successful]
                solutions = [r.metrics.get('n_solutions', 0) for r in successful]
                analysis_data.append({
                    'Algorithm': alg_name,
                    'Program': prog_name,
                    'HV_Mean': np.mean(hvs),
                    'Solutions_Mean': np.mean(solutions)
                })
    
    if analysis_data:
        df = pd.DataFrame(analysis_data)
        analysis = perform_comprehensive_analysis(df)
        report_text = generate_statistical_report(
            analysis, 
            output_path / "statistical_report.txt"
        )
        print("Statistical report generated")
    
    print(f"Complete report saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create visualization dashboard for MO results'
    )
    parser.add_argument('results_file', help='Path to results pickle file')
    parser.add_argument('--output-dir', default='reports',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    create_comparison_report(args.results_file, args.output_dir)