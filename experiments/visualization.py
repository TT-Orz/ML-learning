"""
Visualization utilities for experimental results.

This module provides functions to generate publication-quality figures
including paired scatter plots, bar charts, box plots, heatmaps, and
critical difference diagrams.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_paired_comparison(results_df, method_a='native', method_b='onehot',
                           output_path=None):
    """
    Create paired scatter plot comparing two methods.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results with columns for both methods' accuracies.

    method_a, method_b : str
        Column name suffixes for the two methods.

    output_path : str or None
        Path to save figure. If None, doesn't save.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Extract scores
    col_a = f'{method_a}_accuracy_mean'
    col_b = f'{method_b}_accuracy_mean'

    if col_a not in results_df.columns or col_b not in results_df.columns:
        # Try without _mean suffix
        col_a = f'{method_a}_accuracy'
        col_b = f'{method_b}_accuracy'

    scores_a = results_df[col_a]
    scores_b = results_df[col_b]

    # Scatter plot
    ax.scatter(scores_a, scores_b, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

    # Diagonal line
    min_val = min(scores_a.min(), scores_b.min())
    max_val = max(scores_a.max(), scores_b.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=2)

    # Labels for datasets
    if 'dataset' in results_df.columns:
        for idx, row in results_df.iterrows():
            ax.annotate(row['dataset'], (row[col_a], row[col_b]),
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=8, alpha=0.7)

    ax.set_xlabel(f'{method_a.capitalize()} Accuracy', fontsize=12)
    ax.set_ylabel(f'{method_b.capitalize()} Accuracy', fontsize=12)
    ax.set_title(f'Paired Comparison: {method_a.capitalize()} vs {method_b.capitalize()}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Add win counts
    wins_a = (scores_a > scores_b).sum()
    wins_b = (scores_b > scores_a).sum()
    ties = (scores_a == scores_b).sum()

    textstr = f'{method_a} wins: {wins_a}\n{method_b} wins: {wins_b}\nTies: {ties}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    return fig


def plot_classifier_comparison_bars(results_df, output_path=None):
    """
    Create bar chart comparing multiple classifiers.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results with 'classifier', 'accuracy_mean', 'accuracy_std' columns.

    output_path : str or None
        Path to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Average across datasets if multiple
    if 'dataset' in results_df.columns:
        summary = results_df.groupby('classifier').agg({
            'accuracy_mean': ['mean', 'std']
        }).reset_index()
        summary.columns = ['classifier', 'accuracy_mean', 'accuracy_std']
    else:
        summary = results_df.copy()

    # Sort by accuracy
    summary = summary.sort_values('accuracy_mean', ascending=False)

    x = np.arange(len(summary))
    ax.bar(x, summary['accuracy_mean'], yerr=summary['accuracy_std'],
          capsize=5, alpha=0.7, edgecolor='black', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(summary['classifier'], rotation=45, ha='right')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Classifier Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (acc, std) in enumerate(zip(summary['accuracy_mean'], summary['accuracy_std'])):
        ax.text(i, acc + std + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    return fig


def plot_heatmap(results_df, metric='accuracy_mean', output_path=None):
    """
    Create heatmap of classifier performance across datasets.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results with 'dataset', 'classifier', and metric columns.

    metric : str
        Metric to visualize.

    output_path : str or None
        Path to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Pivot to create matrix
    pivot = results_df.pivot(index='dataset', columns='classifier', values=metric)

    fig, ax = plt.subplots(figsize=(12, len(pivot) * 0.5 + 2))

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax,
               cbar_kws={'label': metric}, linewidths=0.5)

    ax.set_title(f'Classifier Performance Heatmap ({metric})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Classifier', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    return fig


def plot_boxplot_comparison(results_df, output_path=None):
    """
    Create box plot comparing classifier accuracies across datasets.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results with 'classifier' and 'accuracy_mean' columns.

    output_path : str or None
        Path to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Box plot
    classifiers = results_df['classifier'].unique()
    data = [results_df[results_df['classifier'] == clf]['accuracy_mean'].values
            for clf in classifiers]

    bp = ax.boxplot(data, labels=classifiers, patch_artist=True)

    # Color the boxes
    colors = sns.color_palette('Set2', len(classifiers))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(classifiers, rotation=45, ha='right')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Distribution Across Datasets', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    return fig


def plot_learning_curves(hyperparam_results, output_path=None):
    """
    Plot learning curves showing performance vs n_estimators.

    Parameters
    ----------
    hyperparam_results : dict
        Results from hyperparameter study with 'n_estimators', 'accuracies', 'times'.

    output_path : str or None
        Path to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    n_estimators = hyperparam_results['n_estimators']
    accuracies = hyperparam_results['accuracies']
    times = hyperparam_results['times']

    # Accuracy vs n_estimators
    ax1.plot(n_estimators, accuracies, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    if 'std_devs' in hyperparam_results:
        std_devs = hyperparam_results['std_devs']
        ax1.fill_between(n_estimators,
                        np.array(accuracies) - np.array(std_devs),
                        np.array(accuracies) + np.array(std_devs),
                        alpha=0.2)

    ax1.set_xlabel('Number of Estimators', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Learning Curve: Accuracy vs Ensemble Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Time vs n_estimators
    ax2.plot(n_estimators, times, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Number of Estimators', fontsize=12)
    ax2.set_ylabel('Training Time (seconds)', fontsize=12)
    ax2.set_title('Computational Cost vs Ensemble Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    return fig


def plot_critical_difference_diagram(results_df, metric='accuracy_mean', output_path=None):
    """
    Create critical difference diagram for comparing multiple classifiers.

    Note: This is a simplified version. For full CD diagrams, consider using
    the Orange library or similar tools.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results with 'dataset', 'classifier', and metric columns.

    metric : str
        Metric to use for ranking.

    output_path : str or None
        Path to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Calculate average ranks
    ranks = []

    for dataset in results_df['dataset'].unique():
        dataset_results = results_df[results_df['dataset'] == dataset].copy()
        dataset_results['rank'] = dataset_results[metric].rank(ascending=False)

        for _, row in dataset_results.iterrows():
            ranks.append({
                'classifier': row['classifier'],
                'rank': row['rank']
            })

    ranks_df = pd.DataFrame(ranks)
    avg_ranks = ranks_df.groupby('classifier')['rank'].mean().sort_values()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(avg_ranks))
    ax.barh(y_pos, avg_ranks.values, alpha=0.7, edgecolor='black', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(avg_ranks.index)
    ax.set_xlabel('Average Rank', fontsize=12)
    ax.set_title('Average Rank Across Datasets (lower is better)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add rank values
    for i, rank in enumerate(avg_ranks.values):
        ax.text(rank + 0.1, i, f'{rank:.2f}', va='center', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    return fig


def create_all_rq1_visualizations(rq1_results, output_dir='figures'):
    """
    Create all visualizations for RQ1 experiments.

    Parameters
    ----------
    rq1_results : pandas.DataFrame
        Results from RQ1 experiments.

    output_dir : str
        Directory to save figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Generating RQ1 visualizations...")

    # 1. Paired comparison
    print("  1/3: Paired scatter plot...")
    plot_paired_comparison(rq1_results, 'native', 'onehot',
                          output_path=os.path.join(output_dir, 'rq1_paired_comparison.pdf'))

    # 2. Bar chart
    print("  2/3: Bar chart...")
    comparison_data = []
    for _, row in rq1_results.iterrows():
        comparison_data.append({
            'dataset': row['dataset'],
            'method': 'Native',
            'accuracy': row['native_accuracy_mean'],
            'std': row['native_accuracy_std']
        })
        comparison_data.append({
            'dataset': row['dataset'],
            'method': 'OneHot',
            'accuracy': row['onehot_accuracy_mean'],
            'std': row['onehot_accuracy_std']
        })

    comparison_df = pd.DataFrame(comparison_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    datasets = rq1_results['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.35

    native_acc = rq1_results['native_accuracy_mean'].values
    native_std = rq1_results['native_accuracy_std'].values
    onehot_acc = rq1_results['onehot_accuracy_mean'].values
    onehot_std = rq1_results['onehot_accuracy_std'].values

    ax.bar(x - width/2, native_acc, width, label='Native', yerr=native_std,
          capsize=5, alpha=0.7, edgecolor='black')
    ax.bar(x + width/2, onehot_acc, width, label='OneHot', yerr=onehot_std,
          capsize=5, alpha=0.7, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('RQ1: Native vs One-Hot Encoding', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rq1_bar_comparison.pdf'), bbox_inches='tight')
    plt.close()

    # 3. Difference plot
    print("  3/3: Difference plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    differences = rq1_results['difference'].values
    datasets = rq1_results['dataset'].values

    colors = ['green' if d > 0 else 'red' for d in differences]
    ax.barh(datasets, differences, color=colors, alpha=0.7, edgecolor='black')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Accuracy Difference (Native - OneHot)', fontsize=12)
    ax.set_title('RQ1: Performance Difference', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rq1_difference_plot.pdf'), bbox_inches='tight')
    plt.close()

    print("  ✓ RQ1 visualizations complete")


def create_all_rq2_visualizations(rq2_results, output_dir='figures'):
    """
    Create all visualizations for RQ2 experiments.

    Parameters
    ----------
    rq2_results : pandas.DataFrame
        Results from RQ2 experiments.

    output_dir : str
        Directory to save figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Generating RQ2 visualizations...")

    # 1. Heatmap
    print("  1/4: Heatmap...")
    plot_heatmap(rq2_results, metric='accuracy_mean',
                output_path=os.path.join(output_dir, 'rq2_heatmap.pdf'))

    # 2. Box plot
    print("  2/4: Box plot...")
    plot_boxplot_comparison(rq2_results,
                           output_path=os.path.join(output_dir, 'rq2_boxplot.pdf'))

    # 3. Bar chart
    print("  3/4: Bar chart...")
    plot_classifier_comparison_bars(rq2_results,
                                   output_path=os.path.join(output_dir, 'rq2_bar_comparison.pdf'))

    # 4. Critical difference diagram
    print("  4/4: Critical difference diagram...")
    plot_critical_difference_diagram(rq2_results,
                                    output_path=os.path.join(output_dir, 'rq2_critical_difference.pdf'))

    print("  ✓ RQ2 visualizations complete")


if __name__ == '__main__':
    # Test with dummy data
    print("Testing visualization functions...")

    # Create dummy RQ1 data
    rq1_data = pd.DataFrame({
        'dataset': ['dataset1', 'dataset2', 'dataset3'],
        'native_accuracy_mean': [0.85, 0.78, 0.92],
        'native_accuracy_std': [0.02, 0.03, 0.01],
        'onehot_accuracy_mean': [0.83, 0.80, 0.90],
        'onehot_accuracy_std': [0.02, 0.02, 0.02],
        'difference': [0.02, -0.02, 0.02]
    })

    create_all_rq1_visualizations(rq1_data, output_dir='figures')

    print("\n✓ Visualization tests complete")
