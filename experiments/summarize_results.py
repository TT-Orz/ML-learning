"""
Generate LaTeX tables and summary statistics for the paper.

This module creates publication-ready tables and summaries for inclusion
in the coursework report.
"""

import numpy as np
import pandas as pd


def generate_latex_table_rq1(rq1_results):
    """
    Generate LaTeX table for RQ1 results.

    Format: Dataset | Native | One-Hot | Difference | Winner

    Parameters
    ----------
    rq1_results : pandas.DataFrame
        Results from RQ1 experiments.

    Returns
    -------
    latex_table : str
        LaTeX formatted table.
    """
    latex = []

    latex.append(r"\begin{table}[htbp]")
    latex.append(r"    \centering")
    latex.append(r"    \caption{RQ1: Native Categorical Handling vs One-Hot Encoding}")
    latex.append(r"    \label{tab:rq1_results}")
    latex.append(r"    \begin{tabular}{lcccc}")
    latex.append(r"        \toprule")
    latex.append(r"        \textbf{Dataset} & \textbf{Native} & \textbf{One-Hot} & \textbf{Difference} & \textbf{Winner} \\")
    latex.append(r"        \midrule")

    for _, row in rq1_results.iterrows():
        dataset = row['dataset'].replace('_', r'\_')
        native_acc = f"{row['native_accuracy_mean']:.3f}"
        native_std = f"{row['native_accuracy_std']:.3f}"
        onehot_acc = f"{row['onehot_accuracy_mean']:.3f}"
        onehot_std = f"{row['onehot_accuracy_std']:.3f}"
        diff = f"{row['difference']:+.3f}"

        winner = row['winner']
        if winner == 'Native':
            winner_mark = r"\textbf{Native}"
        elif winner == 'OneHot':
            winner_mark = r"\textbf{OneHot}"
        else:
            winner_mark = "Tie"

        latex.append(
            f"        {dataset} & "
            f"${native_acc} \\pm {native_std}$ & "
            f"${onehot_acc} \\pm {onehot_std}$ & "
            f"${diff}$ & "
            f"{winner_mark} \\\\"
        )

    latex.append(r"        \midrule")

    # Add averages
    avg_native = rq1_results['native_accuracy_mean'].mean()
    avg_onehot = rq1_results['onehot_accuracy_mean'].mean()
    avg_diff = rq1_results['difference'].mean()

    latex.append(
        f"        \\textbf{{Average}} & "
        f"$\\mathbf{{{avg_native:.3f}}}$ & "
        f"$\\mathbf{{{avg_onehot:.3f}}}$ & "
        f"$\\mathbf{{{avg_diff:+.3f}}}$ & "
        f" \\\\"
    )

    # Add win counts
    wins_native = (rq1_results['winner'] == 'Native').sum()
    wins_onehot = (rq1_results['winner'] == 'OneHot').sum()

    latex.append(
        f"        \\textbf{{Wins}} & "
        f"$\\mathbf{{{wins_native}}}$ & "
        f"$\\mathbf{{{wins_onehot}}}$ & "
        f" & "
        f" \\\\"
    )

    latex.append(r"        \bottomrule")
    latex.append(r"    \end{tabular}")
    latex.append(r"\end{table}")

    return '\n'.join(latex)


def generate_latex_table_rq2(rq2_results):
    """
    Generate LaTeX table for RQ2 results.

    Format: Dataset | Classifier1 | Classifier2 | ... | Best

    Parameters
    ----------
    rq2_results : pandas.DataFrame
        Results from RQ2 experiments.

    Returns
    -------
    latex_table : str
        LaTeX formatted table.
    """
    # Pivot table to get classifiers as columns
    pivot = rq2_results.pivot(index='dataset', columns='classifier', values='accuracy_mean')

    classifiers = pivot.columns.tolist()
    datasets = pivot.index.tolist()

    latex = []

    latex.append(r"\begin{table}[htbp]")
    latex.append(r"    \centering")
    latex.append(r"    \caption{RQ2: Comparison with Baseline Classifiers}")
    latex.append(r"    \label{tab:rq2_results}")

    # Adjust column count (dataset + classifiers + best)
    n_cols = len(classifiers) + 2
    col_spec = 'l' + 'c' * (n_cols - 1)

    latex.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    latex.append(r"        \toprule")

    # Header - abbreviate classifier names if too long
    header = r"        \textbf{Dataset}"
    for clf in classifiers:
        clf_abbrev = clf.replace('TreeEnsemble', 'TE').replace('_OneHot', '').replace('_Native', '(N)')
        clf_abbrev = clf_abbrev.replace('RandomForest', 'RF').replace('GradientBoosting', 'GB')
        clf_abbrev = clf_abbrev.replace('HistGradientBoosting', 'HGB').replace('AdaBoost', 'AB')
        header += f" & \\textbf{{{clf_abbrev}}}"

    header += r" & \textbf{Best} \\"

    latex.append(header)
    latex.append(r"        \midrule")

    # Data rows
    for dataset in datasets:
        dataset_clean = dataset.replace('_', r'\_')
        row = f"        {dataset_clean}"

        best_acc = pivot.loc[dataset].max()
        best_clf = pivot.loc[dataset].idxmax()

        for clf in classifiers:
            acc = pivot.loc[dataset, clf]

            if pd.isna(acc):
                row += " & ---"
            else:
                # Bold the best accuracy
                if acc == best_acc:
                    row += f" & $\\mathbf{{{acc:.3f}}}$"
                else:
                    row += f" & ${acc:.3f}$"

        # Add best classifier
        best_abbrev = best_clf.replace('TreeEnsemble', 'TE').replace('_OneHot', '').replace('_Native', '(N)')
        row += f" & {best_abbrev} \\\\"

        latex.append(row)

    latex.append(r"        \midrule")

    # Average row
    row = r"        \textbf{Average}"
    avg_accs = pivot.mean()

    best_avg = avg_accs.max()

    for clf in classifiers:
        avg = avg_accs[clf]

        if pd.isna(avg):
            row += " & ---"
        else:
            if avg == best_avg:
                row += f" & $\\mathbf{{{avg:.3f}}}$"
            else:
                row += f" & ${avg:.3f}$"

    row += r" & \\"
    latex.append(row)

    # Rank row
    row = r"        \textbf{Avg. Rank}"

    ranks = []
    for dataset in datasets:
        dataset_ranks = pivot.loc[dataset].rank(ascending=False)
        ranks.append(dataset_ranks)

    avg_ranks = pd.concat(ranks, axis=1).mean(axis=1)

    for clf in classifiers:
        rank = avg_ranks[clf]
        row += f" & ${rank:.2f}$"

    row += r" & \\"
    latex.append(row)

    latex.append(r"        \bottomrule")
    latex.append(r"    \end{tabular}")
    latex.append(r"\end{table}")

    return '\n'.join(latex)


def generate_summary_statistics(rq1_results, rq2_results):
    """
    Generate overall summary statistics for the experiments.

    Parameters
    ----------
    rq1_results : pandas.DataFrame
        RQ1 results.

    rq2_results : pandas.DataFrame
        RQ2 results.

    Returns
    -------
    summary : str
        Text summary.
    """
    lines = []

    lines.append("=" * 80)
    lines.append("EXPERIMENTAL RESULTS SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    # RQ1 Summary
    lines.append("RQ1: Native vs One-Hot Encoding")
    lines.append("-" * 80)

    wins_native = (rq1_results['winner'] == 'Native').sum()
    wins_onehot = (rq1_results['winner'] == 'OneHot').sum()
    ties = (rq1_results['winner'] == 'Tie').sum()
    total = len(rq1_results)

    lines.append(f"Total datasets: {total}")
    lines.append(f"Native wins: {wins_native} ({100*wins_native/total:.1f}%)")
    lines.append(f"OneHot wins: {wins_onehot} ({100*wins_onehot/total:.1f}%)")
    lines.append(f"Ties: {ties} ({100*ties/total:.1f}%)")
    lines.append("")

    avg_native = rq1_results['native_accuracy_mean'].mean()
    std_native = rq1_results['native_accuracy_mean'].std()
    avg_onehot = rq1_results['onehot_accuracy_mean'].mean()
    std_onehot = rq1_results['onehot_accuracy_mean'].std()

    lines.append(f"Average accuracy (Native): {avg_native:.4f} ± {std_native:.4f}")
    lines.append(f"Average accuracy (OneHot): {avg_onehot:.4f} ± {std_onehot:.4f}")
    lines.append(f"Difference: {avg_native - avg_onehot:+.4f}")
    lines.append("")

    avg_time_native = rq1_results['native_time'].mean()
    avg_time_onehot = rq1_results['onehot_time'].mean()

    lines.append(f"Average time (Native): {avg_time_native:.2f}s")
    lines.append(f"Average time (OneHot): {avg_time_onehot:.2f}s")
    lines.append(f"Speedup: {avg_time_onehot / avg_time_native:.2f}x")
    lines.append("")

    # RQ2 Summary
    lines.append("RQ2: Comparison with Baselines")
    lines.append("-" * 80)

    # Average accuracy per classifier
    avg_by_clf = rq2_results.groupby('classifier')['accuracy_mean'].agg(['mean', 'std'])
    avg_by_clf = avg_by_clf.sort_values('mean', ascending=False)

    lines.append("Average accuracy by classifier:")
    for clf, row in avg_by_clf.iterrows():
        lines.append(f"  {clf:30s}: {row['mean']:.4f} ± {row['std']:.4f}")

    lines.append("")

    # Win counts
    win_counts = {}
    for dataset in rq2_results['dataset'].unique():
        dataset_results = rq2_results[rq2_results['dataset'] == dataset]
        winner = dataset_results.loc[dataset_results['accuracy_mean'].idxmax(), 'classifier']

        if winner not in win_counts:
            win_counts[winner] = 0
        win_counts[winner] += 1

    lines.append("Win count by classifier:")
    for clf, wins in sorted(win_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {clf:30s}: {wins}/{len(rq2_results['dataset'].unique())}")

    lines.append("")

    # Average ranks
    ranks = []
    for dataset in rq2_results['dataset'].unique():
        dataset_results = rq2_results[rq2_results['dataset'] == dataset].copy()
        dataset_results['rank'] = dataset_results['accuracy_mean'].rank(ascending=False)
        ranks.append(dataset_results[['classifier', 'rank']])

    all_ranks = pd.concat(ranks)
    avg_ranks = all_ranks.groupby('classifier')['rank'].mean().sort_values()

    lines.append("Average rank by classifier (lower is better):")
    for clf, rank in avg_ranks.items():
        lines.append(f"  {clf:30s}: {rank:.2f}")

    lines.append("")
    lines.append("=" * 80)

    return '\n'.join(lines)


def generate_dataset_table(datasets):
    """
    Generate LaTeX table describing datasets.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets.

    Returns
    -------
    latex_table : str
        LaTeX formatted table.
    """
    from experiments.data_loader import get_dataset_info

    latex = []

    latex.append(r"\begin{table}[htbp]")
    latex.append(r"    \centering")
    latex.append(r"    \caption{Dataset Characteristics}")
    latex.append(r"    \label{tab:datasets}")
    latex.append(r"    \begin{tabular}{lccccc}")
    latex.append(r"        \toprule")
    latex.append(r"        \textbf{Dataset} & \textbf{Samples} & \textbf{Features} & \textbf{Classes} & \textbf{Imbalanced} & \textbf{Avg. Categories} \\")
    latex.append(r"        \midrule")

    for dataset_name, (X, y) in datasets.items():
        info = get_dataset_info(X, y)

        dataset_clean = dataset_name.replace('_', r'\_')
        imbalanced = 'Yes' if info['is_imbalanced'] else 'No'

        latex.append(
            f"        {dataset_clean} & "
            f"{info['n_samples']} & "
            f"{info['n_features']} & "
            f"{info['n_classes']} & "
            f"{imbalanced} & "
            f"{info['avg_categories_per_feature']:.1f} \\\\"
        )

    latex.append(r"        \bottomrule")
    latex.append(r"    \end{tabular}")
    latex.append(r"\end{table}")

    return '\n'.join(latex)


if __name__ == '__main__':
    # Test with dummy data
    print("Testing summarize_results functions...")

    # Create dummy RQ1 data
    rq1_data = pd.DataFrame({
        'dataset': ['dataset1', 'dataset2', 'dataset3'],
        'native_accuracy_mean': [0.85, 0.78, 0.92],
        'native_accuracy_std': [0.02, 0.03, 0.01],
        'onehot_accuracy_mean': [0.83, 0.80, 0.90],
        'onehot_accuracy_std': [0.02, 0.02, 0.02],
        'difference': [0.02, -0.02, 0.02],
        'winner': ['Native', 'OneHot', 'Native'],
        'native_time': [10.5, 8.2, 15.3],
        'onehot_time': [12.1, 9.5, 18.2]
    })

    print("\nGenerating RQ1 LaTeX table...")
    rq1_latex = generate_latex_table_rq1(rq1_data)
    print(rq1_latex)

    print("\n✓ summarize_results tests complete")
