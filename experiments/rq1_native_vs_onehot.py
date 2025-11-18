"""
RQ1: Does TreeEnsembleClassifier perform better on average when handling
categorical features natively than when used with one-hot encoding?

This module compares TreeEnsembleClassifier performance with native categorical
handling vs one-hot encoding preprocessing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import time

from solution.tree_ensemble import TreeEnsembleClassifier
from experiments.evaluation import evaluate_classifier
from experiments.data_loader import get_dataset_info


def run_rq1_experiments(datasets, n_estimators=200, cv=5, random_state=42):
    """
    Run RQ1 experiments: compare native categorical handling vs one-hot encoding.

    This function evaluates TreeEnsembleClassifier in two configurations:
    1. Native: Direct use of categorical features (no preprocessing)
    2. OneHot: One-hot encoding applied before classification

    Parameters
    ----------
    datasets : dict
        Dictionary mapping dataset_name -> (X, y).

    n_estimators : int, default=200
        Number of estimators in the ensemble.

    cv : int, default=5
        Number of cross-validation folds.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    results_df : pandas.DataFrame
        Results for each dataset with columns:
        - dataset: dataset name
        - n_samples, n_features, n_classes: dataset characteristics
        - native_accuracy_mean, native_accuracy_std: native approach metrics
        - onehot_accuracy_mean, onehot_accuracy_std: one-hot approach metrics
        - native_balanced_acc, onehot_balanced_acc: balanced accuracy
        - native_time, onehot_time: computation time
        - difference: native - onehot accuracy
        - winner: which approach performed better
    """
    print("=" * 80)
    print("RQ1: Native Categorical Handling vs One-Hot Encoding")
    print("=" * 80)
    print(f"Configuration: n_estimators={n_estimators}, cv={cv}")
    print("=" * 80 + "\n")

    results = []

    for dataset_name, (X, y) in datasets.items():
        print(f"\n[{dataset_name.upper()}]")
        print("-" * 80)

        # Get dataset info
        info = get_dataset_info(X, y)
        print(f"  Samples: {info['n_samples']}, Features: {info['n_features']}, Classes: {info['n_classes']}")

        # 1. NATIVE CATEGORICAL HANDLING
        print(f"  [1/2] Evaluating Native approach...", end=' ')
        clf_native = TreeEnsembleClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

        start_time = time.time()
        scores_native = evaluate_classifier(clf_native, X, y, cv=cv, random_state=random_state)
        native_time = time.time() - start_time

        print(f"✓ Acc: {scores_native['accuracy_mean']:.4f} ± {scores_native['accuracy_std']:.4f}")

        # 2. ONE-HOT ENCODING APPROACH
        print(f"  [2/2] Evaluating One-Hot approach...", end=' ')

        # We need to apply one-hot encoding inside each CV fold to avoid data leakage
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

        onehot_accuracies = []
        onehot_balanced_accs = []
        onehot_aurocs = []
        onehot_fit_times = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Apply one-hot encoding
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_train_encoded = encoder.fit_transform(X_train)
            X_test_encoded = encoder.transform(X_test)

            # Train classifier
            clf_onehot = TreeEnsembleClassifier(n_estimators=n_estimators, random_state=random_state)

            fold_start = time.time()
            clf_onehot.fit(X_train_encoded, y_train)
            fold_time = time.time() - fold_start

            # Evaluate
            y_pred = clf_onehot.predict(X_test_encoded)
            acc = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)

            onehot_accuracies.append(acc)
            onehot_balanced_accs.append(balanced_acc)
            onehot_fit_times.append(fold_time)

        onehot_time = time.time() - start_time

        scores_onehot_mean = np.mean(onehot_accuracies)
        scores_onehot_std = np.std(onehot_accuracies)
        balanced_acc_onehot_mean = np.mean(onehot_balanced_accs)

        print(f"✓ Acc: {scores_onehot_mean:.4f} ± {scores_onehot_std:.4f}")

        # Compare results
        difference = scores_native['accuracy_mean'] - scores_onehot_mean
        winner = 'Native' if difference > 0 else ('OneHot' if difference < 0 else 'Tie')

        print(f"  Difference: {difference:+.4f} (Winner: {winner})")
        print(f"  Time - Native: {native_time:.2f}s, OneHot: {onehot_time:.2f}s")

        # Store results
        results.append({
            'dataset': dataset_name,
            'n_samples': info['n_samples'],
            'n_features': info['n_features'],
            'n_classes': info['n_classes'],
            'native_accuracy_mean': scores_native['accuracy_mean'],
            'native_accuracy_std': scores_native['accuracy_std'],
            'native_balanced_acc': scores_native['balanced_accuracy_mean'],
            'native_time': native_time,
            'onehot_accuracy_mean': scores_onehot_mean,
            'onehot_accuracy_std': scores_onehot_std,
            'onehot_balanced_acc': balanced_acc_onehot_mean,
            'onehot_time': onehot_time,
            'difference': difference,
            'winner': winner
        })

    results_df = pd.DataFrame(results)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    wins_native = (results_df['winner'] == 'Native').sum()
    wins_onehot = (results_df['winner'] == 'OneHot').sum()
    ties = (results_df['winner'] == 'Tie').sum()

    print(f"  Native wins:   {wins_native}/{len(results_df)}")
    print(f"  OneHot wins:   {wins_onehot}/{len(results_df)}")
    print(f"  Ties:          {ties}/{len(results_df)}")
    print(f"  Average difference: {results_df['difference'].mean():+.4f}")
    print(f"  Std difference:     {results_df['difference'].std():.4f}")

    # Average accuracies
    avg_native = results_df['native_accuracy_mean'].mean()
    avg_onehot = results_df['onehot_accuracy_mean'].mean()

    print(f"\n  Average accuracy:")
    print(f"    Native:  {avg_native:.4f}")
    print(f"    OneHot:  {avg_onehot:.4f}")
    print(f"    Difference: {avg_native - avg_onehot:+.4f}")

    # Average time
    avg_time_native = results_df['native_time'].mean()
    avg_time_onehot = results_df['onehot_time'].mean()

    print(f"\n  Average time:")
    print(f"    Native:  {avg_time_native:.2f}s")
    print(f"    OneHot:  {avg_time_onehot:.2f}s")
    print(f"    Speedup: {avg_time_onehot / avg_time_native:.2f}x")

    print("\n" + "=" * 80)

    return results_df


def analyze_rq1_results(results_df):
    """
    Perform detailed analysis of RQ1 results.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results from run_rq1_experiments.

    Returns
    -------
    analysis : dict
        Dictionary with detailed analysis including:
        - win/loss/tie counts
        - average accuracies
        - correlation with dataset characteristics
        - statistical significance (to be added)
    """
    analysis = {}

    # Win/Loss/Tie analysis
    analysis['wins_native'] = (results_df['winner'] == 'Native').sum()
    analysis['wins_onehot'] = (results_df['winner'] == 'OneHot').sum()
    analysis['ties'] = (results_df['winner'] == 'Tie').sum()
    analysis['total_datasets'] = len(results_df)

    # Average performance
    analysis['avg_accuracy_native'] = results_df['native_accuracy_mean'].mean()
    analysis['avg_accuracy_onehot'] = results_df['onehot_accuracy_mean'].mean()
    analysis['avg_difference'] = results_df['difference'].mean()
    analysis['std_difference'] = results_df['difference'].std()

    # Balanced accuracy
    analysis['avg_balanced_acc_native'] = results_df['native_balanced_acc'].mean()
    analysis['avg_balanced_acc_onehot'] = results_df['onehot_balanced_acc'].mean()

    # Time analysis
    analysis['avg_time_native'] = results_df['native_time'].mean()
    analysis['avg_time_onehot'] = results_df['onehot_time'].mean()
    analysis['speedup_factor'] = analysis['avg_time_onehot'] / analysis['avg_time_native']

    # Correlation with dataset size
    if 'n_samples' in results_df.columns:
        analysis['corr_difference_samples'] = results_df['difference'].corr(results_df['n_samples'])
        analysis['corr_difference_features'] = results_df['difference'].corr(results_df['n_features'])

    # Best and worst cases
    best_native_idx = results_df['difference'].idxmax()
    worst_native_idx = results_df['difference'].idxmin()

    analysis['best_native_dataset'] = results_df.loc[best_native_idx, 'dataset']
    analysis['best_native_difference'] = results_df.loc[best_native_idx, 'difference']

    analysis['worst_native_dataset'] = results_df.loc[worst_native_idx, 'dataset']
    analysis['worst_native_difference'] = results_df.loc[worst_native_idx, 'difference']

    return analysis


def print_rq1_analysis(analysis):
    """
    Print formatted analysis of RQ1 results.

    Parameters
    ----------
    analysis : dict
        Analysis dictionary from analyze_rq1_results.
    """
    print("\n" + "=" * 80)
    print("RQ1 DETAILED ANALYSIS")
    print("=" * 80)

    print(f"\n1. Win/Loss/Tie Distribution:")
    print(f"   Native wins:  {analysis['wins_native']}/{analysis['total_datasets']} ({100*analysis['wins_native']/analysis['total_datasets']:.1f}%)")
    print(f"   OneHot wins:  {analysis['wins_onehot']}/{analysis['total_datasets']} ({100*analysis['wins_onehot']/analysis['total_datasets']:.1f}%)")
    print(f"   Ties:         {analysis['ties']}/{analysis['total_datasets']} ({100*analysis['ties']/analysis['total_datasets']:.1f}%)")

    print(f"\n2. Average Performance:")
    print(f"   Native accuracy:  {analysis['avg_accuracy_native']:.4f}")
    print(f"   OneHot accuracy:  {analysis['avg_accuracy_onehot']:.4f}")
    print(f"   Difference:       {analysis['avg_difference']:+.4f} ± {analysis['std_difference']:.4f}")

    print(f"\n3. Balanced Accuracy:")
    print(f"   Native:  {analysis['avg_balanced_acc_native']:.4f}")
    print(f"   OneHot:  {analysis['avg_balanced_acc_onehot']:.4f}")

    print(f"\n4. Computational Efficiency:")
    print(f"   Average time (Native):  {analysis['avg_time_native']:.2f}s")
    print(f"   Average time (OneHot):  {analysis['avg_time_onehot']:.2f}s")
    print(f"   Speedup factor:         {analysis['speedup_factor']:.2f}x")

    if 'corr_difference_samples' in analysis:
        print(f"\n5. Correlation with Dataset Characteristics:")
        print(f"   Difference vs n_samples:  {analysis['corr_difference_samples']:.3f}")
        print(f"   Difference vs n_features: {analysis['corr_difference_features']:.3f}")

    print(f"\n6. Extreme Cases:")
    print(f"   Best for Native:  {analysis['best_native_dataset']} (diff: {analysis['best_native_difference']:+.4f})")
    print(f"   Worst for Native: {analysis['worst_native_dataset']} (diff: {analysis['worst_native_difference']:+.4f})")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    # Test RQ1 experiments
    from experiments.data_loader import load_categorical_datasets

    print("Loading datasets...")
    datasets = load_categorical_datasets()

    print(f"\nRunning RQ1 experiments on {len(datasets)} datasets...")
    results_df = run_rq1_experiments(datasets, n_estimators=100, cv=5, random_state=42)

    # Analyze results
    analysis = analyze_rq1_results(results_df)
    print_rq1_analysis(analysis)

    # Save results
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'rq1_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Save analysis
    analysis_file = os.path.join(output_dir, 'rq1_analysis.txt')
    with open(analysis_file, 'w') as f:
        f.write("RQ1 Analysis\n")
        f.write("=" * 80 + "\n\n")
        for key, value in analysis.items():
            f.write(f"{key}: {value}\n")

    print(f"Analysis saved to: {analysis_file}")
