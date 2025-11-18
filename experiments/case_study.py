"""
Case study: Detailed analysis of a single dataset.

This module performs in-depth analysis of one dataset including:
- Dataset characteristics
- Detailed classifier comparison
- Hyperparameter sensitivity
- Confusion matrices
- Feature importance (if applicable)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import time

from solution.tree_ensemble import TreeEnsembleClassifier
from experiments.evaluation import evaluate_with_confusion_matrix
from experiments.data_loader import get_dataset_info
from experiments.rq2_baseline_comparison import create_classifiers


def case_study_analysis(dataset_name, X, y, cv=5, random_state=42):
    """
    Perform comprehensive case study on a single dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    X : array-like
        Features.

    y : array-like
        Labels.

    cv : int, default=5
        Number of CV folds.

    random_state : int, default=42
        Random seed.

    Returns
    -------
    results : dict
        Comprehensive results including:
        - dataset_info: dataset characteristics
        - classifier_comparison: performance of all classifiers
        - confusion_matrices: confusion matrices for each classifier
        - hyperparameter_study: sensitivity to n_estimators
        - timing_analysis: computation time vs accuracy trade-offs
    """
    print("=" * 80)
    print(f"CASE STUDY: {dataset_name.upper()}")
    print("=" * 80 + "\n")

    results = {}

    # 1. Dataset Information
    print("[1/5] Analyzing dataset characteristics...")
    info = get_dataset_info(X, y)
    results['dataset_info'] = info

    print(f"  Samples:          {info['n_samples']}")
    print(f"  Features:         {info['n_features']}")
    print(f"  Classes:          {info['n_classes']}")
    print(f"  Class distribution: {info['class_distribution']}")
    print(f"  Imbalanced:       {info['is_imbalanced']}")
    print(f"  Avg categories/feature: {info['avg_categories_per_feature']:.1f}")

    # 2. Classifier Comparison
    print("\n[2/5] Comparing all classifiers...")
    classifiers = create_classifiers(n_estimators=200, random_state=random_state)

    from experiments.evaluation import compare_classifiers
    comparison_results = compare_classifiers(classifiers, X, y, cv=cv, random_state=random_state)
    results['classifier_comparison'] = comparison_results

    print("\n  Results:")
    print(comparison_results[['classifier', 'accuracy_mean', 'balanced_accuracy_mean', 'fit_time_mean']].to_string(index=False))

    # 3. Confusion Matrices
    print("\n[3/5] Computing confusion matrices...")
    results['confusion_matrices'] = {}

    # Get confusion matrix for top 3 classifiers
    top_3 = comparison_results.nlargest(3, 'accuracy_mean')['classifier'].tolist()

    for clf_name in top_3:
        print(f"  Computing for {clf_name}...")
        clf = classifiers[clf_name]

        cm_results = evaluate_with_confusion_matrix(clf, X, y, cv=cv, random_state=random_state)
        results['confusion_matrices'][clf_name] = cm_results['confusion_matrix']

    # 4. Hyperparameter Sensitivity
    print("\n[4/5] Analyzing hyperparameter sensitivity...")
    n_estimators_list = [10, 25, 50, 100, 200, 500]

    hyperparam_results = study_hyperparameter_sensitivity(
        X, y, n_estimators_list=n_estimators_list, cv=cv, random_state=random_state
    )
    results['hyperparameter_study'] = hyperparam_results

    print("  n_estimators | Accuracy | Time")
    print("  " + "-" * 40)
    for n_est, acc, time_val in zip(n_estimators_list,
                                     hyperparam_results['accuracies'],
                                     hyperparam_results['times']):
        print(f"  {n_est:12d} | {acc:.4f}  | {time_val:.2f}s")

    # 5. Timing Analysis
    print("\n[5/5] Performing timing analysis...")
    timing_results = analyze_timing(classifiers, X, y, random_state=random_state)
    results['timing_analysis'] = timing_results

    print("  Classifier | Time | Accuracy")
    print("  " + "-" * 50)
    for clf_name, time_val, acc in zip(timing_results['classifiers'],
                                       timing_results['times'],
                                       timing_results['accuracies']):
        print(f"  {clf_name:30s} | {time_val:5.2f}s | {acc:.4f}")

    print("\n" + "=" * 80)
    print("CASE STUDY COMPLETE")
    print("=" * 80)

    return results


def study_hyperparameter_sensitivity(X, y, n_estimators_list, cv=5, random_state=42):
    """
    Study sensitivity to n_estimators parameter.

    Parameters
    ----------
    X, y : array-like
        Data and labels.

    n_estimators_list : list
        List of n_estimators values to test.

    cv : int
        Number of CV folds.

    random_state : int
        Random seed.

    Returns
    -------
    results : dict
        Dictionary with n_estimators, accuracies, and times.
    """
    from experiments.evaluation import evaluate_classifier

    accuracies = []
    std_devs = []
    times = []

    for n_est in n_estimators_list:
        clf = TreeEnsembleClassifier(n_estimators=n_est, random_state=random_state)

        start_time = time.time()
        scores = evaluate_classifier(clf, X, y, cv=cv, random_state=random_state)
        elapsed = time.time() - start_time

        accuracies.append(scores['accuracy_mean'])
        std_devs.append(scores['accuracy_std'])
        times.append(elapsed)

    return {
        'n_estimators': n_estimators_list,
        'accuracies': accuracies,
        'std_devs': std_devs,
        'times': times
    }


def analyze_timing(classifiers, X, y, random_state=42):
    """
    Analyze timing and accuracy trade-offs for different classifiers.

    Parameters
    ----------
    classifiers : dict
        Dictionary of classifiers.

    X, y : array-like
        Data and labels.

    random_state : int
        Random seed.

    Returns
    -------
    results : dict
        Dictionary with timing and accuracy information.
    """
    # Use a single train/test split for faster timing analysis
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    clf_names = []
    times = []
    accuracies = []

    for clf_name, clf in classifiers.items():
        try:
            start_time = time.time()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            elapsed = time.time() - start_time

            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_test, y_pred)

            clf_names.append(clf_name)
            times.append(elapsed)
            accuracies.append(acc)
        except Exception as e:
            print(f"  Error with {clf_name}: {e}")

    return {
        'classifiers': clf_names,
        'times': times,
        'accuracies': accuracies
    }


def plot_case_study_results(results, output_dir='figures'):
    """
    Generate plots for case study results.

    Parameters
    ----------
    results : dict
        Results from case_study_analysis.

    output_dir : str
        Directory to save figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Confusion matrices
    plot_confusion_matrices(results['confusion_matrices'], output_dir)

    # 2. Hyperparameter sensitivity
    plot_hyperparameter_sensitivity(results['hyperparameter_study'], output_dir)

    # 3. Timing vs Accuracy
    plot_timing_vs_accuracy(results['timing_analysis'], output_dir)

    # 4. Classifier comparison bar chart
    plot_classifier_comparison(results['classifier_comparison'], output_dir)


def plot_confusion_matrices(confusion_matrices, output_dir):
    """Plot confusion matrices for each classifier."""
    n_classifiers = len(confusion_matrices)

    if n_classifiers == 0:
        return

    fig, axes = plt.subplots(1, n_classifiers, figsize=(6 * n_classifiers, 5))

    if n_classifiers == 1:
        axes = [axes]

    for idx, (clf_name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
        ax.set_title(clf_name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'case_study_confusion_matrices.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close()


def plot_hyperparameter_sensitivity(hyperparam_results, output_dir):
    """Plot hyperparameter sensitivity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    n_estimators = hyperparam_results['n_estimators']
    accuracies = hyperparam_results['accuracies']
    times = hyperparam_results['times']

    # Accuracy vs n_estimators
    ax1.plot(n_estimators, accuracies, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Estimators', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy vs Number of Estimators', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Time vs n_estimators
    ax2.plot(n_estimators, times, 'o-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Number of Estimators', fontsize=12)
    ax2.set_ylabel('Training Time (s)', fontsize=12)
    ax2.set_title('Training Time vs Number of Estimators', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'case_study_hyperparameter_sensitivity.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close()


def plot_timing_vs_accuracy(timing_results, output_dir):
    """Plot timing vs accuracy trade-off."""
    fig, ax = plt.subplots(figsize=(10, 6))

    classifiers = timing_results['classifiers']
    times = timing_results['times']
    accuracies = timing_results['accuracies']

    # Scatter plot
    ax.scatter(times, accuracies, s=100, alpha=0.6)

    # Annotate points
    for i, clf_name in enumerate(classifiers):
        # Shorten long names
        short_name = clf_name.replace('_OneHot', '').replace('TreeEnsemble', 'TE')
        ax.annotate(short_name, (times[i], accuracies[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Training Time (seconds)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Training Time Trade-off', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'case_study_timing_vs_accuracy.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close()


def plot_classifier_comparison(comparison_df, output_dir):
    """Plot classifier comparison as bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    classifiers = comparison_df['classifier']
    accuracies = comparison_df['accuracy_mean']
    stds = comparison_df['accuracy_std']

    x = np.arange(len(classifiers))

    ax.bar(x, accuracies, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, rotation=45, ha='right')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Classifier Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'case_study_classifier_comparison.pdf'),
                bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == '__main__':
    # Test case study
    from experiments.data_loader import load_categorical_datasets

    print("Loading datasets...")
    datasets = load_categorical_datasets()

    # Select the largest dataset for case study
    largest_dataset = None
    largest_size = 0

    for name, (X, y) in datasets.items():
        size = X.shape[0]
        if size > largest_size:
            largest_size = size
            largest_dataset = name

    if largest_dataset is not None:
        print(f"\nPerforming case study on: {largest_dataset}")

        X, y = datasets[largest_dataset]
        results = case_study_analysis(largest_dataset, X, y, cv=5, random_state=42)

        # Generate plots
        print("\nGenerating plots...")
        plot_case_study_results(results, output_dir='figures')

        # Save results
        print("\nSaving results...")
        import json

        output_file = 'results/case_study_results.json'
        os.makedirs('results', exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        results_serializable = convert_to_serializable(results)

        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"Results saved to: {output_file}")
        print("Plots saved to: figures/")
    else:
        print("No datasets available for case study")
