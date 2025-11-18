"""
RQ2: How does TreeEnsembleClassifier compare with scikit-learn ensemble baselines
using one-hot encoding and other classifiers on categorical datasets?

This module compares TreeEnsembleClassifier against standard sklearn baselines
including RandomForest, HistGradientBoosting, and others.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import warnings

from solution.tree_ensemble import TreeEnsembleClassifier
from experiments.evaluation import compare_classifiers
from experiments.data_loader import get_dataset_info

warnings.filterwarnings('ignore')


def create_classifiers(n_estimators=200, random_state=42):
    """
    Create dictionary of classifiers for comparison.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of estimators for ensemble methods.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    classifiers : dict
        Dictionary mapping classifier_name -> classifier_object.
    """
    classifiers = {}

    # 1. TreeEnsembleClassifier (Native categorical handling)
    classifiers['TreeEnsemble_Native'] = TreeEnsembleClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    # 2. TreeEnsembleClassifier with One-Hot Encoding
    classifiers['TreeEnsemble_OneHot'] = Pipeline([
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ('classifier', TreeEnsembleClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        ))
    ])

    # 3. RandomForestClassifier with One-Hot Encoding
    classifiers['RandomForest_OneHot'] = Pipeline([
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_features='sqrt'
        ))
    ])

    # 4. HistGradientBoostingClassifier (native categorical support)
    # Note: HistGradientBoosting requires categorical features to be marked
    # We'll wrap it to handle this automatically
    classifiers['HistGradientBoosting'] = HistGradientBoostingClassifierWrapper(
        max_iter=n_estimators,
        random_state=random_state
    )

    # 5. GradientBoostingClassifier with One-Hot Encoding
    classifiers['GradientBoosting_OneHot'] = Pipeline([
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ('classifier', GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_features='sqrt'
        ))
    ])

    # 6. AdaBoost with Decision Tree
    classifiers['AdaBoost_OneHot'] = Pipeline([
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ('classifier', AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state),
            n_estimators=n_estimators,
            random_state=random_state
        ))
    ])

    return classifiers


class HistGradientBoostingClassifierWrapper:
    """
    Wrapper for HistGradientBoostingClassifier to handle categorical features.

    HistGradientBoostingClassifier can handle categorical features natively,
    but requires them to be specified. This wrapper automatically detects
    and marks categorical features.
    """

    def __init__(self, max_iter=100, random_state=None):
        self.max_iter = max_iter
        self.random_state = random_state
        self.clf = None
        self.categorical_mask = None

    def _identify_categorical_features(self, X):
        """Identify which features are categorical."""
        n_features = X.shape[1]
        categorical_mask = np.zeros(n_features, dtype=bool)

        for i in range(n_features):
            feature = X[:, i]
            # Check if feature is non-numeric
            if not np.issubdtype(feature.dtype, np.number):
                categorical_mask[i] = True
            # Or if it has few unique values (likely categorical)
            elif len(np.unique(feature)) <= 10:
                categorical_mask[i] = True

        return categorical_mask

    def fit(self, X, y):
        """Fit the classifier."""
        X = np.asarray(X)

        # Identify categorical features
        self.categorical_mask = self._identify_categorical_features(X)

        # Convert categorical features to integers
        X_processed = X.copy()
        for i in range(X.shape[1]):
            if self.categorical_mask[i]:
                unique_vals = np.unique(X[:, i])
                val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
                X_processed[:, i] = np.array([val_to_idx[val] for val in X[:, i]])

        X_processed = X_processed.astype(float)

        # Create classifier
        categorical_features = np.where(self.categorical_mask)[0].tolist()

        self.clf = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            random_state=self.random_state,
            categorical_features=categorical_features if len(categorical_features) > 0 else None
        )

        self.clf.fit(X_processed, y)
        return self

    def predict(self, X):
        """Predict class labels."""
        X = np.asarray(X)

        # Convert categorical features to integers using training mapping
        X_processed = X.copy()
        for i in range(X.shape[1]):
            if self.categorical_mask[i]:
                unique_vals = np.unique(X[:, i])
                val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
                X_processed[:, i] = np.array([val_to_idx.get(val, -1) for val in X[:, i]])

        X_processed = X_processed.astype(float)

        return self.clf.predict(X_processed)

    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.asarray(X)

        # Convert categorical features to integers
        X_processed = X.copy()
        for i in range(X.shape[1]):
            if self.categorical_mask[i]:
                unique_vals = np.unique(X[:, i])
                val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
                X_processed[:, i] = np.array([val_to_idx.get(val, -1) for val in X[:, i]])

        X_processed = X_processed.astype(float)

        return self.clf.predict_proba(X_processed)


def run_rq2_experiments(datasets, n_estimators=200, cv=5, random_state=42):
    """
    Run RQ2 experiments: compare TreeEnsemble with sklearn baselines.

    Parameters
    ----------
    datasets : dict
        Dictionary mapping dataset_name -> (X, y).

    n_estimators : int, default=200
        Number of estimators for ensemble methods.

    cv : int, default=5
        Number of cross-validation folds.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    results_df : pandas.DataFrame
        Results for each (dataset, classifier) pair with columns:
        - dataset: dataset name
        - classifier: classifier name
        - accuracy_mean, accuracy_std: accuracy metrics
        - balanced_accuracy_mean: balanced accuracy
        - auroc_mean: AUROC
        - fit_time_mean: training time
    """
    print("=" * 80)
    print("RQ2: Comparison with Sklearn Baselines")
    print("=" * 80)
    print(f"Configuration: n_estimators={n_estimators}, cv={cv}")
    print("=" * 80 + "\n")

    # Create classifiers
    classifiers = create_classifiers(n_estimators=n_estimators, random_state=random_state)

    all_results = []

    for dataset_name, (X, y) in datasets.items():
        print(f"\n[{dataset_name.upper()}]")
        print("-" * 80)

        # Get dataset info
        info = get_dataset_info(X, y)
        print(f"  Samples: {info['n_samples']}, Features: {info['n_features']}, Classes: {info['n_classes']}")

        # Evaluate all classifiers
        results_df = compare_classifiers(classifiers, X, y, cv=cv, random_state=random_state)

        # Add dataset name
        results_df['dataset'] = dataset_name

        # Add dataset characteristics
        results_df['n_samples'] = info['n_samples']
        results_df['n_features'] = info['n_features']
        results_df['n_classes'] = info['n_classes']

        all_results.append(results_df)

        # Print summary for this dataset
        best_idx = results_df['accuracy_mean'].idxmax()
        best_classifier = results_df.loc[best_idx, 'classifier']
        best_accuracy = results_df.loc[best_idx, 'accuracy_mean']

        print(f"\n  Best classifier: {best_classifier} (Acc: {best_accuracy:.4f})")

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Reorder columns
    cols = ['dataset', 'classifier', 'n_samples', 'n_features', 'n_classes',
            'accuracy_mean', 'accuracy_std', 'balanced_accuracy_mean',
            'auroc_mean', 'fit_time_mean', 'predict_time_mean']
    combined_results = combined_results[cols]

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Average accuracy per classifier
    avg_by_classifier = combined_results.groupby('classifier')['accuracy_mean'].agg(['mean', 'std', 'count'])
    avg_by_classifier = avg_by_classifier.sort_values('mean', ascending=False)

    print("\nAverage Accuracy by Classifier:")
    print("-" * 80)
    for classifier, row in avg_by_classifier.iterrows():
        print(f"  {classifier:30s}: {row['mean']:.4f} ± {row['std']:.4f} (n={int(row['count'])})")

    # Win count per classifier
    print("\n\nWin Count by Classifier:")
    print("-" * 80)

    win_counts = {}
    for dataset_name in combined_results['dataset'].unique():
        dataset_results = combined_results[combined_results['dataset'] == dataset_name]
        best_idx = dataset_results['accuracy_mean'].idxmax()
        winner = dataset_results.loc[best_idx, 'classifier']

        if winner not in win_counts:
            win_counts[winner] = 0
        win_counts[winner] += 1

    for classifier, wins in sorted(win_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {classifier:30s}: {wins}/{len(datasets)} datasets")

    # Average time
    print("\n\nAverage Training Time by Classifier:")
    print("-" * 80)
    avg_time = combined_results.groupby('classifier')['fit_time_mean'].mean().sort_values()
    for classifier, time_val in avg_time.items():
        print(f"  {classifier:30s}: {time_val:.3f}s")

    print("\n" + "=" * 80)

    return combined_results


def analyze_rq2_results(results_df):
    """
    Perform detailed analysis of RQ2 results.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results from run_rq2_experiments.

    Returns
    -------
    analysis : dict
        Dictionary with detailed analysis.
    """
    analysis = {}

    # Average accuracy per classifier
    avg_acc = results_df.groupby('classifier')['accuracy_mean'].agg(['mean', 'std'])
    analysis['avg_accuracy'] = avg_acc.to_dict()

    # Rankings
    classifier_rankings = []
    for dataset in results_df['dataset'].unique():
        dataset_results = results_df[results_df['dataset'] == dataset].copy()
        dataset_results['rank'] = dataset_results['accuracy_mean'].rank(ascending=False)

        for _, row in dataset_results.iterrows():
            classifier_rankings.append({
                'dataset': dataset,
                'classifier': row['classifier'],
                'rank': row['rank']
            })

    rankings_df = pd.DataFrame(classifier_rankings)
    avg_rank = rankings_df.groupby('classifier')['rank'].mean().sort_values()
    analysis['average_rank'] = avg_rank.to_dict()

    # Win counts
    win_counts = {}
    for dataset in results_df['dataset'].unique():
        dataset_results = results_df[results_df['dataset'] == dataset]
        winner = dataset_results.loc[dataset_results['accuracy_mean'].idxmax(), 'classifier']

        if winner not in win_counts:
            win_counts[winner] = 0
        win_counts[winner] += 1

    analysis['win_counts'] = win_counts
    analysis['total_datasets'] = len(results_df['dataset'].unique())

    # Time analysis
    avg_time = results_df.groupby('classifier')['fit_time_mean'].mean()
    analysis['avg_time'] = avg_time.to_dict()

    return analysis


def compare_native_vs_baselines(results_df):
    """
    Specifically compare TreeEnsemble_Native against all baselines.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results from run_rq2_experiments.

    Returns
    -------
    comparison : pandas.DataFrame
        Pairwise comparison of TreeEnsemble_Native vs each baseline.
    """
    comparisons = []

    native_results = results_df[results_df['classifier'] == 'TreeEnsemble_Native']

    for classifier in results_df['classifier'].unique():
        if classifier == 'TreeEnsemble_Native':
            continue

        baseline_results = results_df[results_df['classifier'] == classifier]

        # Merge on dataset
        merged = pd.merge(
            native_results[['dataset', 'accuracy_mean']],
            baseline_results[['dataset', 'accuracy_mean']],
            on='dataset',
            suffixes=('_native', '_baseline')
        )

        # Compare
        merged['difference'] = merged['accuracy_mean_native'] - merged['accuracy_mean_baseline']

        wins = (merged['difference'] > 0).sum()
        losses = (merged['difference'] < 0).sum()
        ties = (merged['difference'] == 0).sum()

        avg_diff = merged['difference'].mean()

        comparisons.append({
            'baseline': classifier,
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'avg_difference': avg_diff,
            'win_rate': wins / len(merged)
        })

    comparison_df = pd.DataFrame(comparisons)
    comparison_df = comparison_df.sort_values('avg_difference', ascending=False)

    return comparison_df


if __name__ == '__main__':
    # Test RQ2 experiments
    from experiments.data_loader import load_categorical_datasets

    print("Loading datasets...")
    datasets = load_categorical_datasets()

    print(f"\nRunning RQ2 experiments on {len(datasets)} datasets...")
    results_df = run_rq2_experiments(datasets, n_estimators=100, cv=5, random_state=42)

    # Analyze results
    print("\n\nPerforming detailed analysis...")
    analysis = analyze_rq2_results(results_df)

    print("\nAverage Rankings:")
    for classifier, rank in sorted(analysis['average_rank'].items(), key=lambda x: x[1]):
        print(f"  {classifier:30s}: {rank:.2f}")

    # Compare native vs baselines
    print("\n\nTreeEnsemble_Native vs Baselines:")
    comparison_df = compare_native_vs_baselines(results_df)
    print(comparison_df.to_string(index=False))

    # Save results
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'rq2_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")

    comparison_file = os.path.join(output_dir, 'rq2_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Comparison saved to: {comparison_file}")
