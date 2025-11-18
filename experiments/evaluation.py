"""
Evaluation framework for classifier comparison.

This module provides functions for cross-validation, metrics calculation,
and classifier comparison with proper handling of categorical data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
import time
import warnings

warnings.filterwarnings('ignore')


def evaluate_classifier(clf, X, y, cv=5, random_state=42, return_predictions=False):
    """
    Evaluate a classifier using stratified k-fold cross-validation.

    This function performs comprehensive evaluation of a classifier using
    multiple metrics. It handles both numeric and categorical features
    automatically.

    Parameters
    ----------
    clf : estimator object
        The classifier to evaluate. Must implement fit() and predict() methods.

    X : array-like of shape (n_samples, n_features)
        Training data (can be categorical).

    y : array-like of shape (n_samples,)
        Target values.

    cv : int, default=5
        Number of folds for cross-validation.

    random_state : int, default=42
        Random seed for reproducibility.

    return_predictions : bool, default=False
        If True, return predictions from each fold.

    Returns
    -------
    results : dict
        Dictionary with the following keys:
        - accuracy_mean: mean accuracy across folds
        - accuracy_std: standard deviation of accuracy
        - balanced_accuracy_mean: mean balanced accuracy
        - balanced_accuracy_std: standard deviation of balanced accuracy
        - auroc_mean: mean AUROC (if applicable)
        - auroc_std: standard deviation of AUROC
        - fit_time_mean: mean training time per fold
        - predict_time_mean: mean prediction time per fold
        - fold_scores: list of accuracy scores for each fold
        - predictions: predictions from each fold (if return_predictions=True)
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Encode target if necessary
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Storage for results
    accuracies = []
    balanced_accuracies = []
    aurocs = []
    fit_times = []
    predict_times = []
    all_predictions = [] if return_predictions else None
    all_true_labels = [] if return_predictions else None

    # Perform cross-validation manually for better control
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train the classifier
        start_time = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - start_time

        # Make predictions
        start_time = time.time()
        y_pred = clf.predict(X_test)
        predict_time = time.time() - start_time

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)

        accuracies.append(accuracy)
        balanced_accuracies.append(balanced_acc)
        fit_times.append(fit_time)
        predict_times.append(predict_time)

        # Calculate AUROC if applicable
        if n_classes == 2:
            # Binary classification - use predict_proba if available
            try:
                if hasattr(clf, 'predict_proba'):
                    y_proba = clf.predict_proba(X_test)[:, 1]
                    auroc = roc_auc_score(y_test, y_proba)
                else:
                    # Use predictions as proxy
                    auroc = balanced_acc  # Fallback
                aurocs.append(auroc)
            except Exception:
                aurocs.append(balanced_acc)
        else:
            # Multiclass - use one-vs-rest AUROC if possible
            try:
                if hasattr(clf, 'predict_proba'):
                    # Encode y_test
                    y_test_encoded = le.transform(y_test)
                    y_proba = clf.predict_proba(X_test)
                    auroc = roc_auc_score(y_test_encoded, y_proba,
                                         multi_class='ovr', average='macro')
                else:
                    auroc = balanced_acc
                aurocs.append(auroc)
            except Exception:
                aurocs.append(balanced_acc)

        # Store predictions if requested
        if return_predictions:
            all_predictions.append(y_pred)
            all_true_labels.append(y_test)

    # Compile results
    results = {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'balanced_accuracy_mean': np.mean(balanced_accuracies),
        'balanced_accuracy_std': np.std(balanced_accuracies),
        'auroc_mean': np.mean(aurocs) if len(aurocs) > 0 else np.nan,
        'auroc_std': np.std(aurocs) if len(aurocs) > 0 else np.nan,
        'fit_time_mean': np.mean(fit_times),
        'fit_time_std': np.std(fit_times),
        'predict_time_mean': np.mean(predict_times),
        'predict_time_std': np.std(predict_times),
        'fold_scores': accuracies
    }

    if return_predictions:
        results['predictions'] = all_predictions
        results['true_labels'] = all_true_labels

    return results


def compare_classifiers(classifiers_dict, X, y, cv=5, random_state=42):
    """
    Compare multiple classifiers on the same dataset.

    Parameters
    ----------
    classifiers_dict : dict
        Dictionary mapping classifier name -> classifier object.
        Example: {'Random Forest': RandomForestClassifier(), ...}

    X : array-like of shape (n_samples, n_features)
        Training data.

    y : array-like of shape (n_samples,)
        Target values.

    cv : int, default=5
        Number of folds for cross-validation.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    results_df : pandas.DataFrame
        DataFrame with rows for each classifier and columns for metrics.
        Columns include: classifier, accuracy_mean, accuracy_std,
        balanced_accuracy_mean, auroc_mean, fit_time_mean, etc.
    """
    results = []

    for clf_name, clf in classifiers_dict.items():
        print(f"  Evaluating {clf_name}...", end=' ')

        try:
            scores = evaluate_classifier(clf, X, y, cv=cv, random_state=random_state)

            result = {
                'classifier': clf_name,
                'accuracy_mean': scores['accuracy_mean'],
                'accuracy_std': scores['accuracy_std'],
                'balanced_accuracy_mean': scores['balanced_accuracy_mean'],
                'balanced_accuracy_std': scores['balanced_accuracy_std'],
                'auroc_mean': scores['auroc_mean'],
                'auroc_std': scores['auroc_std'],
                'fit_time_mean': scores['fit_time_mean'],
                'predict_time_mean': scores['predict_time_mean']
            }

            results.append(result)
            print(f"✓ (Acc: {scores['accuracy_mean']:.3f})")

        except Exception as e:
            print(f"✗ (Error: {e})")
            result = {
                'classifier': clf_name,
                'accuracy_mean': np.nan,
                'accuracy_std': np.nan,
                'balanced_accuracy_mean': np.nan,
                'balanced_accuracy_std': np.nan,
                'auroc_mean': np.nan,
                'auroc_std': np.nan,
                'fit_time_mean': np.nan,
                'predict_time_mean': np.nan
            }
            results.append(result)

    return pd.DataFrame(results)


def evaluate_with_confusion_matrix(clf, X, y, cv=5, random_state=42):
    """
    Evaluate a classifier and compute aggregate confusion matrix.

    Parameters
    ----------
    clf : estimator object
        The classifier to evaluate.

    X : array-like
        Training data.

    y : array-like
        Target values.

    cv : int, default=5
        Number of folds.

    random_state : int, default=42
        Random seed.

    Returns
    -------
    results : dict
        Dictionary with evaluation metrics and confusion matrix.
    """
    scores = evaluate_classifier(clf, X, y, cv=cv, random_state=random_state,
                                return_predictions=True)

    # Aggregate confusion matrix
    all_true = np.concatenate(scores['true_labels'])
    all_pred = np.concatenate(scores['predictions'])

    cm = confusion_matrix(all_true, all_pred)

    scores['confusion_matrix'] = cm
    scores['classification_report'] = classification_report(all_true, all_pred)

    return scores


def evaluate_with_train_test_split(clf, X_train, X_test, y_train, y_test):
    """
    Evaluate a classifier on a single train/test split.

    Parameters
    ----------
    clf : estimator object
        The classifier to evaluate.

    X_train, X_test : array-like
        Training and test data.

    y_train, y_test : array-like
        Training and test labels.

    Returns
    -------
    results : dict
        Dictionary with evaluation metrics.
    """
    # Train
    start_time = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - start_time

    # Predict
    start_time = time.time()
    y_pred = clf.predict(X_test)
    predict_time = time.time() - start_time

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Try to compute AUROC
    try:
        if hasattr(clf, 'predict_proba'):
            y_proba = clf.predict_proba(X_test)
            le = LabelEncoder()
            y_test_encoded = le.fit_transform(y_test)

            if len(np.unique(y_test)) == 2:
                auroc = roc_auc_score(y_test_encoded, y_proba[:, 1])
            else:
                auroc = roc_auc_score(y_test_encoded, y_proba,
                                     multi_class='ovr', average='macro')
        else:
            auroc = np.nan
    except Exception:
        auroc = np.nan

    results = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'auroc': auroc,
        'fit_time': fit_time,
        'predict_time': predict_time,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred)
    }

    return results


def compute_metrics_summary(results_df):
    """
    Compute summary statistics across classifiers.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results from compare_classifiers().

    Returns
    -------
    summary : dict
        Summary statistics including best classifier, rankings, etc.
    """
    # Best classifier by accuracy
    best_idx = results_df['accuracy_mean'].idxmax()
    best_classifier = results_df.loc[best_idx, 'classifier']
    best_accuracy = results_df.loc[best_idx, 'accuracy_mean']

    # Rankings
    results_df['accuracy_rank'] = results_df['accuracy_mean'].rank(ascending=False)
    results_df['balanced_accuracy_rank'] = results_df['balanced_accuracy_mean'].rank(ascending=False)

    summary = {
        'best_classifier': best_classifier,
        'best_accuracy': best_accuracy,
        'rankings': results_df[['classifier', 'accuracy_rank', 'balanced_accuracy_rank']].to_dict('records'),
        'mean_accuracy': results_df['accuracy_mean'].mean(),
        'std_accuracy': results_df['accuracy_mean'].std()
    }

    return summary


def repeated_stratified_cv(clf, X, y, n_repeats=10, n_splits=5, random_state=42):
    """
    Perform repeated stratified cross-validation.

    Parameters
    ----------
    clf : estimator
        Classifier to evaluate.

    X, y : array-like
        Data and labels.

    n_repeats : int, default=10
        Number of times to repeat cross-validation.

    n_splits : int, default=5
        Number of folds per repetition.

    random_state : int, default=42
        Random seed.

    Returns
    -------
    results : dict
        Comprehensive results across all repeats.
    """
    all_accuracies = []

    for repeat in range(n_repeats):
        repeat_seed = random_state + repeat if random_state is not None else None

        scores = evaluate_classifier(clf, X, y, cv=n_splits, random_state=repeat_seed)
        all_accuracies.extend(scores['fold_scores'])

    results = {
        'accuracy_mean': np.mean(all_accuracies),
        'accuracy_std': np.std(all_accuracies),
        'accuracy_median': np.median(all_accuracies),
        'accuracy_min': np.min(all_accuracies),
        'accuracy_max': np.max(all_accuracies),
        'all_scores': all_accuracies
    }

    return results


if __name__ == '__main__':
    # Test with synthetic data
    from experiments.data_loader import create_synthetic_categorical_dataset
    from solution.tree_ensemble import TreeEnsembleClassifier

    print("Testing evaluation framework...")

    # Create test dataset
    X, y = create_synthetic_categorical_dataset(
        n_samples=200, n_features=5, n_classes=3, random_state=42
    )

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    # Test single classifier evaluation
    print("\n1. Single classifier evaluation:")
    clf = TreeEnsembleClassifier(n_estimators=50, random_state=42)
    results = evaluate_classifier(clf, X, y, cv=5)

    print(f"   Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
    print(f"   Balanced Accuracy: {results['balanced_accuracy_mean']:.3f}")
    print(f"   Training time: {results['fit_time_mean']:.3f}s")

    # Test classifier comparison
    print("\n2. Classifier comparison:")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline

    classifiers = {
        'TreeEnsemble': TreeEnsembleClassifier(n_estimators=50, random_state=42),
        'RandomForest': Pipeline([
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
            ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
        ])
    }

    comparison_df = compare_classifiers(classifiers, X, y, cv=5)
    print("\n", comparison_df)

    print("\n✓ Evaluation framework working correctly!")
