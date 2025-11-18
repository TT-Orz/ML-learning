"""
Data loading and preprocessing utilities for categorical datasets.

This module provides functions to load categorical datasets, apply one-hot
encoding, and compute dataset statistics.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.datasets import fetch_openml
from collections import Counter
import os
import warnings

warnings.filterwarnings('ignore')


def load_categorical_datasets():
    """
    Load all categorical datasets from various sources.

    This function loads multiple categorical datasets suitable for
    classification tasks. Datasets are loaded from:
    - Local files in the data/ directory (if available)
    - UCI Machine Learning Repository via sklearn
    - Synthetic datasets for testing

    Returns
    -------
    datasets : dict
        Dictionary mapping dataset_name -> (X, y), where:
        - X is a numpy array of shape (n_samples, n_features) with categorical features
        - y is a numpy array of shape (n_samples,) with class labels
    """
    datasets = {}

    print("Loading categorical datasets...")

    # 1. Try loading Mushroom dataset
    try:
        print("  Loading Mushroom dataset...")
        X, y = load_mushroom_dataset()
        if X is not None:
            datasets['mushroom'] = (X, y)
            print(f"    ✓ Mushroom: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"    ✗ Failed to load Mushroom: {e}")

    # 2. Try loading Tic-Tac-Toe dataset
    try:
        print("  Loading Tic-Tac-Toe dataset...")
        X, y = load_tic_tac_toe_dataset()
        if X is not None:
            datasets['tic_tac_toe'] = (X, y)
            print(f"    ✓ Tic-Tac-Toe: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"    ✗ Failed to load Tic-Tac-Toe: {e}")

    # 3. Try loading Car Evaluation dataset
    try:
        print("  Loading Car Evaluation dataset...")
        X, y = load_car_evaluation_dataset()
        if X is not None:
            datasets['car_evaluation'] = (X, y)
            print(f"    ✓ Car Evaluation: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"    ✗ Failed to load Car Evaluation: {e}")

    # 4. Try loading Chess (King-Rook vs King-Pawn) dataset
    try:
        print("  Loading Chess (kr-vs-kp) dataset...")
        X, y = load_chess_dataset()
        if X is not None:
            datasets['chess_kr_vs_kp'] = (X, y)
            print(f"    ✓ Chess: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"    ✗ Failed to load Chess: {e}")

    # 5. Try loading Balloons dataset
    try:
        print("  Loading Balloons dataset...")
        X, y = load_balloons_dataset()
        if X is not None:
            datasets['balloons'] = (X, y)
            print(f"    ✓ Balloons: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"    ✗ Failed to load Balloons: {e}")

    # 6. Create synthetic categorical datasets if no real datasets loaded
    if len(datasets) == 0:
        print("  Creating synthetic categorical datasets...")
        datasets['synthetic_small'] = create_synthetic_categorical_dataset(
            n_samples=200, n_features=5, n_classes=2, n_categories_per_feature=4,
            random_state=42
        )
        datasets['synthetic_medium'] = create_synthetic_categorical_dataset(
            n_samples=500, n_features=10, n_classes=3, n_categories_per_feature=5,
            random_state=43
        )
        datasets['synthetic_large'] = create_synthetic_categorical_dataset(
            n_samples=1000, n_features=15, n_classes=4, n_categories_per_feature=6,
            random_state=44
        )
        print(f"    ✓ Created 3 synthetic datasets")

    print(f"\nTotal datasets loaded: {len(datasets)}\n")
    return datasets


def load_mushroom_dataset():
    """
    Load the Mushroom dataset from OpenML.

    Returns
    -------
    X : ndarray
        Features (all categorical).
    y : ndarray
        Target (edible vs poisonous).
    """
    try:
        data = fetch_openml('mushroom', version=1, as_frame=True, parser='auto')
        X = data.data.values  # All categorical features
        y = data.target.values
        return X, y
    except Exception as e:
        print(f"      Error loading from OpenML: {e}")
        return None, None


def load_tic_tac_toe_dataset():
    """
    Load the Tic-Tac-Toe Endgame dataset from OpenML.

    Returns
    -------
    X : ndarray
        Features (board positions).
    y : ndarray
        Target (positive/negative outcome).
    """
    try:
        data = fetch_openml('tic-tac-toe', version=1, as_frame=True, parser='auto')
        X = data.data.values
        y = data.target.values
        return X, y
    except Exception as e:
        print(f"      Error loading from OpenML: {e}")
        return None, None


def load_car_evaluation_dataset():
    """
    Load the Car Evaluation dataset from OpenML.

    Returns
    -------
    X : ndarray
        Features (buying price, maintenance, doors, etc.).
    y : ndarray
        Target (car acceptability).
    """
    try:
        data = fetch_openml('car', version=1, as_frame=True, parser='auto')
        X = data.data.values
        y = data.target.values
        return X, y
    except Exception as e:
        print(f"      Error loading from OpenML: {e}")
        return None, None


def load_chess_dataset():
    """
    Load the Chess (King-Rook vs King-Pawn) dataset from OpenML.

    Returns
    -------
    X : ndarray
        Features (chess board positions).
    y : ndarray
        Target (win/loss).
    """
    try:
        data = fetch_openml('kr-vs-kp', version=1, as_frame=True, parser='auto')
        X = data.data.values
        y = data.target.values
        return X, y
    except Exception as e:
        print(f"      Error loading from OpenML: {e}")
        return None, None


def load_balloons_dataset():
    """
    Load the Balloons dataset.

    Returns
    -------
    X : ndarray
        Features (color, size, act, age).
    y : ndarray
        Target (inflated T/F).
    """
    # Create a simple balloons dataset manually
    data = [
        ['YELLOW', 'SMALL', 'STRETCH', 'ADULT', 'T'],
        ['YELLOW', 'SMALL', 'STRETCH', 'CHILD', 'F'],
        ['YELLOW', 'SMALL', 'DIP', 'ADULT', 'F'],
        ['YELLOW', 'SMALL', 'DIP', 'CHILD', 'F'],
        ['YELLOW', 'LARGE', 'STRETCH', 'ADULT', 'T'],
        ['YELLOW', 'LARGE', 'STRETCH', 'CHILD', 'F'],
        ['YELLOW', 'LARGE', 'DIP', 'ADULT', 'F'],
        ['YELLOW', 'LARGE', 'DIP', 'CHILD', 'F'],
        ['PURPLE', 'SMALL', 'STRETCH', 'ADULT', 'T'],
        ['PURPLE', 'SMALL', 'STRETCH', 'CHILD', 'F'],
        ['PURPLE', 'SMALL', 'DIP', 'ADULT', 'F'],
        ['PURPLE', 'SMALL', 'DIP', 'CHILD', 'F'],
        ['PURPLE', 'LARGE', 'STRETCH', 'ADULT', 'T'],
        ['PURPLE', 'LARGE', 'STRETCH', 'CHILD', 'F'],
        ['PURPLE', 'LARGE', 'DIP', 'ADULT', 'F'],
        ['PURPLE', 'LARGE', 'DIP', 'CHILD', 'F'],
    ]

    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]

    return X, y


def create_synthetic_categorical_dataset(n_samples=500, n_features=10,
                                         n_classes=3, n_categories_per_feature=5,
                                         random_state=None):
    """
    Create a synthetic categorical dataset.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of categorical features.
    n_classes : int
        Number of classes.
    n_categories_per_feature : int
        Number of categories per feature.
    random_state : int or None
        Random seed.

    Returns
    -------
    X : ndarray
        Categorical features.
    y : ndarray
        Class labels.
    """
    rng = np.random.RandomState(random_state)

    # Create categorical features
    X = np.empty((n_samples, n_features), dtype=object)

    for feature_idx in range(n_features):
        # Generate categories as strings
        categories = [f'cat{i}' for i in range(n_categories_per_feature)]
        X[:, feature_idx] = rng.choice(categories, size=n_samples)

    # Create target with some structure
    # Make some features more predictive than others
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # Use first few features to determine class
        predictive_features = X[i, :min(3, n_features)]
        # Hash the features to get a consistent class
        feature_hash = hash(tuple(predictive_features)) % n_classes
        y[i] = feature_hash

    # Add some noise
    noise_indices = rng.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_indices] = rng.choice(n_classes, size=len(noise_indices))

    return X, y


def apply_one_hot_encoding(X_train, X_test=None):
    """
    Apply one-hot encoding to categorical features.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training data with categorical features.

    X_test : array-like of shape (n_samples, n_features) or None
        Test data with categorical features. If None, only fit on train.

    Returns
    -------
    X_train_encoded : ndarray
        One-hot encoded training data.

    X_test_encoded : ndarray or None
        One-hot encoded test data (if X_test was provided).

    encoder : OneHotEncoder
        Fitted encoder object.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Fit on training data
    X_train_encoded = encoder.fit_transform(X_train)

    # Transform test data if provided
    if X_test is not None:
        X_test_encoded = encoder.transform(X_test)
        return X_train_encoded, X_test_encoded, encoder
    else:
        return X_train_encoded, None, encoder


def encode_target(y_train, y_test=None):
    """
    Encode target labels as integers.

    Parameters
    ----------
    y_train : array-like
        Training labels.

    y_test : array-like or None
        Test labels.

    Returns
    -------
    y_train_encoded : ndarray
        Encoded training labels.

    y_test_encoded : ndarray or None
        Encoded test labels (if y_test was provided).

    encoder : LabelEncoder
        Fitted label encoder.
    """
    encoder = LabelEncoder()

    y_train_encoded = encoder.fit_transform(y_train)

    if y_test is not None:
        y_test_encoded = encoder.transform(y_test)
        return y_train_encoded, y_test_encoded, encoder
    else:
        return y_train_encoded, None, encoder


def get_dataset_info(X, y):
    """
    Get summary statistics for a dataset.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Features.

    y : array-like of shape (n_samples,)
        Labels.

    Returns
    -------
    info : dict
        Dictionary with dataset statistics:
        - n_samples: number of samples
        - n_features: number of features
        - n_classes: number of unique classes
        - class_distribution: counts of each class
        - class_balance_ratio: ratio of min to max class size
        - is_imbalanced: whether dataset is imbalanced (ratio < 0.5)
        - feature_categories: number of categories per feature
        - total_categories: total number of unique categories across all features
    """
    X = np.asarray(X)
    y = np.asarray(y)

    n_samples, n_features = X.shape
    classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    class_distribution = dict(zip(classes, class_counts))

    # Calculate class balance
    min_class_size = class_counts.min()
    max_class_size = class_counts.max()
    balance_ratio = min_class_size / max_class_size if max_class_size > 0 else 0
    is_imbalanced = balance_ratio < 0.5

    # Analyze feature categories
    feature_categories = []
    total_categories = 0

    for feature_idx in range(n_features):
        unique_vals = np.unique(X[:, feature_idx])
        n_categories = len(unique_vals)
        feature_categories.append(n_categories)
        total_categories += n_categories

    info = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_classes': n_classes,
        'class_distribution': class_distribution,
        'class_balance_ratio': balance_ratio,
        'is_imbalanced': is_imbalanced,
        'feature_categories': feature_categories,
        'total_categories': total_categories,
        'avg_categories_per_feature': np.mean(feature_categories),
        'max_categories_in_feature': max(feature_categories),
        'min_categories_in_feature': min(feature_categories)
    }

    return info


def print_dataset_summary(datasets):
    """
    Print a summary of all loaded datasets.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets from load_categorical_datasets().
    """
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    for dataset_name, (X, y) in datasets.items():
        info = get_dataset_info(X, y)

        print(f"\n{dataset_name.upper()}")
        print("-" * 80)
        print(f"  Samples:          {info['n_samples']}")
        print(f"  Features:         {info['n_features']}")
        print(f"  Classes:          {info['n_classes']}")
        print(f"  Class distribution: {info['class_distribution']}")
        print(f"  Balance ratio:    {info['class_balance_ratio']:.3f}")
        print(f"  Imbalanced:       {info['is_imbalanced']}")
        print(f"  Avg categories/feature: {info['avg_categories_per_feature']:.1f}")
        print(f"  Total categories: {info['total_categories']}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    # Test the data loader
    datasets = load_categorical_datasets()
    print_dataset_summary(datasets)

    # Test one-hot encoding on first dataset
    if len(datasets) > 0:
        dataset_name = list(datasets.keys())[0]
        X, y = datasets[dataset_name]

        print(f"\nTesting one-hot encoding on {dataset_name}...")
        X_encoded, _, encoder = apply_one_hot_encoding(X)
        print(f"  Original shape: {X.shape}")
        print(f"  Encoded shape:  {X_encoded.shape}")
        print(f"  Expansion factor: {X_encoded.shape[1] / X.shape[1]:.2f}x")
