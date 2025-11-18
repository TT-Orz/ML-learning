"""
Tree Ensemble Classifier for Categorical Data

This module implements an ensemble of decision stumps for categorical data
classification. It follows sklearn conventions and supports various diversity
strategies to improve generalization.

Author: COMP3222 Coursework Part 2
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from .decision_stump import DecisionStumpClassifier


class TreeEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    An ensemble of decision stumps for categorical data classification.

    This classifier creates multiple decision stumps and combines their
    predictions using either soft voting (averaging probabilities) or
    hard voting (majority vote). Diversity is achieved through random
    feature subsets, bootstrap sampling, and/or varying quality measures.

    The classifier is compatible with sklearn's interface and can be used
    with cross-validation, grid search, and other sklearn utilities.

    Parameters
    ----------
    n_estimators : int, default=200
        The number of decision stumps in the ensemble.

    average_probas : bool, default=True
        If True, use soft voting (average predicted probabilities).
        If False, use hard voting (majority vote on predicted classes).

    random_state : int or None, default=None
        Seed for random number generator. Ensures reproducible results
        when set to a specific value.

    Attributes
    ----------
    estimators_ : list of DecisionStumpClassifier
        The collection of fitted decision stumps.

    classes_ : numpy.ndarray
        Unique class labels in the training data.

    label_encoder_ : LabelEncoder
        Encoder used to transform class labels to 0, 1, ..., n_classes - 1.

    n_features_per_stump_ : int
        Number of features randomly selected for each stump.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import cross_val_score
    >>> X = np.array([['yes', 'no'], ['yes', 'yes'], ['no', 'yes'],
    ...               ['no', 'no'], ['yes', 'no'], ['no', 'yes']], dtype=object)
    >>> y = np.array(['A', 'A', 'B', 'B', 'A', 'B'])
    >>> ensemble = TreeEnsembleClassifier(n_estimators=50, random_state=42)
    >>> ensemble.fit(X, y)
    >>> predictions = ensemble.predict(X)
    >>> print(predictions)
    ['A' 'A' 'B' 'B' 'A' 'B']
    """

    def __init__(self, n_estimators=200, average_probas=True, random_state=None):
        """Initialize the tree ensemble classifier."""
        self.n_estimators = n_estimators
        self.average_probas = average_probas
        self.random_state = random_state

    def _validate_input(self, X):
        """
        Validate that input contains only categorical data (no floats).

        Parameters
        ----------
        X : numpy.ndarray
            Input data to validate.

        Raises
        ------
        ValueError
            If X contains floating-point values or has invalid shape.
        """
        if X.size == 0:
            raise ValueError("Input X cannot be empty")

        if len(X.shape) != 2:
            raise ValueError("Input X must be 2-dimensional")

        # Check if any element is a float type
        if np.issubdtype(X.dtype, np.floating):
            raise ValueError("Input X contains floating-point values. "
                           "Only categorical (string/integer) data is supported.")

        # Also check for actual float values in object arrays
        if X.dtype == object:
            for val in X.flat:
                if isinstance(val, float) and not np.isnan(val):
                    raise ValueError("Input X contains floating-point values. "
                                   "Only categorical (string/integer) data is supported.")

    def fit(self, X, y):
        """
        Fit the ensemble of decision stumps to the training data.

        Creates n_estimators decision stumps, each trained on a random
        subset of features. Uses bootstrap sampling and/or random quality
        measures to ensure diversity in the ensemble.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training data. Must contain only categorical values.

        y : array-like of shape (n_samples,)
            Target labels. Can be any type (strings, integers, etc.).

        Returns
        -------
        self : TreeEnsembleClassifier
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If X contains floating-point values or has incompatible shape with y.
        """
        # Validate input
        self._validate_input(X)

        if X.shape[0] != len(y):
            raise ValueError(f"X and y must have same number of samples. "
                           f"Got X: {X.shape[0]}, y: {len(y)}")

        n_samples, n_features = X.shape

        # Encode labels to 0, 1, ..., n_classes - 1
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        # Determine number of features per stump (random feature subsets)
        # Use sqrt(n_features) as recommended for random forests
        self.n_features_per_stump_ = max(1, int(np.sqrt(n_features)))

        # Initialize list to store fitted stumps
        self.estimators_ = []

        # Quality measures to randomly choose from (for additional diversity)
        quality_measures = ["ig", "gain_ratio", "chi2", "chi2_yates"]

        # Create and fit each stump
        for i in range(self.n_estimators):
            # Set random seed for this stump (ensures reproducibility)
            if self.random_state is not None:
                stump_seed = self.random_state + i
            else:
                stump_seed = None

            # Randomly select quality measure (adds diversity)
            if self.random_state is not None:
                measure_rng = np.random.RandomState(self.random_state + i * 1000)
            else:
                measure_rng = np.random.RandomState()
            quality_measure = measure_rng.choice(quality_measures)

            # Create decision stump with random feature subset
            stump = DecisionStumpClassifier(
                n_attributes=self.n_features_per_stump_,
                quality_measure=quality_measure,
                random_state=stump_seed,
                alpha=1.0
            )

            # Bootstrap sampling (sample with replacement)
            if self.random_state is not None:
                bootstrap_rng = np.random.RandomState(self.random_state + i * 2000)
            else:
                bootstrap_rng = np.random.RandomState()

            bootstrap_indices = bootstrap_rng.choice(
                n_samples,
                size=n_samples,
                replace=True
            )

            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y_encoded[bootstrap_indices]

            # Fit the stump
            stump.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(stump)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for each sample.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : numpy.ndarray of shape (n_samples, n_classes)
            Probability distributions over classes for each sample.

        Notes
        -----
        If average_probas=True, returns the average of probability predictions
        from all stumps (soft voting).

        If average_probas=False, returns the proportion of votes for each class
        from all stumps (hard voting converted to probabilities).

        Raises
        ------
        ValueError
            If the classifier has not been fitted yet.
        """
        if not hasattr(self, 'estimators_'):
            raise ValueError("Classifier has not been fitted yet. Call fit() first.")

        self._validate_input(X)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        if self.average_probas:
            # Soft voting: average probabilities
            proba_sum = np.zeros((n_samples, n_classes))

            for stump in self.estimators_:
                proba_sum += stump.predict_proba(X)

            proba = proba_sum / self.n_estimators

        else:
            # Hard voting: count votes and convert to proportions
            vote_counts = np.zeros((n_samples, n_classes))

            for stump in self.estimators_:
                predictions = stump.predict(X)
                for i, pred in enumerate(predictions):
                    vote_counts[i, pred] += 1

            proba = vote_counts / self.n_estimators

        return proba

    def predict(self, X):
        """
        Predict class labels for each sample.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        predictions : numpy.ndarray of shape (n_samples,)
            Predicted class labels in original encoding.

        Notes
        -----
        If average_probas=True, returns argmax of averaged probabilities (soft voting).
        If average_probas=False, returns majority class from all stumps (hard voting).
        Ties are broken by choosing the smallest class index.
        """
        if not hasattr(self, 'estimators_'):
            raise ValueError("Classifier has not been fitted yet. Call fit() first.")

        self._validate_input(X)

        if self.average_probas:
            # Soft voting: get class with highest average probability
            proba = self.predict_proba(X)
            y_encoded = np.argmax(proba, axis=1)
        else:
            # Hard voting: get majority class
            n_samples = X.shape[0]
            n_classes = len(self.classes_)
            vote_counts = np.zeros((n_samples, n_classes), dtype=int)

            for stump in self.estimators_:
                predictions = stump.predict(X)
                for i, pred in enumerate(predictions):
                    vote_counts[i, pred] += 1

            # argmax handles tie-breaking by returning smallest index
            y_encoded = np.argmax(vote_counts, axis=1)

        # Decode back to original labels
        predictions = self.label_encoder_.inverse_transform(y_encoded)

        return predictions


if __name__ == '__main__':
    # Test with the whisky dataset
    print("Testing TreeEnsembleClassifier")
    print("=" * 70)

    # Whisky dataset
    X = np.array([
        ['yes', 'no', 'yes'],   # Islay
        ['yes', 'yes', 'yes'],  # Islay
        ['yes', 'no', 'no'],    # Islay
        ['yes', 'no', 'no'],    # Islay
        ['no', 'yes', 'no'],    # Islay
        ['no', 'yes', 'yes'],   # Speyside
        ['no', 'yes', 'yes'],   # Speyside
        ['no', 'yes', 'yes'],   # Speyside
        ['no', 'no', 'yes'],    # Speyside
        ['no', 'no', 'yes']     # Speyside
    ], dtype=object)

    y = np.array(['Islay', 'Islay', 'Islay', 'Islay', 'Islay',
                  'Speyside', 'Speyside', 'Speyside', 'Speyside', 'Speyside'])

    # Test with soft voting
    print("\nTest 1: Soft Voting (average_probas=True)")
    print("-" * 70)

    ensemble_soft = TreeEnsembleClassifier(
        n_estimators=100,
        average_probas=True,
        random_state=42
    )
    ensemble_soft.fit(X, y)
    predictions_soft = ensemble_soft.predict(X)
    accuracy_soft = np.mean(predictions_soft == y)

    print(f"Number of estimators: {len(ensemble_soft.estimators_)}")
    print(f"Features per stump: {ensemble_soft.n_features_per_stump_}")
    print(f"Training accuracy: {accuracy_soft:.3f}")

    # Show probability predictions for first few samples
    proba_soft = ensemble_soft.predict_proba(X[:3])
    print(f"\nProbabilities for first 3 samples:")
    for i, p in enumerate(proba_soft):
        print(f"  Sample {i}: Islay={p[0]:.3f}, Speyside={p[1]:.3f} -> {predictions_soft[i]}")

    # Test with hard voting
    print("\n" + "=" * 70)
    print("Test 2: Hard Voting (average_probas=False)")
    print("-" * 70)

    ensemble_hard = TreeEnsembleClassifier(
        n_estimators=100,
        average_probas=False,
        random_state=42
    )
    ensemble_hard.fit(X, y)
    predictions_hard = ensemble_hard.predict(X)
    accuracy_hard = np.mean(predictions_hard == y)

    print(f"Training accuracy: {accuracy_hard:.3f}")

    # Show vote proportions for first few samples
    proba_hard = ensemble_hard.predict_proba(X[:3])
    print(f"\nVote proportions for first 3 samples:")
    for i, p in enumerate(proba_hard):
        print(f"  Sample {i}: Islay={p[0]:.3f}, Speyside={p[1]:.3f} -> {predictions_hard[i]}")

    # Test with different ensemble sizes
    print("\n" + "=" * 70)
    print("Test 3: Effect of Ensemble Size")
    print("-" * 70)

    for n_est in [10, 50, 100, 200]:
        ensemble = TreeEnsembleClassifier(
            n_estimators=n_est,
            average_probas=True,
            random_state=42
        )
        ensemble.fit(X, y)
        predictions = ensemble.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"n_estimators={n_est:3d}: accuracy={accuracy:.3f}")

    # Test sklearn compatibility
    print("\n" + "=" * 70)
    print("Test 4: sklearn Compatibility")
    print("-" * 70)

    from sklearn.model_selection import cross_val_score

    # Create a slightly larger dataset for cross-validation
    X_cv = np.tile(X, (3, 1))  # Repeat 3 times
    y_cv = np.tile(y, 3)

    ensemble_cv = TreeEnsembleClassifier(n_estimators=50, random_state=42)

    try:
        scores = cross_val_score(ensemble_cv, X_cv, y_cv, cv=3)
        print(f"Cross-validation scores: {scores}")
        print(f"Mean CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    except Exception as e:
        print(f"Cross-validation test: {str(e)}")

    # Test deterministic behavior
    print("\n" + "=" * 70)
    print("Test 5: Deterministic Behavior")
    print("-" * 70)

    ensemble1 = TreeEnsembleClassifier(n_estimators=50, random_state=42)
    ensemble1.fit(X, y)
    pred1 = ensemble1.predict(X)

    ensemble2 = TreeEnsembleClassifier(n_estimators=50, random_state=42)
    ensemble2.fit(X, y)
    pred2 = ensemble2.predict(X)

    print(f"Predictions match: {np.array_equal(pred1, pred2)}")

    # Test with different random states
    ensemble3 = TreeEnsembleClassifier(n_estimators=50, random_state=123)
    ensemble3.fit(X, y)
    pred3 = ensemble3.predict(X)

    print(f"Different random_state gives different results: {not np.array_equal(pred1, pred3)}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
