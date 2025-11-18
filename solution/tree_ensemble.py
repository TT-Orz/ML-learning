"""
Tree Ensemble Classifier with native categorical feature support.

An ensemble of decision stumps using bootstrap aggregating (bagging) for
improved performance. This implementation natively handles categorical features
without requiring one-hot encoding.
"""

import numpy as np
from collections import Counter
from .decision_stump import DecisionStumpClassifier


class TreeEnsembleClassifier:
    """
    Ensemble of Decision Stumps using Bagging.

    This classifier creates an ensemble of decision stumps, each trained on
    a bootstrap sample of the training data. It natively handles categorical
    features without one-hot encoding.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of decision stumps in the ensemble.

    criterion : str, default='gini'
        The function to measure split quality. Supported criteria:
        - 'gini': Gini impurity
        - 'entropy': Information gain

    max_features : int, float, str or None, default=None
        The number of features to consider when looking for the best split:
        - If int, consider max_features features.
        - If float, max_features is a fraction and int(max_features * n_features).
        - If 'sqrt', max_features=sqrt(n_features).
        - If 'log2', max_features=log2(n_features).
        - If None, max_features=n_features.

    bootstrap : bool, default=True
        Whether to use bootstrap samples for training each stump.

    random_state : int or None, default=None
        Random seed for reproducibility.

    n_jobs : int, default=1
        Number of parallel jobs (not implemented, for API compatibility).

    Attributes
    ----------
    estimators_ : list
        The collection of fitted decision stumps.

    classes_ : ndarray
        The unique class labels.

    n_features_ : int
        The number of features when fit is performed.
    """

    def __init__(self, n_estimators=100, criterion='gini', max_features=None,
                 bootstrap=True, random_state=None, n_jobs=1):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimators_ = []
        self.classes_ = None
        self.n_features_ = None

    def _get_max_features(self, n_features):
        """Calculate the number of features to use for each split."""
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        else:
            return n_features

    def _bootstrap_sample(self, X, y, rng):
        """
        Create a bootstrap sample of the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        rng : numpy.random.RandomState
            Random number generator.

        Returns
        -------
        X_sample : array
            Bootstrap sample of X.

        y_sample : array
            Bootstrap sample of y.
        """
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def _feature_sample(self, X, rng):
        """
        Sample features for a single estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        rng : numpy.random.RandomState
            Random number generator.

        Returns
        -------
        X_sample : array
            Data with sampled features.

        feature_indices : array
            Indices of selected features.
        """
        n_features = X.shape[1]
        max_features = self._get_max_features(n_features)

        if max_features == n_features:
            return X, np.arange(n_features)

        feature_indices = rng.choice(n_features, size=max_features, replace=False)
        return X[:, feature_indices], feature_indices

    def fit(self, X, y):
        """
        Build an ensemble of decision stumps from the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can contain categorical features (strings, objects, or integers).

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_ = np.unique(y)

        # Initialize random state
        if self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
        else:
            rng = np.random.RandomState()

        self.estimators_ = []

        # Train each decision stump
        for i in range(self.n_estimators):
            # Create bootstrap sample if requested
            if self.bootstrap:
                X_train, y_train = self._bootstrap_sample(X, y, rng)
            else:
                X_train, y_train = X, y

            # Sample features if max_features is set
            if self.max_features is not None:
                X_train_sampled, feature_indices = self._feature_sample(X_train, rng)
            else:
                X_train_sampled = X_train
                feature_indices = np.arange(n_features)

            # Train decision stump
            # Use different random state for each stump
            stump_random_state = None if self.random_state is None else self.random_state + i
            stump = DecisionStumpClassifier(
                criterion=self.criterion,
                random_state=stump_random_state
            )
            stump.fit(X_train_sampled, y_train)

            # Store the stump along with its feature indices
            self.estimators_.append({
                'stump': stump,
                'feature_indices': feature_indices
            })

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels (majority vote from all stumps).
        """
        X = np.asarray(X)
        n_samples = X.shape[0]

        # Collect predictions from all estimators
        all_predictions = np.empty((n_samples, self.n_estimators), dtype=object)

        for i, estimator_dict in enumerate(self.estimators_):
            stump = estimator_dict['stump']
            feature_indices = estimator_dict['feature_indices']

            # Select the features used by this stump
            X_subset = X[:, feature_indices]

            # Get predictions
            predictions = stump.predict(X_subset)
            all_predictions[:, i] = predictions

        # Majority voting
        final_predictions = []
        for i in range(n_samples):
            vote_counts = Counter(all_predictions[i, :])
            final_predictions.append(vote_counts.most_common(1)[0][0])

        # Convert to array with appropriate dtype
        final_predictions = np.array(final_predictions)

        return final_predictions

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        The predicted class probabilities are the fraction of stumps
        predicting each class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Collect predictions from all estimators
        all_predictions = np.empty((n_samples, self.n_estimators), dtype=object)

        for i, estimator_dict in enumerate(self.estimators_):
            stump = estimator_dict['stump']
            feature_indices = estimator_dict['feature_indices']

            # Select the features used by this stump
            X_subset = X[:, feature_indices]

            # Get predictions
            predictions = stump.predict(X_subset)
            all_predictions[:, i] = predictions

        # Calculate probabilities
        proba = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            vote_counts = Counter(all_predictions[i, :])
            for j, cls in enumerate(self.classes_):
                proba[i, j] = vote_counts.get(cls, 0) / self.n_estimators

        return proba

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
