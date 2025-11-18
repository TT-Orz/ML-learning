"""
Decision Stump Classifier for Categorical Data

This module implements a one-level decision tree classifier that works exclusively
with categorical features. It supports various split quality measures and includes
Laplace smoothing for probability estimation.

Author: COMP3222 Coursework Part 2
"""

import numpy as np
from .attribute_quality import (
    information_gain,
    information_gain_ratio,
    chi_squared,
    chi_squared_yates
)


class DecisionStumpClassifier:
    """
    A one-level decision tree classifier for categorical data.

    This classifier selects the single best attribute to split on based on
    a specified quality measure, then uses class distributions within each
    attribute value for prediction. It applies Laplace smoothing to handle
    unseen categories gracefully.

    Parameters
    ----------
    n_attributes : int or None, default=None
        Number of attributes to randomly consider when selecting the best split.
        If None, all attributes are considered.

    quality_measure : str, default="ig"
        The criterion to evaluate split quality. Options:
        - "ig": Information Gain
        - "gain_ratio": Information Gain Ratio
        - "chi2": Chi-Squared statistic
        - "chi2_yates": Chi-Squared with Yates's correction

    random_state : int or None, default=None
        Seed for random number generator. Used when n_attributes < total attributes
        to ensure reproducible attribute sampling.

    alpha : float, default=1.0
        Laplace smoothing parameter. Applied when estimating probabilities:
        P(class | value) = (count + alpha) / (total + alpha × n_classes)

    Attributes
    ----------
    att_index : int
        The column index of the selected split attribute.

    classes_ : numpy.ndarray
        Unique class labels in the training data, sorted.

    class_counts_ : dict
        Maps each attribute value to an array of class counts.
        Key: attribute value, Value: array of shape (n_classes,)

    root_prior_ : numpy.ndarray
        Overall class probability distribution, used for unseen attribute values.
        Shape: (n_classes,)

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([['yes', 'no'], ['yes', 'yes'], ['no', 'yes']], dtype=object)
    >>> y = np.array([0, 0, 1])
    >>> stump = DecisionStumpClassifier(quality_measure="ig")
    >>> stump.fit(X, y)
    >>> predictions = stump.predict(X)
    >>> print(predictions)
    [0 0 1]
    """

    def __init__(self, n_attributes=None, quality_measure="ig",
                 random_state=None, alpha=1.0):
        """Initialize the decision stump classifier."""
        self.n_attributes = n_attributes
        self.quality_measure = quality_measure
        self.random_state = random_state
        self.alpha = alpha

        # Attributes set during fit
        self.att_index = None
        self.classes_ = None
        self.class_counts_ = None
        self.root_prior_ = None

    def _validate_input(self, X):
        """
        Validate that input contains only categorical data (no floats).

        Parameters
        ----------
        X : numpy.ndarray
            Input data to validate.

        Raises
        ------
        TypeError
            If X contains floating-point values.
        ValueError
            If X is empty or has invalid shape.
        """
        if X.size == 0:
            raise ValueError("Input X cannot be empty")

        if len(X.shape) != 2:
            raise ValueError("Input X must be 2-dimensional")

        # Check if any element is a float type
        # We need to check the dtype and actual values
        if np.issubdtype(X.dtype, np.floating):
            raise TypeError("Input X contains floating-point values. "
                          "Only categorical (string/integer) data is supported.")

        # Also check for actual float values in object arrays
        if X.dtype == object:
            for val in X.flat:
                if isinstance(val, float) and not np.isnan(val):
                    raise TypeError("Input X contains floating-point values. "
                                  "Only categorical (string/integer) data is supported.")

    def fit(self, X, y):
        """
        Fit the decision stump to the training data.

        Selects the best attribute based on the specified quality measure,
        then stores class distributions for each attribute value.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training data. Must contain only categorical values (strings or integers).

        y : numpy.ndarray of shape (n_samples,)
            Target labels. Should be encoded as 0, 1, ..., n_classes - 1.

        Returns
        -------
        self : DecisionStumpClassifier
            Returns self for method chaining.

        Raises
        ------
        TypeError
            If X contains floating-point values.
        ValueError
            If X and y have incompatible shapes.
        """
        # Validate input
        self._validate_input(X)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples. "
                           f"Got X: {X.shape[0]}, y: {y.shape[0]}")

        n_samples, n_features = X.shape

        # Store unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Calculate root prior (overall class distribution)
        class_counts_total = np.bincount(y, minlength=n_classes)
        self.root_prior_ = (class_counts_total + self.alpha) / \
                          (n_samples + self.alpha * n_classes)

        # Determine which attributes to consider
        if self.n_attributes is None:
            # Consider all attributes
            candidate_attributes = np.arange(n_features)
        else:
            # Randomly sample n_attributes without replacement
            if self.random_state is not None:
                rng = np.random.RandomState(self.random_state)
            else:
                rng = np.random.RandomState()

            n_to_sample = min(self.n_attributes, n_features)
            candidate_attributes = rng.choice(n_features, size=n_to_sample,
                                            replace=False)

        # Select quality measure function
        quality_functions = {
            "ig": information_gain,
            "gain_ratio": information_gain_ratio,
            "chi2": chi_squared,
            "chi2_yates": chi_squared_yates
        }

        if self.quality_measure not in quality_functions:
            raise ValueError(f"Unknown quality measure: {self.quality_measure}. "
                           f"Choose from {list(quality_functions.keys())}")

        quality_func = quality_functions[self.quality_measure]

        # Evaluate each candidate attribute
        best_score = -np.inf
        best_attribute = None

        for att_idx in candidate_attributes:
            # Build contingency table for this attribute
            # Rows: attribute values, Columns: class labels
            att_values = X[:, att_idx]
            unique_values = np.unique(att_values)

            # Create contingency table
            contingency_table = []
            for val in unique_values:
                mask = att_values == val
                class_counts = np.bincount(y[mask], minlength=n_classes)
                contingency_table.append(class_counts)

            contingency_table = np.array(contingency_table)

            # Calculate quality score
            score = quality_func(contingency_table)

            # Update best attribute (tie-breaking: choose lowest index)
            if score > best_score or (score == best_score and
                                     (best_attribute is None or att_idx < best_attribute)):
                best_score = score
                best_attribute = att_idx

        # Store the selected attribute
        self.att_index = best_attribute

        # Store class counts for each value of the selected attribute
        self.class_counts_ = {}
        att_values = X[:, self.att_index]

        for val in np.unique(att_values):
            mask = att_values == val
            class_counts = np.bincount(y[mask], minlength=n_classes)
            self.class_counts_[val] = class_counts

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for each sample.

        Uses Laplace smoothing to estimate probabilities. For attribute values
        not seen during training, returns the root prior distribution.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : numpy.ndarray of shape (n_samples, n_classes)
            Probability distributions over classes for each sample.
            Each row sums to 1.0.

        Raises
        ------
        ValueError
            If the classifier has not been fitted yet.
        """
        if self.att_index is None:
            raise ValueError("Classifier has not been fitted yet. Call fit() first.")

        self._validate_input(X)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Initialize probability matrix
        proba = np.zeros((n_samples, n_classes))

        # Get the attribute values for the selected attribute
        att_values = X[:, self.att_index]

        for i, val in enumerate(att_values):
            if val in self.class_counts_:
                # Known attribute value: use stored counts with Laplace smoothing
                counts = self.class_counts_[val]
                total = counts.sum()
                proba[i] = (counts + self.alpha) / (total + self.alpha * n_classes)
            else:
                # Unseen attribute value: use root prior
                proba[i] = self.root_prior_

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
            Predicted class labels.

        Notes
        -----
        In case of ties (equal probabilities), the smallest class index is returned.
        """
        # Get probability distributions
        proba = self.predict_proba(X)

        # Return class with highest probability (argmax)
        # argmax automatically does tie-breaking by choosing smallest index
        predictions = np.argmax(proba, axis=1)

        return predictions


if __name__ == '__main__':
    # Test with the whisky dataset from Part 1
    print("Testing DecisionStumpClassifier")
    print("=" * 70)

    # Whisky dataset
    # Features: [Peaty, Winey, Postcode-starts-PA]
    # Classes: 0 = Islay, 1 = Speyside
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

    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Test with different quality measures
    quality_measures = ["ig", "gain_ratio", "chi2", "chi2_yates"]

    for measure in quality_measures:
        print(f"\nTesting with quality measure: {measure}")
        print("-" * 70)

        stump = DecisionStumpClassifier(quality_measure=measure, random_state=42)
        stump.fit(X, y)

        print(f"Selected attribute index: {stump.att_index}")
        print(f"Attribute name: ['Peaty', 'Winey', 'Postcode-PA'][{stump.att_index}]")

        predictions = stump.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"Training accuracy: {accuracy:.3f}")

        # Show probability predictions for first few samples
        proba = stump.predict_proba(X[:3])
        print(f"\nProbabilities for first 3 samples:")
        for i, p in enumerate(proba):
            print(f"  Sample {i}: Islay={p[0]:.3f}, Speyside={p[1]:.3f} -> Pred={predictions[i]}")

    # Test with random attribute selection
    print("\n" + "=" * 70)
    print("Testing random attribute selection (n_attributes=2)")
    print("-" * 70)

    stump = DecisionStumpClassifier(n_attributes=2, quality_measure="ig", random_state=42)
    stump.fit(X, y)
    predictions = stump.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Selected from 2 random attributes: index {stump.att_index}")
    print(f"Training accuracy: {accuracy:.3f}")

    # Test unseen category handling
    print("\n" + "=" * 70)
    print("Testing unseen category handling")
    print("-" * 70)

    X_test = np.array([
        ['yes', 'no', 'yes'],      # Known combination
        ['maybe', 'no', 'yes'],    # Unknown value in selected attribute
    ], dtype=object)

    proba_test = stump.predict_proba(X_test)
    pred_test = stump.predict(X_test)

    print(f"Test sample with known values:")
    print(f"  Probabilities: Islay={proba_test[0][0]:.3f}, Speyside={proba_test[0][1]:.3f}")
    print(f"  Prediction: {pred_test[0]}")

    print(f"\nTest sample with unseen value (uses root prior):")
    print(f"  Probabilities: Islay={proba_test[1][0]:.3f}, Speyside={proba_test[1][1]:.3f}")
    print(f"  Prediction: {pred_test[1]}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
