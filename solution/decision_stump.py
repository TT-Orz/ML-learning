"""
Decision Stump Classifier with native categorical feature support.

A decision stump is a one-level decision tree that makes predictions based on
a single feature. This implementation natively handles categorical features
without requiring one-hot encoding.
"""

import numpy as np
from collections import Counter


class DecisionStumpClassifier:
    """
    Decision Stump Classifier that natively handles categorical features.

    This classifier creates a simple one-level decision tree by selecting
    the best feature and split that maximizes information gain (or minimizes
    Gini impurity).

    Parameters
    ----------
    criterion : str, default='gini'
        The function to measure split quality. Supported criteria:
        - 'gini': Gini impurity
        - 'entropy': Information gain

    random_state : int or None, default=None
        Random seed for reproducibility when breaking ties.

    Attributes
    ----------
    best_feature_ : int
        Index of the feature used for splitting.

    split_value_ : any
        The value used for splitting (for categorical features, this defines
        the subset of categories that go to the left child).

    left_prediction_ : any
        Predicted class for samples going to the left child.

    right_prediction_ : any
        Predicted class for samples going to the right child.

    is_categorical_ : bool
        Whether the best feature is categorical.

    categories_left_ : set
        Set of categories that go to the left child (for categorical features).
    """

    def __init__(self, criterion='gini', random_state=None):
        self.criterion = criterion
        self.random_state = random_state
        self.best_feature_ = None
        self.split_value_ = None
        self.left_prediction_ = None
        self.right_prediction_ = None
        self.is_categorical_ = None
        self.categories_left_ = None

    def _is_categorical_feature(self, X, feature_idx):
        """
        Determine if a feature is categorical.

        A feature is considered categorical if it's not numeric or if it has
        fewer unique values suggesting discrete categories.
        """
        feature = X[:, feature_idx]

        # Check if feature is non-numeric (strings, objects)
        if not np.issubdtype(feature.dtype, np.number):
            return True

        # For numeric features, check if they appear to be categorical
        # (e.g., integers with few unique values)
        unique_vals = np.unique(feature)
        if len(unique_vals) <= 10 and np.all(feature == feature.astype(int)):
            return True

        return False

    def _gini_impurity(self, y):
        """Calculate Gini impurity for a set of labels."""
        if len(y) == 0:
            return 0

        counts = Counter(y)
        impurity = 1.0
        n = len(y)

        for count in counts.values():
            prob = count / n
            impurity -= prob ** 2

        return impurity

    def _entropy(self, y):
        """Calculate entropy for a set of labels."""
        if len(y) == 0:
            return 0

        counts = Counter(y)
        entropy = 0.0
        n = len(y)

        for count in counts.values():
            if count > 0:
                prob = count / n
                entropy -= prob * np.log2(prob)

        return entropy

    def _information_gain(self, y, left_y, right_y):
        """Calculate information gain for a split."""
        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)

        if self.criterion == 'gini':
            parent_impurity = self._gini_impurity(y)
            left_impurity = self._gini_impurity(left_y)
            right_impurity = self._gini_impurity(right_y)
        else:  # entropy
            parent_impurity = self._entropy(y)
            left_impurity = self._entropy(left_y)
            right_impurity = self._entropy(right_y)

        # Weighted average of child impurities
        weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity

        return parent_impurity - weighted_impurity

    def _try_categorical_split(self, X, y, feature_idx):
        """
        Try all possible binary splits for a categorical feature.

        For categorical features, we try splitting categories into two groups.
        We use a greedy approach: for each category, try putting it in the left
        group vs right group.
        """
        feature = X[:, feature_idx]
        categories = np.unique(feature)

        if len(categories) <= 1:
            return None, -np.inf

        best_gain = -np.inf
        best_split = None

        # Try each category as the basis for splitting
        for cat in categories:
            # Split: this category vs all others
            left_mask = feature == cat
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_y = y[left_mask]
            right_y = y[right_mask]

            gain = self._information_gain(y, left_y, right_y)

            if gain > best_gain:
                best_gain = gain
                best_split = {cat}  # Set containing the category for left split

        # Also try grouping categories by their most common class
        if len(categories) > 2:
            category_class_map = {}
            for cat in categories:
                cat_mask = feature == cat
                cat_y = y[cat_mask]
                if len(cat_y) > 0:
                    most_common = Counter(cat_y).most_common(1)[0][0]
                    category_class_map[cat] = most_common

            # Group categories by their most common class
            classes = set(category_class_map.values())
            for target_class in classes:
                left_cats = {cat for cat, cls in category_class_map.items()
                            if cls == target_class}

                if len(left_cats) == 0 or len(left_cats) == len(categories):
                    continue

                left_mask = np.isin(feature, list(left_cats))
                right_mask = ~left_mask

                left_y = y[left_mask]
                right_y = y[right_mask]

                if len(left_y) > 0 and len(right_y) > 0:
                    gain = self._information_gain(y, left_y, right_y)

                    if gain > best_gain:
                        best_gain = gain
                        best_split = left_cats

        return best_split, best_gain

    def _try_numeric_split(self, X, y, feature_idx):
        """
        Try all possible splits for a numeric feature.

        For numeric features, we sort the values and try splitting at midpoints.
        """
        feature = X[:, feature_idx]
        unique_values = np.unique(feature)

        if len(unique_values) <= 1:
            return None, -np.inf

        best_gain = -np.inf
        best_threshold = None

        # Try splitting at midpoints between consecutive unique values
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2

            left_mask = feature <= threshold
            right_mask = ~left_mask

            left_y = y[left_mask]
            right_y = y[right_mask]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            gain = self._information_gain(y, left_y, right_y)

            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold

        return best_threshold, best_gain

    def fit(self, X, y):
        """
        Fit the decision stump to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

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

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Try all features and find the best split
        best_overall_gain = -np.inf

        for feature_idx in range(n_features):
            is_categorical = self._is_categorical_feature(X, feature_idx)

            if is_categorical:
                split_value, gain = self._try_categorical_split(X, y, feature_idx)
            else:
                split_value, gain = self._try_numeric_split(X, y, feature_idx)

            if gain > best_overall_gain:
                best_overall_gain = gain
                self.best_feature_ = feature_idx
                self.split_value_ = split_value
                self.is_categorical_ = is_categorical

        # If no good split was found, predict the most common class
        if self.best_feature_ is None:
            most_common = Counter(y).most_common(1)[0][0]
            self.left_prediction_ = most_common
            self.right_prediction_ = most_common
            return self

        # Determine predictions for left and right children
        if self.is_categorical_:
            self.categories_left_ = self.split_value_
            feature = X[:, self.best_feature_]
            left_mask = np.isin(feature, list(self.categories_left_))
        else:
            feature = X[:, self.best_feature_]
            left_mask = feature <= self.split_value_

        right_mask = ~left_mask

        left_y = y[left_mask]
        right_y = y[right_mask]

        # Predict most common class in each child
        if len(left_y) > 0:
            self.left_prediction_ = Counter(left_y).most_common(1)[0][0]
        else:
            self.left_prediction_ = Counter(y).most_common(1)[0][0]

        if len(right_y) > 0:
            self.right_prediction_ = Counter(right_y).most_common(1)[0][0]
        else:
            self.right_prediction_ = Counter(y).most_common(1)[0][0]

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
            Predicted class labels.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        predictions = np.empty(n_samples, dtype=object)

        if self.best_feature_ is None:
            # No split was made, return the default prediction
            predictions[:] = self.left_prediction_
            return predictions

        feature = X[:, self.best_feature_]

        if self.is_categorical_:
            left_mask = np.isin(feature, list(self.categories_left_))
        else:
            left_mask = feature <= self.split_value_

        predictions[left_mask] = self.left_prediction_
        predictions[~left_mask] = self.right_prediction_

        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        For decision stumps, probabilities are 0 or 1 based on the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        predictions = self.predict(X)
        classes = np.unique(predictions)
        n_classes = len(classes)
        n_samples = len(predictions)

        proba = np.zeros((n_samples, n_classes))

        for i, pred in enumerate(predictions):
            class_idx = np.where(classes == pred)[0][0]
            proba[i, class_idx] = 1.0

        return proba
