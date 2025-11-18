"""
Unit tests for DecisionStumpClassifier.

Tests the decision stump classifier functionality including input validation,
prediction, probability estimation, and handling of edge cases.

Author: COMP3222 Coursework Part 2
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import solution module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solution.decision_stump import DecisionStumpClassifier


class TestInputValidation:
    """Test input validation and error handling."""

    def test_rejects_float_array(self):
        """Test that float arrays are rejected."""
        X = np.array([[1.5, 2.3], [3.1, 4.7]])  # Float array
        y = np.array([0, 1])

        stump = DecisionStumpClassifier()
        with pytest.raises(TypeError, match="floating-point"):
            stump.fit(X, y)

    def test_rejects_mixed_float_in_object_array(self):
        """Test that object arrays with float values are rejected."""
        X = np.array([['yes', 1.5], ['no', 2.3]], dtype=object)
        y = np.array([0, 1])

        stump = DecisionStumpClassifier()
        with pytest.raises(TypeError, match="floating-point"):
            stump.fit(X, y)

    def test_accepts_integer_values(self):
        """Test that integer values are accepted."""
        X = np.array([[1, 2], [3, 4], [1, 4]], dtype=int)
        y = np.array([0, 1, 0])

        stump = DecisionStumpClassifier()
        # Should not raise an error
        stump.fit(X, y)
        assert stump.att_index is not None

    def test_accepts_string_values(self):
        """Test that string values are accepted."""
        X = np.array([['yes', 'no'], ['no', 'yes'], ['yes', 'yes']], dtype=object)
        y = np.array([0, 1, 0])

        stump = DecisionStumpClassifier()
        # Should not raise an error
        stump.fit(X, y)
        assert stump.att_index is not None

    def test_empty_input(self):
        """Test that empty input raises error."""
        X = np.array([])
        y = np.array([])

        stump = DecisionStumpClassifier()
        with pytest.raises(ValueError):
            stump.fit(X, y)

    def test_mismatched_shapes(self):
        """Test that mismatched X and y shapes raise error."""
        X = np.array([['yes', 'no'], ['no', 'yes']], dtype=object)
        y = np.array([0])  # Only one label for two samples

        stump = DecisionStumpClassifier()
        with pytest.raises(ValueError, match="same number of samples"):
            stump.fit(X, y)


class TestFitting:
    """Test fitting behavior."""

    def test_fit_whisky_dataset(self):
        """Test fitting on the whisky dataset."""
        X = np.array([
            ['yes', 'no', 'yes'],
            ['yes', 'yes', 'yes'],
            ['yes', 'no', 'no'],
            ['yes', 'no', 'no'],
            ['no', 'yes', 'no'],
            ['no', 'yes', 'yes'],
            ['no', 'yes', 'yes'],
            ['no', 'yes', 'yes'],
            ['no', 'no', 'yes'],
            ['no', 'no', 'yes']
        ], dtype=object)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        stump = DecisionStumpClassifier(quality_measure="ig")
        stump.fit(X, y)

        # Should select an attribute
        assert stump.att_index is not None
        assert 0 <= stump.att_index < 3

        # Should store classes
        assert len(stump.classes_) == 2
        assert set(stump.classes_) == {0, 1}

        # Should store class counts
        assert stump.class_counts_ is not None
        assert len(stump.class_counts_) > 0

        # Should have root prior
        assert stump.root_prior_ is not None
        assert len(stump.root_prior_) == 2
        assert abs(sum(stump.root_prior_) - 1.0) < 0.01  # Should sum to 1

    def test_different_quality_measures(self):
        """Test that different quality measures work."""
        X = np.array([['yes', 'no'], ['no', 'yes'], ['yes', 'yes'],
                      ['no', 'no'], ['yes', 'no'], ['no', 'yes']], dtype=object)
        y = np.array([0, 1, 0, 1, 0, 1])

        for measure in ["ig", "gain_ratio", "chi2", "chi2_yates"]:
            stump = DecisionStumpClassifier(quality_measure=measure)
            stump.fit(X, y)
            assert stump.att_index is not None, \
                f"Failed to fit with quality measure: {measure}"

    def test_n_attributes_selection(self):
        """Test that n_attributes parameter works."""
        X = np.array([['a', 'b', 'c', 'd'],
                      ['a', 'b', 'd', 'c'],
                      ['b', 'a', 'c', 'd']], dtype=object)
        y = np.array([0, 0, 1])

        # Select only 2 attributes
        stump = DecisionStumpClassifier(n_attributes=2, random_state=42)
        stump.fit(X, y)

        # Should still select one attribute
        assert stump.att_index is not None
        assert 0 <= stump.att_index < 4


class TestPrediction:
    """Test prediction functionality."""

    def test_predict_returns_correct_shape(self):
        """Test that predict returns correct shape."""
        X = np.array([['yes', 'no'], ['no', 'yes'], ['yes', 'yes']], dtype=object)
        y = np.array([0, 1, 0])

        stump = DecisionStumpClassifier()
        stump.fit(X, y)

        X_test = np.array([['yes', 'no'], ['no', 'yes']], dtype=object)
        predictions = stump.predict(X_test)

        assert predictions.shape == (2,)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba_returns_probabilities(self):
        """Test that predict_proba returns valid probabilities."""
        X = np.array([['yes', 'no'], ['no', 'yes'], ['yes', 'yes'],
                      ['no', 'no']], dtype=object)
        y = np.array([0, 1, 0, 1])

        stump = DecisionStumpClassifier()
        stump.fit(X, y)

        X_test = np.array([['yes', 'no'], ['no', 'yes']], dtype=object)
        proba = stump.predict_proba(X_test)

        # Check shape
        assert proba.shape == (2, 2)

        # Check that probabilities sum to 1
        for p in proba:
            assert abs(sum(p) - 1.0) < 0.01, f"Probabilities don't sum to 1: {p}"

        # Check that probabilities are in [0, 1]
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_consistency(self):
        """Test that predict is consistent with predict_proba."""
        X = np.array([['yes', 'no'], ['no', 'yes'], ['yes', 'yes']], dtype=object)
        y = np.array([0, 1, 0])

        stump = DecisionStumpClassifier()
        stump.fit(X, y)

        X_test = np.array([['yes', 'no'], ['no', 'yes']], dtype=object)
        predictions = stump.predict(X_test)
        proba = stump.predict_proba(X_test)

        # predict should return argmax of predict_proba
        for i, pred in enumerate(predictions):
            assert pred == np.argmax(proba[i]), \
                f"Inconsistent prediction: predict={pred}, argmax(proba)={np.argmax(proba[i])}"

    def test_unseen_category_handling(self):
        """Test handling of unseen categories."""
        X_train = np.array([['yes', 'no'], ['no', 'yes'], ['yes', 'yes']], dtype=object)
        y_train = np.array([0, 1, 0])

        stump = DecisionStumpClassifier()
        stump.fit(X_train, y_train)

        # Test with unseen category
        X_test = np.array([['maybe', 'unknown']], dtype=object)
        proba = stump.predict_proba(X_test)

        # Should use root prior
        assert proba.shape == (1, 2)
        assert abs(sum(proba[0]) - 1.0) < 0.01  # Should sum to 1

        # Should not crash
        pred = stump.predict(X_test)
        assert pred.shape == (1,)


class TestDeterministicBehavior:
    """Test deterministic behavior with random_state."""

    def test_same_random_state_same_results(self):
        """Test that same random_state gives same results."""
        X = np.array([['a', 'b', 'c', 'd'],
                      ['a', 'b', 'd', 'c'],
                      ['b', 'a', 'c', 'd'],
                      ['b', 'b', 'd', 'c']], dtype=object)
        y = np.array([0, 0, 1, 1])

        stump1 = DecisionStumpClassifier(n_attributes=2, random_state=42)
        stump1.fit(X, y)
        pred1 = stump1.predict(X)

        stump2 = DecisionStumpClassifier(n_attributes=2, random_state=42)
        stump2.fit(X, y)
        pred2 = stump2.predict(X)

        assert stump1.att_index == stump2.att_index, \
            "Same random_state should select same attribute"
        assert np.array_equal(pred1, pred2), \
            "Same random_state should give same predictions"

    def test_different_random_state_different_results(self):
        """Test that different random_state can give different results."""
        X = np.array([['a', 'b', 'c', 'd'],
                      ['a', 'b', 'd', 'c'],
                      ['b', 'a', 'c', 'd'],
                      ['b', 'b', 'd', 'c']], dtype=object)
        y = np.array([0, 0, 1, 1])

        stump1 = DecisionStumpClassifier(n_attributes=2, random_state=42)
        stump1.fit(X, y)

        stump2 = DecisionStumpClassifier(n_attributes=2, random_state=123)
        stump2.fit(X, y)

        # May or may not be different, but should work without errors
        assert stump1.att_index is not None
        assert stump2.att_index is not None


class TestLaplaceSmoothing:
    """Test Laplace smoothing behavior."""

    def test_laplace_smoothing_applied(self):
        """Test that Laplace smoothing prevents zero probabilities."""
        # Create a dataset where one class is missing for a value
        X = np.array([['yes'], ['yes'], ['no']], dtype=object)
        y = np.array([0, 0, 1])

        stump = DecisionStumpClassifier(alpha=1.0)
        stump.fit(X, y)

        # 'yes' has only class 0, but should have non-zero probability for class 1
        X_test = np.array([['yes']], dtype=object)
        proba = stump.predict_proba(X_test)

        # Both probabilities should be positive (due to Laplace smoothing)
        assert proba[0, 0] > 0, "Class 0 probability should be positive"
        assert proba[0, 1] > 0, "Class 1 probability should be positive (Laplace smoothing)"

    def test_different_alpha_values(self):
        """Test that different alpha values affect probabilities."""
        X = np.array([['yes'], ['yes'], ['no']], dtype=object)
        y = np.array([0, 0, 1])

        stump1 = DecisionStumpClassifier(alpha=0.5)
        stump1.fit(X, y)

        stump2 = DecisionStumpClassifier(alpha=2.0)
        stump2.fit(X, y)

        X_test = np.array([['yes']], dtype=object)
        proba1 = stump1.predict_proba(X_test)
        proba2 = stump2.predict_proba(X_test)

        # Different alpha should give different probabilities
        assert not np.allclose(proba1, proba2), \
            "Different alpha values should give different probabilities"


class TestWhiskyDataset:
    """Comprehensive tests on the whisky dataset."""

    def test_whisky_ig_accuracy(self):
        """Test accuracy on whisky dataset with information gain."""
        X = np.array([
            ['yes', 'no', 'yes'],
            ['yes', 'yes', 'yes'],
            ['yes', 'no', 'no'],
            ['yes', 'no', 'no'],
            ['no', 'yes', 'no'],
            ['no', 'yes', 'yes'],
            ['no', 'yes', 'yes'],
            ['no', 'yes', 'yes'],
            ['no', 'no', 'yes'],
            ['no', 'no', 'yes']
        ], dtype=object)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        stump = DecisionStumpClassifier(quality_measure="ig")
        stump.fit(X, y)
        predictions = stump.predict(X)
        accuracy = np.mean(predictions == y)

        # Should achieve reasonable accuracy (> 0.5 for binary problem)
        assert accuracy > 0.5, f"Accuracy should be > 0.5, got {accuracy:.3f}"

    def test_whisky_best_attribute(self):
        """Test that the best attribute is selected."""
        X = np.array([
            ['yes', 'no', 'yes'],
            ['yes', 'yes', 'yes'],
            ['yes', 'no', 'no'],
            ['yes', 'no', 'no'],
            ['no', 'yes', 'no'],
            ['no', 'yes', 'yes'],
            ['no', 'yes', 'yes'],
            ['no', 'yes', 'yes'],
            ['no', 'no', 'yes'],
            ['no', 'no', 'yes']
        ], dtype=object)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        # Feature 0 (Peaty) should be the best with IG
        stump = DecisionStumpClassifier(quality_measure="ig")
        stump.fit(X, y)

        # Peaty attribute should be selected (index 0)
        assert stump.att_index == 0, \
            f"Expected Peaty (index 0) to be selected, got index {stump.att_index}"


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
