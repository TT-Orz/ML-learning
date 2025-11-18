"""
COMP3222 Machine Learning Coursework - Part 2
Decision Tree Components for Categorical Data Classification

This package provides implementations of:
- Attribute quality measures (Information Gain, Gain Ratio, Chi-Squared)
- Decision Stump Classifier (one-level decision tree)
- Tree Ensemble Classifier (ensemble of decision stumps)
"""

from .attribute_quality import (
    information_gain,
    information_gain_ratio,
    chi_squared,
    chi_squared_yates
)
from .decision_stump import DecisionStumpClassifier
from .tree_ensemble import TreeEnsembleClassifier

__all__ = [
    'information_gain',
    'information_gain_ratio',
    'chi_squared',
    'chi_squared_yates',
    'DecisionStumpClassifier',
    'TreeEnsembleClassifier'
]
