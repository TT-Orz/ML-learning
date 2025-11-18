"""
COMP3222 Machine Learning Coursework - Solution Package
Part 2: Custom ensemble classifier with native categorical support
"""

from .decision_stump import DecisionStumpClassifier
from .tree_ensemble import TreeEnsembleClassifier

__all__ = ['DecisionStumpClassifier', 'TreeEnsembleClassifier']
