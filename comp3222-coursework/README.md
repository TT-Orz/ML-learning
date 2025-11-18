# COMP3222 Machine Learning Coursework - Part 2

Implementation of decision tree components for categorical data classification.

## Project Structure

```
comp3222-coursework/
├── solution/
│   ├── __init__.py
│   ├── attribute_quality.py      # Quality measures for attribute splits
│   ├── decision_stump.py          # One-level decision tree classifier
│   └── tree_ensemble.py           # Ensemble of decision stumps
└── tests/
    ├── __init__.py
    ├── test_attribute_quality.py  # Unit tests for quality measures
    └── test_decision_stump.py     # Unit tests for decision stump
```

## Installation

Install required dependencies:

```bash
pip install numpy scikit-learn pytest
```

## Components

### 1. Attribute Quality Measures (`solution/attribute_quality.py`)

Four functions to evaluate the quality of attribute splits:

- **`information_gain(table)`**: Calculates information gain in bits
- **`information_gain_ratio(table)`**: Calculates gain ratio (normalized IG)
- **`chi_squared(table)`**: Calculates Pearson χ² statistic
- **`chi_squared_yates(table)`**: Calculates χ² with Yates's correction for 2×2 tables

**Input**: 2D numpy array (contingency table) where rows = attribute values, columns = class labels

**Example**:
```python
import numpy as np
from solution.attribute_quality import information_gain

table = np.array([[4, 0], [1, 5]])  # Peaty attribute from whisky dataset
ig = information_gain(table)
print(f"Information Gain: {ig:.3f} bits")  # Output: 0.610 bits
```

**Run demo**:
```bash
python -m solution.attribute_quality
```

### 2. Decision Stump Classifier (`solution/decision_stump.py`)

A one-level decision tree classifier for categorical data only.

**Features**:
- Supports multiple quality measures (IG, Gain Ratio, Chi-Squared)
- Random feature subset selection
- Laplace smoothing for probability estimation
- Handles unseen categories with root prior
- Rejects floating-point inputs

**Example**:
```python
import numpy as np
from solution.decision_stump import DecisionStumpClassifier

# Whisky dataset (categorical features only)
X = np.array([
    ['yes', 'no', 'yes'],
    ['yes', 'yes', 'yes'],
    ['no', 'yes', 'no']
], dtype=object)
y = np.array([0, 0, 1])

# Create and train classifier
stump = DecisionStumpClassifier(quality_measure="ig", random_state=42)
stump.fit(X, y)

# Make predictions
predictions = stump.predict(X)
probabilities = stump.predict_proba(X)
```

**Run demo**:
```bash
python -m solution.decision_stump
```

### 3. Tree Ensemble Classifier (`solution/tree_ensemble.py`)

An ensemble of decision stumps compatible with scikit-learn.

**Features**:
- Multiple diversity strategies:
  - Random feature subsets (√n_features per stump)
  - Bootstrap sampling
  - Random quality measure selection
- Soft voting (average probabilities) or hard voting (majority vote)
- scikit-learn compatible (works with cross-validation, grid search, etc.)
- Deterministic with random_state

**Example**:
```python
import numpy as np
from solution.tree_ensemble import TreeEnsembleClassifier
from sklearn.model_selection import cross_val_score

X = np.array([['yes', 'no'], ['no', 'yes'], ['yes', 'yes']], dtype=object)
y = np.array(['A', 'B', 'A'])

# Create ensemble with 100 stumps
ensemble = TreeEnsembleClassifier(n_estimators=100, random_state=42)
ensemble.fit(X, y)

# Make predictions
predictions = ensemble.predict(X)

# Use with sklearn tools
scores = cross_val_score(ensemble, X, y, cv=3)
```

**Run demo**:
```bash
python -m solution.tree_ensemble
```

## Testing

Run all unit tests:

```bash
python -m pytest tests/ -v
```

Run specific test file:

```bash
python -m pytest tests/test_attribute_quality.py -v
python -m pytest tests/test_decision_stump.py -v
```

## Implementation Details

### Attribute Quality Measures

All functions handle edge cases:
- Zero divisions (returns 0.0)
- Empty tables
- Pure nodes (entropy = 0)
- 0 × log₂(0) = 0 convention

Yates's correction is applied only to 2×2 tables; larger tables return standard χ².

### Decision Stump Classifier

**Key design choices**:
- Uses contingency tables for efficient computation
- Tie-breaking: selects lowest column index
- Laplace smoothing: `(count + α) / (total + α × n_classes)`
- Root prior: overall class distribution for unseen values

**Input validation**:
- Rejects float dtype arrays
- Rejects object arrays containing float values
- Accepts integers, strings, None, NaN (treated as categories)

### Tree Ensemble Classifier

**Diversity strategies**:
- **Feature subsets**: Each stump sees √n_features random features
- **Bootstrap sampling**: Each stump trained on n_samples with replacement
- **Quality measures**: Randomly selects from [ig, gain_ratio, chi2, chi2_yates]

**Voting methods**:
- `average_probas=True`: Soft voting (average predict_proba)
- `average_probas=False`: Hard voting (majority vote)

## Test Coverage

### Attribute Quality (`test_attribute_quality.py`)
- ✓ Hand-calculated values from whisky dataset
- ✓ Perfect splits (IG = 1.0)
- ✓ Uninformative splits (IG ≈ 0)
- ✓ Independence testing (χ² = 0)
- ✓ Yates correction behavior
- ✓ Edge cases (empty tables, single rows, zeros)

### Decision Stump (`test_decision_stump.py`)
- ✓ Float rejection
- ✓ Deterministic behavior with random_state
- ✓ Valid probability outputs (sum to 1)
- ✓ Unseen category handling
- ✓ Laplace smoothing verification
- ✓ All quality measures
- ✓ Whisky dataset predictions

## Performance

On the whisky dataset (10 samples, 3 features, 2 classes):

| Classifier | Quality Measure | Accuracy |
|-----------|----------------|----------|
| Decision Stump | IG | 90% |
| Decision Stump | Gain Ratio | 90% |
| Decision Stump | Chi-Squared | 90% |
| Tree Ensemble (100) | Mixed | 90% |

## Notes

- **Categorical data only**: All classifiers reject floating-point inputs
- **Deterministic**: Setting `random_state` ensures reproducible results
- **sklearn compatible**: TreeEnsembleClassifier inherits from BaseEstimator and ClassifierMixin
- **Comprehensive docstrings**: All functions include detailed documentation

## References

- Lecture 2: Decision Trees
- Quinlan, J. R. (1986). Induction of decision trees. Machine Learning, 1(1), 81-106.
- Yates, F. (1934). Contingency tables involving small numbers and the χ² test.

## Author

COMP3222 Coursework Part 2
MSc Machine Learning
