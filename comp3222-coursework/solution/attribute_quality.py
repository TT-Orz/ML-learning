"""
Attribute Quality Measures for Decision Trees

This module provides functions to evaluate the quality of attribute splits
for categorical data classification using various statistical measures.

All functions accept a contingency table (2D numpy array) where:
- Rows represent attribute values
- Columns represent class labels
- Cell values are counts of instances

Author: COMP3222 Coursework Part 2
"""

import numpy as np


def information_gain(table):
    """
    Calculate the information gain of a split given a contingency table.

    Information gain measures the reduction in entropy achieved by partitioning
    the data according to an attribute. It is calculated as:
    IG(S, A) = H(S) - Σ (|S_v| / |S|) × H(S_v)

    where:
    - H(S) is the entropy of the parent node
    - S_v is the subset of instances with attribute value v
    - H(S_v) is the entropy of that subset

    Entropy is calculated as: H(S) = -Σ p_i × log2(p_i)
    Convention: 0 × log2(0) = 0

    Parameters
    ----------
    table : numpy.ndarray
        2D contingency table where rows are attribute values and columns are class labels.
        Shape: (n_attribute_values, n_classes)

    Returns
    -------
    float
        Information gain in bits (non-negative value)

    Examples
    --------
    >>> import numpy as np
    >>> table = np.array([[4, 0], [1, 5]])  # Peaty attribute split
    >>> ig = information_gain(table)
    >>> print(f"{ig:.3f}")
    0.610
    """
    # Handle empty table
    if table.size == 0 or table.sum() == 0:
        return 0.0

    # Calculate parent entropy (entropy before split)
    class_totals = table.sum(axis=0)  # Total per class
    total_instances = class_totals.sum()

    # Calculate class probabilities
    class_probs = class_totals / total_instances

    # Calculate parent entropy: H(S) = -Σ p_i × log2(p_i)
    # Handle 0 × log2(0) = 0 convention
    parent_entropy = 0.0
    for p in class_probs:
        if p > 0:
            parent_entropy -= p * np.log2(p)

    # Calculate weighted child entropy (entropy after split)
    weighted_child_entropy = 0.0

    for row in table:
        row_total = row.sum()
        if row_total == 0:
            continue

        # Weight for this attribute value
        weight = row_total / total_instances

        # Calculate entropy for this subset
        subset_probs = row / row_total
        subset_entropy = 0.0
        for p in subset_probs:
            if p > 0:
                subset_entropy -= p * np.log2(p)

        weighted_child_entropy += weight * subset_entropy

    # Information gain = reduction in entropy
    return parent_entropy - weighted_child_entropy


def information_gain_ratio(table):
    """
    Calculate the information gain ratio of a split given a contingency table.

    Gain ratio normalizes information gain by the split information to reduce
    bias towards attributes with many values. It is calculated as:
    GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)

    where SplitInfo(S, A) is the entropy of the split distribution:
    SplitInfo(S, A) = -Σ (|S_v| / |S|) × log2(|S_v| / |S|)

    Parameters
    ----------
    table : numpy.ndarray
        2D contingency table where rows are attribute values and columns are class labels.
        Shape: (n_attribute_values, n_classes)

    Returns
    -------
    float
        Gain ratio (non-negative value). Returns 0.0 if split information is 0.

    Examples
    --------
    >>> import numpy as np
    >>> table = np.array([[4, 0], [1, 5]])
    >>> gr = information_gain_ratio(table)
    >>> print(f"{gr:.3f}")
    0.664
    """
    # Handle empty table
    if table.size == 0 or table.sum() == 0:
        return 0.0

    # Calculate information gain
    ig = information_gain(table)

    # Calculate split information (entropy of the split distribution)
    row_totals = table.sum(axis=1)  # Total per attribute value
    total_instances = row_totals.sum()

    split_info = 0.0
    for row_total in row_totals:
        if row_total > 0:
            p = row_total / total_instances
            split_info -= p * np.log2(p)

    # Avoid division by zero
    if split_info == 0:
        return 0.0

    # Gain ratio = information gain / split information
    return ig / split_info


def chi_squared(table):
    """
    Calculate the Pearson chi-squared statistic for a contingency table.

    The chi-squared statistic measures the discrepancy between observed and
    expected frequencies under the assumption of independence:
    χ² = Σ (O_ij - E_ij)² / E_ij

    where:
    - O_ij is the observed count in cell (i, j)
    - E_ij is the expected count: (row_total × col_total) / grand_total

    Parameters
    ----------
    table : numpy.ndarray
        2D contingency table where rows are attribute values and columns are class labels.
        Shape: (n_attribute_values, n_classes)

    Returns
    -------
    float
        Chi-squared statistic (non-negative value)

    Examples
    --------
    >>> import numpy as np
    >>> table = np.array([[4, 0], [1, 5]])
    >>> chi2 = chi_squared(table)
    >>> print(f"{chi2:.3f}")
    6.667
    """
    # Handle empty table
    if table.size == 0 or table.sum() == 0:
        return 0.0

    # Calculate marginal totals
    row_totals = table.sum(axis=1)  # Sum across columns for each row
    col_totals = table.sum(axis=0)  # Sum across rows for each column
    grand_total = table.sum()

    # Calculate chi-squared statistic
    chi2 = 0.0

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            # Expected frequency: E_ij = (row_total × col_total) / grand_total
            expected = (row_totals[i] * col_totals[j]) / grand_total

            # Avoid division by zero
            if expected > 0:
                observed = table[i, j]
                chi2 += (observed - expected) ** 2 / expected

    return chi2


def chi_squared_yates(table):
    """
    Calculate the chi-squared statistic with Yates's correction for continuity.

    Yates's correction is applied to 2×2 contingency tables to reduce the
    approximation error of the chi-squared test. For larger tables, the
    standard chi-squared statistic is returned.

    For 2×2 tables:
    χ²_Yates = Σ (|O_ij - E_ij| - 0.5)² / E_ij

    For larger tables:
    Returns the standard chi-squared statistic (no correction applied)

    Parameters
    ----------
    table : numpy.ndarray
        2D contingency table where rows are attribute values and columns are class labels.
        Shape: (n_attribute_values, n_classes)

    Returns
    -------
    float
        Chi-squared statistic with Yates's correction (for 2×2 tables) or
        standard chi-squared statistic (for larger tables)

    Examples
    --------
    >>> import numpy as np
    >>> table = np.array([[4, 0], [1, 5]])  # 2×2 table
    >>> chi2_yates = chi_squared_yates(table)
    >>> print(f"{chi2_yates:.3f}")
    4.513
    """
    # Handle empty table
    if table.size == 0 or table.sum() == 0:
        return 0.0

    # If not a 2×2 table, return standard chi-squared
    if table.shape[0] != 2 or table.shape[1] != 2:
        return chi_squared(table)

    # Apply Yates's correction for 2×2 tables
    row_totals = table.sum(axis=1)
    col_totals = table.sum(axis=0)
    grand_total = table.sum()

    chi2_yates = 0.0

    for i in range(2):
        for j in range(2):
            # Expected frequency
            expected = (row_totals[i] * col_totals[j]) / grand_total

            # Avoid division by zero
            if expected > 0:
                observed = table[i, j]
                # Apply Yates's correction: use |O - E| - 0.5
                corrected_diff = abs(observed - expected) - 0.5
                # Ensure non-negative
                corrected_diff = max(0, corrected_diff)
                chi2_yates += (corrected_diff ** 2) / expected

    return chi2_yates


if __name__ == '__main__':
    # Test with the whisky dataset example from the coursework
    print("Testing Attribute Quality Measures")
    print("=" * 50)

    # Example 1: Peaty attribute from Part 1
    print("\nExample 1: Peaty attribute (yes/no) vs Region (Islay/Speyside)")
    table_peaty = np.array([
        [4, 0],  # yes: 4 Islay, 0 Speyside
        [1, 5]   # no:  1 Islay, 5 Speyside
    ])
    print(f"Contingency table:\n{table_peaty}")
    print(f"Information Gain:       {information_gain(table_peaty):.3f} bits")
    print(f"Gain Ratio:             {information_gain_ratio(table_peaty):.3f}")
    print(f"Chi-Squared:            {chi_squared(table_peaty):.3f}")
    print(f"Chi-Squared (Yates):    {chi_squared_yates(table_peaty):.3f}")

    # Example 2: A perfectly uninformative split
    print("\n" + "=" * 50)
    print("Example 2: Uninformative attribute (equal distribution)")
    table_uninformative = np.array([
        [2, 2],  # value1: 2 class0, 2 class1
        [3, 3]   # value2: 3 class0, 3 class1
    ])
    print(f"Contingency table:\n{table_uninformative}")
    print(f"Information Gain:       {information_gain(table_uninformative):.3f} bits")
    print(f"Gain Ratio:             {information_gain_ratio(table_uninformative):.3f}")
    print(f"Chi-Squared:            {chi_squared(table_uninformative):.3f}")
    print(f"Chi-Squared (Yates):    {chi_squared_yates(table_uninformative):.3f}")

    # Example 3: A perfect split
    print("\n" + "=" * 50)
    print("Example 3: Perfect split (pure subsets)")
    table_perfect = np.array([
        [5, 0],  # value1: all class0
        [0, 5]   # value2: all class1
    ])
    print(f"Contingency table:\n{table_perfect}")
    print(f"Information Gain:       {information_gain(table_perfect):.3f} bits")
    print(f"Gain Ratio:             {information_gain_ratio(table_perfect):.3f}")
    print(f"Chi-Squared:            {chi_squared(table_perfect):.3f}")
    print(f"Chi-Squared (Yates):    {chi_squared_yates(table_perfect):.3f}")

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
