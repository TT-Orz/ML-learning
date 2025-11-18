"""
Unit tests for attribute quality measures.

Tests the information gain, gain ratio, chi-squared, and chi-squared with
Yates correction functions against hand-calculated values and edge cases.

Author: COMP3222 Coursework Part 2
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import solution module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solution.attribute_quality import (
    information_gain,
    information_gain_ratio,
    chi_squared,
    chi_squared_yates
)


class TestInformationGain:
    """Test cases for information_gain function."""

    def test_peaty_attribute(self):
        """Test IG calculation for Peaty attribute from Part 1."""
        # Peaty attribute: yes=[4 Islay, 0 Speyside], no=[1 Islay, 5 Speyside]
        table = np.array([[4, 0], [1, 5]])
        expected = 0.610  # Hand-calculated value
        actual = information_gain(table)
        assert abs(actual - expected) < 0.01, \
            f"Expected IG ≈ {expected}, got {actual:.3f}"

    def test_perfect_split(self):
        """Test IG for a perfect split (all subsets are pure)."""
        # Perfect split: each value maps to exactly one class
        table = np.array([[5, 0], [0, 5]])
        # Parent entropy: -(0.5*log2(0.5) + 0.5*log2(0.5)) = 1.0
        # Child entropy: 0 (both subsets are pure)
        # IG = 1.0 - 0 = 1.0
        expected = 1.0
        actual = information_gain(table)
        assert abs(actual - expected) < 0.01, \
            f"Perfect split should have IG=1.0, got {actual:.3f}"

    def test_no_split(self):
        """Test IG for a useless split (identical distributions)."""
        # No information: same class distribution in all subsets
        table = np.array([[2, 2], [3, 3]])
        # Should have IG ≈ 0 (no reduction in entropy)
        expected = 0.0
        actual = information_gain(table)
        assert abs(actual - expected) < 0.01, \
            f"Useless split should have IG≈0, got {actual:.3f}"

    def test_pure_parent(self):
        """Test IG when parent is already pure."""
        # All instances belong to one class
        table = np.array([[3, 0], [2, 0]])
        # Parent is pure, so parent entropy = 0, IG = 0
        expected = 0.0
        actual = information_gain(table)
        assert abs(actual - expected) < 0.01, \
            f"Pure parent should have IG=0, got {actual:.3f}"

    def test_empty_table(self):
        """Test IG with empty table."""
        table = np.array([])
        expected = 0.0
        actual = information_gain(table)
        assert actual == expected, \
            f"Empty table should return IG=0, got {actual}"

    def test_three_way_split(self):
        """Test IG with three attribute values."""
        # Three-way split
        table = np.array([[3, 1], [2, 2], [1, 3]])
        actual = information_gain(table)
        # Should be positive (some information gain)
        assert actual > 0, \
            f"Three-way split should have positive IG, got {actual:.3f}"
        # Should be less than 1 (not perfect)
        assert actual < 1, \
            f"Imperfect split should have IG < 1, got {actual:.3f}"


class TestInformationGainRatio:
    """Test cases for information_gain_ratio function."""

    def test_peaty_attribute(self):
        """Test gain ratio for Peaty attribute."""
        table = np.array([[4, 0], [1, 5]])
        # IG ≈ 0.610
        # Split info: -(4/10*log2(4/10) + 6/10*log2(6/10)) ≈ 0.971
        # GR ≈ 0.610 / 0.971 ≈ 0.628
        expected = 0.628
        actual = information_gain_ratio(table)
        assert abs(actual - expected) < 0.05, \
            f"Expected GR ≈ {expected}, got {actual:.3f}"

    def test_perfect_split(self):
        """Test gain ratio for perfect split."""
        table = np.array([[5, 0], [0, 5]])
        # IG = 1.0, Split info = 1.0, GR = 1.0
        expected = 1.0
        actual = information_gain_ratio(table)
        assert abs(actual - expected) < 0.01, \
            f"Perfect balanced split should have GR=1.0, got {actual:.3f}"

    def test_many_values(self):
        """Test that gain ratio penalizes attributes with many values."""
        # Two-way split
        table_two = np.array([[3, 2], [2, 3]])
        gr_two = information_gain_ratio(table_two)

        # Five-way split with same IG
        table_five = np.array([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
        gr_five = information_gain_ratio(table_five)

        # Five-way split should have lower gain ratio (penalty for many values)
        assert gr_five < gr_two, \
            f"Many-valued attribute should have lower GR. Got {gr_five:.3f} vs {gr_two:.3f}"

    def test_zero_split_info(self):
        """Test gain ratio when split info is zero."""
        # Single attribute value (degenerate case)
        table = np.array([[5, 5]])
        # Split info = 0, should return 0.0 to avoid division by zero
        expected = 0.0
        actual = information_gain_ratio(table)
        assert actual == expected, \
            f"Zero split info should return 0.0, got {actual}"


class TestChiSquared:
    """Test cases for chi_squared function."""

    def test_peaty_attribute(self):
        """Test chi-squared for Peaty attribute."""
        table = np.array([[4, 0], [1, 5]])
        # Expected values:
        # E_11 = (4*5)/10 = 2.0, E_12 = (4*5)/10 = 2.0
        # E_21 = (6*5)/10 = 3.0, E_22 = (6*5)/10 = 3.0
        # χ² = (4-2)²/2 + (0-2)²/2 + (1-3)²/3 + (5-3)²/3
        #    = 2 + 2 + 1.333 + 1.333 = 6.667
        expected = 6.667
        actual = chi_squared(table)
        assert abs(actual - expected) < 0.01, \
            f"Expected χ² ≈ {expected}, got {actual:.3f}"

    def test_independence(self):
        """Test chi-squared when attribute and class are independent."""
        # Proportions are the same in all rows
        table = np.array([[10, 10], [10, 10]])
        # Should have χ² = 0 (perfect independence)
        expected = 0.0
        actual = chi_squared(table)
        assert abs(actual - expected) < 0.01, \
            f"Independent variables should have χ²=0, got {actual:.3f}"

    def test_perfect_association(self):
        """Test chi-squared for perfect association."""
        table = np.array([[10, 0], [0, 10]])
        # Strong association, high χ² value
        actual = chi_squared(table)
        assert actual > 15, \
            f"Perfect association should have high χ², got {actual:.3f}"

    def test_three_by_three(self):
        """Test chi-squared with 3×3 table."""
        table = np.array([[5, 3, 2], [2, 4, 4], [3, 3, 4]])
        actual = chi_squared(table)
        # Should be positive
        assert actual >= 0, \
            f"Chi-squared must be non-negative, got {actual:.3f}"


class TestChiSquaredYates:
    """Test cases for chi_squared_yates function."""

    def test_peaty_attribute_2x2(self):
        """Test Yates correction for 2×2 Peaty attribute table."""
        table = np.array([[4, 0], [1, 5]])
        # With Yates correction: |O - E| - 0.5
        # χ²_Yates = (|4-2|-0.5)²/2 + (|0-2|-0.5)²/2 +
        #            (|1-3|-0.5)²/3 + (|5-3|-0.5)²/3
        #          = 1.5²/2 + 1.5²/2 + 1.5²/3 + 1.5²/3
        #          = 1.125 + 1.125 + 0.75 + 0.75 = 3.75
        expected = 3.75
        actual = chi_squared_yates(table)
        assert abs(actual - expected) < 0.01, \
            f"Expected χ²_Yates ≈ {expected}, got {actual:.3f}"

    def test_yates_smaller_than_standard(self):
        """Test that Yates correction gives smaller value for 2×2 tables."""
        table = np.array([[4, 0], [1, 5]])
        chi2_standard = chi_squared(table)
        chi2_yates = chi_squared_yates(table)
        assert chi2_yates < chi2_standard, \
            f"Yates correction should reduce χ². Got {chi2_yates:.3f} vs {chi2_standard:.3f}"

    def test_non_2x2_returns_standard(self):
        """Test that non-2×2 tables return standard chi-squared."""
        # 2×3 table (not 2×2)
        table = np.array([[3, 2, 1], [2, 3, 4]])
        chi2_standard = chi_squared(table)
        chi2_yates = chi_squared_yates(table)
        assert abs(chi2_yates - chi2_standard) < 0.001, \
            f"Non-2×2 table should return standard χ². Got {chi2_yates:.3f} vs {chi2_standard:.3f}"

    def test_3x3_returns_standard(self):
        """Test that 3×3 tables return standard chi-squared."""
        table = np.array([[5, 3, 2], [2, 4, 4], [3, 3, 4]])
        chi2_standard = chi_squared(table)
        chi2_yates = chi_squared_yates(table)
        assert abs(chi2_yates - chi2_standard) < 0.001, \
            f"3×3 table should return standard χ². Got {chi2_yates:.3f} vs {chi2_standard:.3f}"


class TestEdgeCases:
    """Test edge cases for all functions."""

    def test_all_functions_with_zeros(self):
        """Test all functions handle zero cells correctly."""
        table = np.array([[10, 0], [0, 10]])

        ig = information_gain(table)
        gr = information_gain_ratio(table)
        chi2 = chi_squared(table)
        chi2_y = chi_squared_yates(table)

        # All should return valid (non-NaN) results
        assert not np.isnan(ig), "IG returned NaN"
        assert not np.isnan(gr), "GR returned NaN"
        assert not np.isnan(chi2), "χ² returned NaN"
        assert not np.isnan(chi2_y), "χ²_Yates returned NaN"

    def test_all_functions_with_single_row(self):
        """Test all functions with single row table."""
        table = np.array([[5, 5]])

        ig = information_gain(table)
        gr = information_gain_ratio(table)
        chi2 = chi_squared(table)
        chi2_y = chi_squared_yates(table)

        # All should return valid results (likely 0)
        assert not np.isnan(ig), "IG returned NaN"
        assert not np.isnan(gr), "GR returned NaN"
        assert not np.isnan(chi2), "χ² returned NaN"
        assert not np.isnan(chi2_y), "χ²_Yates returned NaN"


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
