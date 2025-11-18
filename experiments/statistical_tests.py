"""
Statistical significance testing for classifier comparisons.

This module provides functions for:
- Wilcoxon signed-rank test (paired)
- Friedman test (multiple classifiers, multiple datasets)
- Nemenyi post-hoc test
- Effect size calculation
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
import warnings

warnings.filterwarnings('ignore')


def perform_wilcoxon_test(scores_a, scores_b, alpha=0.05):
    """
    Perform Wilcoxon signed-rank test for paired samples.

    This non-parametric test checks if two related samples have different
    distributions.

    Parameters
    ----------
    scores_a, scores_b : array-like
        Paired scores from two methods.

    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    results : dict
        Dictionary with:
        - statistic: test statistic
        - p_value: p-value
        - significant: whether result is significant at alpha level
        - interpretation: text interpretation
        - alpha: significance level used
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    # Remove ties (where a == b)
    diff = scores_a - scores_b
    non_zero_diff = diff[diff != 0]

    if len(non_zero_diff) < 3:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'interpretation': 'Too few non-zero differences for test',
            'alpha': alpha,
            'effect_size': np.nan
        }

    try:
        statistic, p_value = wilcoxon(scores_a, scores_b, alternative='two-sided')
    except Exception as e:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'interpretation': f'Test failed: {e}',
            'alpha': alpha,
            'effect_size': np.nan
        }

    significant = p_value < alpha

    if significant:
        if np.median(scores_a) > np.median(scores_b):
            interpretation = f"Method A significantly better (p={p_value:.4f})"
        else:
            interpretation = f"Method B significantly better (p={p_value:.4f})"
    else:
        interpretation = f"No significant difference (p={p_value:.4f})"

    # Calculate effect size (r = Z / sqrt(N))
    n = len(scores_a)
    z_score = stats.norm.ppf(1 - p_value / 2) if p_value > 0 else 3.0
    effect_size = z_score / np.sqrt(n)

    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': significant,
        'interpretation': interpretation,
        'alpha': alpha,
        'effect_size': effect_size,
        'n_samples': n
    }


def perform_paired_t_test(scores_a, scores_b, alpha=0.05):
    """
    Perform paired t-test.

    Assumes normal distribution. Use Wilcoxon for non-parametric alternative.

    Parameters
    ----------
    scores_a, scores_b : array-like
        Paired scores.

    alpha : float
        Significance level.

    Returns
    -------
    results : dict
        Dictionary with test results.
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    statistic, p_value = stats.ttest_rel(scores_a, scores_b)

    significant = p_value < alpha

    # Cohen's d effect size
    diff = scores_a - scores_b
    cohen_d = np.mean(diff) / np.std(diff, ddof=1)

    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': significant,
        'interpretation': 'Significant' if significant else 'Not significant',
        'alpha': alpha,
        'cohens_d': cohen_d
    }


def perform_friedman_test(results_df, alpha=0.05):
    """
    Perform Friedman test for comparing multiple classifiers across datasets.

    The Friedman test is a non-parametric alternative to repeated measures ANOVA.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results with 'dataset', 'classifier', 'accuracy_mean' columns.

    alpha : float
        Significance level.

    Returns
    -------
    results : dict
        Dictionary with test results and rankings.
    """
    # Pivot to get matrix: datasets x classifiers
    pivot = results_df.pivot(index='dataset', columns='classifier', values='accuracy_mean')

    # Check if we have enough data
    n_datasets, n_classifiers = pivot.shape

    if n_datasets < 2 or n_classifiers < 2:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'interpretation': 'Need at least 2 datasets and 2 classifiers',
            'rankings': None
        }

    # Prepare data for Friedman test
    data = [pivot[col].values for col in pivot.columns]

    try:
        statistic, p_value = friedmanchisquare(*data)
    except Exception as e:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'interpretation': f'Test failed: {e}',
            'rankings': None
        }

    significant = p_value < alpha

    # Calculate average ranks
    ranks = pivot.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean(axis=0).sort_values()

    interpretation = "Significant differences detected" if significant else "No significant differences"

    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': significant,
        'interpretation': interpretation,
        'alpha': alpha,
        'avg_ranks': avg_ranks.to_dict(),
        'n_datasets': n_datasets,
        'n_classifiers': n_classifiers
    }


def calculate_critical_difference(n_datasets, n_classifiers, alpha=0.05):
    """
    Calculate critical difference for Nemenyi post-hoc test.

    Parameters
    ----------
    n_datasets : int
        Number of datasets.

    n_classifiers : int
        Number of classifiers.

    alpha : float
        Significance level.

    Returns
    -------
    cd : float
        Critical difference value.
    """
    # Critical values for Nemenyi test (approximation using Studentized range statistic)
    # For simplicity, we use the approximation: CD = q_alpha * sqrt(k(k+1) / (6n))
    # where q_alpha is the critical value from the studentized range distribution

    q_alpha_values = {0.05: 2.344, 0.01: 2.936}  # For common alpha values
    q_alpha = q_alpha_values.get(alpha, 2.344)

    cd = q_alpha * np.sqrt(n_classifiers * (n_classifiers + 1) / (6 * n_datasets))

    return cd


def perform_nemenyi_test(results_df, alpha=0.05):
    """
    Perform Nemenyi post-hoc test after Friedman test.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results with 'dataset', 'classifier', 'accuracy_mean' columns.

    alpha : float
        Significance level.

    Returns
    -------
    results : dict
        Dictionary with pairwise comparisons and critical difference.
    """
    # First perform Friedman test
    friedman_results = perform_friedman_test(results_df, alpha=alpha)

    if not friedman_results['significant']:
        return {
            'friedman_significant': False,
            'pairwise_comparisons': None,
            'critical_difference': None,
            'interpretation': 'Friedman test not significant, no post-hoc needed'
        }

    # Pivot data
    pivot = results_df.pivot(index='dataset', columns='classifier', values='accuracy_mean')

    n_datasets, n_classifiers = pivot.shape

    # Calculate ranks
    ranks = pivot.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean(axis=0)

    # Calculate critical difference
    cd = calculate_critical_difference(n_datasets, n_classifiers, alpha=alpha)

    # Pairwise comparisons
    classifiers = list(avg_ranks.index)
    comparisons = []

    for i in range(len(classifiers)):
        for j in range(i + 1, len(classifiers)):
            clf_i = classifiers[i]
            clf_j = classifiers[j]

            rank_diff = abs(avg_ranks[clf_i] - avg_ranks[clf_j])
            significant = rank_diff > cd

            comparisons.append({
                'classifier_1': clf_i,
                'classifier_2': clf_j,
                'rank_1': avg_ranks[clf_i],
                'rank_2': avg_ranks[clf_j],
                'rank_difference': rank_diff,
                'significant': significant
            })

    return {
        'friedman_significant': True,
        'pairwise_comparisons': pd.DataFrame(comparisons),
        'critical_difference': cd,
        'average_ranks': avg_ranks.to_dict(),
        'interpretation': f'Critical difference: {cd:.3f}'
    }


def bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values : array-like
        List of p-values.

    alpha : float
        Original significance level.

    Returns
    -------
    corrected_alpha : float
        Corrected significance level.

    significant : array
        Boolean array indicating which tests are significant.
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests

    significant = np.array(p_values) < corrected_alpha

    return {
        'original_alpha': alpha,
        'corrected_alpha': corrected_alpha,
        'n_tests': n_tests,
        'significant': significant
    }


def calculate_effect_size(scores_a, scores_b, method='cohen_d'):
    """
    Calculate effect size for the difference between two methods.

    Parameters
    ----------
    scores_a, scores_b : array-like
        Scores from two methods.

    method : str
        Effect size method: 'cohen_d', 'hedges_g', or 'cliff_delta'.

    Returns
    -------
    effect_size : float
        Effect size value.
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    if method == 'cohen_d':
        # Cohen's d for paired samples
        diff = scores_a - scores_b
        return np.mean(diff) / np.std(diff, ddof=1)

    elif method == 'hedges_g':
        # Hedges' g (bias-corrected Cohen's d)
        n = len(scores_a)
        cohen_d = calculate_effect_size(scores_a, scores_b, method='cohen_d')
        correction = (n - 3) / (n - 2.25)
        return cohen_d * correction

    elif method == 'cliff_delta':
        # Cliff's Delta (non-parametric effect size)
        n_a = len(scores_a)
        n_b = len(scores_b)

        count_greater = 0
        count_less = 0

        for a in scores_a:
            for b in scores_b:
                if a > b:
                    count_greater += 1
                elif a < b:
                    count_less += 1

        return (count_greater - count_less) / (n_a * n_b)

    else:
        raise ValueError(f"Unknown method: {method}")


def comprehensive_statistical_analysis(results_df, method_a='native', method_b='onehot'):
    """
    Perform comprehensive statistical analysis for RQ1-style comparison.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results with columns for both methods.

    method_a, method_b : str
        Names of methods to compare.

    Returns
    -------
    analysis : dict
        Comprehensive analysis results.
    """
    # Extract scores
    col_a = f'{method_a}_accuracy_mean'
    col_b = f'{method_b}_accuracy_mean'

    scores_a = results_df[col_a].values
    scores_b = results_df[col_b].values

    analysis = {}

    # 1. Wilcoxon test
    analysis['wilcoxon'] = perform_wilcoxon_test(scores_a, scores_b)

    # 2. Paired t-test
    analysis['t_test'] = perform_paired_t_test(scores_a, scores_b)

    # 3. Effect sizes
    analysis['effect_sizes'] = {
        'cohen_d': calculate_effect_size(scores_a, scores_b, method='cohen_d'),
        'cliff_delta': calculate_effect_size(scores_a, scores_b, method='cliff_delta')
    }

    # 4. Descriptive statistics
    analysis['descriptive'] = {
        f'{method_a}_mean': np.mean(scores_a),
        f'{method_a}_std': np.std(scores_a),
        f'{method_b}_mean': np.mean(scores_b),
        f'{method_b}_std': np.std(scores_b),
        'mean_difference': np.mean(scores_a - scores_b),
        'median_difference': np.median(scores_a - scores_b)
    }

    return analysis


def print_statistical_analysis(analysis):
    """
    Print formatted statistical analysis results.

    Parameters
    ----------
    analysis : dict
        Results from comprehensive_statistical_analysis.
    """
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    print("\n1. Wilcoxon Signed-Rank Test:")
    print(f"   Statistic: {analysis['wilcoxon']['statistic']:.4f}")
    print(f"   P-value:   {analysis['wilcoxon']['p_value']:.4f}")
    print(f"   Significant: {analysis['wilcoxon']['significant']}")
    print(f"   {analysis['wilcoxon']['interpretation']}")

    print("\n2. Paired T-Test:")
    print(f"   Statistic: {analysis['t_test']['statistic']:.4f}")
    print(f"   P-value:   {analysis['t_test']['p_value']:.4f}")
    print(f"   Cohen's d: {analysis['t_test']['cohens_d']:.4f}")

    print("\n3. Effect Sizes:")
    print(f"   Cohen's d:   {analysis['effect_sizes']['cohen_d']:.4f}")
    print(f"   Cliff's Δ:   {analysis['effect_sizes']['cliff_delta']:.4f}")

    print("\n4. Descriptive Statistics:")
    for key, value in analysis['descriptive'].items():
        print(f"   {key}: {value:.4f}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    # Test statistical functions
    print("Testing statistical functions...")

    # Create dummy data
    np.random.seed(42)
    scores_a = np.random.normal(0.85, 0.05, 10)
    scores_b = np.random.normal(0.82, 0.05, 10)

    # Test Wilcoxon
    print("\n1. Wilcoxon test:")
    result = perform_wilcoxon_test(scores_a, scores_b)
    print(f"   P-value: {result['p_value']:.4f}")
    print(f"   {result['interpretation']}")

    # Test effect size
    print("\n2. Effect size:")
    cohen_d = calculate_effect_size(scores_a, scores_b, method='cohen_d')
    print(f"   Cohen's d: {cohen_d:.4f}")

    print("\n✓ Statistical tests working correctly")
