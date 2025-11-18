"""
Master script to run all experiments for Part 3.

This script orchestrates the complete experimental evaluation:
1. Load datasets
2. Run RQ1 experiments (Native vs One-Hot)
3. Run RQ2 experiments (Baseline comparison)
4. Run case study
5. Perform statistical analysis
6. Generate visualizations
7. Create summary tables
"""

import sys
import os
import time
import warnings

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

from experiments.data_loader import load_categorical_datasets, print_dataset_summary
from experiments.rq1_native_vs_onehot import run_rq1_experiments, analyze_rq1_results, print_rq1_analysis
from experiments.rq2_baseline_comparison import run_rq2_experiments, analyze_rq2_results
from experiments.case_study import case_study_analysis, plot_case_study_results
from experiments.statistical_tests import (comprehensive_statistical_analysis,
                                          print_statistical_analysis,
                                          perform_friedman_test,
                                          perform_nemenyi_test)
from experiments.visualization import (create_all_rq1_visualizations,
                                      create_all_rq2_visualizations)


def main():
    """
    Run all experiments for Part 3.
    """
    start_time = time.time()

    print("=" * 80)
    print("COMP3222 MACHINE LEARNING COURSEWORK")
    print("Part 3: Experimental Evaluation of Classifiers on Categorical Datasets")
    print("=" * 80)
    print()

    # Configuration
    N_ESTIMATORS = 200  # Number of estimators for ensemble methods
    CV_FOLDS = 5        # Number of cross-validation folds
    RANDOM_STATE = 42   # Random seed for reproducibility

    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    # =========================================================================
    # STEP 1: Load Datasets
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Loading Categorical Datasets")
    print("=" * 80)

    datasets = load_categorical_datasets()

    if len(datasets) == 0:
        print("\n❌ ERROR: No datasets loaded. Cannot continue.")
        return

    print_dataset_summary(datasets)

    # =========================================================================
    # STEP 2: RQ1 Experiments - Native vs One-Hot Encoding
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: RQ1 Experiments - Native vs One-Hot Encoding")
    print("=" * 80)

    rq1_start = time.time()
    rq1_results = run_rq1_experiments(
        datasets,
        n_estimators=N_ESTIMATORS,
        cv=CV_FOLDS,
        random_state=RANDOM_STATE
    )
    rq1_time = time.time() - rq1_start

    # Save RQ1 results
    rq1_results.to_csv('results/rq1_results.csv', index=False)
    print(f"\n✓ RQ1 results saved to: results/rq1_results.csv")
    print(f"  Time elapsed: {rq1_time:.2f}s")

    # Analyze RQ1 results
    rq1_analysis = analyze_rq1_results(rq1_results)
    print_rq1_analysis(rq1_analysis)

    # Statistical analysis for RQ1
    print("\n" + "-" * 80)
    print("RQ1 Statistical Analysis")
    print("-" * 80)

    rq1_stats = comprehensive_statistical_analysis(
        rq1_results,
        method_a='native',
        method_b='onehot'
    )
    print_statistical_analysis(rq1_stats)

    # Generate RQ1 visualizations
    print("\n" + "-" * 80)
    print("Generating RQ1 Visualizations")
    print("-" * 80)

    create_all_rq1_visualizations(rq1_results, output_dir='figures')
    print("✓ RQ1 visualizations saved to: figures/")

    # =========================================================================
    # STEP 3: RQ2 Experiments - Baseline Comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: RQ2 Experiments - Baseline Comparison")
    print("=" * 80)

    rq2_start = time.time()
    rq2_results = run_rq2_experiments(
        datasets,
        n_estimators=N_ESTIMATORS,
        cv=CV_FOLDS,
        random_state=RANDOM_STATE
    )
    rq2_time = time.time() - rq2_start

    # Save RQ2 results
    rq2_results.to_csv('results/rq2_results.csv', index=False)
    print(f"\n✓ RQ2 results saved to: results/rq2_results.csv")
    print(f"  Time elapsed: {rq2_time:.2f}s")

    # Analyze RQ2 results
    rq2_analysis = analyze_rq2_results(rq2_results)

    # Friedman test for RQ2
    print("\n" + "-" * 80)
    print("RQ2 Statistical Analysis - Friedman Test")
    print("-" * 80)

    friedman_results = perform_friedman_test(rq2_results, alpha=0.05)
    print(f"  Friedman statistic: {friedman_results['statistic']:.4f}")
    print(f"  P-value: {friedman_results['p_value']:.4f}")
    print(f"  Significant: {friedman_results['significant']}")
    print(f"  {friedman_results['interpretation']}")

    if friedman_results['significant']:
        print("\n  Average ranks:")
        for clf, rank in sorted(friedman_results['avg_ranks'].items(), key=lambda x: x[1]):
            print(f"    {clf:30s}: {rank:.2f}")

        # Nemenyi post-hoc test
        print("\n" + "-" * 80)
        print("Nemenyi Post-Hoc Test")
        print("-" * 80)

        nemenyi_results = perform_nemenyi_test(rq2_results, alpha=0.05)
        print(f"  Critical difference: {nemenyi_results['critical_difference']:.3f}")

        if nemenyi_results['pairwise_comparisons'] is not None:
            significant_pairs = nemenyi_results['pairwise_comparisons'][
                nemenyi_results['pairwise_comparisons']['significant']
            ]
            print(f"  Significant pairwise differences: {len(significant_pairs)}")

            if len(significant_pairs) > 0:
                print("\n  Significant pairs:")
                for _, row in significant_pairs.iterrows():
                    print(f"    {row['classifier_1']} vs {row['classifier_2']}: "
                          f"rank diff = {row['rank_difference']:.2f}")

    # Generate RQ2 visualizations
    print("\n" + "-" * 80)
    print("Generating RQ2 Visualizations")
    print("-" * 80)

    create_all_rq2_visualizations(rq2_results, output_dir='figures')
    print("✓ RQ2 visualizations saved to: figures/")

    # =========================================================================
    # STEP 4: Case Study - Detailed Analysis of One Dataset
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Case Study - Detailed Analysis")
    print("=" * 80)

    # Select largest dataset for case study
    largest_dataset = None
    largest_size = 0

    for name, (X, y) in datasets.items():
        size = X.shape[0]
        if size > largest_size:
            largest_size = size
            largest_dataset = name

    if largest_dataset is not None:
        print(f"\nSelected dataset for case study: {largest_dataset}")

        case_study_start = time.time()
        X, y = datasets[largest_dataset]

        case_study_results = case_study_analysis(
            largest_dataset, X, y,
            cv=CV_FOLDS,
            random_state=RANDOM_STATE
        )
        case_study_time = time.time() - case_study_start

        # Save case study results
        import json

        def convert_to_serializable(obj):
            import numpy as np
            import pandas as pd

            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        case_study_serializable = convert_to_serializable(case_study_results)

        with open('results/case_study_results.json', 'w') as f:
            json.dump(case_study_serializable, f, indent=2)

        print(f"\n✓ Case study results saved to: results/case_study_results.json")
        print(f"  Time elapsed: {case_study_time:.2f}s")

        # Generate case study plots
        print("\nGenerating case study visualizations...")
        plot_case_study_results(case_study_results, output_dir='figures')
        print("✓ Case study visualizations saved to: figures/")
    else:
        print("\n⚠ No suitable dataset found for case study")

    # =========================================================================
    # STEP 5: Generate Summary Tables
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Generating Summary Tables")
    print("=" * 80)

    from experiments.summarize_results import (generate_latex_table_rq1,
                                              generate_latex_table_rq2,
                                              generate_summary_statistics)

    # RQ1 LaTeX table
    print("\nGenerating RQ1 LaTeX table...")
    rq1_latex = generate_latex_table_rq1(rq1_results)
    with open('results/rq1_table.tex', 'w') as f:
        f.write(rq1_latex)
    print("✓ Saved to: results/rq1_table.tex")

    # RQ2 LaTeX table
    print("\nGenerating RQ2 LaTeX table...")
    rq2_latex = generate_latex_table_rq2(rq2_results)
    with open('results/rq2_table.tex', 'w') as f:
        f.write(rq2_latex)
    print("✓ Saved to: results/rq2_table.tex")

    # Summary statistics
    print("\nGenerating summary statistics...")
    summary = generate_summary_statistics(rq1_results, rq2_results)
    with open('results/summary_statistics.txt', 'w') as f:
        f.write(summary)
    print("✓ Saved to: results/summary_statistics.txt")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE")
    print("=" * 80)

    print(f"\nTotal execution time: {total_time:.2f}s ({total_time/60:.1f} minutes)")

    print("\n📁 Results saved to:")
    print("   - results/rq1_results.csv")
    print("   - results/rq2_results.csv")
    print("   - results/case_study_results.json")
    print("   - results/rq1_table.tex")
    print("   - results/rq2_table.tex")
    print("   - results/summary_statistics.txt")

    print("\n📊 Figures saved to:")
    print("   - figures/rq1_*.pdf")
    print("   - figures/rq2_*.pdf")
    print("   - figures/case_study_*.pdf")

    print("\n✅ All experiments completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Experiments interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
