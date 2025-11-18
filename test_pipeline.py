"""
Quick test of the experimental pipeline.

This script runs a mini version of the experiments for testing purposes.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.data_loader import create_synthetic_categorical_dataset
from experiments.rq1_native_vs_onehot import run_rq1_experiments
from experiments.visualization import create_all_rq1_visualizations

print("=" * 80)
print("QUICK PIPELINE TEST")
print("=" * 80)

# Create small test datasets
print("\n1. Creating test datasets...")
datasets = {
    'test_small': create_synthetic_categorical_dataset(100, 5, 2, 42),
    'test_medium': create_synthetic_categorical_dataset(150, 6, 3, 43),
}
print(f"   Created {len(datasets)} test datasets")

# Run mini RQ1 experiment
print("\n2. Running mini RQ1 experiment...")
print("   (Using n_estimators=20, cv=3 for speed)")

results = run_rq1_experiments(
    datasets,
    n_estimators=20,
    cv=3,
    random_state=42
)

print("\n3. Results:")
print(results[['dataset', 'native_accuracy_mean', 'onehot_accuracy_mean', 'difference', 'winner']])

# Save results
os.makedirs('test_results', exist_ok=True)
results.to_csv('test_results/test_rq1_results.csv', index=False)
print("\n   Results saved to: test_results/test_rq1_results.csv")

# Generate visualizations
print("\n4. Generating visualizations...")
os.makedirs('test_figures', exist_ok=True)
create_all_rq1_visualizations(results, output_dir='test_figures')
print("   Figures saved to: test_figures/")

print("\n" + "=" * 80)
print("✅ PIPELINE TEST SUCCESSFUL!")
print("=" * 80)
print("\nThe experimental framework is working correctly.")
print("To run full experiments, use: python experiments/run_all_experiments.py")
