# COMP3222 Part 3: Experimental Evaluation

This directory contains the complete experimental framework for evaluating classifiers on categorical datasets.

## Project Structure

```
ML-learning/
├── solution/                      # Part 2: Classifier implementations
│   ├── __init__.py
│   ├── decision_stump.py         # Decision stump with categorical support
│   └── tree_ensemble.py          # Ensemble classifier
│
├── experiments/                   # Part 3: Experimental evaluation
│   ├── __init__.py
│   ├── data_loader.py            # Dataset loading and preprocessing
│   ├── evaluation.py             # Cross-validation and metrics
│   ├── rq1_native_vs_onehot.py   # RQ1 experiments
│   ├── rq2_baseline_comparison.py # RQ2 experiments
│   ├── case_study.py             # Detailed case study analysis
│   ├── visualization.py          # Plotting functions
│   ├── statistical_tests.py      # Statistical significance tests
│   ├── run_all_experiments.py    # Master experiment runner
│   └── summarize_results.py      # LaTeX table generation
│
├── results/                       # Experimental results (created by scripts)
│   ├── rq1_results.csv
│   ├── rq2_results.csv
│   ├── case_study_results.json
│   ├── rq1_table.tex
│   ├── rq2_table.tex
│   └── summary_statistics.txt
│
├── figures/                       # Generated figures (created by scripts)
│   ├── rq1_*.pdf
│   ├── rq2_*.pdf
│   └── case_study_*.pdf
│
├── data/                          # Categorical datasets (optional)
│
├── requirements.txt               # Python dependencies
└── EXPERIMENTS_README.md          # This file
```

## Research Questions

### RQ1: Native vs One-Hot Encoding
**Question:** Does TreeEnsembleClassifier perform better on average when handling categorical features natively than when used with one-hot encoding?

**Approach:**
- Compare TreeEnsembleClassifier on raw categorical data vs one-hot encoded data
- Use same hyperparameters and CV splits for fair comparison
- Evaluate on multiple categorical datasets
- Statistical significance testing with Wilcoxon signed-rank test

### RQ2: Baseline Comparison
**Question:** How does TreeEnsembleClassifier compare with scikit-learn ensemble baselines using one-hot encoding and other classifiers on categorical datasets?

**Approach:**
- Compare against RandomForest, GradientBoosting, HistGradientBoosting, AdaBoost
- Use consistent evaluation protocol across all classifiers
- Statistical testing with Friedman test + Nemenyi post-hoc
- Analyze computational efficiency

## Installation

### 1. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import sklearn; import numpy; import pandas; print('All dependencies installed!')"
```

## Running Experiments

### Quick Start: Run All Experiments
```bash
cd experiments
python run_all_experiments.py
```

This will:
1. Load categorical datasets
2. Run RQ1 experiments (Native vs One-Hot)
3. Run RQ2 experiments (Baseline comparison)
4. Perform case study on largest dataset
5. Generate statistical analyses
6. Create visualizations
7. Generate LaTeX tables

**Expected runtime:** 10-30 minutes (depending on number of datasets and hardware)

### Run Individual Experiments

#### RQ1 Only
```bash
python experiments/rq1_native_vs_onehot.py
```

#### RQ2 Only
```bash
python experiments/rq2_baseline_comparison.py
```

#### Case Study Only
```bash
python experiments/case_study.py
```

### Test Individual Components

#### Test Data Loading
```bash
python experiments/data_loader.py
```

#### Test Evaluation Framework
```bash
python experiments/evaluation.py
```

#### Test Visualizations
```bash
python experiments/visualization.py
```

## Configuration

Edit the following parameters in `run_all_experiments.py`:

```python
N_ESTIMATORS = 200  # Number of estimators for ensemble methods
CV_FOLDS = 5        # Number of cross-validation folds
RANDOM_STATE = 42   # Random seed for reproducibility
```

For faster testing, reduce `N_ESTIMATORS` to 50-100.

## Datasets

The framework automatically loads datasets from OpenML:
- Mushroom
- Tic-Tac-Toe
- Car Evaluation
- Chess (King-Rook vs King-Pawn)
- Balloons

If OpenML is unavailable, synthetic datasets are generated automatically.

### Adding Custom Datasets

To add your own categorical dataset, edit `experiments/data_loader.py`:

```python
def load_my_dataset():
    """Load custom dataset."""
    # Load your data
    X = ...  # Shape: (n_samples, n_features) with categorical values
    y = ...  # Shape: (n_samples,) with class labels
    return X, y

# Add to load_categorical_datasets():
datasets['my_dataset'] = load_my_dataset()
```

## Output Files

### Results (CSV/JSON)
- `results/rq1_results.csv`: RQ1 detailed results
- `results/rq2_results.csv`: RQ2 detailed results
- `results/case_study_results.json`: Case study analysis
- `results/summary_statistics.txt`: Overall summary

### Figures (PDF)
- `figures/rq1_paired_comparison.pdf`: Scatter plot comparing native vs one-hot
- `figures/rq1_bar_comparison.pdf`: Bar chart for RQ1
- `figures/rq1_difference_plot.pdf`: Difference visualization
- `figures/rq2_heatmap.pdf`: Heatmap of classifier performance
- `figures/rq2_boxplot.pdf`: Box plot comparison
- `figures/rq2_critical_difference.pdf`: Average rank visualization
- `figures/case_study_*.pdf`: Case study visualizations

### LaTeX Tables
- `results/rq1_table.tex`: RQ1 results table for paper
- `results/rq2_table.tex`: RQ2 results table for paper

Include these directly in your LaTeX document:
```latex
\input{results/rq1_table.tex}
```

## Customization

### Modify Classifiers (RQ2)

Edit `experiments/rq2_baseline_comparison.py`, function `create_classifiers()`:

```python
classifiers['MyClassifier'] = Pipeline([
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
    ('classifier', MyClassifier(...))
])
```

### Add New Metrics

Edit `experiments/evaluation.py`, function `evaluate_classifier()`:

```python
# Add your metric to the scoring dictionary
scoring = {
    'accuracy': 'accuracy',
    'my_metric': my_custom_scorer
}
```

### Customize Plots

Edit `experiments/visualization.py` to modify:
- Color schemes
- Figure sizes
- Font sizes
- Plot styles

## Troubleshooting

### Issue: Datasets fail to load
**Solution:** Check internet connection (for OpenML). Synthetic datasets will be created as fallback.

### Issue: Out of memory
**Solution:**
- Reduce `N_ESTIMATORS` in configuration
- Process fewer datasets at once
- Use smaller `CV_FOLDS`

### Issue: Slow execution
**Solution:**
- Reduce `N_ESTIMATORS` (try 50 or 100)
- Reduce `CV_FOLDS` (try 3)
- Comment out case study in `run_all_experiments.py`

### Issue: Import errors
**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue: sklearn warnings
**Solution:** These are usually harmless. To suppress:
```python
import warnings
warnings.filterwarnings('ignore')
```

## Code Quality

The codebase follows these principles:
- **Reproducibility:** Fixed random seeds throughout
- **No data leakage:** One-hot encoding inside CV folds
- **Comprehensive metrics:** Multiple metrics reported with confidence intervals
- **Statistical rigor:** Proper significance testing with multiple comparison correction
- **Documentation:** All functions have docstrings

## Testing

### Unit Tests
```bash
# Test each module individually
python experiments/data_loader.py
python experiments/evaluation.py
python experiments/statistical_tests.py
python experiments/visualization.py
```

### Integration Test
```bash
# Run quick version of full pipeline
# (Edit run_all_experiments.py to set N_ESTIMATORS=50, CV_FOLDS=3)
python experiments/run_all_experiments.py
```

## Performance Tips

1. **Parallel Processing:** Set `n_jobs=-1` in classifiers for multi-core usage
2. **Caching:** Results are saved to CSV - you can skip re-running completed experiments
3. **Incremental:** Run RQ1 and RQ2 separately if needed
4. **Profiling:** Use `time` command to measure execution time

## Citation

If you use this framework, please cite:

```
COMP3222 Machine Learning Coursework
Part 3: Experimental Evaluation of Classifiers on Categorical Datasets
University of Southampton, 2024
```

## Support

For issues or questions:
1. Check this README
2. Review inline documentation in Python files
3. Check error messages and stack traces
4. Verify all dependencies are installed

## License

This code is for educational purposes as part of COMP3222 coursework.

## Acknowledgments

- Datasets from UCI Machine Learning Repository via OpenML
- Scikit-learn for baseline implementations
- Matplotlib/Seaborn for visualizations
