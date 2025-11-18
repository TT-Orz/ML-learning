# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Test (< 1 minute)

```bash
# Test the framework with synthetic data
python test_pipeline.py
```

This will:
- Create 2 small synthetic datasets
- Run RQ1 experiments
- Generate visualizations
- Save results to `test_results/` and `test_figures/`

## Run Full Experiments

### Option 1: Run Everything (15-30 minutes)

```bash
cd experiments
python run_all_experiments.py
```

This will run:
1. RQ1: Native vs One-Hot Encoding
2. RQ2: Baseline Comparison
3. Case Study
4. Statistical Analysis
5. Visualization Generation

Results will be saved to `results/` and `figures/`.

### Option 2: Run Individual Experiments

```bash
# RQ1 only (5-10 minutes)
python experiments/rq1_native_vs_onehot.py

# RQ2 only (10-15 minutes)
python experiments/rq2_baseline_comparison.py

# Case study only (3-5 minutes)
python experiments/case_study.py
```

## Faster Testing

For quicker testing, edit `experiments/run_all_experiments.py` and change:

```python
N_ESTIMATORS = 50   # Instead of 200
CV_FOLDS = 3        # Instead of 5
```

## View Results

### CSV Results
```bash
# View RQ1 results
cat results/rq1_results.csv

# View RQ2 results
cat results/rq2_results.csv

# View summary
cat results/summary_statistics.txt
```

### Figures
```bash
# View generated figures
ls figures/

# Open a figure (Linux with display)
xdg-open figures/rq1_paired_comparison.pdf
```

### LaTeX Tables
```bash
# View LaTeX tables for your paper
cat results/rq1_table.tex
cat results/rq2_table.tex
```

## Typical Workflow

1. **Test the framework:**
   ```bash
   python test_pipeline.py
   ```

2. **Run full experiments:**
   ```bash
   python experiments/run_all_experiments.py
   ```

3. **Check results:**
   ```bash
   cat results/summary_statistics.txt
   ```

4. **View figures:**
   ```bash
   ls figures/
   ```

5. **Include tables in your paper:**
   ```latex
   \input{results/rq1_table.tex}
   \input{results/rq2_table.tex}
   ```

## Troubleshooting

### ImportError: No module named 'X'
```bash
pip install -r requirements.txt
```

### Slow execution
Reduce `N_ESTIMATORS` and `CV_FOLDS` in the main script.

### Out of memory
Process datasets individually or reduce ensemble size.

### Figures not generating
Check that `matplotlib` and `seaborn` are installed:
```bash
pip install matplotlib seaborn
```

## File Locations

- **Source code:** `solution/` and `experiments/`
- **Results:** `results/*.csv`, `results/*.json`
- **Figures:** `figures/*.pdf`
- **LaTeX tables:** `results/*.tex`

## Next Steps

1. Review the experimental results
2. Analyze the figures
3. Interpret statistical tests
4. Write your conclusions
5. Include tables and figures in your report

For detailed documentation, see `EXPERIMENTS_README.md`.
