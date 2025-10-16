# Experiment 15: LOF Comparison - Quick Start Guide

**Status**: ✅ Implementation Complete & Tested  
**Date**: October 14, 2025

---

## What This Experiment Does

Tests whether **Local Outlier Factor (LOF)** - a local density-based anomaly detector - can outperform the current best approach (**Mahalanobis distance**) for HTTP anomaly detection.

**Your Insight**: HTTP traffic forms multiple clusters (API calls, forms, static assets), and anomalies should be measured against their **nearest cluster**, not the global distribution.

---

## Quick Test (30 seconds) ✅ VERIFIED WORKING

```bash
cd /Users/joaquincamponario/Documents/Personal/neuralshield/experiments/15_lof_comparison
uv run python test_lof_quick.py
```

**Result**: LOF detector is working correctly!

---

## Run Full Experiment (Recommended)

### Option 1: Run Everything (2-3 hours)

```bash
cd /Users/joaquincamponario/Documents/Personal/neuralshield/experiments/15_lof_comparison
uv run python run_experiment.py
```

This tests:

- CSIC with preprocessing
- CSIC without preprocessing
- SR_BH with preprocessing
- SR_BH without preprocessing

Each variant runs both LOF and Mahalanobis for direct comparison.

### Option 2: Start with CSIC Only (~10 minutes)

If you want faster results, edit `run_experiment.py` and comment out SR_BH experiments (lines 252-268), then run:

```bash
uv run python run_experiment.py
```

---

## Hyperparameter Search (Optional)

Find optimal `n_neighbors` before running full experiment:

### CSIC (~5 minutes)

```bash
cd /Users/joaquincamponario/Documents/Personal/neuralshield/experiments/15_lof_comparison

uv run python hyperparameter_search.py \
    ../03_secbert_comparison/secbert_with_preprocessing/csic_train_embeddings_converted.npz \
    ../03_secbert_comparison/secbert_with_preprocessing/csic_test_embeddings_converted.npz \
    hyperparameter_search_csic \
    --n-neighbors 50 100 150 200 300 500

# View results
cat hyperparameter_search_csic/best_config.json
```

### SR_BH (~30-60 minutes)

```bash
uv run python hyperparameter_search.py \
    ../08_secbert_srbh/with_preprocessing/train_embeddings.npz \
    ../08_secbert_srbh/with_preprocessing/test_embeddings.npz \
    hyperparameter_search_srbh \
    --n-neighbors 50 100 150 200 300 500

# View results
cat hyperparameter_search_srbh/best_config.json
```

---

## View Results

### After Full Experiment

```bash
cd /Users/joaquincamponario/Documents/Personal/neuralshield/experiments/15_lof_comparison

# View comparison summary
cat comparison_summary.json | jq '.[] | {
  experiment: .lof.experiment,
  lof_recall: .lof.recall,
  maha_recall: .mahalanobis.recall,
  improvement: ((.lof.recall - .mahalanobis.recall) / .mahalanobis.recall * 100)
}'

# View specific results
cat csic_with_preprocessing/lof/results.json | jq
cat srbh_with_preprocessing/lof/results.json | jq

# Open visualizations
open csic_with_preprocessing/lof/confusion_matrix.png
open csic_with_preprocessing/lof/score_distribution.png
```

---

## Success Criteria

**Baseline (from Experiment 08):**

- CSIC with preprocessing: 49.26% recall @ 5% FPR (Mahalanobis)
- SR_BH with preprocessing: 54.12% recall @ 5% FPR (Mahalanobis)

**Target:**

- **Minimum**: LOF matches Mahalanobis (49-54%)
- **Success**: LOF improves by 2-5pp (51-57%)
- **Strong success**: LOF improves by 5pp+ (57%+)

---

## What to Expect

**Training time:**

- CSIC: ~30 seconds per detector
- SR_BH: ~10-30 seconds per detector

**Inference time:**

- CSIC (50K samples): ~5 seconds
- SR_BH (807K samples): ~30-60 seconds

**Total experiment time:**

- CSIC variants: ~5 minutes each
- SR_BH variants: ~30-45 minutes each
- **Full experiment: 2-3 hours**

---

## Files Created

```
experiments/15_lof_comparison/
├── START_HERE.md                   # This file
├── README.md                       # Overview
├── EXPERIMENT_PLAN.md              # Detailed methodology
├── IMPLEMENTATION_SUMMARY.md       # Technical details
├── RUN_COMMANDS.md                 # Complete command reference
├── test_lof_quick.py              # Quick test (verified working)
├── hyperparameter_search.py        # Find optimal n_neighbors
└── run_experiment.py               # Main experiment runner
```

**Core implementation:**

- `src/neuralshield/anomaly/lof.py` - LOF detector class

---

## Next Steps After Results

1. Analyze `comparison_summary.json`
2. Compare visualizations (confusion matrices, score distributions)
3. Update `experiments/CONSOLIDATED_RESULTS.md` with findings
4. If successful: Test on other embeddings (BGE-small, ByT5)
5. If unsuccessful: Visualize embeddings to understand why

---

## Quick Reference

```bash
# Navigate to experiment
cd /Users/joaquincamponario/Documents/Personal/neuralshield/experiments/15_lof_comparison

# Quick test
uv run python test_lof_quick.py

# Full experiment
uv run python run_experiment.py

# View results
cat comparison_summary.json | jq
```

---

## Questions?

See `RUN_COMMANDS.md` for detailed commands and troubleshooting.
