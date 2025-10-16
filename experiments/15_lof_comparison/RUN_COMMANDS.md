# Experiment 15: LOF Comparison - Run Commands

**Date**: October 14, 2025  
**Status**: Ready to Execute

---

## Quick Test (Recommended First Step)

Test LOF detector on a small subset to verify everything works:

```bash
cd experiments/15_lof_comparison
uv run python test_lof_quick.py
```

**Expected output:**

- LOF and Mahalanobis both train successfully
- Scores are computed
- Quick recall comparison printed
- Should complete in ~30 seconds

---

## Hyperparameter Search (Optional but Recommended)

Find optimal `n_neighbors` for each dataset:

### CSIC Hyperparameter Search

```bash
cd experiments/15_lof_comparison

uv run python hyperparameter_search.py \
    ../03_secbert_comparison/secbert_with_preprocessing/csic_train_embeddings_converted.npz \
    ../03_secbert_comparison/secbert_with_preprocessing/csic_test_embeddings_converted.npz \
    hyperparameter_search_csic \
    --n-neighbors 50 100 150 200 300 500
```

**Expected time:** ~5-10 minutes  
**Output:** `hyperparameter_search_csic/best_config.json`

### SR_BH Hyperparameter Search

```bash
cd experiments/15_lof_comparison

uv run python hyperparameter_search.py \
    ../08_secbert_srbh/with_preprocessing/train_embeddings.npz \
    ../08_secbert_srbh/with_preprocessing/test_embeddings.npz \
    hyperparameter_search_srbh \
    --n-neighbors 50 100 150 200 300 500
```

**Expected time:** ~30-60 minutes (larger dataset)  
**Output:** `hyperparameter_search_srbh/best_config.json`

### View Results

```bash
# Best configuration for CSIC
cat hyperparameter_search_csic/best_config.json

# Best configuration for SR_BH
cat hyperparameter_search_srbh/best_config.json

# All results
cat hyperparameter_search_csic/hyperparameter_results.json | jq '.[] | {n_neighbors, recall, f1}'
```

---

## Full Experiment (Main Run)

Run complete LOF vs Mahalanobis comparison on all variants:

```bash
cd experiments/15_lof_comparison
uv run python run_experiment.py
```

**This will test:**

1. CSIC with preprocessing
2. CSIC without preprocessing
3. SR_BH with preprocessing
4. SR_BH without preprocessing

**Expected time:**

- CSIC variants: ~5 minutes each
- SR_BH variants: ~30-45 minutes each
- **Total: ~2-3 hours**

**Output structure:**

```
15_lof_comparison/
├── csic_with_preprocessing/
│   ├── lof/
│   │   ├── results.json
│   │   ├── confusion_matrix.png
│   │   └── score_distribution.png
│   └── mahalanobis/
│       └── [same files]
├── csic_without_preprocessing/
│   └── [same structure]
├── srbh_with_preprocessing/
│   └── [same structure]
├── srbh_without_preprocessing/
│   └── [same structure]
└── comparison_summary.json
```

---

## Analyze Results

### View Comparison Summary

```bash
cd experiments/15_lof_comparison

# Pretty print comparison
cat comparison_summary.json | jq '.[] | {
  experiment: .lof.experiment,
  lof_recall: .lof.recall,
  maha_recall: .mahalanobis.recall,
  improvement: ((.lof.recall - .mahalanobis.recall) / .mahalanobis.recall * 100)
}'
```

### View Specific Results

```bash
# CSIC with preprocessing - LOF
cat csic_with_preprocessing/lof/results.json | jq '{recall, precision, f1_score, accuracy}'

# CSIC with preprocessing - Mahalanobis
cat csic_with_preprocessing/mahalanobis/results.json | jq '{recall, precision, f1_score, accuracy}'

# SR_BH with preprocessing - LOF
cat srbh_with_preprocessing/lof/results.json | jq '{recall, precision, f1_score, accuracy}'
```

### View Visualizations

```bash
# Open confusion matrices
open csic_with_preprocessing/lof/confusion_matrix.png
open csic_with_preprocessing/mahalanobis/confusion_matrix.png

# Open score distributions
open csic_with_preprocessing/lof/score_distribution.png
open srbh_with_preprocessing/lof/score_distribution.png
```

---

## Expected Results

### Success Criteria

**Minimum viable:**

- LOF recall: 49-54% (matches Mahalanobis)
- No errors, clean execution

**Success:**

- LOF recall: 51-57% (+2-5pp improvement)
- OR significant cross-dataset improvement

**Strong success:**

- LOF recall: 57%+ (+5pp+ improvement)
- Clear evidence of local density benefits

### Baseline Comparison (from Experiment 08)

| Dataset | Variant               | Mahalanobis Recall | Target LOF Recall |
| ------- | --------------------- | ------------------ | ----------------- |
| CSIC    | With preprocessing    | 49.26%             | >51%              |
| CSIC    | Without preprocessing | 43.69%             | >46%              |
| SR_BH   | With preprocessing    | 54.12%             | >56%              |
| SR_BH   | Without preprocessing | 48.18%             | >50%              |

---

## Troubleshooting

### If test_lof_quick.py fails:

```bash
# Check if LOF is properly registered
uv run python -c "from neuralshield.anomaly import LOFDetector; print('✅ LOF imported')"

# Check embeddings exist
ls -lh ../03_secbert_comparison/secbert_with_preprocessing/*.npz
ls -lh ../08_secbert_srbh/with_preprocessing/*.npz
```

### If hyperparameter search is slow:

- Reduce n_neighbors range: `--n-neighbors 100 200`
- Use smaller subset (edit script to use `train_embeddings[:10000]`)

### If full experiment is too slow:

- Comment out SR_BH experiments in `run_experiment.py`
- Run CSIC only first (faster, ~10 minutes total)

### Memory issues:

- SR_BH test set is large (807K samples)
- Ensure at least 16GB RAM available
- Close other applications

---

## Next Steps After Results

1. **Analyze comparison_summary.json**
2. **Update CONSOLIDATED_RESULTS.md** with findings
3. **If successful:** Test on other embeddings (BGE-small, ByT5)
4. **If unsuccessful:** Visualize embeddings to understand why
5. **Document insights** in experiment README

---

## Monitoring Progress

```bash
# Watch logs in real-time
tail -f experiments/15_lof_comparison/experiment.log

# Check which experiments completed
ls -la experiments/15_lof_comparison/*/lof/results.json
```

---

## Quick Reference

```bash
# Navigate to experiment
cd experiments/15_lof_comparison

# Quick test (30 seconds)
uv run python test_lof_quick.py

# Hyperparameter search CSIC (~5 min)
uv run python hyperparameter_search.py \
    ../03_secbert_comparison/secbert_with_preprocessing/csic_train_embeddings.npz \
    ../03_secbert_comparison/secbert_with_preprocessing/csic_test_embeddings.npz \
    hyperparameter_search_csic

# Full experiment (2-3 hours)
uv run python run_experiment.py

# View results
cat comparison_summary.json | jq
```
