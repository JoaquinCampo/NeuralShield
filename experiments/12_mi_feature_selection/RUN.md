# Quick Execution Guide - Experiment 12

## Step 1: Run the main experiment

```bash
uv run python experiments/12_mi_feature_selection/test_mi_secbert_mahalanobis.py
```

**Expected output**:

```
[1/4] Loading embeddings...
CSIC train: (47000, 768)
CSIC test: (50065, 768)
SR_BH attacks: (382620, 768)

[2/4] Computing MI scores...
Computing MI: 47,000 normal + 382,620 attacks
MI scores computed. Range: [0.000123, 0.045678]

[3/4] Testing different K values...
Testing k=50... Recall=0.XXXX, Precision=0.XXXX, F1=0.XXXX
Testing k=100... Recall=0.XXXX, Precision=0.XXXX, F1=0.XXXX
Testing k=200... Recall=0.XXXX, Precision=0.XXXX, F1=0.XXXX
Testing k=300... Recall=0.XXXX, Precision=0.XXXX, F1=0.XXXX
Testing k=400... Recall=0.XXXX, Precision=0.XXXX, F1=0.XXXX
Testing k=768... Recall=0.4926, Precision=0.9081, F1=0.6387

[4/4] Saving results...

SUMMARY
Baseline (all 768 dims): 0.4926 recall @ 5% FPR
Best (k=XXX): 0.XXXX recall @ 5% FPR
✅ MI selection improves by +X.X%!
```

**Time**: ~5 minutes

---

## Step 2: Analyze and visualize

```bash
uv run python experiments/12_mi_feature_selection/analyze_mi_scores.py
```

**Expected output**:

```
Analyzing MI feature selection results...
Loaded MI scores: (768,)
Loaded 6 experiment results

Generating visualizations...
Saved MI distribution plot
Saved top dimensions plot
Saved recall vs k plot

MI SCORES STATISTICS
Min:    0.000123
Max:    0.045678
Mean:   0.012345
Median: 0.009876
Std:    0.008765

Top 10 Dimensions:
  1. Dimension 542: 0.045678
  2. Dimension 123: 0.043210
  ...

✅ Analysis complete!
```

**Generates**:

- `results/plots/mi_distribution.png`
- `results/plots/top_50_dimensions.png`
- `results/plots/recall_vs_k.png`

---

## Step 3: Review results

Check `experiments/12_mi_feature_selection/results/metrics_comparison.json`:

```json
[
  {"k": 50, "recall": 0.XXXX, "precision": 0.XXXX, "f1_score": 0.XXXX},
  {"k": 100, "recall": 0.XXXX, "precision": 0.XXXX, "f1_score": 0.XXXX},
  {"k": 200, "recall": 0.XXXX, "precision": 0.XXXX, "f1_score": 0.XXXX},
  {"k": 300, "recall": 0.XXXX, "precision": 0.XXXX, "f1_score": 0.XXXX},
  {"k": 400, "recall": 0.XXXX, "precision": 0.XXXX, "f1_score": 0.XXXX},
  {"k": 768, "recall": 0.4926, "precision": 0.9081, "f1_score": 0.6387}
]
```

---

## Decision Tree

**If best_recall > 0.51** (≥3% improvement):
→ **Integrate MI into production**
→ Save `selected_dims_k100.npy` to `models/`
→ Update inference pipeline

**If best_recall ≈ 0.49** (no improvement):
→ **Use for efficiency only** (faster inference)
→ Document neutral result

**If best_recall < 0.48** (worse):
→ **Abandon MI approach**
→ Try PCA or other dimension reduction methods
