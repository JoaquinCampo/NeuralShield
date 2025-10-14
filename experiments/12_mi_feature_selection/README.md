# Experiment 12: MI-Based Feature Selection

**Status**: Ready to run  
**Date**: October 14, 2025

---

## Quick Summary

Tests whether **mutual information (MI)** based dimension selection improves SecBERT + Mahalanobis detection performance.

**Core Idea**: Use MI on generic attacks (SR_BH) to identify which of SecBERT's 768 dimensions best discriminate attacks from normal requests, then train Mahalanobis detector using only those dimensions.

---

## Hypothesis

**H1**: Not all 768 SecBERT dimensions are equally useful for attack detection.

**H2**: Selecting top-K dimensions by MI will improve recall while reducing noise and computational cost.

**H3**: 100-200 dimensions might outperform all 768 dimensions.

---

## Methodology

### Data Flow

```
CSIC Normal (47K)  ──┐
                     ├──> Compute MI Scores (768 values)
SR_BH Attacks (382K) ┘

                     ↓

              Select Top K Dims
              (e.g., K=100)

                     ↓

CSIC Normal (47K)    ──> Train Mahalanobis (on K dims only)

                     ↓

CSIC Test (50K)      ──> Evaluate
```

### Key Insight

**MI computation** uses attack data to find discriminative dimensions, but **detector training** uses only normal data (pure anomaly detection).

---

## Quick Start

### Run Experiment

```bash
# Main experiment (5-10 minutes)
uv run python experiments/12_mi_feature_selection/test_mi_secbert_mahalanobis.py

# Analyze results
uv run python experiments/12_mi_feature_selection/analyze_mi_scores.py
```

### What Gets Tested

| K   | Description                |
| --- | -------------------------- |
| 50  | Minimal dimensionality     |
| 100 | Paper's optimal for TF-IDF |
| 200 | Medium reduction           |
| 300 | Light reduction            |
| 400 | Minimal reduction          |
| 768 | Baseline (all dims, no MI) |

---

## Expected Outcomes

### Scenario A: MI Helps

```
k=100: 55% recall ← Better than baseline!
k=768: 49% recall
```

**Conclusion**: Integrate MI dimension selection into production.

### Scenario B: MI Doesn't Help

```
k=100: 45% recall ← Worse than baseline
k=768: 49% recall
```

**Conclusion**: SecBERT's 768 dimensions are already optimal, skip MI.

### Scenario C: MI Equals Baseline

```
k=100: 49% recall ← Same as baseline
k=768: 49% recall
```

**Conclusion**: No benefit, but faster inference (use k=100 for speed).

---

## Files Generated

```
results/
├── mi_scores.npy                   # (768,) MI score per dimension
├── selected_dims_k50.npy           # Top 50 dimension indices
├── selected_dims_k100.npy          # Top 100 dimension indices
├── selected_dims_k200.npy          # Top 200 dimension indices
├── metrics_comparison.json         # Performance for all K values
└── plots/
    ├── mi_distribution.png         # Histogram of MI scores
    ├── top_50_dimensions.png       # Bar chart of top dims
    └── recall_vs_k.png             # Performance curve
```

---

## Comparison to Paper

**Paper's approach** (sparse features):

- TF-IDF (5000 tokens) → MI → Select 100 tokens → BoW + One-Class SVM
- Result: 78-92% TPR @ 2-5% FPR

**Our approach** (dense features):

- SecBERT (768 dims) → MI → Select 100 dims → Mahalanobis
- Result: TBD

**Key difference**: Applying MI to dense semantic embeddings (novel).

---

## Success Criteria

- ✅ Successfully compute MI scores for all 768 dimensions
- ✅ Test multiple K values (50, 100, 200, 300, 400, 768)
- ✅ Generate comparison metrics and visualizations
- ✅ Determine if MI selection improves over baseline
- ✅ If successful (≥3% improvement), integrate into production pipeline

---

## Time Estimate

| Task            | Time      |
| --------------- | --------- |
| Load embeddings | 5 sec     |
| Compute MI      | 30 sec    |
| Test 6 K values | 3 min     |
| Generate plots  | 30 sec    |
| **Total**       | **5 min** |

All embeddings are pre-computed from Experiments 03 & 08.

---

## Next Steps

**If MI helps (≥3% improvement)**:

1. Create production `MIFeatureSelector` class
2. Save selected dimensions to `models/`
3. Update inference pipeline
4. Test on SR_BH dataset (cross-validation)

**If MI doesn't help**:

1. Document negative result
2. Investigate why (all dims useful? MI doesn't work on dense embeddings?)
3. Try alternative: PCA-based dimensionality reduction
