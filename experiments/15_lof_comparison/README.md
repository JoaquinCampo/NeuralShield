# Experiment 15: Local Outlier Factor (LOF) Comparison

**Goal:** Test if local density-based anomaly detection outperforms global methods (Mahalanobis)

**Key Hypothesis:** HTTP traffic forms multiple clusters (different endpoints, user types, etc.), and local density analysis will better capture anomalies than global covariance.

---

## Motivation

Current best performance (Experiment 08):

- **SecBERT + Mahalanobis**: 54.12% recall @ 5% FPR (SR_BH)
- **SecBERT + Mahalanobis**: 49.26% recall @ 5% FPR (CSIC)

**Problem with global Mahalanobis:**

- Fits one multivariate Gaussian to ALL training data
- Assumes single cohesive cluster
- Misses local density variations

**Why LOF might be better:**

- HTTP traffic is naturally multimodal (API calls, forms, static assets, admin panels)
- Different endpoint types form distinct clusters
- Anomalies should be measured against their nearest cluster, not global distribution

---

## Experiment Structure

```
15_lof_comparison/
├── README.md                       # This file
├── EXPERIMENT_PLAN.md              # Detailed methodology
├── run_experiment.py               # Main experiment runner
├── hyperparameter_search.py        # Tune n_neighbors
├── with_preprocessing/             # LOF with preprocessing
├── without_preprocessing/          # LOF without preprocessing
├── cross_dataset_srbh_to_csic/     # Cross-dataset test
└── cross_dataset_csic_to_srbh/     # Cross-dataset test
```

---

## Quick Start

### Run Full Experiment (All Variants)

```bash
cd experiments/15_lof_comparison
uv run python run_experiment.py
```

This will test:

1. SR_BH with/without preprocessing
2. CSIC with/without preprocessing
3. Cross-dataset generalization (both directions)

### Hyperparameter Search

```bash
uv run python hyperparameter_search.py \
    --train-embeddings /path/to/train.npz \
    --test-embeddings /path/to/test.npz \
    --n-neighbors 50 100 200 500
```

---

## Key Questions

1. **Same-dataset performance:** Can LOF beat 49-54% recall?
2. **Cross-dataset generalization:** Can LOF improve on 10% recall?
3. **Optimal n_neighbors:** What's the sweet spot for HTTP traffic?
4. **Preprocessing impact:** Does preprocessing help LOF as much as Mahalanobis?

---

## Expected Results

**Best case:** LOF captures local cluster structure better → 60%+ recall

**Worst case:** kNN search overhead without performance gain → same 49-54% recall

**Most likely:** Modest improvement on same-dataset, significant improvement on cross-dataset (local patterns more transferable)

---

## Next Steps After Results

1. Compare LOF vs Mahalanobis on per-endpoint basis
2. Visualize clusters to understand where LOF helps
3. Try hybrid approach (LOF for clustered data, Mahalanobis for uniform data)
4. Test other local methods (LoOP, LOCI)
