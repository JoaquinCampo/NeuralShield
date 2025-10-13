# Experiment 06: Mahalanobis Distance

## Objective

Test if Mahalanobis distance (via sklearn's `EmpiricalCovariance`) outperforms IsolationForest.

## Why Mahalanobis?

Uses empirical covariance to compute statistical distance:

- Accounts for feature correlations (IsolationForest doesn't)
- Fast and stable (no iterative robust estimation)
- Battle-tested sklearn implementation
- Zero hyperparameters to tune

## Baseline to Beat

**BGE-small + IsolationForest:** 27-28% recall @ 5% FPR

## Quick Start

### With Preprocessing

```bash
uv run python experiments/06_mahalanobis_comparison/test_mahalanobis.py \
  experiments/02_dense_embeddings_comparison/dense_with_preprocessing/embeddings.npz \
  experiments/02_dense_embeddings_comparison/dense_with_preprocessing/test_embeddings.npz \
  experiments/06_mahalanobis_comparison/with_preprocessing \
  --max-fpr 0.05 \
  --wandb \
  --wandb-run-name "mahalanobis-bge-with-prep"
```

### Without Preprocessing

```bash
uv run python experiments/06_mahalanobis_comparison/test_mahalanobis.py \
  experiments/02_dense_embeddings_comparison/dense_without_preprocessing/embeddings.npz \
  experiments/02_dense_embeddings_comparison/dense_without_preprocessing/test_embeddings.npz \
  experiments/06_mahalanobis_comparison/without_preprocessing \
  --max-fpr 0.05 \
  --wandb \
  --wandb-run-name "mahalanobis-bge-no-prep"
```

## Expected Results

| Scenario  | Target | vs IsolationForest |
| --------- | ------ | ------------------ |
| Minimum   | 27%    | Match baseline     |
| Good      | 35%    | +30%               |
| Excellent | 40%+   | +48%               |
