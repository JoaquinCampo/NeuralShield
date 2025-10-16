# Experiment 17: GMM + SecBERT

## Objective

Test Gaussian Mixture Model (GMM) as an anomaly detector with SecBERT embeddings and compare with existing Mahalanobis results.

## Hypothesis

GMM should perform better than single-distribution Mahalanobis when:

- Normal traffic has multiple distinct clusters or modes
- The data is naturally multi-modal rather than unimodal

## Approach

1. Use precomputed SecBERT embeddings from experiment 03
2. Fit GMM with varying numbers of components (1-10)
3. Use negative log-likelihood as anomaly score
4. Compare against Mahalanobis baseline from experiment 03

## Key Differences: GMM vs Mahalanobis

| Aspect                  | Mahalanobis                   | GMM                           |
| ----------------------- | ----------------------------- | ----------------------------- |
| Distribution assumption | Single Gaussian               | Mixture of K Gaussians        |
| Anomaly score           | Squared distance to centroid  | Negative log-likelihood       |
| Parameters              | None (just mean + covariance) | n_components, covariance_type |
| Best for                | Unimodal normal data          | Multi-modal normal data       |

## Scripts

- `test_gmm_secbert.py`: Test GMM with specific configuration
- `compare_components.py`: Compare different numbers of components (1-10)

## Expected Outcomes

1. Optimal number of components identified via BIC/AIC
2. Performance comparison showing whether multi-modal modeling helps
3. Decision on whether GMM is worth the added complexity over Mahalanobis

## Usage

```bash
# Compare different numbers of components
uv run experiments/17_gmm_secbert/compare_components.py \
  experiments/03_secbert_comparison/secbert_with_preprocessing/csic_train_embeddings_converted.npz \
  experiments/03_secbert_comparison/secbert_with_preprocessing/csic_test_embeddings_converted.npz \
  experiments/17_gmm_secbert/component_comparison

# Test specific configuration
uv run experiments/17_gmm_secbert/test_gmm_secbert.py \
  experiments/03_secbert_comparison/secbert_with_preprocessing/csic_train_embeddings_converted.npz \
  experiments/03_secbert_comparison/secbert_with_preprocessing/csic_test_embeddings_converted.npz \
  experiments/17_gmm_secbert/gmm_3components \
  --n-components 3 \
  --covariance-type full
```

## Results

See individual result subdirectories and `RESULTS.md` after running experiments.
