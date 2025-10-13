# Experiment 10: TF-IDF with PCA Dimensionality Reduction + Mahalanobis

## Overview

This experiment tests Mahalanobis distance detection on TF-IDF embeddings with PCA dimensionality reduction to 300 components.

## Hypothesis

- Reducing TF-IDF dimensions from thousands to 300 using PCA may improve Mahalanobis detection
- Lower dimensionality could reduce noise and computational cost while preserving signal

## Method

1. Generate TF-IDF embeddings (with and without preprocessing)
2. Apply PCA to reduce to 300 dimensions
3. Fit Mahalanobis (Empirical Covariance) on reduced training embeddings
4. Evaluate on test set with 5% FPR constraint

## Running the Experiment

```bash
# Generate embeddings
uv run experiments/10_tfidf_pca_mahalanobis/generate_embeddings.py

# Run with preprocessing
uv run experiments/10_tfidf_pca_mahalanobis/test_pca_mahalanobis.py \
  experiments/10_tfidf_pca_mahalanobis/with_preprocessing/train_embeddings.npz \
  experiments/10_tfidf_pca_mahalanobis/with_preprocessing/test_embeddings.npz \
  experiments/10_tfidf_pca_mahalanobis/with_preprocessing \
  --n-components 300 \
  --max-fpr 0.05

# Run without preprocessing
uv run experiments/10_tfidf_pca_mahalanobis/test_pca_mahalanobis.py \
  experiments/10_tfidf_pca_mahalanobis/without_preprocessing/train_embeddings.npz \
  experiments/10_tfidf_pca_mahalanobis/without_preprocessing/test_embeddings.npz \
  experiments/10_tfidf_pca_mahalanobis/without_preprocessing \
  --n-components 300 \
  --max-fpr 0.05
```

## Results

Results will be saved in:

- `with_preprocessing/results.json`
- `without_preprocessing/results.json`
- Visualizations: `score_distribution.png`, `confusion_matrix.png`
