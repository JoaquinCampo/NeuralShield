#!/bin/bash
set -e

echo "=================================================="
echo "TF-IDF PCA Mahalanobis Experiment"
echo "=================================================="

# Step 1: Generate embeddings
echo ""
echo "Step 1: Generating embeddings..."
uv run experiments/10_tfidf_pca_mahalanobis/generate_embeddings.py

# Step 2: Run with preprocessing
echo ""
echo "Step 2: Running with preprocessing..."
uv run experiments/10_tfidf_pca_mahalanobis/test_pca_mahalanobis.py \
  experiments/10_tfidf_pca_mahalanobis/with_preprocessing/train_embeddings.npz \
  experiments/10_tfidf_pca_mahalanobis/with_preprocessing/test_embeddings.npz \
  experiments/10_tfidf_pca_mahalanobis/with_preprocessing \
  --n-components 300 \
  --max-fpr 0.05

# Step 3: Run without preprocessing
echo ""
echo "Step 3: Running without preprocessing..."
uv run experiments/10_tfidf_pca_mahalanobis/test_pca_mahalanobis.py \
  experiments/10_tfidf_pca_mahalanobis/without_preprocessing/train_embeddings.npz \
  experiments/10_tfidf_pca_mahalanobis/without_preprocessing/test_embeddings.npz \
  experiments/10_tfidf_pca_mahalanobis/without_preprocessing \
  --n-components 300 \
  --max-fpr 0.05

echo ""
echo "=================================================="
echo "Experiment complete!"
echo "=================================================="
echo ""
echo "Results saved in:"
echo "  - experiments/10_tfidf_pca_mahalanobis/with_preprocessing/"
echo "  - experiments/10_tfidf_pca_mahalanobis/without_preprocessing/"
