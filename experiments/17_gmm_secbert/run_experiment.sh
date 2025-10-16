#!/bin/bash
# Run GMM + SecBERT experiment

set -e

TRAIN_EMB="experiments/03_secbert_comparison/secbert_with_preprocessing/csic_train_embeddings_converted.npz"
TEST_EMB="experiments/03_secbert_comparison/secbert_with_preprocessing/csic_test_embeddings_converted.npz"

echo "========================================"
echo "GMM + SecBERT Experiment"
echo "========================================"
echo ""

# First, compare different numbers of components
echo "Step 1: Compare different n_components (1-10)..."
uv run experiments/17_gmm_secbert/compare_components.py \
  $TRAIN_EMB \
  $TEST_EMB \
  experiments/17_gmm_secbert/component_comparison \
  --min-components 1 \
  --max-components 10 \
  --max-fpr 0.05

echo ""
echo "Step 2: Test best configurations..."

# Test a few specific configurations based on common patterns
for n_comp in 1 3 5; do
  echo ""
  echo "Testing n_components=$n_comp..."
  uv run experiments/17_gmm_secbert/test_gmm_secbert.py \
    $TRAIN_EMB \
    $TEST_EMB \
    experiments/17_gmm_secbert/gmm_${n_comp}components \
    --n-components $n_comp \
    --covariance-type full \
    --max-fpr 0.05
done

echo ""
echo "========================================"
echo "Experiment complete!"
echo "========================================"
echo ""
echo "Results saved in experiments/17_gmm_secbert/"
echo ""
echo "Baseline (Mahalanobis): Recall=49.26%, F1=63.87%"
echo "Check component_comparison.json for GMM results"

