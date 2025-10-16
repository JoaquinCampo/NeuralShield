# How to Run Experiment 17: GMM + SecBERT

## Quick Start

```bash
# Run full experiment
./experiments/17_gmm_secbert/run_experiment.sh
```

This will:

1. Compare n_components from 1 to 10
2. Test best configurations (1, 3, 5 components)
3. Generate visualizations and results

## Individual Scripts

### Compare Components

Test different numbers of Gaussian components:

```bash
uv run experiments/17_gmm_secbert/compare_components.py \
  experiments/03_secbert_comparison/secbert_with_preprocessing/csic_train_embeddings_converted.npz \
  experiments/03_secbert_comparison/secbert_with_preprocessing/csic_test_embeddings_converted.npz \
  experiments/17_gmm_secbert/component_comparison \
  --min-components 1 \
  --max-components 10 \
  --max-fpr 0.05
```

Output:

- `component_comparison.json`: Metrics for each n_components
- `component_comparison.png`: Visualization comparing all configurations

### Test Specific Configuration

Run GMM with specific hyperparameters:

```bash
uv run experiments/17_gmm_secbert/test_gmm_secbert.py \
  experiments/03_secbert_comparison/secbert_with_preprocessing/csic_train_embeddings_converted.npz \
  experiments/03_secbert_comparison/secbert_with_preprocessing/csic_test_embeddings_converted.npz \
  experiments/17_gmm_secbert/gmm_3components \
  --n-components 3 \
  --covariance-type full \
  --max-fpr 0.05
```

Output:

- `results.json`: Performance metrics
- `gmm_detector.joblib`: Trained model
- `score_distribution.png`: Score distributions
- `confusion_matrix.png`: Confusion matrix

## Expected Runtime

- Component comparison (1-10): ~5-10 min
- Single configuration test: ~1-2 min

## Baseline Comparison

From Experiment 03 (SecBERT + Mahalanobis):

- Recall: 49.26%
- Precision: 90.81%
- F1-Score: 63.87%
- FPR: 5.00%

Goal: Check if GMM (multi-modal) improves over single-Gaussian Mahalanobis.
