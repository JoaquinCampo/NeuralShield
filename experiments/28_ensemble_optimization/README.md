# Experiment 28: Ensemble Optimization with Beautiful WandB Visualizations

**Purpose**: Optimize ensemble of TF-IDF+PCA+LOF and SecBERT+Mahalanobis with comprehensive WandB logging and beautiful visualizations.

**WandB Project**: `neuralshield-v2`

---

## Quick Start

```bash
cd experiments/28_ensemble_optimization
uv run python optimize_ensemble.py
```

---

## What This Experiment Does

1. **Loads Pre-Generated Embeddings**:
   - TF-IDF+PCA+LOF embeddings (from Experiment 26)
   - SecBERT+Mahalanobis embeddings (from Experiment 03)

2. **Weight Optimization Sweep**:
   - Tests 21 different fusion weights (0.0 to 1.0)
   - Logs metrics at each step (time-series in WandB)
   - Finds optimal weight for best F1-score

3. **Beautiful Visualizations**:
   - Score distribution comparison
   - ROC curves comparison
   - Agreement Venn diagrams
   - Performance comparison bars
   - Weight optimization curves

4. **Comprehensive Metrics**:
   - Component model metrics (LOF, SecBERT)
   - Ensemble metrics at optimal weight
   - Improvement over best component

---

## Prerequisites

### Required Embeddings

**LOF Embeddings** (TF-IDF+PCA) - **Converted with labels**:
- Train: `experiments/28_ensemble_optimization/csic_lof_train_embeddings.npz`
- Test: `experiments/28_ensemble_optimization/csic_lof_test_embeddings.npz`
- Model: `experiments/26_tfidf_pca_lof_csic/lof_tfidf_pca200_k15.joblib`

**SecBERT Embeddings** (already have labels):
- Train: `embeddings/SecBert/train_embeddings_compact.npz`
- Test: `embeddings/SecBert/test_embeddings_compact.npz`
- Model: `experiments/03_secbert_comparison/secbert_mahalanobis_with_preprocessing/csic_mahalanobis_model.joblib`

### Converting Embeddings (If Needed)

If embeddings don't have labels, use the conversion script:

```bash
# Convert LOF train embeddings
uv run python experiments/28_ensemble_optimization/add_labels_to_embeddings.py \
  experiments/26_tfidf_pca_lof_csic/csic_train_embeddings.npz \
  src/neuralshield/data/CSIC/train.jsonl \
  experiments/28_ensemble_optimization/csic_lof_train_embeddings.npz

# Convert LOF test embeddings
uv run python experiments/28_ensemble_optimization/add_labels_to_embeddings.py \
  experiments/26_tfidf_pca_lof_csic/csic_test_embeddings.npz \
  src/neuralshield/data/CSIC/test.jsonl \
  experiments/28_ensemble_optimization/csic_lof_test_embeddings.npz
```

---

## Usage

### Basic Run

```bash
uv run python experiments/28_ensemble_optimization/optimize_ensemble.py
```

### Custom Paths

```bash
uv run python experiments/28_ensemble_optimization/optimize_ensemble.py \
  --lof-train-embeddings path/to/lof_train.npz \
  --lof-test-embeddings path/to/lof_test.npz \
  --lof-model path/to/lof_model.joblib \
  --secbert-train-embeddings path/to/secbert_train.npz \
  --secbert-test-embeddings path/to/secbert_test.npz \
  --secbert-model path/to/mahalanobis_model.joblib
```

### Custom Parameters

```bash
uv run python experiments/28_ensemble_optimization/optimize_ensemble.py \
  --max-fpr 0.05 \
  --weight-steps 41 \
  --wandb-project neuralshield-v2 \
  --wandb-run-name "my-ensemble-run"
```

---

## Outputs

### WandB Dashboard

The experiment logs to WandB project `neuralshield-v2` with:

**Time-Series Plots**:
- `weight_optimization/recall` - Recall vs weight
- `weight_optimization/precision` - Precision vs weight
- `weight_optimization/f1_score` - F1 vs weight
- `weight_optimization/fpr` - FPR vs weight
- `weight_optimization/roc_auc` - ROC AUC vs weight

**Static Visualizations**:
- `visualizations/score_distributions` - Score distribution comparison
- `visualizations/roc_curves` - ROC curves comparison
- `visualizations/agreement_venn` - Detection overlap Venn diagrams
- `visualizations/performance_comparison` - Bar chart comparison
- `visualizations/weight_optimization` - Weight sweep curves

**Summary Metrics**:
- `best_weight` - Optimal fusion weight
- `best_recall` - Best recall achieved
- `best_f1` - Best F1-score
- `improvement_vs_best_component` - Improvement over best single model

### Local Files

- `results/results.json` - Complete results including weight sweep
- Visualizations saved to WandB (can also be saved locally if needed)

---

## Expected Results

Based on previous experiments:
- **LOF**: ~64% recall @ 5% FPR
- **SecBERT**: ~49% recall @ 5% FPR
- **Ensemble**: Expected ~80% recall @ ~9% FPR (OR logic) or optimized weight for better FPR control

---

## Design Decisions

1. **Standardization**: Scores are Z-score standardized before fusion (default: `--standardize`)
2. **Weight Range**: 0.0 (all SecBERT) to 1.0 (all LOF) in equal steps
3. **Threshold**: Set at 5% FPR on normal samples for each weight
4. **Optimization**: Maximizes F1-score (can be changed to recall if needed)

---

## See Also

- `docs/WANDB_VISUALIZATION_GUIDE.md` - Visualization design principles
- `docs/CSIC_DATASET_GUIDE.md` - CSIC dataset usage
- Experiment 25 - Agreement analysis between models
- Experiment 18 - Previous ensemble experiments

