# Dataset Naming Convention

## Overview

All embedding and model files now follow a consistent naming pattern that includes the dataset identifier.

## Naming Pattern

```
<dataset>_<type>_<variant>.<ext>

dataset: csic, srbh, etc.
type: train_embeddings, test_embeddings, model, vectorizer
variant: optional (e.g., best, converted, 0.15)
ext: .npz, .joblib
```

## Examples

**CSIC Dataset Files**:

- `csic_train_embeddings.npz` - Training embeddings
- `csic_test_embeddings.npz` - Test embeddings
- `csic_model.joblib` - Trained IsolationForest model
- `csic_best_model.joblib` - Best model from hyperparameter search
- `csic_vectorizer.joblib` - TF-IDF vectorizer
- `csic_model_0.15.joblib` - Model with contamination=0.15

**Future SR_BH Dataset Files**:

- `srbh_train_embeddings.npz`
- `srbh_test_embeddings.npz`
- `srbh_model.joblib`
- etc.

## File Inventory

After renaming (October 2025):

### Experiment 01: TF-IDF

- `csic_train_embeddings.npz`
- `csic_test_embeddings.npz`
- `csic_vectorizer.joblib`
- `csic_model.joblib`
- `csic_model_{0.15,0.2,0.25,0.3}.joblib`

### Experiment 02: Dense Embeddings (BGE-small)

- `csic_train_embeddings.npz`
- `csic_test_embeddings.npz`
- `csic_model.joblib`
- `csic_best_model.joblib`

### Experiment 03: SecBERT

- `csic_train_embeddings.npz`
- `csic_test_embeddings.npz`
- `csic_train_embeddings_converted.npz` (for Mahalanobis)
- `csic_test_embeddings_converted.npz`
- `csic_best_model.joblib`
- `csic_mahalanobis_model.joblib`

### Experiment 04: ColBERT + MUVERA

- `csic_train_embeddings.npz`
- `csic_test_embeddings.npz`
- `csic_best_model.joblib`

### Experiment 05: ByT5

- `csic_train_embeddings.npz`
- `csic_test_embeddings.npz`
- `csic_best_model.joblib`

### Experiment 06: Mahalanobis

- `csic_mahalanobis_model.joblib`

## Benefits

1. **Clear provenance**: Know which dataset each file comes from
2. **Multi-dataset support**: Can have CSIC and SR_BH files side-by-side
3. **Consistency**: Uniform naming across all experiments
4. **Future-proof**: Easy to add new datasets (e.g., `httparchive_`, `custom_`)

## .gitignore

All embedding and model files remain ignored:

```gitignore
*.joblib
*.npz
```

This applies regardless of the dataset prefix.
