# NeuralShield Experiments

This directory contains systematic experiments to evaluate and improve the NeuralShield anomaly detection system.

## Experiment Naming Convention

Experiments are numbered sequentially with descriptive names:
- `01_name_of_experiment/` - Brief description
- `02_next_experiment/` - Brief description

## Completed Experiments

### 01_tfidf_preprocessing_comparison

**Objective**: Determine if the preprocessing pipeline improves anomaly detection performance

**Approach**: 
- Train two IsolationForest models on TF-IDF embeddings
- One with preprocessing pipeline, one without
- Compare across contamination values: 0.1, 0.15, 0.2, 0.25, 0.3

**Key Findings**:
- ✅ Preprocessing slightly improves precision (+3-4pp) and reduces FPR
- ❌ Both models have extremely low recall (<1%)
- ⚠️ **Root cause**: TF-IDF embeddings are too sparse (99%+ zeros), causing training scores to cluster
- **Conclusion**: Cannot fairly evaluate preprocessing impact until we fix the embedding issue

**Status**: ✅ Complete

**Next Steps**: Test with dense semantic embeddings (FastEmbed/BERT)

---

## Planned Experiments

### 02_dense_embeddings_comparison (Next)

Compare TF-IDF vs dense semantic embeddings (FastEmbed) with IsolationForest

### Future Ideas

- Different anomaly detection algorithms (OCSVM, AutoEncoder, etc.)
- Supervised vs unsupervised approaches
- Feature engineering impact
- Ensemble methods
