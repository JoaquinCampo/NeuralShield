# Experiment 02: Dense Embeddings Comparison

**Status**: 🔄 Ready to Run  
**Date**: September 30, 2025

---

## Quick Summary

Testing if **dense semantic embeddings** (FastEmbed/BERT-like) improve anomaly detection over sparse TF-IDF embeddings.

**Why**: Experiment 01 showed TF-IDF has extremely low recall (<1%) due to sparse, concentrated embeddings.

**Hypothesis**: Dense embeddings will achieve >10% recall (vs <1% with TF-IDF).

---

## Files

- `EXPERIMENT_PLAN.md` - Detailed experimental design and methodology
- `RUN_EXPERIMENT.md` - Step-by-step execution guide
- `analyze_scores.py` - Training score distribution analysis tool

---

## Quick Start

```bash
# See detailed instructions in RUN_EXPERIMENT.md

# 1. Generate embeddings (~10-15 min)
uv run python -m scripts.fastembed_embed ...

# 2. Train models (~1 min)
uv run python -m scripts.train_anomaly ...

# 3. Analyze scores (~10 sec)
uv run python experiments/02_dense_embeddings_comparison/analyze_scores.py ...

# 4. Test models (~5-10 min)
uv run python -m scripts.test_anomaly ...
```

---

## Success Criteria

- ✅ Recall > 10% (minimum viable)
- ✅ Training score std dev > 0.01
- ✅ F1-Score > 15%

---

## Results

_To be filled after experiment completion_
