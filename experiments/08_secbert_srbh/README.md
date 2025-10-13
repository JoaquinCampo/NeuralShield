# Experiment 08: SecBERT on SR_BH_2020 Dataset

**Status**: 🔄 Ready to Run  
**Date**: October 13, 2025

---

## Quick Summary

Testing **SecBERT** (cybersecurity-focused BERT) on the **SR_BH_2020** dataset (9x larger than CSIC) to evaluate:

1. Performance on larger-scale dataset (907K samples)
2. Domain-specific embeddings vs general-purpose (BGE-small)
3. Scalability of preprocessing pipeline

**Hypothesis**: SecBERT's security-domain pretraining may improve detection on diverse attack types.

---

## Dataset

**SR_BH_2020** (Security Research Black Hat 2020):

- **Total**: 907,815 HTTP requests
- **Train**: 100,000 valid samples
- **Test**: 807,815 samples (425K valid + 382K attacks)
- **Attack types**: 13 CAPEC categories (SQL injection, path traversal, etc.)

**Comparison to CSIC**:

- 9.3x larger dataset
- More diverse attack patterns
- More balanced distribution (58% valid vs 42% attack)

---

## Model

**SecBERT** (`jackaduma/SecBERT`):

- BERT-base pretrained on security corpora
- Output dimension: 768
- Cybersecurity-focused vocabulary
- Expected advantage on security-specific patterns

---

## Experiment Structure

### Variants

1. **Without preprocessing**: Raw HTTP requests → SecBERT
2. **With preprocessing**: Structured requests → SecBERT (recommended)

### Files

- `with_preprocessing/` - Results with full preprocessing pipeline
- `without_preprocessing/` - Results with raw HTTP text
- `EXPERIMENT_PLAN.md` - Detailed experimental design
- `RUN_EXPERIMENT.md` - Step-by-step execution guide

---

## Quick Start

See `RUN_EXPERIMENT.md` for detailed commands.

**TL;DR**:

```bash
# 1. Generate embeddings (WITH preprocessing)
uv run python -m scripts.secbert_embed \
  src/neuralshield/data/SR_BH_2020/train.jsonl \
  experiments/08_secbert_srbh/with_preprocessing/train_embeddings.npz \
  --use-pipeline --batch-size 32 --device cuda

# 2. Train model
uv run python -m scripts.train_anomaly \
  experiments/08_secbert_srbh/with_preprocessing/train_embeddings.npz \
  experiments/08_secbert_srbh/with_preprocessing/model.joblib

# 3. Test
uv run python -m scripts.secbert_embed \
  src/neuralshield/data/SR_BH_2020/test.jsonl \
  experiments/08_secbert_srbh/with_preprocessing/test_embeddings.npz \
  --use-pipeline --batch-size 32 --device cuda

uv run python -m scripts.test_anomaly_precomputed \
  experiments/08_secbert_srbh/with_preprocessing/test_embeddings.npz \
  experiments/08_secbert_srbh/with_preprocessing/model.joblib
```

---

## Success Criteria

- ✅ Complete embedding generation for 900K+ samples
- ✅ Compare performance with CSIC baseline (SecBERT achieved ~14% recall on CSIC)
- ✅ Evaluate preprocessing impact
- ✅ Assess scalability of pipeline

---

## Expected Results

Based on CSIC experiments:

- SecBERT (no prep): ~10-15% recall @ 5% FPR
- SecBERT (with prep): ~18-25% recall @ 5% FPR
- May vary due to different attack distribution in SR_BH

---

## Notes

- GPU strongly recommended (4-6 hours on CPU vs ~30-60 min on GPU)
- Test set is 8x train size - expect longer processing
- Request body excluded per project constraints
