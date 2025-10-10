# Experiment 03: SecBERT vs BGE-small Comparison

**Date**: October 10, 2025  
**Status**: 🔄 Planning  
**Author**: NeuralShield Team

---

## Objective

Determine if **domain-specific embeddings** (SecBERT) outperform general-purpose dense embeddings (BGE-small) for HTTP request anomaly detection.

## Hypothesis

SecBERT will significantly improve recall because:

1. **Trained on cybersecurity text**: APTnotes, threat reports, CASIE dataset
2. **Security vocabulary**: Custom wordpiece vocab for attack patterns
3. **Domain knowledge**: Should recognize SQL injection, XSS, path traversal patterns better than general models

## Comparison Matrix

| Model                | Type         | Training Data       | Dimensions | Size  |
| -------------------- | ------------ | ------------------- | ---------- | ----- |
| **SecBERT** (new)    | Cyber-BERT   | Security reports    | 768        | 84M   |
| BGE-small (baseline) | General BERT | Wikipedia, web text | 384        | 67M   |
| TF-IDF (Exp 01)      | Sparse       | Training set only   | 5000       | Small |

---

## Research Questions

1. Does domain-specific training beat general-purpose embeddings?
2. Is 768-dim SecBERT worth the extra compute vs 384-dim BGE-small?
3. Do security-specific word embeddings help detect attacks in HTTP requests?
4. How does preprocessing impact SecBERT vs BGE-small?

---

## Methodology

### Phase 1: Setup SecBERT Encoder

Need to create a new encoder wrapper since SecBERT isn't in FastEmbed:

```python
# New encoder: neuralshield/encoding/models/secbert.py
from transformers import AutoTokenizer, AutoModel
```

**Model:** `jackaduma/SecBERT`  
**Library:** HuggingFace Transformers  
**Max tokens:** 512 (standard BERT)

### Phase 2: Generate Embeddings

Use existing dump_embeddings.py with secbert encoder:

```bash
# WITHOUT preprocessing - Train
uv run python -m neuralshield.encoding.dump_embeddings \
  src/neuralshield/data/CSIC/train.jsonl \
  experiments/03_secbert_comparison/secbert_without_preprocessing/embeddings.npz \
  --encoder secbert \
  --model-name jackaduma/SecBERT \
  --use-pipeline false \
  --device cpu \
  --batch-size 32

# WITHOUT preprocessing - Test
uv run python -m neuralshield.encoding.dump_embeddings \
  src/neuralshield/data/CSIC/test.jsonl \
  experiments/03_secbert_comparison/secbert_without_preprocessing/test_embeddings.npz \
  --encoder secbert \
  --model-name jackaduma/SecBERT \
  --use-pipeline false \
  --device cpu \
  --batch-size 32

# WITH preprocessing - Train
uv run python -m neuralshield.encoding.dump_embeddings \
  src/neuralshield/data/CSIC/train.jsonl \
  experiments/03_secbert_comparison/secbert_with_preprocessing/embeddings.npz \
  --encoder secbert \
  --model-name jackaduma/SecBERT \
  --use-pipeline true \
  --device cpu \
  --batch-size 32

# WITH preprocessing - Test
uv run python -m neuralshield.encoding.dump_embeddings \
  src/neuralshield/data/CSIC/test.jsonl \
  experiments/03_secbert_comparison/secbert_with_preprocessing/test_embeddings.npz \
  --encoder secbert \
  --model-name jackaduma/SecBERT \
  --use-pipeline true \
  --device cpu \
  --batch-size 32
```

### Phase 3: Hyperparameter Search

Reuse existing `hyperparameter_search.py`:

```bash
uv run python experiments/02_dense_embeddings_comparison/hyperparameter_search.py \
  experiments/03_secbert_comparison/secbert_without_preprocessing/embeddings.npz \
  experiments/03_secbert_comparison/secbert_without_preprocessing/test_embeddings.npz \
  --max-fpr 0.05 \
  --wandb \
  --wandb-run-name "exp03-secbert-no-pipeline"
```

### Phase 4: Comparison

Compare 3 approaches:

1. **TF-IDF** (Exp 01): 0.96% recall at <0.5% FPR
2. **BGE-small** (Exp 02): 27% recall (with prep) / 13.8% (without prep) at 5% FPR
3. **SecBERT** (Exp 03): ??? recall at 5% FPR

---

## Success Criteria

### Minimum Viable Success

- ✅ SecBERT recall > BGE-small recall (>27% at FPR ≤ 5%)
- ✅ Shows measurable benefit of domain-specific training

### Strong Success

- ✅ SecBERT recall > 35% at FPR ≤ 5%
- ✅ Clear improvement across all FPR levels
- ✅ Better F1-score than BGE-small

### Exceptional Success

- ✅ SecBERT recall > 45% at FPR ≤ 5%
- ✅ Establishes SecBERT as best-in-class for WAF anomaly detection
- ✅ Justifies the extra compute cost (768 vs 384 dims)

---

## Implementation Steps

### Step 1: Create SecBERT Encoder

- [x] Create `neuralshield/encoding/models/secbert.py`
- [x] Implement `SecBERTEncoder` class
- [x] Register with encoder factory
- [ ] Test on sample requests

### Step 2: Use Existing Infrastructure

- [x] Use `dump_embeddings.py` with secbert encoder
- [x] Handle 768-dimensional embeddings
- [x] Support batch processing

### Step 3: Generate Embeddings

- [ ] Train embeddings (without preprocessing)
- [ ] Train embeddings (with preprocessing)
- [ ] Test embeddings (without preprocessing)
- [ ] Test embeddings (with preprocessing)

### Step 4: Run Hyperparameter Search

- [ ] Search without preprocessing
- [ ] Search with preprocessing
- [ ] Analyze Pareto frontiers

### Step 5: Comparative Analysis

- [ ] Create comparison plots (3-way: TF-IDF vs BGE vs SecBERT)
- [ ] Statistical significance testing
- [ ] Document findings in RESULTS.md

---

## Expected Timeline

- **Day 1**: Create SecBERT encoder + embedding script (~3 hours)
- **Day 2**: Generate embeddings (~1-2 hours, depending on GPU)
- **Day 3**: Run hyperparameter search (~30 min)
- **Day 4**: Analysis and documentation (~2 hours)

**Total**: ~7 hours active work

---

## Technical Considerations

### Compute Requirements

- **SecBERT**: 84M params, 768 dims → slower than BGE-small
- **Memory**: ~500MB model + batch embeddings
- **Speed**: Expect 2-3x slower than BGE-small

### Integration Options

**Option A**: Create dedicated `secbert.py` encoder  
**Option B**: Extend `fastembed.py` to support HuggingFace models  
**Recommendation**: Option A (cleaner, more explicit)

### Max Token Length

- SecBERT: 512 tokens (standard BERT)
- HTTP requests: Most fit, but some very long
- **Solution**: Truncate to 512 tokens (acceptable loss)

---

## Risk Mitigation

### Potential Issue: SecBERT performs worse

- **Reason**: HTTP requests != cyber threat reports
- **Mitigation**: Still valuable negative result (publish findings)

### Potential Issue: Compute too slow

- **Mitigation**: Use smaller batch sizes, run on GPU if available

### Potential Issue: 768 dims overfit with IsolationForest

- **Mitigation**: Try dimensionality reduction (PCA to 384 dims)

---

## Next Steps After Experiment

**If SecBERT wins**:

- Experiment 04: Fine-tune SecBERT on HTTP requests
- Experiment 05: Ensemble (SecBERT + BGE-small)

**If BGE-small wins**:

- Experiment 04: Try larger BGE models (base, large)
- Experiment 05: Different anomaly detectors (OCSVM, AutoEncoder)

---

## Notes

- SecBERT paper: ["SecBERT: A Pretrained Language Model for Cyber Security Text"](https://huggingface.co/jackaduma/SecBERT)
- Original repo: [jackaduma/SecBERT](https://github.com/jackaduma/SecBERT)
- Trained on: APTnotes, Stucco-Data, CASIE, SemEval-2018 Task 8

**Key insight**: This is our first attempt at using domain-specific embeddings. Even if it doesn't win, we learn whether domain specificity matters for this task.
