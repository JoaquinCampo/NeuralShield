# Experiment 05: ByT5 Byte-Level Embeddings for HTTP Anomaly Detection

**Date**: October 10, 2025  
**Status**: 🚀 Ready to Execute  
**Author**: NeuralShield Team

---

## Objective

Determine if **byte-level tokenization** (ByT5) outperforms word-piece tokenization (BGE, SecBERT, ColBERT) for HTTP request anomaly detection.

## Hypothesis

ByT5 will achieve the highest recall because:

1. **Byte-level tokenization preserves attack patterns**

   - Single-character edits matter: `%27` vs `%28`
   - SQL injection markers: `';--`, `' OR '1'='1`
   - Path traversal: `../` vs `..%2f`
   - Unicode tricks, null bytes, CRLF visible

2. **No preprocessing needed**

   - Raw strings preserve attack signatures
   - No tokenization artifacts
   - No vocabulary limitations

3. **Mean+Max pooling captures anomalies**

   - Mean: Global request context
   - Max: Rare spikes (attack indicators)
   - Both critical for detection

4. **Proven T5 architecture**
   - Strong semantic understanding
   - Well-studied, reliable
   - Not ONNX-dependent (unlike ColBERT)

## Comparison Matrix

| Model                   | Tokenization | Vectors | Dimensions | Preprocessing | Complexity |
| ----------------------- | ------------ | ------- | ---------- | ------------- | ---------- |
| **ByT5** (new)          | Byte-level   | 1       | 2944       | None needed   | Low        |
| ColBERT+MUVERA (Exp 04) | Word-piece   | N→1     | 10,240     | Optional      | High       |
| SecBERT (Exp 03)        | Word-piece   | 1       | 768        | Optional      | Medium     |
| BGE-small (Exp 02)      | Word-piece   | 1       | 384        | Optional      | Low        |
| TF-IDF (Exp 01)         | Word         | 1       | 5000       | No            | Very Low   |

---

## Research Questions

1. Does byte-level tokenization beat word-piece for HTTP requests?
2. Is "no preprocessing" better than preprocessing for attack detection?
3. Can mean+max pooling capture anomalies as well as complex aggregations?
4. Is ByT5's 2944 dims a good middle ground (vs BGE's 384 or ColBERT's 10,240)?

---

## Methodology

### Phase 1: ByT5 Encoder Implementation

**Architecture:**

```python
class ByT5Encoder:
    def encode(self, request: str):
        # 1. Byte-level tokenization (preserves all characters)
        tokens = tokenizer(request, max_length=1024)

        # 2. T5 encoder (1472 hidden dims)
        hidden_states = model(**tokens).last_hidden_state

        # 3. Mean pooling (global context)
        mean_pool = hidden_states.mean(dim=1)  # (1472,)

        # 4. Max pooling (rare spikes)
        max_pool = hidden_states.max(dim=1)    # (1472,)

        # 5. Concatenate
        combined = cat([mean_pool, max_pool])  # (2944,)

        # 6. L2 normalize (sensitive to differences)
        return normalize(combined)
```

**Key Design Decisions:**

- **No [CLS] token**: ByT5 doesn't have one; mean+max is better anyway
- **Max length 1024**: Handles most requests; truncate longer ones
- **L2 normalization**: Makes embeddings sensitive to subtle differences
- **No preprocessing**: Raw strings preserve attack patterns

### Phase 2: Generate Embeddings

**Without Preprocessing:**

```bash
# Train (36k samples)
python src/scripts/byt5_embed.py \
  src/neuralshield/data/CSIC/train.jsonl \
  experiments/05_byt5_comparison/without_preprocessing/train_embeddings.npz \
  --batch-size 64 --device cuda

# Test (19k samples)
python src/scripts/byt5_embed.py \
  src/neuralshield/data/CSIC/test.jsonl \
  experiments/05_byt5_comparison/without_preprocessing/test_embeddings.npz \
  --batch-size 64 --device cuda
```

**With Preprocessing:**

```bash
# Same commands with --use-pipeline flag
```

**Expected Time:** ~10-15 min total on A100

### Phase 3: Hyperparameter Search

Reuse existing infrastructure:

```bash
python experiments/02_dense_embeddings_comparison/hyperparameter_search.py \
  experiments/05_byt5_comparison/without_preprocessing/train_embeddings.npz \
  experiments/05_byt5_comparison/without_preprocessing/test_embeddings.npz \
  --max-fpr 0.05 --wandb --wandb-run-name "byt5-no-prep"
```

**Search Space:** 72 configurations

- contamination: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
- n_estimators: [100, 200, 300, 500]
- max_samples: ["auto", 256, 512, 1024]

**Expected Time:** ~45-60 min per scenario on A100

### Phase 4: Comparison

Compare all 5 approaches at 5% FPR:

| Experiment | Model          | Tokenization   | Recall     | F1       | Preprocessing | Status      |
| ---------- | -------------- | -------------- | ---------- | -------- | ------------- | ----------- |
| Exp 01     | TF-IDF         | Word           | 0.96%      | 0.02     | No            | Baseline    |
| Exp 02     | BGE-small      | Word-piece     | 13.81%     | 0.24     | No            | Good        |
| **Exp 05** | **ByT5**       | **Byte**       | **14.36%** | **0.24** | **No**        | **Good**    |
| **Exp 05** | **ByT5**       | **Byte**       | **20.63%** | **0.33** | **Yes**       | **Better**  |
| **Exp 02** | **BGE-small**  | **Word-piece** | **27-28%** | **0.42** | **Yes**       | **🏆 BEST** |
| Exp 04     | ColBERT+MUVERA | Word-piece     | Failed     | N/A      | Yes           | Failed      |

---

## Success Criteria

### Minimum Viable Success

- ✅ ByT5 recall > BGE-small (>27% at FPR ≤ 5%)
- ✅ Demonstrates byte-level tokenization value

### Strong Success

- ✅ ByT5 recall > 35% at FPR ≤ 5%
- ✅ Best performance among all approaches
- ✅ Clear benefit of no preprocessing

### Exceptional Success

- ✅ ByT5 recall > 45% at FPR ≤ 5%
- ✅ Establishes byte-level as standard for HTTP anomaly detection
- ✅ Simpler than ColBERT, better than everything else

---

## Implementation Checklist

### Step 1: Encoder Implementation

- [x] Create `src/neuralshield/encoding/models/byt5.py`
- [x] Implement byte-level tokenization
- [x] Implement mean+max pooling
- [x] Add L2 normalization
- [x] Register encoder in factory

### Step 2: Embedding Script

- [x] Create `src/scripts/byt5_embed.py`
- [x] Support batch processing
- [x] Support GPU acceleration
- [x] Support preprocessing pipeline

### Step 3: Generate Embeddings

- [x] Train embeddings (without preprocessing)
- [x] Test embeddings (without preprocessing)
- [x] Train embeddings (with preprocessing)
- [x] Test embeddings (with preprocessing)

### Step 4: Hyperparameter Search

- [x] Search without preprocessing
- [x] Search with preprocessing
- [x] Identify best configurations

### Step 5: Testing

- [x] Test best model (without preprocessing)
- [x] Test best model (with preprocessing)
- [x] Log all results to wandb

### Step 6: Comparative Analysis

- [x] Create 5-way comparison table
- [x] Analyze why ByT5 wins/loses
- [x] Document findings

---

## Technical Considerations

### Byte-Level Advantages

- **Preserves attacks**: Single-char edits visible
- **No vocabulary**: Any byte sequence valid
- **Unicode safe**: No tokenization artifacts
- **Minimal preprocessing**: Raw is better

### Potential Challenges

- **Longer sequences**: Bytes > words
- **GPU memory**: Larger attention matrices
- **Speed**: More tokens to process

### Mitigation

- Max length 1024 (handles 99% of requests)
- Batch size 64 (A100 can handle)
- Truncation acceptable for long requests

---

## Expected Timeline

- **Setup** (done): Encoder + script creation
- **Embedding generation**: ~15 min (4 files)
- **Hyperparameter search**: ~2 hours (2 scenarios)
- **Testing & analysis**: ~30 min
- **Total**: ~2.5-3 hours

---

## Next Steps After Experiment

**If ByT5 wins:**

- Deploy as production encoder
- Write paper on byte-level for HTTP anomaly detection
- Explore ByT5-base (larger model)

**If ByT5 loses:**

- Analyze: Why did word-piece win?
- Try ByT5 with different pooling
- Ensemble best approaches

---

## Key Insight

**Byte-level tokenization is theoretically perfect for HTTP anomaly detection** because attacks often rely on single-character exploits that word-piece tokenizers miss. This experiment will definitively test that hypothesis.

---

## Results & Conclusions

**Date Completed**: October 12, 2025

### Final Metrics

**ByT5 (without preprocessing):**

- Recall: 14.36% @ 5.37% FPR
- Precision: 72.83%
- F1-Score: 0.24
- Configuration: contamination=0.05, n_estimators=200, max_samples=auto

**ByT5 (with preprocessing):**

- Recall: 20.63% @ 4.83% FPR
- Precision: 81.07%
- F1-Score: 0.33
- Configuration: contamination=0.05, n_estimators=500, max_samples=1024

### Hypothesis Outcome: ❌ REJECTED

**ByT5 did NOT outperform BGE-small.** While byte-level tokenization showed promise, it fell short of word-piece tokenization:

- ByT5 (with prep): 20.63% recall
- BGE-small (with prep): 27-28% recall
- **Gap: 33% worse performance**

### Why ByT5 Lost

1. **Semantic patterns > Byte-level patterns**
   - HTTP attacks are more about semantic meaning than character-level exploits
   - SQL injection patterns like "OR 1=1" are semantic, not byte-level
2. **Dimensionality trade-off**
   - ByT5: 2944 dims (middle ground)
   - BGE: 384 dims (efficient, focused)
   - 2944 dims may be too many for IsolationForest with 47k samples
3. **IsolationForest limitations**
   - Doesn't leverage fine-grained byte-level distinctions
   - Better suited for coarser semantic features
4. **Preprocessing still helps**
   - ByT5 improved 43% with preprocessing (14.36% → 20.63%)
   - Suggests attack patterns benefit from normalization

### Lessons Learned

- **Word-piece tokenization is sufficient** for HTTP anomaly detection
- **Smaller embeddings (384-dim) outperform larger ones (2944-dim)** with IsolationForest
- **Preprocessing is critical** across all embedding types
- **Byte-level tokenization doesn't provide expected advantage** for this task
- **BGE-small with preprocessing is the clear winner** (27-28% recall @ 5% FPR)

### Recommendations

1. **Production deployment**: Use BGE-small with preprocessing
2. **Future work**:
   - Try ByT5 with different anomaly detectors (e.g., neural networks)
   - Test on datasets with more byte-level exploits (unicode, encoding attacks)
   - Explore hybrid approaches (semantic + byte-level features)
3. **Abandon multi-vector approaches** (ColBERT) due to dimensionality issues
