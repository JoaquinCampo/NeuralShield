# Experiment 04: ColBERT + MUVERA Multi-Vector Embeddings

**Date**: October 10, 2025  
**Status**: 🔄 In Progress  
**Author**: NeuralShield Team

---

## Objective

Test **multi-vector embeddings with MUVERA compression** to determine if richer token-level representations improve HTTP request anomaly detection beyond single-vector approaches.

## Hypothesis

ColBERT + MUVERA will improve recall because:

1. **Token-level embeddings**: Each token gets its own vector, preserving fine-grained semantic information
2. **Better attack pattern capture**: SQL injection, XSS patterns have distinct token signatures
3. **MUVERA compression**: Efficiently aggregates multi-vectors to fixed-size without losing critical information
4. **Proven retrieval performance**: ColBERT excels at semantic matching tasks

## Background

### Traditional Single-Vector Approach

- BGE-small, SecBERT: One vector per request
- Loses token-level granularity
- Simple but potentially too coarse

### Multi-Vector Approach (ColBERT)

- One vector per token (variable length)
- Rich semantic representation
- Problem: IsolationForest needs fixed-size input

### MUVERA Solution

- Learns to compress multi-vectors → single fixed-size vector
- Preserves important information via learned aggregation
- Best of both worlds: rich representation + compatibility

## Comparison Matrix

| Model                    | Type          | Vectors/Request | Dimensions | Compression |
| ------------------------ | ------------- | --------------- | ---------- | ----------- |
| **ColBERT+MUVERA** (new) | Multi-vector  | N→1 (learned)   | 128        | MUVERA      |
| SecBERT (Exp 03)         | Single-vector | 1               | 768        | [CLS]       |
| BGE-small (Exp 02)       | Single-vector | 1               | 384        | Mean pool   |
| TF-IDF (Exp 01)          | Sparse        | 1               | 5000       | N/A         |

---

## Research Questions

1. Do multi-vector representations outperform single-vector embeddings?
2. Is MUVERA's learned compression better than simple [CLS] or mean pooling?
3. Can 128-dim MUVERA embeddings compete with 384-768 dim single-vector models?
4. How does preprocessing impact multi-vector representations?

---

## Methodology

### Phase 1: Setup ColBERT + MUVERA Encoder

Created new encoder using FastEmbed's built-in MUVERA support:

```python
# New encoder: neuralshield/encoding/models/colbert_muvera.py
from fastembed import LateInteractionTextEmbedding
from fastembed.postprocess import Muvera
```

**Model**: `colbert-ir/colbertv2.0`  
**Library**: FastEmbed  
**Output dimensions**: 128 (configurable)  
**Max tokens**: 512

### Phase 2: Generate Embeddings

```bash
# WITHOUT preprocessing - Train
uv run python src/scripts/colbert_embed.py \
  src/neuralshield/data/CSIC/train.jsonl \
  experiments/04_colbert_muvera_comparison/colbert_without_preprocessing/embeddings.npz \
  --no-use-pipeline \
  --batch-size 32

# WITHOUT preprocessing - Test
uv run python src/scripts/colbert_embed.py \
  src/neuralshield/data/CSIC/test.jsonl \
  experiments/04_colbert_muvera_comparison/colbert_without_preprocessing/test_embeddings.npz \
  --no-use-pipeline \
  --batch-size 32

# WITH preprocessing - Train
uv run python src/scripts/colbert_embed.py \
  src/neuralshield/data/CSIC/train.jsonl \
  experiments/04_colbert_muvera_comparison/colbert_with_preprocessing/embeddings.npz \
  --use-pipeline \
  --batch-size 32

# WITH preprocessing - Test
uv run python src/scripts/colbert_embed.py \
  src/neuralshield/data/CSIC/test.jsonl \
  experiments/04_colbert_muvera_comparison/colbert_with_preprocessing/test_embeddings.npz \
  --use-pipeline \
  --batch-size 32
```

### Phase 3: Hyperparameter Search

Reuse existing `hyperparameter_search.py`:

```bash
# Without preprocessing
uv run python experiments/02_dense_embeddings_comparison/hyperparameter_search.py \
  experiments/04_colbert_muvera_comparison/colbert_without_preprocessing/embeddings.npz \
  experiments/04_colbert_muvera_comparison/colbert_without_preprocessing/test_embeddings.npz \
  --max-fpr 0.05 \
  --wandb \
  --wandb-run-name "exp04-colbert-muvera-no-pipeline"

# With preprocessing
uv run python experiments/02_dense_embeddings_comparison/hyperparameter_search.py \
  experiments/04_colbert_muvera_comparison/colbert_with_preprocessing/embeddings.npz \
  experiments/04_colbert_muvera_comparison/colbert_with_preprocessing/test_embeddings.npz \
  --max-fpr 0.05 \
  --wandb \
  --wandb-run-name "exp04-colbert-muvera-with-pipeline"
```

### Phase 4: Comparison

Compare 4 approaches at 5% FPR:

1. **TF-IDF** (Exp 01): 0.96% recall
2. **BGE-small** (Exp 02): 27% recall (with prep)
3. **SecBERT** (Exp 03): TBD
4. **ColBERT+MUVERA** (Exp 04): TBD

---

## Success Criteria

### Minimum Viable Success

- ✅ ColBERT+MUVERA recall > BGE-small baseline (>27% at FPR ≤ 5%)
- ✅ Demonstrates multi-vector value for anomaly detection

### Strong Success

- ✅ ColBERT+MUVERA recall > 35% at FPR ≤ 5%
- ✅ Outperforms both BGE-small and SecBERT
- ✅ Better F1-score across multiple FPR thresholds

### Exceptional Success

- ✅ ColBERT+MUVERA recall > 45% at FPR ≤ 5%
- ✅ Best-in-class performance for HTTP anomaly detection
- ✅ Clear evidence that multi-vector > single-vector

---

## Implementation Steps

### Step 1: Create ColBERT + MUVERA Encoder

- [x] Create `neuralshield/encoding/models/colbert_muvera.py`
- [x] Implement `ColBERTMuveraEncoder` class
- [x] Register with encoder factory
- [ ] Test on sample requests

### Step 2: Create Embedding Script

- [ ] Create `src/scripts/colbert_embed.py`
- [ ] Handle batch processing
- [ ] Support pipeline integration

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

- [ ] Create 4-way comparison plots (TF-IDF vs BGE vs SecBERT vs ColBERT)
- [ ] Statistical significance testing
- [ ] Document findings in RESULTS.md

---

## Technical Considerations

### MUVERA Architecture

```
HTTP Request → Tokenize → ColBERT (N vectors) → MUVERA → Single 128-dim vector
```

- **Input**: Variable-length token sequence
- **ColBERT**: N token embeddings (each 128-dim)
- **MUVERA**: Learned aggregation to single 128-dim vector
- **Output**: Fixed-size embedding for IsolationForest

### Compute Requirements

- **Model size**: ColBERT (~440MB) + MUVERA parameters
- **Memory**: ~800MB model + batch embeddings
- **Speed**: Expect similar to BGE-small (both use BERT-scale models)

### Dimensionality Trade-off

- ColBERT+MUVERA: 128 dims (smaller than BGE-small's 384)
- Hypothesis: Multi-vector origin compensates for smaller final size
- If fails: Try MUVERA output_dim=256 or 384

---

## Risk Mitigation

### Potential Issue: MUVERA loses critical information

- **Reason**: Compression from N×128 to 1×128 might be too aggressive
- **Mitigation**: Test different MUVERA dimensions (128, 256, 384)

### Potential Issue: ColBERT too slow

- **Reason**: Processing all tokens adds overhead
- **Mitigation**: Use smaller batch sizes, optimize on GPU

### Potential Issue: Multi-vector doesn't help for anomaly detection

- **Reason**: IsolationForest might not leverage fine-grained semantics
- **Mitigation**: Still valuable negative result (publish findings)

---

## Expected Timeline

- **Hour 1**: Create colbert_embed.py script (~30 min)
- **Hour 2-3**: Generate embeddings (~1-2 hours)
- **Hour 3-4**: Run hyperparameter search (~30 min)
- **Hour 4-5**: Analysis and documentation (~1 hour)

**Total**: ~5 hours active work

---

## Next Steps After Experiment

**If ColBERT+MUVERA wins**:

- Experiment 05: Optimize MUVERA dimensions
- Experiment 06: Ensemble (ColBERT + SecBERT)

**If single-vector wins**:

- Focus on fine-tuning best single-vector model (BGE or SecBERT)
- Explore different anomaly detectors (OCSVM, AutoEncoder)

---

## References

- **MUVERA Paper**: [Qdrant Blog - MUVERA Embeddings](https://qdrant.tech/articles/muvera-embeddings/)
- **ColBERT Paper**: ["ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"](https://arxiv.org/abs/2004.12832)
- **FastEmbed MUVERA**: Built-in support for multi-vector compression

**Key insight**: This is our first attempt at multi-vector representations for anomaly detection. The learned compression via MUVERA should preserve semantic richness while maintaining compatibility with existing infrastructure.
