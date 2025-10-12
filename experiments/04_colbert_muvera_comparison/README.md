# Experiment 04: ColBERT + MUVERA Comparison

Testing multi-vector embeddings with MUVERA compression for HTTP request anomaly detection.

## Quick Start (Single Command)

Run the entire experiment with one command:

```bash
# Local (CPU)
uv run python experiments/04_colbert_muvera_comparison/run_full_experiment.py \
  src/neuralshield/data/CSIC/train.jsonl \
  src/neuralshield/data/CSIC/test.jsonl \
  --batch-size 16 \
  --max-fpr 0.05 \
  --device cpu

# Local (GPU)
uv run python experiments/04_colbert_muvera_comparison/run_full_experiment.py \
  src/neuralshield/data/CSIC/train.jsonl \
  src/neuralshield/data/CSIC/test.jsonl \
  --batch-size 1024 \
  --max-fpr 0.05 \
  --device cuda

# Google Colab
# See COLAB_SETUP.md for detailed instructions
```

This script will:

1. Generate embeddings (with and without preprocessing)
2. Run hyperparameter search (72 configs each)
3. Retrain best models with wandb logging
4. Test both models with comprehensive evaluation

**Estimated time**: 1-2 hours on GPU, 3-4 hours on CPU

## What is ColBERT + MUVERA?

### ColBERT (Contextualized Late Interaction)

- Produces one embedding vector **per token** (multi-vector representation)
- Preserves fine-grained semantic information
- Excellent for retrieval tasks

### MUVERA (Multi-Vector Representation Aggregation)

- Learns to compress N token vectors → 1 fixed-size vector
- Preserves important semantic information via learned aggregation
- Makes multi-vector embeddings compatible with traditional ML models

### Why This Matters for Anomaly Detection

- Attack patterns (SQL injection, XSS) have distinct token-level signatures
- Multi-vector captures these better than single-vector approaches
- MUVERA ensures compatibility with IsolationForest

## Implementation

### ColBERT + MUVERA Encoder

Created `src/neuralshield/encoding/models/colbert_muvera.py`:

- **Model**: `colbert-ir/colbertv2.0`
- **Dimensions**: 128 (after MUVERA compression)
- **Max tokens**: 512
- **Library**: FastEmbed with built-in MUVERA support

### Key Features

- Multi-vector token embeddings with learned compression
- Configurable MUVERA output dimensions
- Batch processing support
- Device-agnostic (CPU/GPU)

## Hypothesis

ColBERT + MUVERA should outperform single-vector approaches because:

1. Token-level embeddings capture fine-grained attack patterns
2. MUVERA compression preserves semantic richness
3. Proven effectiveness in retrieval tasks

## Expected Results

- **Minimum**: > 27% recall at 5% FPR (beat BGE-small)
- **Strong**: > 35% recall at 5% FPR
- **Exceptional**: > 45% recall at 5% FPR

## Comparison Baseline

| Model          | Recall @ 5% FPR | Preprocessing | Dimensions |
| -------------- | --------------- | ------------- | ---------- |
| TF-IDF         | 0.96%           | No            | 5000       |
| BGE-small      | 13.81%          | No            | 384        |
| BGE-small      | 27-28%          | Yes           | 384        |
| SecBERT        | TBD             | TBD           | 768        |
| ColBERT+MUVERA | TBD             | TBD           | 1024       |

## Technical Details

### Architecture Flow

```
HTTP Request
  ↓ Tokenize
  ↓ ColBERT (produces N vectors, each 1024-dim)
  ↓ MUVERA (learned aggregation)
  ↓ Single 1024-dim vector
  ↓ IsolationForest
```

### Why 128 Dimensions?

- Smaller than BGE-small (384) and SecBERT (768)
- Hypothesis: Multi-vector origin compensates for smaller size
- Can increase if needed (MUVERA supports 256, 384, etc.)

## Next Steps

1. Generate embeddings (with/without preprocessing)
2. Run hyperparameter search
3. Compare against BGE-small and SecBERT
4. Analyze if multi-vector representations improve anomaly detection

---

## Results & Conclusions

**Date Completed**: October 10, 2025

### Actual Dimensions

The encoder produced **10,240 dimensions** (not 128 as initially planned), which caused significant issues.

### Final Metrics (With Preprocessing)

- Recall: 93.06%
- **FPR: 92.73%** ← CRITICAL FAILURE
- Precision: 50.15%
- F1-Score: 0.65

**Confusion Matrix:**

- True Positives: 23,326
- False Positives: 23,182 ← Flagged 93% of normal traffic!
- True Negatives: 1,818
- False Negatives: 1,739

### Hypothesis Outcome: ❌ FAILED

ColBERT + MUVERA did NOT outperform single-vector approaches. The model is **completely unusable** for production.

### Why It Failed

1. **Curse of Dimensionality**

   - 10,240 dimensions is far too many for IsolationForest
   - With only 47k training samples, the model couldn't learn meaningful patterns
   - In high-dimensional space, all points appear equally distant

2. **FPR 92.73% is Unacceptable**

   - Model flags 93% of normal traffic as attacks
   - Would generate 23,182 false alarms out of 25,000 normal requests
   - Completely defeats the purpose of anomaly detection

3. **IsolationForest Limitations**
   - Designed for low-to-medium dimensionality (typically < 100-500 dims)
   - Cannot leverage the richness of 10k-dimensional embeddings
   - Multi-vector information gets lost in the noise

### Comparison with Alternatives

| Model              | Recall     | FPR        | Preprocessing | Status        |
| ------------------ | ---------- | ---------- | ------------- | ------------- |
| BGE-small          | 13.81%     | ~5%        | No            | Good          |
| BGE-small          | 27-28%     | ~5%        | Yes           | 🏆 Best       |
| ByT5               | 14.36%     | 5.37%      | No            | Good          |
| ByT5               | 20.63%     | 4.83%      | Yes           | Better        |
| **ColBERT+MUVERA** | **93.06%** | **92.73%** | **Yes**       | **❌ Failed** |

### Lessons Learned

1. **Multi-vector embeddings are incompatible with IsolationForest** at this scale
2. **Dimensionality matters more than semantic richness** for tree-based anomaly detectors
3. **Simpler is better**: BGE-small (384 dims) >> ColBERT+MUVERA (10,240 dims)
4. **MUVERA compression didn't work as expected** - output dimensions far exceeded target

### Recommendations

1. **DO NOT use ColBERT+MUVERA** for IsolationForest-based anomaly detection
2. **Stick with single-vector embeddings** (BGE, ByT5, SecBERT)
3. **Keep dimensionality low** (< 3000 for this dataset size)
4. **Future work**: Try ColBERT with neural network anomaly detectors that can handle high dimensions
5. **Alternative**: Apply PCA dimensionality reduction before IsolationForest (untested)
