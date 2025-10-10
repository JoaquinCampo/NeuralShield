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
