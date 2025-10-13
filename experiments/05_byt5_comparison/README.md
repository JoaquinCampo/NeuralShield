# Experiment 05: ByT5 Byte-Level Embeddings

Testing byte-level embeddings (ByT5) for HTTP request anomaly detection.

## Why ByT5?

**Perfect for HTTP anomaly detection:**

- **Byte-level tokenization** preserves single-character edits
  - `%27` (SQL injection) ≠ `%28`
  - `;--` (comment injection) visible
  - Unicode tricks, null bytes, CRLF captured
- **No preprocessing needed** - raw strings work best
- **Mean+Max pooling** captures both global context and rare spikes
- **Reasonable size**: 2944 dimensions (vs ColBERT's 10,240)

## Quick Start

### 1. Generate Embeddings

**WITHOUT preprocessing (train + test):**

```bash
uv run python src/scripts/byt5_embed.py \
  src/neuralshield/data/CSIC/train.jsonl \
  experiments/05_byt5_comparison/without_preprocessing/train_embeddings.npz \
  --batch-size 64 \
  --device cuda

uv run python src/scripts/byt5_embed.py \
  src/neuralshield/data/CSIC/test.jsonl \
  experiments/05_byt5_comparison/without_preprocessing/test_embeddings.npz \
  --batch-size 64 \
  --device cuda
```

**WITH preprocessing (train + test):**

```bash
uv run python src/scripts/byt5_embed.py \
  src/neuralshield/data/CSIC/train.jsonl \
  experiments/05_byt5_comparison/with_preprocessing/train_embeddings.npz \
  --use-pipeline \
  --batch-size 64 \
  --device cuda

uv run python src/scripts/byt5_embed.py \
  src/neuralshield/data/CSIC/test.jsonl \
  experiments/05_byt5_comparison/with_preprocessing/test_embeddings.npz \
  --use-pipeline \
  --batch-size 64 \
  --device cuda
```

### 2. Run Hyperparameter Search

```bash
# Without preprocessing
uv run python experiments/02_dense_embeddings_comparison/hyperparameter_search.py \
  experiments/05_byt5_comparison/without_preprocessing/train_embeddings.npz \
  experiments/05_byt5_comparison/without_preprocessing/test_embeddings.npz \
  --max-fpr 0.05 \
  --wandb \
  --wandb-run-name "byt5-no-prep"

# With preprocessing
uv run python experiments/02_dense_embeddings_comparison/hyperparameter_search.py \
  experiments/05_byt5_comparison/with_preprocessing/train_embeddings.npz \
  experiments/05_byt5_comparison/with_preprocessing/test_embeddings.npz \
  --max-fpr 0.05 \
  --wandb \
  --wandb-run-name "byt5-with-prep"
```

### 3. Test Models

```bash
# Without preprocessing
uv run python src/scripts/test_anomaly_precomputed.py \
  experiments/05_byt5_comparison/without_preprocessing/test_embeddings.npz \
  experiments/05_byt5_comparison/without_preprocessing/best_model.joblib \
  --wandb \
  --wandb-run-name "byt5-no-prep-test"

# With preprocessing
uv run python src/scripts/test_anomaly_precomputed.py \
  experiments/05_byt5_comparison/with_preprocessing/test_embeddings.npz \
  experiments/05_byt5_comparison/with_preprocessing/best_model.joblib \
  --wandb \
  --wandb-run-name "byt5-with-prep-test"
```

## Technical Details

### Architecture

```
HTTP Request (raw, no preprocessing!)
  ↓ ByT5 Tokenizer (byte-level)
  ↓ T5 Encoder (1472 hidden dims)
  ↓ Mean Pooling (global context)
  ↓ Max Pooling (rare spikes/anomalies)
  ↓ Concatenate [mean, max] → 2944 dims
  ↓ L2 Normalize
  ↓ IsolationForest
```

### Model Details

- **Model**: `google/byt5-small`
- **Hidden size**: 1472
- **Output dimensions**: 2944 (mean+max)
- **Max length**: 1024 tokens
- **Size**: ~580MB

### Why Mean+Max Pooling?

- **Mean**: Captures global request context
- **Max**: Captures rare spikes (critical for attack detection)
- **Both needed**: Attack patterns can be global OR local

## Comparison Results

| Model          | Recall @ 5% FPR | Preprocessing | Dimensions | Status      |
| -------------- | --------------- | ------------- | ---------- | ----------- |
| TF-IDF         | 0.96%           | No            | 5000       | Baseline    |
| BGE-small      | 13.81%          | No            | 384        | Good        |
| **ByT5**       | **14.36%**      | **No**        | **2944**   | **Good**    |
| **ByT5**       | **20.63%**      | **Yes**       | **2944**   | **Better**  |
| **BGE-small**  | **27-28%**      | **Yes**       | **384**    | **🏆 Best** |
| ColBERT+MUVERA | Failed          | Yes           | 10,240     | Failed      |

## Actual Results

**Hypothesis Outcome**: ❌ ByT5 did NOT outperform BGE-small

**Key Findings:**

1. **ByT5 (no prep): 14.36% recall @ 5.37% FPR**
   - Slightly better than BGE-small without preprocessing
   - High precision (72.83%)
2. **ByT5 (with prep): 20.63% recall @ 4.83% FPR**

   - Significant improvement with preprocessing (+6%)
   - Very high precision (81.07%)
   - But still falls short of BGE-small with preprocessing

3. **Winner: BGE-small (with prep) @ 27-28% recall**
   - 33% better recall than ByT5 with preprocessing
   - Byte-level tokenization advantage did not materialize

**Why ByT5 didn't win:**

- HTTP attacks may be more about semantic patterns than byte-level exploits
- Word-piece tokenization (BGE) captures attack semantics better
- ByT5's 2944 dimensions may be a middle ground that's neither optimal
- IsolationForest may not leverage byte-level granularity effectively

## Estimated Time (A100)

- **Embedding generation**: ~10-15 min total (all 4 files)
- **Hyperparameter search**: ~45-60 min per scenario
- **Total**: ~2-2.5 hours
