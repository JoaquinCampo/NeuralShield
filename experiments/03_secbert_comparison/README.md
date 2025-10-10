# Experiment 03: SecBERT Comparison

Testing domain-specific cybersecurity embeddings (SecBERT) vs general-purpose embeddings (BGE-small).

## Quick Start

### Generate Embeddings

```bash
# WITHOUT preprocessing (train + test)
uv run python -m neuralshield.encoding.dump_embeddings \
  src/neuralshield/data/CSIC/train.jsonl \
  experiments/03_secbert_comparison/secbert_without_preprocessing/embeddings.npz \
  --encoder secbert --use-pipeline false --batch-size 32

uv run python -m neuralshield.encoding.dump_embeddings \
  src/neuralshield/data/CSIC/test.jsonl \
  experiments/03_secbert_comparison/secbert_without_preprocessing/test_embeddings.npz \
  --encoder secbert --use-pipeline false --batch-size 32

# WITH preprocessing (train + test)
uv run python -m neuralshield.encoding.dump_embeddings \
  src/neuralshield/data/CSIC/train.jsonl \
  experiments/03_secbert_comparison/secbert_with_preprocessing/embeddings.npz \
  --encoder secbert --use-pipeline true --batch-size 32

uv run python -m neuralshield.encoding.dump_embeddings \
  src/neuralshield/data/CSIC/test.jsonl \
  experiments/03_secbert_comparison/secbert_with_preprocessing/test_embeddings.npz \
  --encoder secbert --use-pipeline true --batch-size 32
```

### Run Hyperparameter Search

```bash
# Without preprocessing
uv run python experiments/02_dense_embeddings_comparison/hyperparameter_search.py \
  experiments/03_secbert_comparison/secbert_without_preprocessing/embeddings.npz \
  experiments/03_secbert_comparison/secbert_without_preprocessing/test_embeddings.npz \
  --max-fpr 0.05

# With preprocessing
uv run python experiments/02_dense_embeddings_comparison/hyperparameter_search.py \
  experiments/03_secbert_comparison/secbert_with_preprocessing/embeddings.npz \
  experiments/03_secbert_comparison/secbert_with_preprocessing/test_embeddings.npz \
  --max-fpr 0.05
```

## Implementation

### SecBERT Encoder

Created `src/neuralshield/encoding/models/secbert.py` following the existing encoder pattern:

- **Model**: `jackaduma/SecBERT`
- **Dimensions**: 768 (2x BGE-small)
- **Max tokens**: 512 (standard BERT)
- **Library**: HuggingFace Transformers

### Key Features

- Uses [CLS] token embedding for sentence representation
- Automatic truncation to 512 tokens
- Batch processing support
- Device-agnostic (CPU/GPU/MPS)
- Clean shutdown with memory management

## Hypothesis

SecBERT should outperform BGE-small because:

1. Trained on cybersecurity text (APTnotes, CASIE)
2. Custom vocabulary for security terms
3. Domain knowledge of attack patterns

## Expected Results

- **Minimum**: > 27% recall at 5% FPR (beat BGE-small)
- **Strong**: > 35% recall at 5% FPR
- **Exceptional**: > 45% recall at 5% FPR

## Comparison Baseline

| Model     | Recall @ 5% FPR | Preprocessing |
| --------- | --------------- | ------------- |
| TF-IDF    | 0.96%           | No            |
| BGE-small | 13.81%          | No            |
| BGE-small | 27-28%          | Yes           |
| SecBERT   | TBD             | TBD           |
