# Experiment 11: Domain-Adapted SecBERT

## Hypothesis

Domain-adapted SecBERT (fine-tuned via MLM on valid HTTP requests) will produce embeddings that better capture HTTP-specific patterns, leading to improved anomaly detection performance compared to the base SecBERT model.

**Expected improvement**: +3-5% recall @ 5% FPR over base SecBERT (49.26% → 52-54%)

## Rationale

The base SecBERT model was pre-trained on cybersecurity text (APTnotes, CASIE, etc.) which provides general security domain knowledge. However, by continuing pre-training on our specific HTTP corpus via Masked Language Modeling (MLM), the model should:

1. **Learn HTTP-specific syntax**: `[METHOD]`, `[URL]`, `[QUERY]`, `[HEADER]` token structures
2. **Understand parameter patterns**: Common query parameter names and value structures
3. **Capture header conventions**: Standard HTTP header formats and values
4. **Improve tokenization efficiency**: Better token-level representations of HTTP constructs

## Training Summary

**Model**: `jackaduma/SecBERT` → Domain-adapted on HTTP corpus
**Corpus**: 47,000 preprocessed valid HTTP requests
**Training approach**: Masked Language Modeling (MLM)

**Hyperparameters**:

- Epochs: 3
- Batch size: 8 (effective: 16 with gradient accumulation)
- Learning rate: 2e-5 (cosine decay with 500-step warmup)
- Masking: 15% random tokens
- Max length: 512 tokens

**Final metrics**:

- Training loss: 0.77
- Validation loss: 0.35
- Perplexity: 1.41 (near-perfect HTTP token prediction!)

**Pooling Strategy**:

- **Mean+max pooling** over last two hidden layers (not [CLS])
- **Mean**: Captures "style" of normal traffic
- **Max**: Catches rare/spiky tokens (weird encodings, duplicate headers)
- **Layer averaging**: Layers -1 and -2 for stability without labels
- **Embedding dimensions**: 1536 (768 mean + 768 max)

## Comparison Baseline

**Base SecBERT + Mahalanobis** (Experiment 03):

- Recall @ 5% FPR: 49.26%
- Preprocessing: Yes
- Model: `jackaduma/SecBERT` (no adaptation)
- Pooling: [CLS] token only (768 dims)

## Experiment Structure

```
11_secbert_adapted/
├── README.md                                  # This file
├── generate_embeddings.py                     # Script to generate embeddings
├── train_mahalanobis.py                       # Script to train Mahalanobis detector
├── with_preprocessing/
│   ├── train_embeddings.npz                  # Adapted SecBERT embeddings (train)
│   ├── test_embeddings.npz                   # Adapted SecBERT embeddings (test)
│   ├── results.json                          # Performance metrics
│   ├── score_distribution.png                # Mahalanobis distance visualization
│   └── confusion_matrix.png                  # Classification results
└── RESULTS.md                                 # Final analysis and comparison
```

## Workflow

1. ✅ Train adapted SecBERT via MLM (completed - perplexity 1.41)
2. ✅ Generate embeddings using adapted model (completed - 1536 dims)
3. **Train Mahalanobis detector** on new embeddings ← Next step
4. Evaluate recall @ 5% FPR
5. Compare with base SecBERT results

## Running the Experiment

```bash
# Train Mahalanobis detector (with W&B logging by default)
uv run python experiments/11_secbert_adapted/train_mahalanobis.py

# With custom W&B run name
uv run python experiments/11_secbert_adapted/train_mahalanobis.py \
  --wandb-run-name "exp11-adapted-secbert-final"

# With custom paths
uv run python experiments/11_secbert_adapted/train_mahalanobis.py \
  --train-embeddings experiments/11_secbert_adapted/with_preprocessing/train_embeddings.npz \
  --test-embeddings experiments/11_secbert_adapted/with_preprocessing/test_embeddings.npz \
  --output-dir experiments/11_secbert_adapted/with_preprocessing \
  --max-fpr 0.05

# Disable W&B logging
uv run python experiments/11_secbert_adapted/train_mahalanobis.py --no-use-wandb
```

**Expected outputs**:

- `results.json` - Performance metrics and comparison
- `score_distribution.png` - Mahalanobis distance distributions
- `confusion_matrix.png` - Classification confusion matrix
- **W&B run** - Metrics, visualizations, and improvement tracking

## Success Criteria

- [ ] Adapted SecBERT achieves >50% recall @ 5% FPR
- [ ] Improvement of +2% or more over base SecBERT (49.26%)
- [ ] Validation loss confirms proper domain adaptation (<1.0 perplexity)
