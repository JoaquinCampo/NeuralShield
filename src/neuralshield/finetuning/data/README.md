# HTTP MLM Training Corpus

**Created**: October 13, 2025  
**Purpose**: Domain adaptation of SecBERT via Masked Language Modeling

---

## Corpus Statistics

| Metric              | Value                                   |
| ------------------- | --------------------------------------- |
| **Total Documents** | 47,000                                  |
| **File Size**       | 37.48 MB                                |
| **Source**          | CSIC training set (valid requests only) |
| **Preprocessing**   | Applied (NeuralShield pipeline)         |

---

## Tokenization Analysis

**Tokenizer**: SecBERT (`jackaduma/SecBERT`)

| Metric                      | Value                 |
| --------------------------- | --------------------- |
| **Average Sequence Length** | 312.9 tokens          |
| **Min Length**              | 285 tokens            |
| **Max Length**              | 461 tokens            |
| **Truncated (≥512)**        | 0 (0.0%)              |
| **Unique Tokens Used**      | 3,362 / 52,000 (6.5%) |

---

## Key Insights

### Good News

- **No truncation**: All sequences fit within 512 token limit
- **Consistent length**: Avg 313 tokens (good for batching)
- **Rich vocabulary**: 3,362 unique tokens from HTTP domain

### Observations

- Only 6.5% of SecBERT's vocabulary is used
- HTTP domain has specialized, limited vocabulary
- Perfect opportunity for domain adaptation

---

## File Format

**Format**: Text corpus with documents separated by double newlines (`\n\n`)

**Example structure**:

```
[METHOD] GET
[URL] /path/to/resource
[QUERY] param=value
[HEADER] User-Agent: ...

[METHOD] POST
[URL] /api/endpoint
...
```

---

## Next Steps

1. **Create MLM training script** → `train_mlm.py`
2. **Train on GPU** (2-4 hours)
3. **Evaluate adapted embeddings**

---

## Usage

**Regenerate corpus**:

```bash
uv run python -m neuralshield.finetuning.prepare_mlm_data \
  src/neuralshield/data/CSIC/train.jsonl \
  src/neuralshield/finetuning/data/http_corpus.txt
```

**Without preprocessing** (raw HTTP):

```bash
uv run python -m neuralshield.finetuning.prepare_mlm_data \
  src/neuralshield/data/CSIC/train.jsonl \
  src/neuralshield/finetuning/data/http_corpus_raw.txt \
  --no-preprocess
```

**Skip analysis** (faster):

```bash
uv run python -m neuralshield.finetuning.prepare_mlm_data \
  src/neuralshield/data/CSIC/train.jsonl \
  src/neuralshield/finetuning/data/http_corpus.txt \
  --no-analyze
```

---

## Data Quality

- ✅ All 47,000 valid requests from training set
- ✅ Preprocessing applied (normalization, structuring)
- ✅ Attack requests excluded (maintains anomaly detection paradigm)
- ✅ Ready for MLM training
