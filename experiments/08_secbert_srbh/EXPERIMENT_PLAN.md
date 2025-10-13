# Experiment 08: SecBERT on SR_BH_2020 - Experimental Design

**Date**: October 13, 2025  
**Status**: 🔄 Planning  
**Author**: NeuralShield Team

---

## Objective

Evaluate **SecBERT** embeddings on the **SR_BH_2020** dataset to:

1. Test performance on large-scale real-world attack data
2. Compare domain-specific (SecBERT) vs general-purpose (BGE-small) embeddings
3. Validate preprocessing pipeline scalability
4. Establish baseline for multi-dataset evaluation

---

## Hypothesis

**H1**: SecBERT's security-domain pretraining improves detection vs general embeddings

**Why**:

- Trained on security corpora (CVEs, exploit descriptions, etc.)
- Specialized vocabulary for attack patterns
- Better semantic understanding of malicious payloads

**H2**: SR_BH's diverse attack types will benefit from domain knowledge

**Why**:

- 13 CAPEC attack categories
- Mix of common (SQLi) and rare (request smuggling) attacks
- More realistic attack distribution

**H3**: Preprocessing is critical for both datasets

**Why**:

- Normalizes encoding variations
- Structures data consistently
- Proven +96% recall gain on CSIC with BGE-small

---

## Dataset: SR_BH_2020

### Statistics

| Metric        | Value                              |
| ------------- | ---------------------------------- |
| Total samples | 907,815                            |
| Train samples | 100,000 (all valid)                |
| Test samples  | 807,815 (425K valid + 382K attack) |
| Valid %       | 58%                                |
| Attack %      | 42%                                |

### Attack Distribution

| CAPEC  | Attack Type           | Count     |
| ------ | --------------------- | --------- |
| 242    | Code Injection        | 253K      |
| 66     | SQL Injection         | 243K      |
| 272    | Protocol Manipulation | 139K      |
| 194    | Fake Source Data      | 54K       |
| 126    | Path Traversal        | 21K       |
| Others | Various               | <20K each |

### Comparison to CSIC

| Metric       | CSIC   | SR_BH    | Ratio        |
| ------------ | ------ | -------- | ------------ |
| Total        | 97,065 | 907,815  | 9.3x         |
| Train        | 47,000 | 100,000  | 2.1x         |
| Test         | 50,065 | 807,815  | 16.1x        |
| Attack types | Mixed  | 13 CAPEC | More diverse |

---

## Model: SecBERT

### Architecture

- **Base**: BERT-base (110M parameters)
- **Pretraining**: Security-specific corpora
- **Embedding dim**: 768
- **Max sequence**: 512 tokens
- **Tokenizer**: WordPiece

### Advantages

1. Domain vocabulary (exploit terms, payloads)
2. Context understanding for security patterns
3. Proven on malware/vulnerability classification

### Known Limitations

1. Slower than lightweight models (BGE-small)
2. Higher memory requirements
3. May overfit to specific attack patterns

---

## Experimental Design

### Variants

**Variant A: Without Preprocessing**

- Raw HTTP request text
- Direct SecBERT encoding
- Baseline comparison

**Variant B: With Preprocessing** (Primary)

- Full 13-step preprocessing pipeline
- Structured canonical format
- Expected best performance

### Pipeline Steps (Variant B)

```
00. FramingCleanup
01. RequestStructurer
08. HeaderUnfoldObsFold
09. HeaderNormalizationDuplicates
10. WhitespaceCollapse
11. DangerousCharactersScriptMixing
12. AbsoluteUrlBuilder
03. UnicodeNFKCAndControl
04. PercentDecodeOnce
05. HtmlEntityDecodeOnce
06. QueryParserAndFlags
07. PathStructureNormalizer
```

---

## Methodology

### Phase 1: Embedding Generation

**Train embeddings**:

```bash
# Without preprocessing
secbert_embed train.jsonl → train_embeddings.npz (100K × 768)

# With preprocessing
secbert_embed train.jsonl --use-pipeline → train_embeddings.npz
```

**Test embeddings**:

```bash
# Same for test.jsonl (807K × 768)
```

**Expected time**:

- CPU: 4-6 hours total
- GPU: 30-60 minutes total

### Phase 2: Anomaly Detection

**Training**:

```python
IsolationForest(
    contamination=0.1,
    n_estimators=300,
    random_state=42
)
```

**Rationale**:

- Proven approach from experiments 01-07
- Works well with dense embeddings
- Fast training even on large datasets

### Phase 3: Evaluation

**Metrics**:

1. **Recall @ 5% FPR** (primary metric)
2. Precision, F1-Score
3. ROC-AUC
4. Per-attack-type performance (if time permits)

**Comparison**:

- SecBERT vs BGE-small (CSIC baseline)
- With vs without preprocessing
- SR_BH vs CSIC dataset characteristics

---

## Expected Results

### Performance Estimates

Based on CSIC experiments (Exp 03):

| Variant             | Expected Recall @ 5% FPR |
| ------------------- | ------------------------ |
| SecBERT (no prep)   | 10-15%                   |
| SecBERT (with prep) | 18-25%                   |

**Note**: May vary due to:

- Different attack distribution
- Larger dataset size
- More diverse attack types

### Key Questions

1. Does SecBERT outperform BGE-small on security data?
2. Does preprocessing help as much (expected: yes)
3. How does performance scale with dataset size?
4. Which attack types are easiest/hardest to detect?

---

## Success Criteria

- ✅ Successfully process 900K+ samples
- ✅ Generate embeddings in reasonable time (<6 hours)
- ✅ Achieve >15% recall @ 5% FPR (with preprocessing)
- ✅ Preprocessing shows improvement (>5pp gain)
- ✅ Results interpretable and reproducible

---

## Implementation Details

### Hardware Requirements

**Minimum**:

- 16GB RAM
- 50GB disk space (embeddings + models)
- CPU only (slow but functional)

**Recommended**:

- NVIDIA GPU with 8GB+ VRAM
- 32GB RAM
- 100GB disk space

### Software Dependencies

All already installed:

- PyTorch + Transformers (for SecBERT)
- scikit-learn (for IsolationForest)
- numpy, loguru, typer, tqdm

### File Outputs

**Per variant**:

- `train_embeddings.npz` - Training embeddings (768 dims)
- `test_embeddings.npz` - Test embeddings (768 dims)
- `model.joblib` - Trained IsolationForest
- `results.json` - Evaluation metrics
- `*.png` - Visualization plots

---

## Timeline

1. **Embedding generation**: 30-360 min (GPU vs CPU)
2. **Model training**: 2-5 min
3. **Testing & evaluation**: 5-10 min
4. **Analysis & documentation**: 30-60 min

**Total**: 1-7 hours depending on hardware

---

## Risks & Mitigations

**Risk 1**: Out of memory on large test set

- **Mitigation**: Batch processing already implemented

**Risk 2**: GPU OOM

- **Mitigation**: Reduce batch size (default 32 → 16 or 8)

**Risk 3**: Slow processing on CPU

- **Mitigation**: Use GPU or run overnight

**Risk 4**: Preprocessing errors on diverse data

- **Mitigation**: Robust error handling already in place

---

## Future Work

If results are promising:

1. Test other embedding models (BGE-base, E5)
2. Hyperparameter tuning for IsolationForest
3. Per-attack-type analysis
4. Cross-dataset evaluation (train CSIC, test SR_BH)
5. Ensemble methods
