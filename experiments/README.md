# NeuralShield Experiments

This directory contains systematic experiments to evaluate and improve the NeuralShield anomaly detection system.

## 📊 Final Results Summary

**Winner**: BGE-small with preprocessing - **27-28% recall @ 5% FPR**

See [FINAL_RESULTS.md](./FINAL_RESULTS.md) for complete analysis.

---

## Completed Experiments

### ✅ 01: TF-IDF Baseline

**Location**: `01_tfidf_preprocessing_comparison/`  
**Status**: Complete

**Objective**: Evaluate TF-IDF embeddings with/without preprocessing

**Results**:

- Recall: 0.96% @ ~5% FPR
- Preprocessing improves precision slightly (+3-4pp)
- TF-IDF is too sparse for effective anomaly detection

**Conclusion**: Not suitable for production

---

### 🏆 02: BGE-small Dense Embeddings (WINNER)

**Location**: `02_dense_embeddings_comparison/`  
**Status**: Complete

**Objective**: Test dense semantic embeddings (FastEmbed)

**Results**:

- **Without preprocessing**: 13.81% recall @ ~5% FPR
- **With preprocessing**: 27-28% recall @ ~5% FPR
- **Improvement**: +96% recall gain

**Conclusion**: Best model overall. Recommended for production.

---

### ⏸️ 03: SecBERT Domain-Specific

**Location**: `03_secbert_comparison/`  
**Status**: Not completed

**Objective**: Test cybersecurity-focused BERT variant

**Note**: Skipped due to time constraints. BGE-small already provides strong results.

---

### ❌ 04: ColBERT + MUVERA (FAILED)

**Location**: `04_colbert_muvera_comparison/`  
**Status**: Complete (Failed)

**Objective**: Test multi-vector embeddings with MUVERA compression

**Results**:

- Recall: 93.06%
- **FPR: 92.73%** ← Catastrophic failure
- Flags 93% of normal traffic as attacks

**Why it failed**:

- Curse of dimensionality (10,240 dims)
- IsolationForest cannot handle such high dimensions

**Conclusion**: Multi-vector embeddings incompatible with IsolationForest

---

### ✅ 05: ByT5 Byte-Level Tokenization

**Location**: `05_byt5_comparison/`  
**Status**: Complete

**Objective**: Test if byte-level tokenization > word-piece for HTTP

**Results**:

- **Without preprocessing**: 14.36% recall @ 5.37% FPR
- **With preprocessing**: 20.63% recall @ 4.83% FPR
- **Improvement**: +43% recall gain

**Hypothesis**: ❌ Rejected - byte-level did not outperform word-piece

**Conclusion**: HTTP attacks are semantic, not byte-level. BGE-small still wins.

---

## Key Findings

### 1. Dimensionality Matters

- **Sweet Spot**: 384 dims (BGE-small)
- **Too High**: > 3000 dims causes issues
- **Catastrophic**: > 10,000 dims fails completely

### 2. Preprocessing is Essential

All models improved significantly with preprocessing:

- BGE-small: +96%
- ByT5: +43%

### 3. Simpler is Better

Simple word-piece tokenization (BGE) > byte-level (ByT5)

### 4. IsolationForest Limitations

Works best with low-to-medium dimensions (< 1000)

---

## Recommendations

### For Production

Use **BGE-small with preprocessing**:

```python
from neuralshield.encoding.models import FastEmbedEncoder
from neuralshield.anomaly.model import IsolationForestDetector

encoder = FastEmbedEncoder(
    model_name="BAAI/bge-small-en-v1.5",
    device="cuda"
)

detector = IsolationForestDetector(
    contamination=0.05,
    n_estimators=200,
    max_samples="auto"
)
```

**Expected Performance**: 27-28% recall @ 5% FPR

### Future Work

1. Try neural network anomaly detectors
2. Ensemble methods
3. Semi-supervised approaches
4. PCA dimensionality reduction

---

## Experiment Metrics

- **Total experiments**: 5
- **Duration**: Sep 28 - Oct 12, 2025
- **Compute time**: ~6 hours
- **Configurations tested**: 360+
- **Best model**: BGE-small (27-28% recall)
- **28x improvement** over TF-IDF baseline

---

## Directory Structure

```
experiments/
├── FINAL_RESULTS.md                    # Comprehensive summary
├── 01_tfidf_preprocessing_comparison/  # Baseline
├── 02_dense_embeddings_comparison/     # Winner
├── 03_secbert_comparison/              # Not completed
├── 04_colbert_muvera_comparison/       # Failed
└── 05_byt5_comparison/                 # Byte-level test
```

---

For detailed analysis, see [FINAL_RESULTS.md](./FINAL_RESULTS.md)
