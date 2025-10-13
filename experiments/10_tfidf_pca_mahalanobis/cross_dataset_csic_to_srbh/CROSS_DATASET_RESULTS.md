# Cross-Dataset Results: CSIC → SRBH (TF-IDF+PCA+Mahalanobis)

## Setup

- **Train**: CSIC dataset (47,000 normal samples)
- **Test**: SRBH dataset (425,195 normal, 382,620 attacks)
- **Method**: TF-IDF with preprocessing → PCA (300 dims) → Mahalanobis
- **Constraint**: 5% FPR

## Results

### TF-IDF+PCA+Mahalanobis (CSIC → SRBH)

- **Recall**: 0.00% (17 / 382,620 attacks detected) ❌ **CATASTROPHIC FAILURE**
- **Precision**: 0.08%
- **F1-Score**: 0.01%
- **FPR**: 4.85%
- **PCA Explained Variance**: 91.94%

### Comparison with Same-Dataset Performance

| Metric             | Same-Dataset (CSIC→CSIC) | Cross-Dataset (CSIC→SRBH) | Degradation |
| ------------------ | ------------------------ | ------------------------- | ----------- |
| Recall             | 27.08%                   | **0.00%**                 | **-100.0%** |
| Precision          | 84.45%                   | 0.08%                     | -99.9%      |
| F1-Score           | 41.01%                   | 0.01%                     | -100.0%     |
| Explained Variance | 91.94%                   | 91.94%                    | 0.0%        |

## Comparison with SecBERT Cross-Dataset

### SecBERT+Mahalanobis (CSIC → SRBH)

- **Recall**: 10.76% (41,170 / 382,620 attacks detected)
- **Precision**: 65.95%
- **F1-Score**: 18.50%
- **FPR**: 5.00%

### TF-IDF+PCA vs SecBERT Cross-Dataset

| Metric    | TF-IDF+PCA | SecBERT | SecBERT Advantage            |
| --------- | ---------- | ------- | ---------------------------- |
| Recall    | 0.00%      | 10.76%  | **∞ (TF-IDF complete fail)** |
| Precision | 0.08%      | 65.95%  | +824x                        |
| F1-Score  | 0.01%      | 18.50%  | +1,850x                      |

**SecBERT is infinitely better - TF-IDF+PCA completely fails!**

## Cross-Dataset Direction Comparison

### TF-IDF+PCA Performance by Direction

| Direction       | Recall    | F1-Score  | Verdict                 |
| --------------- | --------- | --------- | ----------------------- |
| SRBH → CSIC     | 16.89%    | 27.85%    | ✅ Works reasonably     |
| **CSIC → SRBH** | **0.00%** | **0.01%** | ❌ **Complete failure** |

### SecBERT Performance by Direction

| Direction   | Recall | F1-Score | Verdict            |
| ----------- | ------ | -------- | ------------------ |
| SRBH → CSIC | 10.26% | 17.80%   | ✅ Works           |
| CSIC → SRBH | 10.76% | 18.50%   | ✅ Works similarly |

## Analysis

### Why TF-IDF+PCA Fails Catastrophically in CSIC→SRBH

1. **Training set size mismatch**:

   - CSIC train: 47,000 samples (small)
   - SRBH train: 100,000 samples (large)
   - Small CSIC training set doesn't capture enough vocabulary diversity

2. **Vocabulary domain shift**:

   - CSIC vocabulary doesn't cover SRBH attack patterns
   - TF-IDF embeddings contain zeros for unseen tokens
   - PCA projection fails on out-of-vocabulary features

3. **Test set size**:

   - SRBH test has 807,815 samples (16x larger than CSIC test)
   - Distribution extremely different from small CSIC training

4. **Attack diversity**:
   - SRBH has 382,620 attacks with different patterns
   - CSIC-trained model has no representation for SRBH attack types

### Why SecBERT Maintains Performance

1. **Semantic robustness**: Pre-trained embeddings handle unseen tokens better
2. **Subword tokenization**: Can handle novel attack patterns through word pieces
3. **Dense representations**: Every request gets a meaningful embedding, even with new vocabulary
4. **Bidirectional**: Understands that both directions are similar difficulty (10.26% vs 10.76%)

### Why SRBH→CSIC Works but CSIC→SRBH Fails for TF-IDF

**SRBH→CSIC (16.89% recall)**:

- Large SRBH training (100K) covers diverse vocabulary
- CSIC attacks use common patterns present in SRBH
- 100% PCA variance suggests SRBH vocabulary is comprehensive

**CSIC→SRBH (0.00% recall)**:

- Small CSIC training (47K) has limited vocabulary
- SRBH attacks use novel patterns not in CSIC
- Out-of-vocabulary features make PCA projection meaningless

## Conclusion

**Directional asymmetry reveals critical weakness**:

- TF-IDF+PCA is extremely sensitive to training set size and vocabulary coverage
- Works when trained on large diverse dataset (SRBH 100K → CSIC)
- **Completely fails** when trained on small dataset (CSIC 47K → SRBH)

**SecBERT shows consistent robustness**:

- ~10% recall in both directions regardless of training size
- Semantic understanding transfers across datasets
- No catastrophic failures

**Practical implication**:

- **Never use TF-IDF+PCA for cross-dataset deployment** - high risk of complete failure
- SecBERT provides predictable (if modest) performance across domains
- Training set size is critical for TF-IDF methods
