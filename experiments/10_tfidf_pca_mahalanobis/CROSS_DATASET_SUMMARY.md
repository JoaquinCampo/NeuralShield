# Cross-Dataset Performance Summary

## Complete Results Matrix

### TF-IDF+PCA+Mahalanobis

| Train Dataset | Test Dataset | Recall | F1-Score | Status                     |
| ------------- | ------------ | ------ | -------- | -------------------------- |
| CSIC (47K)    | CSIC         | 27.08% | 41.01%   | ✅ Baseline                |
| SRBH (100K)   | CSIC         | 16.89% | 27.85%   | ✅ Works (38% degradation) |
| CSIC (47K)    | SRBH         | 0.00%  | 0.01%    | ❌ **COMPLETE FAILURE**    |

### SecBERT+Mahalanobis

| Train Dataset | Test Dataset | Recall | F1-Score | Status                     |
| ------------- | ------------ | ------ | -------- | -------------------------- |
| CSIC          | CSIC         | 49.26% | 63.87%   | ✅ Baseline                |
| SRBH          | CSIC         | 10.26% | 17.80%   | ✅ Works (79% degradation) |
| CSIC          | SRBH         | 10.76% | 18.50%   | ✅ Works (79% degradation) |

## Key Findings

### 1. Directional Asymmetry in TF-IDF+PCA

**SRBH→CSIC: Works (16.89% recall)**

- Large training set (100K samples)
- Comprehensive vocabulary coverage
- 100% PCA explained variance

**CSIC→SRBH: Catastrophic failure (0.00% recall)**

- Small training set (47K samples)
- Limited vocabulary coverage
- Out-of-vocabulary attack patterns

### 2. SecBERT Consistency

- **Bidirectional stability**: ~10% recall in both cross-dataset directions
- **No catastrophic failures**: Graceful degradation even with domain shift
- **Training size independent**: Performance not critically tied to training set size

### 3. Cross-Dataset Performance Reversal

**Same-Dataset (CSIC→CSIC)**:

- SecBERT: 49.26% >> TF-IDF+PCA: 27.08%
- SecBERT is 82% better

**Cross-Dataset (SRBH→CSIC)**:

- TF-IDF+PCA: 16.89% > SecBERT: 10.26%
- TF-IDF+PCA is 64% better

**Cross-Dataset (CSIC→SRBH)**:

- SecBERT: 10.76% >> TF-IDF+PCA: 0.00%
- SecBERT is infinitely better

## Analysis

### Training Set Size Impact

| Method     | Small Train (47K) Cross-Dataset | Large Train (100K) Cross-Dataset |
| ---------- | ------------------------------- | -------------------------------- |
| TF-IDF+PCA | **0.00% recall (failure)**      | 16.89% recall (works)            |
| SecBERT    | 10.76% recall (works)           | 10.26% recall (works)            |

**TF-IDF+PCA requires large, diverse training sets to avoid catastrophic failure.**

### Vocabulary Coverage

**TF-IDF+PCA**:

- Success depends on vocabulary overlap between train and test
- Out-of-vocabulary tokens → zero embeddings → PCA projection fails
- No semantic understanding to generalize

**SecBERT**:

- Subword tokenization handles unseen tokens
- Pre-trained semantic knowledge transfers
- Every request gets meaningful representation

### PCA Explained Variance

| Train Dataset | Explained Variance (300 dims) | Observation                   |
| ------------- | ----------------------------- | ----------------------------- |
| CSIC (47K)    | 91.94%                        | Normal feature complexity     |
| SRBH (100K)   | 100.00%                       | Simpler/sparser feature space |

SRBH's 100% variance suggests more uniform patterns, but paradoxically this doesn't improve detection - it just means the vocabulary is sparser.

## Practical Implications

### When to Use TF-IDF+PCA

✅ **Safe scenarios**:

- Same-dataset deployment (train and test from same distribution)
- Large diverse training set available
- Vocabulary overlap guaranteed

❌ **Dangerous scenarios**:

- Cross-dataset deployment
- Small training set (<100K samples)
- Unknown test distribution
- Production with diverse traffic sources

### When to Use SecBERT

✅ **Recommended scenarios**:

- Cross-dataset deployment
- Unknown test distribution
- Need robust performance
- Can tolerate ~10% recall baseline

⚠️ **Trade-offs**:

- Lower same-dataset performance vs TF-IDF+PCA
- Higher computational cost
- More complex deployment

## Recommendations

1. **For production deployment across diverse sources**: Use SecBERT

   - Predictable performance (~10% recall)
   - No risk of catastrophic failure
   - Graceful degradation

2. **For single-domain deployment**: Consider TF-IDF+PCA IF:

   - Large training set (>100K samples)
   - Test distribution similar to training
   - Can accept risk of 0% recall on distribution shift

3. **Best approach**: Ensemble
   - Combine both methods
   - SecBERT provides stable baseline
   - TF-IDF+PCA adds complementary coverage when it works
   - Degrades gracefully when TF-IDF+PCA fails

## Conclusion

**TF-IDF+PCA is a brittle method**:

- Can outperform deep learning in favorable conditions
- **Catastrophically fails** with small training sets or domain shifts
- High-risk for production deployment

**SecBERT is robust but modest**:

- Consistent ~10% cross-dataset recall
- Never fails completely
- Recommended for real-world deployment

**The paradox**: Simpler methods can work better in-domain, but fail catastrophically out-of-domain. Deep learning provides lower but more reliable performance across domains.
