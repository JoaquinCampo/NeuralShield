# Cross-Dataset Results: SRBH → CSIC (TF-IDF+PCA+Mahalanobis)

## Setup

- **Train**: SRBH dataset (100,000 normal samples)
- **Test**: CSIC dataset (25,000 normal, 25,065 attacks)
- **Method**: TF-IDF with preprocessing → PCA (300 dims) → Mahalanobis
- **Constraint**: 5% FPR

## Results

### TF-IDF+PCA+Mahalanobis (SRBH → CSIC)

- **Recall**: 16.89% (4,233 / 25,065 attacks detected)
- **Precision**: 79.33%
- **F1-Score**: 27.85%
- **FPR**: 4.41%
- **PCA Explained Variance**: 100.00%

### Comparison with Same-Dataset Performance

| Metric             | Same-Dataset (CSIC→CSIC) | Cross-Dataset (SRBH→CSIC) | Degradation |
| ------------------ | ------------------------ | ------------------------- | ----------- |
| Recall             | 27.08%                   | 16.89%                    | -37.6%      |
| Precision          | 84.45%                   | 79.33%                    | -6.1%       |
| F1-Score           | 41.01%                   | 27.85%                    | -32.1%      |
| Explained Variance | 91.94%                   | 100.00%                   | +8.8%       |

## Comparison with SecBERT Cross-Dataset

### SecBERT+Mahalanobis (SRBH → CSIC)

- **Recall**: 10.26% (2,571 / 25,065 attacks detected)
- **Precision**: 67.29%
- **F1-Score**: 17.80%
- **FPR**: 5.00%

### TF-IDF+PCA vs SecBERT Cross-Dataset

| Metric    | TF-IDF+PCA | SecBERT | TF-IDF Advantage |
| --------- | ---------- | ------- | ---------------- |
| Recall    | 16.89%     | 10.26%  | **+64.6%**       |
| Precision | 79.33%     | 67.29%  | +17.9%           |
| F1-Score  | 27.85%     | 17.80%  | +56.5%           |

**TF-IDF+PCA outperforms SecBERT in cross-dataset settings!**

## Analysis

### Why TF-IDF+PCA Generalizes Better

1. **Dataset-agnostic features**: TF-IDF captures token frequency patterns that transfer across datasets
2. **Simpler representations**: Less prone to overfitting on domain-specific semantics
3. **Character-level patterns**: Preserves URL encoding and special character anomalies
4. **100% variance explained**: SRBH has sparser vocabulary, allowing perfect PCA compression

### Why SecBERT Struggles

1. **Domain shift**: Pre-trained on security text, but SRBH vs CSIC have different attack distributions
2. **Semantic overfitting**: Learns dataset-specific semantic patterns that don't transfer
3. **Context dependence**: SecBERT's contextual embeddings are sensitive to domain shifts

### PCA Variance Anomaly

- **SRBH**: 100% variance explained (300 components)
- **CSIC**: 91.94% variance explained (300 components)

This suggests SRBH has a simpler feature distribution, possibly due to:

- More uniform attack patterns
- Less vocabulary diversity
- More structured request formats

## Conclusion

**Cross-dataset generalization reversal**:

- Same-dataset: SecBERT >> TF-IDF+PCA (49.26% vs 27.08% recall)
- Cross-dataset: TF-IDF+PCA > SecBERT (16.89% vs 10.26% recall)

**Key insight**: Simple frequency-based methods generalize better across datasets than semantic embeddings, despite lower same-dataset performance.

**Practical implication**: For deployment across diverse traffic sources, simpler methods like TF-IDF may be more robust than deep learning approaches.
