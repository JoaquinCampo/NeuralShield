# Results: TF-IDF PCA Mahalanobis

## Overview

Tested Mahalanobis distance detection on TF-IDF embeddings with PCA dimensionality reduction to 300 components.

## Key Findings

### Mahalanobis Distance

#### With Preprocessing

- **PCA Explained Variance**: 91.94% (excellent compression)
- **Accuracy**: 61.00%
- **Precision**: 84.45%
- **Recall**: 27.08%
- **F1-Score**: 41.01%
- **FPR**: 5.00%
- **FNR**: 72.92%

#### Without Preprocessing

- **PCA Explained Variance**: 6.43% (very poor compression)
- **Accuracy**: 50.36%
- **Precision**: 64.49%
- **Recall**: 1.89%
- **F1-Score**: 3.67%
- **FPR**: 1.04%
- **FNR**: 98.11%

### Isolation Forest

#### With Preprocessing

- **PCA Explained Variance**: 91.94% (excellent compression)
- **Accuracy**: 58.26%
- **Precision**: 81.26%
- **Recall**: 21.62%
- **F1-Score**: 34.15%
- **FPR**: 5.00%
- **FNR**: 78.38%

#### Without Preprocessing

- **PCA Explained Variance**: 6.43% (very poor compression)
- **Accuracy**: 50.36%
- **Precision**: 64.49%
- **Recall**: 1.89%
- **F1-Score**: 3.67%
- **FPR**: 1.04%
- **FNR**: 98.11%

## Analysis

### Why PCA Works Better With Preprocessing

1. **With preprocessing**: PCA captures 91.94% variance in 300 dimensions

   - Text preprocessing creates more structured, lower-dimensional feature space
   - PCA effectively compresses meaningful signal

2. **Without preprocessing**: PCA captures only 6.43% variance in 300 dimensions
   - Raw text creates very sparse, high-dimensional features
   - Most information is in rare tokens, which PCA discards as noise
   - Results in catastrophic information loss

### Performance Comparison

The with-preprocessing version achieves:

- 14x better recall (27.08% vs 1.89%)
- 11x better F1-score (41.01% vs 3.67%)
- Better detection at the cost of slightly higher FPR

## Conclusion

**PCA dimensionality reduction is only viable with preprocessing**:

- Preprocessing creates dense, informative features that PCA can compress
- Raw TF-IDF is too sparse for effective PCA compression
- The 27% recall with preprocessing is still poor compared to other methods

**Recommendation**: PCA is not the right approach for this problem. The sparse nature of TF-IDF (even with preprocessing) means most detection power is in rare features that PCA tends to discard.
