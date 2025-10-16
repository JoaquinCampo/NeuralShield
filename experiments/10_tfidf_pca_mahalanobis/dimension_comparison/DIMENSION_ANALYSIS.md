# PCA Dimension Analysis: TF-IDF + Mahalanobis

## Executive Summary

Testing PCA dimensions from 50 to 3000 reveals **optimal performance at 500 components** with **diminishing returns above this threshold**.

**Key finding**: More dimensions do not improve detection. Beyond 500 components, performance degrades due to the curse of dimensionality affecting Mahalanobis distance.

## Results Table

| Components | Variance   | Recall     | F1-Score   | Precision  | True Positives |
| ---------- | ---------- | ---------- | ---------- | ---------- | -------------- |
| 50         | 83.67%     | 7.66%      | 13.61%     | 60.58%     | 1,921          |
| 100        | 87.72%     | 9.76%      | 17.02%     | 66.33%     | 2,447          |
| 200        | 90.47%     | 23.20%     | 36.19%     | 82.30%     | 5,814          |
| **300**    | **91.94%** | **27.08%** | **41.01%** | **84.45%** | **6,788**      |
| **500**    | **93.65%** | **27.92%** | **42.01%** | **84.84%** | **6,997**      |
| 1000       | 96.32%     | 19.53%     | 31.37%     | 79.66%     | 4,895          |
| 2000       | 99.08%     | 14.51%     | 24.28%     | 74.42%     | 3,636          |
| 3000       | 99.97%     | 13.48%     | 22.76%     | 73.00%     | 3,379          |

**Best performance: 500 components (27.92% recall, 42.01% F1)**

## Key Observations

### 1. Performance Peak at 500 Components

- Recall improves from 7.66% (50) → 27.92% (500)
- Beyond 500, recall **decreases** despite capturing more variance
- At 3000 components (99.97% variance), recall drops to 13.48%

### 2. The Curse of Dimensionality

**Why does more information hurt performance?**

Mahalanobis distance becomes unreliable in high dimensions:

1. **Covariance estimation degrades**: With 47K training samples and 3000 dimensions, the covariance matrix becomes poorly conditioned
2. **Distance concentration**: In high dimensions, all points become roughly equidistant
3. **Noise amplification**: Higher dimensions preserve more noise than signal

### 3. Variance vs Performance Disconnect

| Jump        | Variance Gain | Recall Change |
| ----------- | ------------- | ------------- |
| 500 → 1000  | +2.67%        | **-8.39%**    |
| 1000 → 2000 | +2.76%        | **-4.02%**    |
| 2000 → 3000 | +0.89%        | **-1.03%**    |

**Capturing more variance does not improve detection beyond 500 components.**

### 4. Sweet Spot: 300-500 Components

- 300 components: 27.08% recall, 91.94% variance
- 500 components: 27.92% recall, 93.65% variance
- Marginal gain: +0.84% recall for +1.71% variance

**Recommendation**: Use 500 components for best performance, or 300 for efficiency with minimal loss.

## Comparison to Previous Experiment

Original experiment used 300 components:

- Recall: 27.08%
- F1-Score: 41.01%

Optimized at 500 components:

- Recall: 27.92% (+3.1% improvement)
- F1-Score: 42.01% (+2.4% improvement)

**Small but measurable improvement.**

## Analysis: Why Performance Degrades

### Covariance Matrix Conditioning

With training set size N=47,000 and dimensionality d:

| Components | N/d Ratio | Status                 |
| ---------- | --------- | ---------------------- |
| 300        | 156.7     | Well-conditioned       |
| 500        | 94.0      | Acceptable             |
| 1000       | 47.0      | Marginal               |
| 2000       | 23.5      | Poor conditioning      |
| 3000       | 15.7      | Very poor conditioning |

**Rule of thumb**: Need N >> d for reliable covariance estimation. Performance degrades when N/d < 50.

### Distance Concentration

In high dimensions, the ratio of max distance to min distance approaches 1:

- Low dimensions: Clear separation between normal and anomalous
- High dimensions: All distances become similar, threshold becomes arbitrary

## Practical Recommendations

### Best Choice: 500 Components

**Why:**

- Highest recall (27.92%)
- Captures 93.65% variance
- N/d ratio (94) is still acceptable
- +3.1% better than 300 components

**When to use:**

- Production deployment
- Maximum detection coverage needed
- Computational cost acceptable

### Alternative: 300 Components

**Why:**

- Nearly as good (27.08% recall, only -0.84% vs 500)
- Faster computation (40% fewer dimensions)
- Better covariance conditioning (N/d = 156.7)

**When to use:**

- Computational efficiency important
- Real-time constraints
- Minimal performance trade-off acceptable

### Avoid: >1000 Components

Performance degrades significantly:

- 1000: -8.39% recall vs 500
- 2000: -13.41% recall vs 500
- 3000: -14.44% recall vs 500

## Conclusion

**Optimal PCA dimensionality is 500 components**, providing the best balance of:

1. Detection performance (27.92% recall)
2. Variance captured (93.65%)
3. Covariance estimation reliability (N/d = 94)

However, even at optimal dimensionality, **TF-IDF+PCA+Mahalanobis significantly underperforms SecBERT+Mahalanobis** (49.26% recall), making it unsuitable as a primary detector.

**The curse of dimensionality is real**: Adding dimensions beyond 500 hurts rather than helps, demonstrating that Mahalanobis distance requires careful dimensionality tuning to avoid degenerate behavior.
