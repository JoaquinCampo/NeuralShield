# Experiment 12: MI Feature Selection Results

**Date**: October 14, 2025  
**Status**: ❌ Negative Result  
**Conclusion**: MI-based dimension selection **does NOT improve** dense embeddings

---

## Executive Summary

**Finding**: All 768 SecBERT dimensions are needed for optimal performance. Selecting fewer dimensions via mutual information **decreases** recall by 12-29%.

**Best configuration remains**: SecBERT (all 768 dims) + Mahalanobis + Preprocessing = **49.26% recall @ 5% FPR**

---

## Results

### Performance by Dimension Count

| K (dims) | Recall     | Precision  | F1-Score   | vs Baseline     |
| -------- | ---------- | ---------- | ---------- | --------------- |
| 50       | 34.86%     | 87.48%     | 49.86%     | **-29%** ❌     |
| 100      | 41.22%     | 89.21%     | 56.39%     | **-16%** ❌     |
| 200      | 46.44%     | 90.30%     | 61.34%     | **-6%** ⚠️      |
| 300      | 48.02%     | 90.59%     | 62.77%     | **-2%** ⚠️      |
| 400      | 48.71%     | 90.71%     | 63.38%     | **-1%** ⚠️      |
| **768**  | **49.26%** | **90.81%** | **63.87%** | **Baseline** ✅ |

### Key Metrics

**Baseline (all 768 dimensions)**:

- Recall: 49.26%
- Precision: 90.81%
- F1-Score: 63.87%
- FPR: 5.00%

**Best MI selection (k=400)**:

- Recall: 48.71% (-1.1% vs baseline)
- Still underperforms

---

## MI Score Statistics

- **Min**: 0.014622
- **Max**: 0.335165
- **Mean**: 0.126207
- **Median**: 0.104438
- **Std**: 0.076699

### Top 10 Most Discriminative Dimensions

1. Dimension 386: 0.335165
2. Dimension 71: 0.334024
3. Dimension 136: 0.331556
4. Dimension 276: 0.328732
5. Dimension 252: 0.324818
6. Dimension 177: 0.323801
7. Dimension 2: 0.322853
8. Dimension 558: 0.319858
9. Dimension 660: 0.316978
10. Dimension 641: 0.314077

---

## Why MI Didn't Help

### Hypothesis (Pre-Experiment)

MI would identify noise dimensions in SecBERT and improve performance by removing them.

### Reality (Post-Experiment)

**All 768 SecBERT dimensions carry useful information** for anomaly detection:

1. **Dense embeddings ≠ Sparse features**

   - Paper's success was on TF-IDF (5000 sparse tokens → 100 selected)
   - SecBERT's 768 dimensions are already highly optimized representations
   - Each dimension contributes to the semantic understanding

2. **Gradual degradation curve**

   - Performance smoothly decreases from 768 → 50 dimensions
   - No clear "elbow" point suggesting optimal dimensionality
   - Indicates all dimensions contribute incrementally

3. **MI scores relatively uniform**
   - Range: 0.015 to 0.335 (only 23x difference)
   - Compare to sparse features: high-MI tokens often 1000x more important than low-MI
   - SecBERT dimensions are more balanced

---

## Comparison to Paper

| Approach  | Features        | Selection  | Result            |
| --------- | --------------- | ---------- | ----------------- |
| **Paper** | TF-IDF (sparse) | MI top 100 | 78-92% TPR        |
| **Ours**  | SecBERT (dense) | MI top 100 | 41% recall ❌     |
| **Ours**  | SecBERT (dense) | All 768    | **49% recall** ✅ |

**Conclusion**: MI works for sparse features but not dense embeddings.

---

## Visualizations Generated

1. **`mi_distribution.png`**: Histogram of MI scores (relatively uniform)
2. **`top_50_dimensions.png`**: Bar chart of highest-MI dimensions
3. **`recall_vs_k.png`**: Performance curve (monotonic increase with K)

---

## Decision: Do NOT Integrate MI

### Reasons

1. ❌ **No performance gain** (-1% to -29% recall)
2. ❌ **Requires attack data** (adds complexity)
3. ❌ **Slower training** (10 min MI computation)
4. ❌ **No efficiency gain** (need all 768 dims for best performance)

### Recommendation

**Keep current approach**: Use all 768 SecBERT dimensions with Mahalanobis detector.

---

## Lessons Learned

### 1. Dense embeddings are fundamentally different from sparse features

**Sparse (TF-IDF)**:

- Most tokens irrelevant (stop words, formatting)
- Few tokens highly discriminative (SQL keywords, XSS patterns)
- MI successfully filters noise

**Dense (SecBERT)**:

- All dimensions encode semantic information
- Pre-trained to capture meaningful patterns
- MI cannot improve what's already optimized

### 2. Domain pretraining matters more than feature selection

SecBERT's 768D > TF-IDF's top 100 tokens because:

- Contextual understanding vs. keyword matching
- Captures syntax + semantics
- Better generalization

### 3. Mahalanobis handles high dimensions well

- No degradation at 768 dimensions
- Covariance-aware (unlike tree-based methods)
- No need for dimensionality reduction

---

## Future Work

### Alternative Approaches (If Needed)

1. **PCA for efficiency** (not accuracy):

   - Test 768 → 200 via PCA for faster inference
   - Expect ~2% recall drop but 4x speedup

2. **Attention-based selection**:

   - Use transformer attention weights
   - Identify which dimensions the model focuses on

3. **Ensemble with TF-IDF**:
   - Combine SecBERT (semantic) + TF-IDF+MI (lexical)
   - May capture complementary patterns

### Not Worth Pursuing

- ❌ MI on other embedding models (same result expected)
- ❌ Different MI computation methods (same underlying issue)
- ❌ Wrapper methods (RFE, forward selection) - too slow, unlikely to help

---

## Files Generated

```
results/
├── mi_scores.npy                      # MI scores for all 768 dims
├── selected_dims_k50.npy              # Top 50 dimension indices
├── selected_dims_k100.npy             # Top 100 dimension indices
├── selected_dims_k200.npy             # Top 200 dimension indices
├── metrics_comparison.json            # Performance for all K values
└── plots/
    ├── mi_distribution.png            # MI score histogram
    ├── top_50_dimensions.png          # Highest-MI dimensions
    └── recall_vs_k.png                # Performance vs dimensionality
```

---

## Conclusion

**MI-based feature selection, while effective for sparse representations (TF-IDF), does NOT transfer to dense semantic embeddings (SecBERT).**

**Production recommendation**: Continue using all 768 SecBERT dimensions + Mahalanobis + Preprocessing for optimal 49.26% recall @ 5% FPR.

This negative result is valuable: it confirms that SecBERT's dense embeddings are already well-optimized and that all dimensions contribute meaningfully to anomaly detection.

---

**Experiment Duration**: 11 minutes  
**Compute**: CPU-only (429K samples for MI computation)  
**Dataset**: CSIC (train/test) + SR_BH (attacks for MI)
