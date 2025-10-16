# TF-IDF + PCA + Mahalanobis Cross-Dataset Results

## Experiment Overview

**Objective**: Test whether preprocessing improves cross-dataset generalization for anomaly detection.

**Method**: TF-IDF (5000 features) → PCA (300 dimensions) → Mahalanobis distance

**Datasets**:

- SR_BH_2020: 100,000 training samples
- CSIC: 47,000 training samples

**Threshold**: 95th percentile of normal scores (≈5% FPR target)

---

## Results Summary

### SR_BH → CSIC (Train on SR_BH, Test on CSIC)

| Variant                   | Recall | Precision | F1-Score | FPR    | TP  | FP    | TN     | FN     |
| ------------------------- | ------ | --------- | -------- | ------ | --- | ----- | ------ | ------ |
| **With Preprocessing**    | 2.94%  | 38.98%    | 5.46%    | 4.61%  | 736 | 1,152 | 23,848 | 24,329 |
| **Without Preprocessing** | 2.85%  | 40.16%    | 5.32%    | 4.26%  | 714 | 1,064 | 23,936 | 24,351 |
| **Δ (Improvement)**       | +0.09% | -1.18%    | +0.14%   | +0.35% | +22 | +88   | -88    | -22    |

**Key Findings**:

- Minimal recall improvement (+0.09%)
- Slightly higher FPR with preprocessing (+0.35%)
- Both variants show very poor cross-dataset performance (< 3% recall)

---

### CSIC → SR_BH (Train on CSIC, Test on SR_BH)

| Variant                   | Recall | Precision | F1-Score | FPR    | TP     | FP     | TN      | FN      |
| ------------------------- | ------ | --------- | -------- | ------ | ------ | ------ | ------- | ------- |
| **With Preprocessing**    | 6.78%  | 54.95%    | 12.07%   | 5.00%  | 25,936 | 21,260 | 403,935 | 356,684 |
| **Without Preprocessing** | 6.17%  | 52.63%    | 11.05%   | 5.00%  | 23,616 | 21,254 | 403,941 | 359,004 |
| **Δ (Improvement)**       | +0.61% | +2.32%    | +1.02%   | ±0.00% | +2,320 | +6     | -6      | -2,320  |

**Key Findings**:

- Modest recall improvement (+0.61%)
- Better precision with preprocessing (+2.32%)
- Both variants still show poor cross-dataset performance (< 7% recall)
- FPR constraint maintained at 5.00% for both

---

## Analysis

### Preprocessing Impact

**Marginal Improvements**:

- SR_BH → CSIC: +0.09% recall gain
- CSIC → SR_BH: +0.61% recall gain

**Conclusion**: Preprocessing provides minimal benefit for cross-dataset generalization with this approach.

### Cross-Dataset Generalization

**Poor Performance Across All Variants**:

- Best recall: 6.78% (CSIC → SR_BH with preprocessing)
- Worst recall: 2.85% (SR_BH → CSIC without preprocessing)

**Root Cause**: Dataset distribution mismatch. Models trained on one dataset fail to generalize to attacks from another dataset.

### PCA Explained Variance

| Training Dataset     | With Preprocessing | Without Preprocessing |
| -------------------- | ------------------ | --------------------- |
| SR_BH (100k samples) | 86.85%             | 86.85%                |
| CSIC (47k samples)   | 95.39%             | 95.39%                |

**Observation**: CSIC shows higher explained variance with 300 components, suggesting more concentrated feature space.

---

## Comparison to Within-Dataset Results

From previous experiments (Experiment 10):

### SR_BH (Within-Dataset)

- With preprocessing: 93.62% recall @ 5% FPR
- Without preprocessing: 93.75% recall @ 5% FPR

### CSIC (Within-Dataset)

- With preprocessing: 99.89% recall @ 5% FPR
- Without preprocessing: 99.93% recall @ 5% FPR

**Cross-Dataset Performance Drop**:

- SR_BH → CSIC: 93.62% → 2.94% (-90.68 pp)
- CSIC → SR_BH: 99.89% → 6.78% (-93.11 pp)

---

## Conclusions

1. **Preprocessing does not solve cross-dataset generalization**: Gains are marginal (< 1% recall improvement)

2. **Severe distribution mismatch**: 90+ percentage point drops in recall when testing cross-dataset

3. **Dataset-specific features dominate**: TF-IDF + PCA learns dataset-specific patterns that don't transfer

4. **Preprocessing is not the bottleneck**: The fundamental issue is the approach (sparse features + linear dimensionality reduction) rather than preprocessing

---

## Recommendations

1. **Abandon cross-dataset evaluation for this approach**: TF-IDF + PCA + Mahalanobis is not suitable for generalization

2. **Focus on within-dataset optimization**: This approach works well within datasets (> 93% recall)

3. **Explore semantic embeddings for cross-dataset**: Dense embeddings (SecBERT, etc.) may capture transferable attack patterns

4. **Consider ensemble approaches**: Combine multiple dataset-specific models rather than seeking universal detector

---

## Files Generated

```
experiments/14_preprocessing_cross_dataset/tfidf_pca_mahalanobis/
├── srbh_to_csic_with_prep/
│   ├── confusion_matrix.png
│   ├── score_distribution.png
│   ├── results.json
│   └── mahalanobis_detector.joblib
├── srbh_to_csic_without_prep/
│   ├── confusion_matrix.png
│   ├── score_distribution.png
│   ├── results.json
│   └── mahalanobis_detector.joblib
├── csic_to_srbh_with_prep/
│   ├── confusion_matrix.png
│   ├── score_distribution.png
│   ├── results.json
│   └── mahalanobis_detector.joblib
└── csic_to_srbh_without_prep/
    ├── confusion_matrix.png
    ├── score_distribution.png
    ├── results.json
    └── mahalanobis_detector.joblib
```
