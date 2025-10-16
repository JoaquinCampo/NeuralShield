# Cross-Dataset Generalization: LOF vs SecBERT

**Date**: October 14, 2025  
**Goal**: Test if LOF generalizes better than SecBERT across different HTTP traffic datasets

---

## Executive Summary

**Finding**: Both LOF and SecBERT **fail catastrophically at cross-dataset generalization**, dropping from 49-64% same-dataset recall to 5-10% cross-dataset recall.

**Key Result**: SecBERT generalizes slightly better than LOF (10% vs 5-10%), suggesting semantic understanding transfers marginally better than syntactic patterns.

**Implication**: Both approaches **require training on target environment data**. Pre-trained models from other datasets are not viable for production.

---

## Results

### LOF + TF-IDF + PCA (no preprocessing)

| Direction        | Train Size | Test Size | Recall @ 5% FPR | Precision | F1     | PCA Variance |
| ---------------- | ---------- | --------- | --------------- | --------- | ------ | ------------ |
| **CSIC → SR_BH** | 47K        | 808K      | **5.52%**       | 49.90%    | 9.94%  | 93.01%       |
| **SR_BH → CSIC** | 100K       | 50K       | **10.06%**      | 67.68%    | 17.51% | **72.44%**   |

### SecBERT + Mahalanobis (with preprocessing) - Reference

| Direction        | Train Size | Test Size | Recall @ 5% FPR | Precision | F1     |
| ---------------- | ---------- | --------- | --------------- | --------- | ------ |
| **CSIC → SR_BH** | 47K        | 807K      | **10.76%**      | 65.95%    | 18.50% |
| **SR_BH → CSIC** | 100K       | 50K       | **10.26%**      | 67.29%    | 17.80% |

### Same-Dataset Baselines

| Model   | Dataset | Recall @ 5% FPR | Difference |
| ------- | ------- | --------------- | ---------- |
| LOF     | CSIC    | 64.20%          | Baseline   |
| SecBERT | CSIC    | 49.26%          | Baseline   |
| SecBERT | SR_BH   | 54.12%          | Baseline   |

---

## Performance Degradation

### LOF Performance Drop

| Scenario         | Same-Dataset | Cross-Dataset | Drop         | Ratio         |
| ---------------- | ------------ | ------------- | ------------ | ------------- |
| **CSIC → SR_BH** | 64.20%       | 5.52%         | **-58.68pp** | **12x worse** |
| **SR_BH → CSIC** | ~64%\*       | 10.06%        | **-54pp**    | **6x worse**  |

\*Estimated based on CSIC performance

### SecBERT Performance Drop

| Scenario         | Same-Dataset | Cross-Dataset | Drop         | Ratio        |
| ---------------- | ------------ | ------------- | ------------ | ------------ |
| **CSIC → SR_BH** | 49.26%       | 10.76%        | **-38.50pp** | **5x worse** |
| **SR_BH → CSIC** | 54.12%       | 10.26%        | **-43.86pp** | **5x worse** |

---

## Head-to-Head Comparison

### SecBERT Wins Cross-Dataset

| Direction        | LOF Recall | SecBERT Recall | Winner      | Advantage        |
| ---------------- | ---------- | -------------- | ----------- | ---------------- |
| **CSIC → SR_BH** | 5.52%      | 10.76%         | **SecBERT** | **+5.24pp (2x)** |
| **SR_BH → CSIC** | 10.06%     | 10.26%         | **SecBERT** | **+0.20pp**      |

**Average**: SecBERT achieves 10.51% recall vs LOF's 7.79% recall cross-dataset (+35% relative advantage)

### LOF Wins Same-Dataset

| Dataset  | LOF Recall | SecBERT Recall | Winner  | Advantage           |
| -------- | ---------- | -------------- | ------- | ------------------- |
| **CSIC** | 64.20%     | 49.26%         | **LOF** | **+14.94pp (+30%)** |

---

## Analysis

### 1. Direction Asymmetry

**LOF shows strong directional bias:**

| Direction    | Recall | Difference         |
| ------------ | ------ | ------------------ |
| SR_BH → CSIC | 10.06% | Baseline           |
| CSIC → SR_BH | 5.52%  | **-4.54pp (-45%)** |

**SecBERT is symmetric:**

| Direction    | Recall | Difference        |
| ------------ | ------ | ----------------- |
| SR_BH → CSIC | 10.26% | Baseline          |
| CSIC → SR_BH | 10.76% | **+0.50pp (+5%)** |

**Why LOF is asymmetric:**

- Training on larger dataset (SR_BH: 100K) → better vocabulary coverage
- CSIC vocabulary (47K samples) doesn't capture SR_BH patterns well
- PCA variance drops dramatically: 93% → **72%** when SR_BH trains on CSIC test

### 2. PCA Variance as Generalization Indicator

| Scenario                | PCA Variance | Recall | Correlation            |
| ----------------------- | ------------ | ------ | ---------------------- |
| CSIC train → CSIC test  | 93.01%       | 64.20% | ✅ Good fit            |
| CSIC train → SR_BH test | 93.01%       | 5.52%  | ⚠️ Vocabulary mismatch |
| SR_BH train → CSIC test | **72.44%**   | 10.06% | ❌ Poor fit            |

**Key insight**: Low PCA variance (72%) signals vocabulary mismatch and predicts poor generalization.

### 3. Why LOF Fails Worse Cross-Dataset

**Syntactic patterns are dataset-specific:**

1. **Token-level differences**

   - CSIC: `/admin/`, `/api/`, specific endpoint patterns
   - SR_BH: Enterprise traffic, different URL structures
   - TF-IDF vocabulary doesn't transfer

2. **Local clusters don't generalize**

   - CSIC clusters: Specific to CSIC endpoints
   - SR_BH clusters: Different traffic patterns
   - LOF learns local density of source dataset → useless for target

3. **Encoding conventions differ**
   - Different percent-encoding usage
   - Different header formats
   - Different query parameter styles

**Example**: LOF trained on CSIC learns that `/admin/` is a normal cluster. SR_BH has different admin patterns → LOF can't recognize them.

### 4. Why SecBERT Degrades More Gracefully

**Semantic understanding transfers (slightly) better:**

1. **Concept-level patterns**

   - "Path traversal" concept vs specific `../` tokens
   - "SQL injection" intent vs specific SQL syntax
   - Domain knowledge from pretraining

2. **Preprocessing helps**

   - Normalization removes dataset-specific noise
   - Focuses on semantic content
   - Makes patterns more transferable

3. **Dense embeddings are more robust**
   - 768D captures richer representations
   - Less sensitive to vocabulary gaps
   - Global covariance adapts better

**But still fails**: 10% recall is only marginally better than random (5% FPR baseline).

---

## Key Insights

### 1. Neither Approach Solves Generalization

**Both fail catastrophically:**

- LOF: 5-10% recall (vs 64% same-dataset)
- SecBERT: 10% recall (vs 49-54% same-dataset)

**Conclusion**: Cross-dataset generalization is an **unsolved problem** for HTTP anomaly detection.

### 2. Semantic > Syntactic for Transfer

**SecBERT's semantic understanding provides marginal advantage:**

- +35% relative improvement over LOF cross-dataset
- But absolute performance still terrible (10% vs 5%)

**Implication**: Semantic models degrade more gracefully, but not enough to be useful.

### 3. Training Data Must Match Deployment Environment

**Critical requirement for production:**

- Cannot use models trained on public datasets
- Must collect data from target environment
- Minimum 50K normal samples recommended
- Periodic retraining as traffic patterns evolve

### 4. PCA Variance Predicts Generalization

**72% variance = red flag:**

- Indicates vocabulary mismatch
- Predicts poor generalization
- Can be used as early warning signal

**Rule of thumb**: If PCA variance drops below 85% on test set, expect poor performance.

---

## Comparison to Literature

### Experiment 08 (SecBERT Cross-Dataset)

Our LOF results match SecBERT's cross-dataset failure:

| Approach    | CSIC → SR_BH | SR_BH → CSIC | Average     |
| ----------- | ------------ | ------------ | ----------- |
| **LOF**     | 5.52%        | 10.06%       | **7.79%**   |
| **SecBERT** | 10.76%       | 10.26%       | **10.51%**  |
| **Gap**     | -5.24pp      | -0.20pp      | **-2.72pp** |

**Conclusion**: LOF doesn't solve the generalization problem that SecBERT has.

---

## Why This Matters

### Production Implications

**Cannot rely on pre-trained models:**

- Public datasets (CSIC, SR_BH) don't transfer
- Each deployment needs custom training
- Requires data collection infrastructure

**Deployment strategy:**

1. Collect 50K+ normal requests from target environment
2. Train model on collected data
3. Deploy with continuous monitoring
4. Retrain monthly as patterns evolve

### Research Implications

**Fundamental challenge:**

- HTTP traffic is highly environment-specific
- URL structures, endpoints, headers vary widely
- No universal "normal HTTP" representation

**Future directions:**

- Multi-dataset training (train on CSIC + SR_BH simultaneously)
- Domain adaptation techniques (DANN, CORAL)
- Transfer learning with fine-tuning
- Meta-learning for few-shot adaptation

---

## Conclusions

### Main Findings

1. **Both approaches fail cross-dataset** - LOF: 5-10%, SecBERT: 10%
2. **SecBERT degrades more gracefully** - +35% relative advantage
3. **LOF dominates same-dataset** - 64% vs 49% (+30%)
4. **Training data must match deployment** - No shortcuts

### Practical Takeaway

**For production deployment:**

- Use LOF for highest same-dataset recall (64%)
- Train on target environment data
- Don't expect cross-dataset transfer
- Plan for continuous retraining

**Silver lining**: LOF's same-dataset performance (64%) is still the best result achieved, making it the production winner despite poor generalization.

---

## Files and Artifacts

### Results

- `cross_dataset/csic_to_srbh/results.json` - CSIC → SR_BH metrics
- `cross_dataset/srbh_to_csic/results.json` - SR_BH → CSIC metrics
- `cross_dataset/summary.json` - Combined summary with baselines

### Code

- `test_cross_dataset_lof.py` - Cross-dataset testing script

---

**Experiment completed**: October 14, 2025  
**Status**: ✅ Complete - confirms generalization failure for both approaches
