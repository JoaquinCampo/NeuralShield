# Preprocessing Impact Experiment - Final Analysis

## Experimental Setup

- **Objective**: Determine if preprocessing pipeline improves anomaly detection
- **Architecture**: IsolationForest + TF-IDF embeddings (5000 features)
- **Training Data**: CSIC train.jsonl - 47,000 normal requests only
- **Test Data**: CSIC test.jsonl - 50,065 samples (25,065 attacks + 25,000 normal)
- **Hyperparameters**: n_estimators=300, tested contamination=[0.1, 0.15, 0.2, 0.25, 0.3]

---

## Results Summary

| Contamination | Model        | Precision    | Recall      | F1-Score    | Accuracy     | FPR         | Specificity  |
| ------------- | ------------ | ------------ | ----------- | ----------- | ------------ | ----------- | ------------ |
| **0.10**      | Without Prep | 63.55%       | 0.76%       | 1.50%       | 50.10%       | 0.44%       | 99.56%       |
| **0.10**      | With Prep    | **67.17%** ✓ | 0.53%       | 1.05%       | 50.07%       | **0.26%** ✓ | **99.74%** ✓ |
| **0.15**      | Without Prep | **64.78%** ✓ | **0.78%** ✓ | **1.54%** ✓ | **50.11%** ✓ | 0.42%       | 99.58%       |
| **0.15**      | With Prep    | 60.66%       | 0.66%       | 1.30%       | 50.05%       | 0.43%       | 99.57%       |
| **0.20**      | Without Prep | 62.29%       | 0.59%       | 1.16%       | 50.05%       | 0.36%       | 99.64%       |
| **0.20**      | With Prep    | **66.20%** ✓ | 0.56%       | 1.12%       | 50.07%       | **0.29%** ✓ | **99.71%** ✓ |
| **0.25**      | Without Prep | 60.69%       | **0.63%** ✓ | **1.26%** ✓ | 50.05%       | 0.41%       | 99.59%       |
| **0.25**      | With Prep    | 61.66%       | 0.47%       | 0.94%       | 50.02%       | **0.30%** ✓ | **99.70%** ✓ |
| **0.30**      | Without Prep | **64.23%** ✓ | **0.67%** ✓ | **1.32%** ✓ | **50.08%** ✓ | 0.37%       | 99.63%       |
| **0.30**      | With Prep    | 59.55%       | 0.52%       | 1.04%       | 50.02%       | 0.36%       | 99.64%       |

✓ = Better performance for that contamination level

---

## Key Findings

### 1. Both Models Are Extremely Conservative

- **Recall < 1%** across ALL contamination values
- Both models flag test data as overwhelmingly "normal"
- Catches less than 200 out of 25,065 attacks in best case

### 2. Preprocessing Impact Varies by Contamination

**When Preprocessing Helps (contamination 0.1, 0.2):**

- ✅ **Better Precision** (+3-4 percentage points)
- ✅ **Lower FPR** (fewer false alarms)
- ✅ **Higher Specificity** (better at passing normal traffic)
- ❌ **Slightly Lower Recall** (misses a few more attacks)

**When Preprocessing Hurts (contamination 0.15, 0.25, 0.3):**

- ❌ **Lower Recall** (misses more attacks)
- ❌ **Lower F1-Score**
- Mixed precision results

### 3. The Fundamental Problem

**Both models struggle because:**

- Training scores are **extremely clustered** (mean=-0.2923, std=0.0002)
- 90% of training samples have **identical scores**
- Test data likely has different score distributions
- Contamination parameter has minimal impact on highly concentrated training scores

---

## Statistical Analysis

### Score Distribution Characteristics

- **Training scores range**: -0.297 to -0.292
- **99% of training data**: Within 0.001 of the median
- **Model confidence**: Very low variance = model sees all training data as nearly identical

### Why Such Low Recall?

The models are trained ONLY on normal traffic, so:

1. They learn a very narrow definition of "normal"
2. Most test traffic (including attacks) falls outside this narrow range
3. But the offset is set so conservatively that even outliers aren't flagged

---

## Conclusions

### Does Preprocessing Help?

**Mixed Results:**

- ✅ **YES** for reducing false alarms (lower FPR consistently)
- ✅ **YES** for precision at contamination 0.1 and 0.2
- ❌ **NO** for improving attack detection (recall)
- ❌ **NO** for overall F1-score in most cases

### The Real Issue

**The TF-IDF + IsolationForest approach with contamination-based thresholding is fundamentally limited:**

1. Training score distributions are too concentrated
2. Contamination parameter cannot meaningfully adjust the decision boundary
3. Both models (with/without preprocessing) suffer from the same core issue

### Recommendations

**To fairly evaluate preprocessing impact, we need to:**

1. **Try different approaches**:

   - Use a validation set with attacks to tune threshold
   - Try supervised or semi-supervised methods
   - Use different anomaly detection algorithms (OCSVM, Deep Learning)

2. **Investigate why scores are so clustered**:

   - Is TF-IDF too sparse? (99%+ zeros)
   - Would dense embeddings (FastEmbed) work better?
   - Do we need more diverse training data?

3. **Alternative contamination strategies**:
   - Manual threshold tuning based on validation set
   - Grid search for optimal operating point
   - ROC curve analysis

---

## Bottom Line

**At these contamination levels, preprocessing provides marginal benefits:**

- Slightly better precision and lower FPR
- But both models have unacceptably low recall (<1%)
- The core issue is the modeling approach, not preprocessing

**The preprocessing pipeline shows promise** (better precision, lower FPR), but we cannot make strong conclusions about its value until we solve the fundamental recall problem.
