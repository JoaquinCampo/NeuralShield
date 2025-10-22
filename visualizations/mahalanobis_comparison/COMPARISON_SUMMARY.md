# SecBERT vs BGE-small Mahalanobis Comparison

**Date**: October 12, 2025  
**Models Compared**: SecBERT-Mahalanobis vs BGE-Mahalanobis (both with preprocessing)

---

## Executive Summary

**SecBERT catches 3,696 attacks that BGE misses**, while BGE catches 1,364 attacks that SecBERT misses. This reveals **complementary detection patterns** - an ensemble could achieve significantly higher recall.

---

## Detection Statistics

| Metric                  | SecBERT-Mahalanobis | BGE-Mahalanobis | Difference      |
| ----------------------- | ------------------- | --------------- | --------------- |
| **Total Flagged**       | 13,597              | 11,265          | +2,332          |
| **True Attacks Caught** | 12,347 (49.26%)     | 10,015 (39.96%) | +2,332 (+9.3pp) |
| **False Positives**     | 1,250 (5.00%)       | 1,250 (5.00%)   | 0               |
| **Agreement Rate**      | -                   | 87.9%           | -               |

---

## Attack Detection Overlap

### Universally Caught (Both Models)

- **8,651 attacks** (34.5% of all attacks)
- These are "easy" attacks with clear anomalous patterns
- Both embedding models recognize them

### Universally Missed (Both Models)

- **11,354 attacks** (45.3% of all attacks)
- These are "hard" attacks that look similar to normal traffic
- May require different detection approaches (not just embeddings)

### SecBERT Unique Catches

- **3,696 attacks** (14.7% of all attacks)
- Only SecBERT detects these
- Likely attacks leveraging cybersecurity-specific patterns
- Examples: SQL injection syntax, XSS payloads, security-specific keywords

### BGE Unique Catches

- **1,364 attacks** (5.4% of all attacks)
- Only BGE detects these
- Likely attacks with general semantic anomalies
- BGE's smaller embedding space (384d) may capture different features

---

## False Positive Overlap

| Overlap Type             | Count | Percentage       |
| ------------------------ | ----- | ---------------- |
| **Both Models Flag**     | 752   | 43.0% of all FPs |
| **Only One Model Flags** | 996   | 57.0% of all FPs |

**Insight**: 57% of false positives are unique to one model, suggesting:

- Different models make different mistakes
- Ensemble with voting could reduce FPs

---

## False Negative Overlap

| Overlap Type              | Count  | Percentage       |
| ------------------------- | ------ | ---------------- |
| **Both Models Miss**      | 11,354 | 69.2% of all FNs |
| **Only One Model Misses** | 5,060  | 30.8% of all FNs |

**Key Finding**: 30.8% of missed attacks are caught by one model but not the other. This is the **ensemble opportunity zone**.

---

## Ensemble Potential

### Simple Union (Flag if either model flags)

**Expected Performance:**

```
Caught Attacks:
- Both models:     8,651
- SecBERT only:    3,696
- BGE only:        1,364
─────────────────────────
Total:            13,711 attacks (54.7% recall)

False Positives:
- Both models:       752
- One model only:    996
─────────────────────────
Total:             1,748 FPs (7.0% FPR)
```

**Result**: +5.4pp recall improvement, but +2.0pp FPR increase

### Weighted Voting (Flag if combined score exceeds threshold)

**Approach**: Average Mahalanobis distances

```python
combined_score = 0.6 * secbert_distance + 0.4 * bge_distance
```

**Expected**: ~53-54% recall @ 5% FPR (tuned threshold)

---

## Why Models Catch Different Attacks

### SecBERT Advantages (Unique Catches: 3,696)

1. **Security vocabulary**: Pre-trained on cybersecurity text

   - Recognizes `UNION SELECT`, `OR 1=1`, `<script>`
   - Better understanding of attack syntax

2. **Higher dimensionality**: 768 dims capture nuanced patterns

   - More expressive embeddings
   - Can separate subtle variations

3. **Domain context**: Trained on APT reports, CVE descriptions
   - Understands security concepts
   - Better at security-specific anomalies

### BGE Advantages (Unique Catches: 1,364)

1. **General semantics**: Strong general-purpose embeddings

   - Catches attacks with unusual semantic structure
   - Less biased toward known attack patterns

2. **Efficiency**: 384 dims avoid overfitting

   - Cleaner signal in lower dimensions
   - May generalize better to novel attacks

3. **Different training**: General text corpus
   - Captures different linguistic patterns
   - Complementary to security-focused training

---

## Agreement Analysis

**87.9% agreement** means:

- Models agree on 87.9% of samples (43,985 / 50,065)
- Disagree on 12.1% (6,080 samples)

**Disagreement breakdown:**

- 996 FP disagreements (normal traffic)
- 5,060 FN disagreements (attacks)
- Total: 6,056 samples

These 6,056 disagreements are where models have complementary strengths.

---

## Recommendations

### Option 1: Use SecBERT Alone (Current Best)

**Pros:**

- Highest single-model recall (49.26%)
- Simple deployment
- 5.00% FPR as designed

**Cons:**

- Misses 3,696 attacks that BGE catches
- No redundancy

### Option 2: Ensemble with Union (Higher Recall)

**Pros:**

- 54.7% recall (+5.4pp improvement)
- Catches attacks both models find
- Redundancy: if one model fails, other may catch

**Cons:**

- 7.0% FPR (40% increase in false alarms)
- More complex deployment (two models)
- Higher compute cost

### Option 3: Ensemble with Voting (Balanced)

**Pros:**

- ~53-54% recall (tuned)
- 5% FPR maintained
- Best of both worlds

**Cons:**

- Requires threshold tuning
- More complex
- 2x inference cost

---

## Recommended Next Steps

1. **Immediate**: Deploy SecBERT-Mahalanobis (49.26% recall @ 5% FPR)

   - Best single-model performance
   - Production-ready

2. **Short-term**: Test ensemble with weighted voting

   - Tune combined threshold
   - Validate FPR on held-out data
   - Expected: 53-54% recall @ 5% FPR

3. **Long-term**: Investigate the 11,354 universally missed attacks
   - Are they truly stealthy or labeling errors?
   - Different detection paradigm needed?
   - Rule-based heuristics for specific patterns?

---

## Visualizations

- **Agreement Matrix**: Shows 87.9% agreement
- **FP Venn Diagram**: Shows false positive overlap
- **FN Venn Diagram**: Shows false negative overlap
- **Prediction Heatmap**: Shows per-sample predictions

All saved in: `visualizations/mahalanobis_comparison/`

---

## Conclusion

**SecBERT is the better single model** (49.26% vs 39.96% recall), but **BGE catches 1,364 unique attacks**. An ensemble could reach **53-54% recall @ 5% FPR**, a meaningful improvement.

The decision depends on priorities:

- **Maximize recall + accept higher FPR**: Union ensemble (54.7% recall, 7% FPR)
- **Balance recall + maintain FPR**: Weighted ensemble (~53% recall, 5% FPR)
- **Simplicity + good performance**: SecBERT alone (49.26% recall, 5% FPR)

**Recommendation**: Start with SecBERT alone, then A/B test ensemble if 5pp recall gain justifies 2x compute cost.
