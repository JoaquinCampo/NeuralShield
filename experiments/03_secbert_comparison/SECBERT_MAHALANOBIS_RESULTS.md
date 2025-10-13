# SecBERT + Mahalanobis Distance Results

**Date**: October 12, 2025  
**Key Finding**: SecBERT + Mahalanobis is the new production winner

---

## Executive Summary

**SecBERT with Mahalanobis distance achieves 49.26% recall @ 5% FPR**, representing:

- **+76% improvement** over BGE-small + IsolationForest (27-28%)
- **+23% improvement** over BGE-small + Mahalanobis (39.96%)
- **+13% improvement** over SecBERT without preprocessing (43.69%)

### Winner: SecBERT + Mahalanobis + Preprocessing

- **Recall**: 49.26%
- **Precision**: 90.81%
- **F1-Score**: 63.87%
- **FPR**: 5.00%
- **Accuracy**: 72.10%
- **Dimensions**: 768
- **Model**: jackaduma/SecBERT
- **Preprocessing**: Required

---

## Complete Results Table

| Model       | Detector        | Preprocessing | Recall @ 5% FPR | Precision  | F1         | Improvement        |
| ----------- | --------------- | ------------- | --------------- | ---------- | ---------- | ------------------ |
| **SecBERT** | **Mahalanobis** | **Yes**       | **49.26%**      | **90.81%** | **63.87%** | **Baseline (NEW)** |
| SecBERT     | Mahalanobis     | No            | 43.69%          | 89.75%     | 58.77%     | -11%               |
| BGE-small   | Mahalanobis     | Yes           | 39.96%          | 88.90%     | 55.13%     | -19%               |
| BGE-small   | Mahalanobis     | No            | 29.45%          | 85.52%     | 43.81%     | -40%               |
| BGE-small   | IsolationForest | Yes           | 27-28%          | ~81%       | ~42%       | -43%               |
| SecBERT     | IsolationForest | No            | 15.48%          | 76.33%     | 25.74%     | -69%               |
| BGE-small   | IsolationForest | No            | 13.81%          | ~76%       | ~24%       | -72%               |
| SecBERT     | IsolationForest | Yes           | 12.32%          | 72.09%     | 21.05%     | -75%               |
| TF-IDF      | IsolationForest | No            | 0.96%           | ~72%       | ~2%        | -98%               |

---

## Key Findings

### 1. Mahalanobis Unlocks SecBERT's Potential

**SecBERT's performance by detector:**

- With IsolationForest: 12-15% recall (curse of dimensionality)
- With Mahalanobis: 44-49% recall (3-4x improvement!)

**Why it works:**

- Mahalanobis handles 768 dimensions well (covariance-aware)
- IsolationForest fails at high dimensions (treats axes independently)
- SecBERT's cybersecurity vocabulary shines when properly leveraged

### 2. Domain-Specific Pre-training Matters

**SecBERT vs BGE-small (both with Mahalanobis + preprocessing):**

- SecBERT: 49.26% recall
- BGE-small: 39.96% recall
- **+23% improvement** from domain-specific training

**SecBERT advantages:**

- Pre-trained on cybersecurity text (APTnotes, CASIE, Stucco-Data)
- Custom vocabulary for security terms (SQL injection, XSS, etc.)
- Better semantic understanding of attack patterns

### 3. Preprocessing Still Critical

**Impact of preprocessing:**

- Without: 43.69% recall
- With: 49.26% recall
- **+13% improvement**

Preprocessing normalizes HTTP artifacts, making attacks more distinguishable.

### 4. The Detector Choice is Crucial

**For high-dimensional embeddings (768 dims):**

- Mahalanobis: 49.26% recall ✓
- IsolationForest: 12.32% recall ✗

The right detector is as important as the right embedding model.

---

## Performance Breakdown

### Confusion Matrix (SecBERT + Mahalanobis + Preprocessing)

|                 | Predicted Normal | Predicted Attack |
| --------------- | ---------------- | ---------------- |
| **True Normal** | 23,750 (TN)      | 1,250 (FP)       |
| **True Attack** | 12,718 (FN)      | 12,347 (TP)      |

### Metrics

- **True Positives**: 12,347 (49.26% of attacks caught)
- **False Positives**: 1,250 (5.00% of normal requests flagged)
- **True Negatives**: 23,750 (95.00% of normal requests passed)
- **False Negatives**: 12,718 (50.74% of attacks missed)

### Statistical Summary

- **Detection Rate**: Nearly 1 in 2 attacks caught
- **False Alarm Rate**: 1 in 20 normal requests flagged
- **Precision**: 9 in 10 alerts are real attacks
- **Accuracy**: 7 in 10 overall classifications correct

---

## Why This Combination Works

### SecBERT Strengths

1. **Domain vocabulary**: Understands "UNION SELECT", "OR 1=1", "../..", etc.
2. **Security context**: Pre-trained on attack descriptions and APT reports
3. **Rich embeddings**: 768 dimensions capture nuanced patterns

### Mahalanobis Strengths

1. **Covariance-aware**: Accounts for feature correlations
2. **Dimension-agnostic**: Handles 768 dims without degradation
3. **Statistically sound**: Based on multivariate Gaussian distance
4. **Fast**: Sub-second training, no hyperparameter tuning

### Preprocessing Contribution

1. **Normalization**: Lowercase, whitespace, URL decoding
2. **Artifact removal**: BOM, control characters
3. **Consistent structure**: `[METHOD]`, `[URL]`, `[QUERY]`, `[HEADER]` tags

---

## Production Recommendation

### Deploy SecBERT + Mahalanobis + Preprocessing

**Configuration:**

```python
from neuralshield.encoding.models import SecBERTEncoder
from neuralshield.anomaly import MahalanobisDetector
from neuralshield.preprocessing import preprocess

# Initialize encoder
encoder = SecBERTEncoder(
    model_name="default",  # jackaduma/SecBERT
    device="cuda"  # or "cpu"
)

# Initialize detector
detector = MahalanobisDetector(name="production")

# Train on valid requests only
train_requests = load_valid_requests()
preprocessed = [preprocess(req) for req in train_requests]
embeddings = encoder.encode(preprocessed)
detector.fit(embeddings)

# Set threshold for 5% FPR
test_normal = load_validation_normal()
normal_embeddings = encoder.encode([preprocess(r) for r in test_normal])
detector.set_threshold(normal_embeddings, max_fpr=0.05)

# Inference
def is_attack(request: str) -> bool:
    processed = preprocess(request)
    embedding = encoder.encode([processed])
    return detector.predict(embedding)[0]
```

**Expected Performance:**

- 49% of attacks detected
- 5% false positive rate
- 91% precision (low false alarms)

---

## Comparison to Prior Winners

### vs BGE-small + IsolationForest (Previous Best)

| Metric        | BGE-small + IsolationForest     | SecBERT + Mahalanobis | Improvement  |
| ------------- | ------------------------------- | --------------------- | ------------ |
| Recall        | 27-28%                          | 49.26%                | **+76%**     |
| Precision     | ~81%                            | 90.81%                | **+12%**     |
| F1-Score      | ~42%                            | 63.87%                | **+52%**     |
| Dimensions    | 384                             | 768                   | 2x           |
| Training Time | Minutes (hyperparameter search) | <1 second             | ~100x faster |

**Winner**: SecBERT + Mahalanobis across all metrics

### vs BGE-small + Mahalanobis (Previous Best Mahalanobis)

| Metric    | BGE-small | SecBERT | Improvement |
| --------- | --------- | ------- | ----------- |
| Recall    | 39.96%    | 49.26%  | **+23%**    |
| Precision | 88.90%    | 90.81%  | **+2%**     |
| F1-Score  | 55.13%    | 63.87%  | **+16%**    |

**Winner**: Domain-specific SecBERT beats general-purpose BGE

---

## Future Work

### Short Term

1. **Threshold tuning**: Test different FPR targets (3%, 7%, 10%)
2. **Ensemble approach**: Combine SecBERT + BGE predictions
3. **Fine-tuning**: Contrastive learning on CSIC dataset

### Medium Term

1. **Other security models**:
   - CyBERT (Mandiant)
   - CyberBERTron
   - VulBERTa
2. **Hybrid detectors**: Mahalanobis + rule-based heuristics
3. **Per-endpoint models**: Different thresholds for different routes

### Long Term

1. **Online learning**: Adapt to new attack patterns
2. **Explainability**: Which features triggered the alert?
3. **Active learning**: Human feedback loop

---

## Conclusion

**SecBERT + Mahalanobis + Preprocessing is the new production champion**, achieving:

- **49.26% recall** @ 5% FPR (best so far)
- **90.81% precision** (low false alarms)
- **63.87% F1-score** (balanced performance)

Key insights:

1. Domain-specific embeddings (SecBERT) > General embeddings (BGE)
2. Covariance-aware detectors (Mahalanobis) > Tree-based (IsolationForest) for high dims
3. Preprocessing remains essential (+13% gain)
4. The combination of all three is critical

This represents a **51x improvement** over the original TF-IDF baseline and establishes a strong foundation for production deployment.

---

**Experiment Duration**: 2 minutes  
**Models Tested**: 2 (SecBERT with/without preprocessing)  
**Detector**: Mahalanobis (EmpiricalCovariance)  
**Dataset**: CSIC HTTP (47K train, 50K test)  
**W&B Runs**:

- https://wandb.ai/joacocampo27-udelar/neuralshield/runs/5xgvi43z (without prep)
- https://wandb.ai/joacocampo27-udelar/neuralshield/runs/5wro39cq (with prep)
