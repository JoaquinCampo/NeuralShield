# Experiment 15: Local Outlier Factor (LOF) - Experimental Design

**Date**: October 14, 2025  
**Status**: 🔄 In Progress  
**Author**: NeuralShield Team

---

## Objective

Evaluate **Local Outlier Factor (LOF)** as an alternative to global Mahalanobis distance for HTTP anomaly detection.

**Primary Goals:**

1. Test if local density analysis outperforms global covariance on same-dataset evaluation
2. Evaluate if local patterns improve cross-dataset generalization
3. Find optimal n_neighbors hyperparameter for HTTP traffic
4. Compare computational cost vs performance gain

---

## Motivation

### Current State (Experiment 08)

**Best performance:**

- SecBERT + Mahalanobis: 54.12% recall @ 5% FPR (SR_BH)
- SecBERT + Mahalanobis: 49.26% recall @ 5% FPR (CSIC)

**Cross-dataset generalization failure:**

- SR_BH → CSIC: 10.26% recall (5x drop)
- CSIC → SR_BH: 10.76% recall (5x drop)

### Problem with Global Mahalanobis

**Assumption:** Normal data forms one multivariate Gaussian

```
Global mean: μ = average(ALL training points)
Global covariance: Σ = covariance(ALL training points)
Distance: d² = (x - μ)ᵀ Σ⁻¹ (x - μ)
```

**Reality:** HTTP traffic is multimodal

- API endpoints: `/api/v1/users`, `/api/v1/products`
- Static assets: `/static/css/*`, `/static/js/*`
- Forms: `/contact`, `/login`, `/register`
- Admin panels: `/admin/*`

Each cluster has different:

- URL patterns
- Header structures
- Query parameter distributions
- Local density characteristics

**Result:** Global Σ gets "stretched" across clusters, loses local structure

### Why LOF Should Work Better

**LOF approach:**

1. For each test point, find k nearest neighbors
2. Compute local reachability density
3. Compare to neighbors' densities
4. Score = ratio of local densities

**Advantages:**

- Adapts to local cluster structure
- Handles multiple clusters naturally
- Robust to global outliers in training data
- Might capture more transferable local patterns

---

## Hypotheses

**H1: Same-dataset improvement**

- LOF will achieve >55% recall @ 5% FPR on SR_BH (vs 54.12% Mahalanobis)
- LOF will achieve >50% recall @ 5% FPR on CSIC (vs 49.26% Mahalanobis)

**Rationale:** HTTP traffic forms natural clusters that LOF can exploit

**H2: Cross-dataset improvement**

- LOF will achieve >15% recall on cross-dataset tests (vs ~10% Mahalanobis)

**Rationale:** Local patterns (e.g., "anomalous relative to similar endpoints") are more transferable than global statistics

**H3: Preprocessing still critical**

- LOF with preprocessing will outperform LOF without preprocessing by 10-20%

**Rationale:** Preprocessing normalizes syntax, allowing LOF to focus on semantic patterns

**H4: Optimal n_neighbors = 100-200**

- Too small (k<50): Noisy local statistics
- Too large (k>500): Loses locality, approaches global
- Sweet spot: 100-200 for HTTP traffic

---

## Experimental Design

### Datasets

**SR_BH_2020:**

- Train: 100K normal samples
- Test: 807K samples (425K valid, 382K attack)
- Embedding: SecBERT (768D)

**CSIC:**

- Train: 47K normal samples
- Test: 50K samples (25K valid, 25K attack)
- Embedding: SecBERT (768D)

### Variants to Test

**1. Same-dataset evaluation (primary):**

- SR_BH with preprocessing
- SR_BH without preprocessing
- CSIC with preprocessing
- CSIC without preprocessing

**2. Cross-dataset evaluation:**

- Train SR_BH (100K) → Test CSIC (50K) with preprocessing
- Train SR_BH (100K) → Test CSIC (50K) without preprocessing
- Train CSIC (47K) → Test SR_BH (807K) with preprocessing
- Train CSIC (47K) → Test SR_BH (807K) without preprocessing

**3. Hyperparameter sweep:**

- n_neighbors: [50, 100, 200, 500]
- Test on SR_BH with preprocessing

### Evaluation Metrics

**Primary metric:** Recall @ 5% FPR

- Threshold set to flag exactly 5% of normal traffic
- Measure what % of attacks are caught

**Secondary metrics:**

- Precision
- F1-score
- Accuracy
- Confusion matrix
- Score distribution plots

**Computational metrics:**

- Training time
- Inference time per sample
- Memory usage

---

## Implementation Details

### LOF Configuration

```python
from neuralshield.anomaly import LOFDetector

detector = LOFDetector(
    n_neighbors=100,      # Primary hyperparameter
    contamination='auto', # Let LOF estimate
    metric='euclidean'    # Standard for embeddings
)
```

### Workflow

```python
# 1. Load embeddings
train_embeddings = load_embeddings("train.npz")  # Normal only
test_embeddings = load_embeddings("test.npz")    # Normal + attack

# 2. Train LOF
detector.fit(train_embeddings)

# 3. Set threshold (5% FPR on test normal data)
test_normal = test_embeddings[labels == 0]
threshold = detector.set_threshold(test_normal, max_fpr=0.05)

# 4. Evaluate on all test data
scores = detector.scores(test_embeddings)
predictions = scores > threshold

# 5. Compute metrics
recall = recall_score(test_labels, predictions)
```

### Comparison Baseline

Run identical evaluation with Mahalanobis for direct comparison:

```python
from neuralshield.anomaly import MahalanobisDetector

baseline = MahalanobisDetector()
baseline.fit(train_embeddings)
# ... same evaluation workflow
```

---

## Expected Results

### Best Case Scenario

**LOF exploits cluster structure:**

- Same-dataset: 60%+ recall (vs 49-54% Mahalanobis)
- Cross-dataset: 20%+ recall (vs 10% Mahalanobis)
- Clear win for local approach

### Worst Case Scenario

**LOF adds overhead without benefit:**

- Same-dataset: 49-54% recall (same as Mahalanobis)
- Cross-dataset: 10% recall (no improvement)
- Slower inference due to kNN search
- Conclusion: HTTP embeddings don't form exploitable clusters

### Most Likely Scenario

**Modest same-dataset improvement, significant cross-dataset improvement:**

- Same-dataset: 52-56% recall (+2-4pp improvement)
- Cross-dataset: 15-18% recall (+5-8pp improvement)
- Local patterns more transferable than global statistics
- Justifies LOF for production use

---

## Success Criteria

**Minimum viable:** LOF matches Mahalanobis performance (49-54% recall)

**Success:** LOF improves by 2-5pp on same-dataset OR 5pp+ on cross-dataset

**Strong success:** LOF improves by 5pp+ on same-dataset AND cross-dataset

---

## Risks and Mitigations

**Risk 1: Computational cost**

- LOF requires kNN search for every test point
- Mitigation: Use approximate nearest neighbors (ANN) if needed

**Risk 2: Hyperparameter sensitivity**

- n_neighbors might need tuning per dataset
- Mitigation: Run hyperparameter sweep first

**Risk 3: Memory usage**

- LOF stores all training data for kNN
- Mitigation: Already storing embeddings for Mahalanobis anyway

**Risk 4: No cluster structure**

- HTTP embeddings might not form clear clusters
- Mitigation: Visualize embeddings with t-SNE/UMAP to verify

---

## Timeline

1. **Implementation** (Day 1): LOF detector class ✅
2. **Hyperparameter search** (Day 1): Find optimal n_neighbors
3. **Same-dataset evaluation** (Day 1-2): SR_BH and CSIC
4. **Cross-dataset evaluation** (Day 2): Both directions
5. **Analysis and visualization** (Day 2-3): Compare to Mahalanobis
6. **Documentation** (Day 3): Update CONSOLIDATED_RESULTS.md

---

## Next Steps After Experiment

**If LOF succeeds:**

1. Test on other embedding models (BGE-small, ByT5)
2. Try other local methods (LoOP, LOCI, LDOF)
3. Implement hybrid LOF+Mahalanobis approach
4. Deploy to production

**If LOF fails:**

1. Visualize embeddings to understand why
2. Try alternative local approaches
3. Consider supervised fine-tuning of embeddings
4. Focus on other improvements (ensemble methods, etc.)

---

## References

**LOF Paper:**

- Breunig et al. (2000). "LOF: Identifying Density-Based Local Outliers"

**Related Work:**

- Experiment 06: Mahalanobis breakthrough (39.96% → 49.26%)
- Experiment 08: SecBERT on SR_BH (54.12% best result)
- Experiment 14: Cross-dataset preprocessing impact

**Implementation:**

- sklearn.neighbors.LocalOutlierFactor
- novelty=True for prediction on new data
