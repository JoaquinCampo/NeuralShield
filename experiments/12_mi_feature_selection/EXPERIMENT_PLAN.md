# Experiment 12: MI-Based Feature Selection - Detailed Plan

**Date**: October 14, 2025  
**Status**: Ready to execute  
**Author**: NeuralShield Team

---

## Objective

Test whether **mutual information (MI)** based dimension selection can improve anomaly detection performance when applied to dense SecBERT embeddings.

---

## Background

### Current Best Performance

**SecBERT + Mahalanobis + Preprocessing** (Experiment 03):

- Recall: 49.26% @ 5% FPR
- Uses all 768 dimensions
- No dimension selection

### Paper's Approach

**TF-IDF + MI + One-Class SVM**:

- Selects 100 tokens via MI from 5000 TF-IDF features
- Achieves 78-92% TPR @ 2-5% FPR
- Uses sparse, interpretable features

### Research Question

**Can MI-based selection improve dense embeddings (SecBERT) as it does for sparse features (TF-IDF)?**

---

## Hypothesis

**H1**: SecBERT's 768 dimensions contain redundancy and noise.

**Reasoning**:

- SecBERT trained for general security text, not HTTP requests
- Some dimensions may encode irrelevant information (writing style, language patterns)
- Selecting attack-discriminative dimensions may improve signal-to-noise ratio

**H2**: 100-200 MI-selected dimensions will outperform all 768.

**Reasoning**:

- Paper showed 100 features optimal for TF-IDF
- Mahalanobis benefits from reduced dimensionality (more stable covariance estimation)
- Less risk of overfitting to training data noise

**H3**: MI on generic attacks (SR_BH) will generalize to different attacks (CSIC).

**Reasoning**:

- Paper demonstrated this for sparse features
- Attack patterns share common tokens across datasets
- MI captures fundamental attack characteristics, not dataset-specific quirks

---

## Methodology

### Phase 1: Data Preparation (Pre-existing)

**CSIC Dataset**:

- Train: 47,000 normal requests → embeddings (47K × 768)
- Test: 50,065 mixed (25K normal + 25K attacks) → embeddings (50K × 768)
- Source: `experiments/03_secbert_comparison/secbert_with_preprocessing/`

**SR_BH Dataset**:

- Test: 807,815 mixed (425K normal + 382K attacks) → embeddings (807K × 768)
- Source: `experiments/08_secbert_srbh/with_preprocessing/`
- **Use only**: 382K attacks (filter by label)

**All embeddings already exist** → No regeneration needed.

### Phase 2: MI Computation

```python
# Combine normal + attacks
X = vstack([
    csic_train_embeddings,    # (47K, 768)
    srbh_attack_embeddings    # (382K, 768)
])
# Shape: (429K, 768)

# Create labels
y = array([0]*47000 + [1]*382000)

# Compute MI for each dimension
mi_scores = mutual_info_classif(X, y, random_state=42)
# Shape: (768,)
```

**Expected time**: 30 seconds on CPU

### Phase 3: Dimension Selection

For each K in {50, 100, 200, 300, 400, 768}:

```python
# Select top K dimensions
top_dims = argsort(mi_scores)[-K:]

# Extract selected dimensions
csic_train_selected = csic_train_embeddings[:, top_dims]  # (47K, K)
csic_test_selected = csic_test_embeddings[:, top_dims]    # (50K, K)
```

### Phase 4: Train & Evaluate

For each K:

```python
# Train Mahalanobis on normal data only
detector = MahalanobisDetector()
detector.fit(csic_train_selected)

# Predict on test set
predictions = detector.predict(csic_test_selected)

# Compute metrics
metrics = compute_metrics_at_fpr(predictions, labels, fpr=0.05)
```

**Expected time**: ~30 seconds per K value

### Phase 5: Analysis

1. Compare recall across K values
2. Identify optimal K
3. Visualize MI score distribution
4. Analyze top dimensions
5. Generate performance curves

---

## Variables Tested

| Variable           | Values                      | Rationale                     |
| ------------------ | --------------------------- | ----------------------------- |
| **K** (dimensions) | 50, 100, 200, 300, 400, 768 | Test minimal → full range     |
| Embedding model    | SecBERT (768D)              | Current best performer        |
| Detector           | Mahalanobis                 | Best for SecBERT              |
| Preprocessing      | With (13 steps)             | Required for best performance |
| MI data source     | SR_BH attacks               | Generic attack corpus         |
| Train data         | CSIC normal                 | Target application            |
| Test data          | CSIC mixed                  | Target application            |

---

## Expected Results

### Prediction Matrix

| K   | Expected Recall | Reasoning                            |
| --- | --------------- | ------------------------------------ |
| 50  | 42-46%          | Too few dimensions, information loss |
| 100 | 50-54%          | Optimal (if MI helps)                |
| 200 | 48-52%          | Good balance                         |
| 300 | 48-50%          | Approaching baseline                 |
| 400 | 48-50%          | Close to baseline                    |
| 768 | 49.26%          | Baseline (known from Exp 03)         |

### Decision Criteria

**Success** (integrate MI):

- Best K achieves ≥51% recall (+3% improvement)
- Best K < 768 (dimensionality reduction)
- Improvement statistically meaningful

**Neutral** (use for efficiency only):

- Best K achieves ~49% recall (no improvement)
- But faster inference with K=100

**Failure** (abandon MI):

- Best K achieves <48% recall (worse than baseline)
- All K < 768 underperform

---

## Risks & Mitigations

### Risk 1: Dense embeddings differ from sparse features

**Issue**: Paper's MI success was on TF-IDF (sparse, interpretable tokens). SecBERT dimensions are dense, latent representations.

**Mitigation**: This is the experiment's purpose. If it fails, we learn that MI doesn't transfer to dense embeddings (valuable negative result).

### Risk 2: SR_BH attacks don't generalize to CSIC

**Issue**: MI computed on SR_BH may select dimensions specific to SR_BH attack distribution.

**Mitigation**: If successful, validate by reversing (train on SR_BH, MI on CSIC attacks). Cross-dataset validation in future experiment.

### Risk 3: MI computation too slow

**Issue**: 429K samples × 768 dims might be slow.

**Mitigation**:

- Already tested: sklearn's mutual_info_classif handles this size (~30 sec)
- Can sample if needed (use 10K attacks instead of 382K)

### Risk 4: Overfitting to MI selection

**Issue**: MI computed on same attack types as test set.

**Mitigation**: Future work tests on unseen attack types. Current experiment establishes if MI helps at all.

---

## Success Metrics

**Primary**: Recall @ 5% FPR

**Secondary**:

- Precision (should stay ≥88%)
- F1-Score
- Computational efficiency (inference time)

**Qualitative**:

- MI score distribution (uniform vs heavy-tailed?)
- Top dimensions interpretability
- Consistency across K values

---

## Deliverables

1. ✅ `results/mi_scores.npy` - MI values for all 768 dimensions
2. ✅ `results/selected_dims_k{50,100,200}.npy` - Selected dimension indices
3. ✅ `results/metrics_comparison.json` - Performance for all K values
4. ✅ `results/plots/mi_distribution.png` - Histogram of MI scores
5. ✅ `results/plots/top_50_dimensions.png` - Bar chart of highest MI dims
6. ✅ `results/plots/recall_vs_k.png` - Performance curve
7. ✅ `RESULTS.md` - Summary and decision

---

## Timeline

| Phase              | Time      | Details                           |
| ------------------ | --------- | --------------------------------- |
| 1. Load embeddings | 5 sec     | Pre-computed, just load from disk |
| 2. Compute MI      | 30 sec    | 429K × 768 matrix                 |
| 3. Test K=50       | 30 sec    | Train + evaluate                  |
| 4. Test K=100      | 30 sec    | Train + evaluate                  |
| 5. Test K=200      | 30 sec    | Train + evaluate                  |
| 6. Test K=300      | 30 sec    | Train + evaluate                  |
| 7. Test K=400      | 30 sec    | Train + evaluate                  |
| 8. Test K=768      | 30 sec    | Baseline validation               |
| 9. Generate plots  | 30 sec    | 3 visualizations                  |
| **Total**          | **5 min** | Full experiment                   |

---

## Future Work

**If successful**:

1. **Experiment 13**: Test on SR_BH dataset (reverse validation)
2. **Experiment 14**: Cross-dataset MI (train CSIC, MI CSIC, test SR_BH)
3. **Production integration**: Create `MIFeatureSelector` class
4. **Per-attack analysis**: Which attack types benefit most from MI selection?

**If unsuccessful**:

1. **Alternative 1**: PCA-based dimensionality reduction (unsupervised)
2. **Alternative 2**: Forward feature selection (wrapper method)
3. **Alternative 3**: SHAP-based feature importance
4. **Analysis**: Why didn't MI work? (all dims useful? wrong K range?)

---

## Notes

- This experiment requires **no new embeddings** (reuses Exp 03 & 08)
- Total compute time: ~5 minutes
- Pure CPU (no GPU needed for MI computation)
- Fully automated (no manual intervention)
- Results immediately actionable (clear decision criteria)
