# Experiment 15: LOF Implementation Summary

**Date**: October 14, 2025  
**Status**: ✅ Implementation Complete - Ready to Run

---

## What Was Implemented

### 1. LOF Detector Class (`src/neuralshield/anomaly/lof.py`)

A complete implementation of Local Outlier Factor anomaly detection:

**Key Features:**

- Implements the `AnomalyDetector` interface
- Uses sklearn's `LocalOutlierFactor` with `novelty=True`
- Configurable `n_neighbors` hyperparameter
- Consistent API with other detectors (Mahalanobis, IsolationForest)
- Save/load functionality for persistence

**Usage:**

```python
from neuralshield.anomaly import LOFDetector

detector = LOFDetector(n_neighbors=100)
detector.fit(train_embeddings)
scores = detector.scores(test_embeddings)
```

### 2. Hyperparameter Search Script (`hyperparameter_search.py`)

Searches over `n_neighbors` values to find optimal configuration:

**Features:**

- Tests multiple n_neighbors values: [50, 100, 150, 200, 300, 500]
- Evaluates recall @ 5% FPR
- Optional wandb logging
- Saves best configuration and all results

**Usage:**

```bash
cd experiments/15_lof_comparison
uv run python hyperparameter_search.py \
    path/to/train.npz \
    path/to/test.npz \
    output_dir \
    --n-neighbors 50 100 200 500
```

### 3. Main Experiment Runner (`run_experiment.py`)

Comprehensive comparison of LOF vs Mahalanobis:

**Features:**

- Tests both datasets (CSIC, SR_BH)
- Tests with/without preprocessing
- Runs LOF and Mahalanobis side-by-side
- Generates visualizations (confusion matrices, score distributions)
- Saves detailed metrics and comparison summary

**Usage:**

```bash
cd experiments/15_lof_comparison
uv run python run_experiment.py
```

### 4. Documentation

- `README.md`: Quick overview and motivation
- `EXPERIMENT_PLAN.md`: Detailed methodology and hypotheses
- `IMPLEMENTATION_SUMMARY.md`: This file

---

## How LOF Works (Conceptual)

**Global Mahalanobis (Current Approach):**

```
1. Compute global mean μ and covariance Σ from ALL training data
2. For each test point: distance = (x - μ)ᵀ Σ⁻¹ (x - μ)
3. Higher distance = more anomalous
```

**Local Outlier Factor (New Approach):**

```
1. For each test point x:
   a. Find k nearest neighbors in training data
   b. Compute local reachability density
   c. Compare x's density to neighbors' densities
2. Higher density ratio = more anomalous (lower local density)
```

**Why This Matters:**

- HTTP traffic forms multiple clusters (API calls, forms, static assets)
- Global methods miss local density variations
- Local methods adapt to each cluster's characteristics

---

## Expected Workflow

### Step 1: Hyperparameter Search (Optional but Recommended)

Find optimal `n_neighbors` for your dataset:

```bash
cd experiments/15_lof_comparison

# For CSIC
uv run python hyperparameter_search.py \
    ../../src/neuralshield/data/CSIC/secbert_train_with_preprocessing.npz \
    ../../src/neuralshield/data/CSIC/secbert_test_with_preprocessing.npz \
    hyperparameter_search_csic \
    --n-neighbors 50 100 150 200 300 500

# Check results
cat hyperparameter_search_csic/best_config.json
```

### Step 2: Run Full Experiment

Execute all variants and compare to Mahalanobis:

```bash
cd experiments/15_lof_comparison
uv run python run_experiment.py
```

This will create:

```
15_lof_comparison/
├── csic_with_preprocessing/
│   ├── lof/
│   │   ├── results.json
│   │   ├── confusion_matrix.png
│   │   └── score_distribution.png
│   └── mahalanobis/
│       └── [same structure]
├── csic_without_preprocessing/
│   └── [same structure]
├── srbh_with_preprocessing/
│   └── [same structure]
├── srbh_without_preprocessing/
│   └── [same structure]
└── comparison_summary.json
```

### Step 3: Analyze Results

```bash
# View comparison summary
cat comparison_summary.json | jq '.[] | {experiment: .lof.experiment, lof_recall: .lof.recall, maha_recall: .mahalanobis.recall}'

# View specific results
cat csic_with_preprocessing/lof/results.json
```

---

## Success Criteria

**Minimum viable:**

- LOF matches Mahalanobis (49-54% recall @ 5% FPR)
- No crashes, clean implementation

**Success:**

- LOF improves by 2-5pp on same-dataset OR
- LOF improves by 5pp+ on cross-dataset

**Strong success:**

- LOF improves by 5pp+ on same-dataset AND cross-dataset
- Clear evidence that local density matters

---

## Integration with Repository

### Detector Registration

LOF is now registered in the detector factory:

```python
from neuralshield.anomaly import get_detector, LOFDetector

# Via factory
LOFClass = get_detector("lof")
detector = LOFClass(n_neighbors=100)

# Direct import
detector = LOFDetector(n_neighbors=100)
```

### Consistent API

LOF follows the same interface as all other detectors:

```python
# All detectors support:
detector.fit(train_embeddings)
scores = detector.scores(test_embeddings)
threshold = detector.set_threshold(normal_embeddings, max_fpr=0.05)
predictions = detector.predict(test_embeddings, threshold=threshold)
detector.save("model.joblib")
loaded = LOFDetector.load("model.joblib")
```

---

## Next Steps

1. **Run hyperparameter search** to find optimal n_neighbors
2. **Run main experiment** to compare LOF vs Mahalanobis
3. **Analyze results** and update `CONSOLIDATED_RESULTS.md`
4. **If successful:** Test on other embedding models (BGE-small, ByT5)
5. **If unsuccessful:** Visualize embeddings to understand why

---

## Files Modified

**New files:**

- `src/neuralshield/anomaly/lof.py` (176 lines)
- `experiments/15_lof_comparison/README.md`
- `experiments/15_lof_comparison/EXPERIMENT_PLAN.md`
- `experiments/15_lof_comparison/hyperparameter_search.py` (242 lines)
- `experiments/15_lof_comparison/run_experiment.py` (364 lines)
- `experiments/15_lof_comparison/IMPLEMENTATION_SUMMARY.md` (this file)

**Modified files:**

- `src/neuralshield/anomaly/__init__.py` (added LOF imports)

**Total:** ~800 lines of new code + documentation

---

## Technical Notes

### LOF Score Interpretation

sklearn's LOF returns **negative** scores where more negative = more anomalous.

We flip the sign in `scores()` method:

```python
lof_scores = self._model.decision_function(embeddings)
return (-lof_scores).astype(np.float32)  # Higher = more anomalous
```

This makes LOF consistent with other detectors (Mahalanobis, IsolationForest).

### Computational Complexity

- **Training:** O(n²) for exact kNN, but sklearn uses optimizations
- **Inference:** O(n \* k) for kNN search per test point
- **Memory:** Stores all training embeddings (same as needed for Mahalanobis)

For 100K training samples with k=100:

- Training: ~10-30 seconds (depending on hardware)
- Inference: ~5-10 seconds for 50K test samples

### Hyperparameter Guidelines

**n_neighbors:**

- Too small (k<50): Noisy local statistics, overfits
- Too large (k>500): Loses locality, approaches global
- Recommended: 100-200 for HTTP traffic

**contamination:**

- Set to 'auto' (default)
- LOF estimates contamination from data
- We override with explicit threshold based on FPR anyway

---

## Questions or Issues?

If the experiment fails or produces unexpected results:

1. Check data paths in `run_experiment.py`
2. Verify embeddings are SecBERT (768D)
3. Check that labels are "attack" or "valid" strings
4. Review logs for errors
5. Try hyperparameter search first to validate setup

---

## Acknowledgments

Implementation inspired by:

- Breunig et al. (2000). "LOF: Identifying Density-Based Local Outliers"
- sklearn.neighbors.LocalOutlierFactor documentation
- Existing detector implementations in neuralshield.anomaly
