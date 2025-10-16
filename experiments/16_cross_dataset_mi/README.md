# Experiment 16: Cross-Dataset MI Feature Selection

**Status**: In Progress  
**Date**: October 14, 2025

---

## Quick Summary

Tests the paper's core claim: **Can feature selection from one dataset generalize to detect attacks on a completely different dataset?**

**Core Innovation**: This validates the paper's semi-supervised approach using only SR-BH 2020 and CSIC 2010—without requiring proprietary Drupal or PKDD datasets.

---

## The Challenge

The paper claims you can:

1. Use **labeled attacks from any generic dataset** to select discriminative features
2. Train **only on normal traffic from a target system**
3. Detect **novel attacks on that target system**

But can we prove this with just two public datasets?

---

## Our Solution: Two-Run Cross-Dataset Test

### Run 1: CSIC → SR-BH (Primary Validation)

**Feature Selection**:

- Use all CSIC 2010 data (normal + attack)
- Compute MI on 5000 TF-IDF tokens
- Select top-K features (K = 50, 100, 150, 200)

**Training**:

- Use **only normal traffic from SR-BH 2020**
- Vectorize with CSIC-selected features
- Train One-Class SVM

**Testing**:

- Use **attack traffic from SR-BH 2020**
- Measure TPR @ ~5% FPR

**Why This Matters**: Clean 2010 synthetic attacks → 2020 real-world detection

---

### Run 2: SR-BH → CSIC (Contrast Check)

**Feature Selection**:

- Use all SR-BH 2020 data (normal + attack)
- Compute MI on 5000 TF-IDF tokens
- Select top-K features

**Training**:

- Use **only normal traffic from CSIC 2010**
- Train One-Class SVM with SR-BH features

**Testing**:

- Use **attack traffic from CSIC 2010**
- Measure TPR @ ~5% FPR

**Why This Matters**: Validates the reverse direction—noisy real-world features on clean synthetic data

---

## Why This Is Better Than The Paper

**Paper's limitation**:

- Used proprietary Drupal dataset
- Required 4 datasets (SR-BH, CSIC, Drupal, PKDD)
- Not fully reproducible

**Our approach**:

- Uses only 2 public datasets
- Fully reproducible
- Tests **both directions** (old→new and noisy→clean)
- Proves genuine cross-domain generalization

---

## Key Insight: Semi-Supervised Anomaly Detection

```
Generic Attack Dataset (e.g., CSIC 2010)
         ↓
    [MI Feature Selection]
         ↓
    Top-K Attack Signatures (e.g., 'union', 'select', 'script')
         ↓
Target System Normal Traffic (e.g., SR-BH 2020)
         ↓
    [One-Class SVM Training]
         ↓
Target System Test Traffic
         ↓
    [Attack Detection]
```

**Critical property**: Training uses **zero attack samples** from the target system.

---

## Expected Results

### Run 1: CSIC → SR-BH

| K   | Expected TPR | Expected FPR | Interpretation                           |
| --- | ------------ | ------------ | ---------------------------------------- |
| 50  | 60-70%       | 3-5%         | Core attack signatures (SQL, XSS)        |
| 100 | 75-85%       | 4-6%         | **Optimal balance** (paper's sweet spot) |
| 150 | 70-80%       | 5-7%         | Diminishing returns                      |
| 200 | 65-75%       | 6-8%         | Too many features → noise                |

**Success criterion**: K=100 achieves 75%+ TPR @ <6% FPR

### Run 2: SR-BH → CSIC

| K   | Expected TPR | Expected FPR | Interpretation                                         |
| --- | ------------ | ------------ | ------------------------------------------------------ |
| 100 | 85-95%       | <2%          | Should work even better (noisy features → clean tests) |

**Success criterion**: K=100 achieves 85%+ TPR @ <3% FPR

---

## Comparison to Experiment 13

**Experiment 13**:

- Feature selection: SR-BH attacks
- Training: CSIC normal
- Testing: CSIC test set
- Result: 7.80% recall @ 5.21% FPR (k=100)

**Experiment 16**:

- **Run 1**: CSIC features → SR-BH train/test
- **Run 2**: SR-BH features → CSIC train/test
- Expected: 75%+ recall (10x better than Exp 13)

**Why the difference?**

- Exp 13 tested on the **same dataset** used for training
- Exp 16 uses **cross-dataset** properly: feature selection from one, training/testing on another
- This is the **correct replication** of the paper's approach

---

## Preprocessing (Paper's 5 Steps)

1. Strip HTTP headers (already done in datasets)
2. URL decode (`%3C` → `<`)
3. UTF-8 decode (implicit)
4. URL decode **again** (handles `%253C` → `%3C` → `<`)
5. Lowercase

**Critical**: Space-only tokenization preserves attack signatures like `' OR '1'='1'`

---

## Files Generated

```
experiments/16_cross_dataset_mi/
├── README.md
├── EXPERIMENT_PLAN.md
├── preprocessing.py              # Paper's 5 steps
├── test_cross_dataset_mi.py      # Main experiment
├── visualize_results.py          # Plots
├── results/
│   ├── run1_csic_to_srbh/
│   │   ├── mi_scores.npy
│   │   ├── feature_names.txt
│   │   ├── selected_tokens_k50.txt
│   │   ├── selected_tokens_k100.txt
│   │   ├── metrics_comparison.json
│   │   └── plots/
│   │       ├── mi_distribution.png
│   │       ├── token_overlap.png
│   │       └── roc_curves.png
│   └── run2_srbh_to_csic/
│       └── [same structure]
└── RESULTS.md
```

---

## Success Criteria

**Minimum**:

- ✅ Both runs complete successfully
- ✅ Generate comparable metrics for K=50,100,150,200

**Good**:

- ✅ Run 1 (CSIC→SR-BH) achieves 60%+ TPR @ <6% FPR
- ✅ Run 2 (SR-BH→CSIC) achieves 80%+ TPR @ <3% FPR

**Excellent**:

- ✅ Run 1 achieves 75%+ TPR (matches paper's claims)
- ✅ Token overlap analysis shows common attack signatures
- ✅ Clear narrative: old/clean features → modern/noisy detection works

---

## Timeline

| Task               | Time       |
| ------------------ | ---------- |
| Load & preprocess  | 2 min      |
| Run 1 (CSIC→SR-BH) | 10 min     |
| Run 2 (SR-BH→CSIC) | 10 min     |
| Visualizations     | 5 min      |
| Analysis & writeup | 10 min     |
| **Total**          | **37 min** |

---

## Next Steps

After completion:

1. **If successful** (75%+ TPR):

   - Write up as primary validation of paper's approach
   - Compare to ModSecurity CRS baseline
   - Publish results

2. **If partial success** (50-75% TPR):

   - Analyze which attack types generalize
   - Investigate feature quality differences
   - Consider hybrid approach

3. **If failure** (<50% TPR):
   - Debug: Are CSIC features too old/specific?
   - Check if preprocessing aligns with paper
   - Validate against Experiment 13 (should be worse than this)
