# Experiment 24: TF-IDF + PCA Cross-Dataset Baseline

This scratch experiment reuses the paper-style TF-IDF + PCA pipeline to test Mahalanobis and LOF detectors across datasets. We fit the vectorizer + PCA on the source dataset’s normal traffic, then evaluate on the target dataset’s test split (normal + attack). Preprocessing follows `paper_preprocess` for consistency with the MI experiments.

## Metrics (Mahalanobis @ 5% FPR)

| Source → Target | Recall | Precision | FPR | Notes |
|-----------------|--------|-----------|-----|-------|
| SR_BH → PKDD    | 0.0000 | 0.0000    | 0.00| Mahalanobis sees identical distribution, never fires |
| PKDD → SR_BH    | 0.0016 | 0.0036    | 0.41| Threshold trained on PKDD overfires on SR-BH headers |

## Metrics (LOF, k=100, threshold from 5% quantile)

| Source → Target | Recall | Precision | FPR | Notes |
|-----------------|--------|-----------|-----|-------|
| SR_BH → PKDD    | 0.0599 | 0.5858    | 0.061| Low recall, modest precision |
| PKDD → SR_BH    | 0.0028 | 0.0076    | 0.326| Extreme false positives |

Outputs for each run are saved as JSON under this directory (e.g. `SRBH_to_PKDD.json`).

**Conclusion:** sparse TF-IDF + PCA features inherit the same dataset-mismatch issues seen in the MI replication. Without aligned token distributions, cross-dataset transfer is very weak.
