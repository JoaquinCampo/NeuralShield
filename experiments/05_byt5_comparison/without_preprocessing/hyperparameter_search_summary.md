# Hyperparameter Search Results

**Date**: 2025-10-12 14:05:17

## Search Configuration

- **Total configurations**: 96
- **FPR constraint**: 0.05
- **Search space**:
  - `contamination`: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
  - `n_estimators`: [100, 200, 300, 500]
  - `max_samples`: ['auto', 256, 512, 1024]

## Pareto Frontier

**Found 2 non-dominated solutions:**

| Rank | Contamination | n_estimators | max_samples | Recall | FPR | F1 | Precision |
|------|---------------|--------------|-------------|--------|-----|----|-----------|
| 1 | 0.05 | 200 | auto | 14.36% | 5.37% | 23.99% | 72.83% |
| 2 | 0.05 | 200 | 256 | 14.36% | 5.37% | 23.99% | 72.83% |

## Best Model (Highest Recall)

**Configuration:**
- Contamination: `0.05`
- n_estimators: `200`
- max_samples: `auto`

**Performance:**
- Recall: `14.36%`
- Precision: `72.83%`
- F1-Score: `23.99%`
- FPR: `5.37%`
- Accuracy: `54.44%`

## Top 10 Models by Recall

| Rank | Contamination | n_estimators | max_samples | Recall | FPR | F1 |
|------|---------------|--------------|-------------|--------|-----|----||
| 1 | 0.30 | 300 | 1024 | 76.00% | 38.70% | 70.83% |
| 2 | 0.30 | 500 | 1024 | 75.69% | 38.81% | 70.60% |
| 3 | 0.30 | 100 | 1024 | 74.71% | 38.06% | 70.26% |
| 4 | 0.30 | 200 | 1024 | 74.44% | 37.74% | 70.20% |
| 5 | 0.30 | 100 | 512 | 73.76% | 37.91% | 69.73% |
| 6 | 0.30 | 500 | 512 | 73.70% | 37.22% | 69.92% |
| 7 | 0.30 | 300 | 512 | 73.55% | 36.86% | 69.94% |
| 8 | 0.30 | 200 | 512 | 72.97% | 36.37% | 69.75% |
| 9 | 0.30 | 500 | auto | 72.73% | 37.00% | 69.39% |
| 10 | 0.30 | 500 | 256 | 72.73% | 37.00% | 69.39% |
