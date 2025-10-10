# Hyperparameter Search Results

**Date**: 2025-10-10 13:45:02

## Search Configuration

- **Total configurations**: 96
- **FPR constraint**: 0.05
- **Search space**:
  - `contamination`: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
  - `n_estimators`: [100, 200, 300, 500]
  - `max_samples`: ['auto', 256, 512, 1024]

## Pareto Frontier

**Found 3 non-dominated solutions:**

| Rank | Contamination | n_estimators | max_samples | Recall | FPR | F1 | Precision |
|------|---------------|--------------|-------------|--------|-----|----|-----------|
| 1 | 0.05 | 300 | 1024 | 27.55% | 4.79% | 41.64% | 85.23% |
| 2 | 0.05 | 100 | auto | 17.23% | 4.71% | 28.26% | 78.57% |
| 3 | 0.05 | 100 | 256 | 17.23% | 4.71% | 28.26% | 78.57% |

## Best Model (Highest Recall)

**Configuration:**
- Contamination: `0.05`
- n_estimators: `300`
- max_samples: `1024`

**Performance:**
- Recall: `27.55%`
- Precision: `85.23%`
- F1-Score: `41.64%`
- FPR: `4.79%`
- Accuracy: `61.34%`

## Top 10 Models by Recall

| Rank | Contamination | n_estimators | max_samples | Recall | FPR | F1 |
|------|---------------|--------------|-------------|--------|-----|----||
| 1 | 0.30 | 200 | 1024 | 57.51% | 29.42% | 61.55% |
| 2 | 0.30 | 100 | 1024 | 57.24% | 29.56% | 61.31% |
| 3 | 0.30 | 300 | 1024 | 57.20% | 29.57% | 61.28% |
| 4 | 0.30 | 500 | 1024 | 57.01% | 29.34% | 61.21% |
| 5 | 0.30 | 200 | 512 | 56.14% | 29.79% | 60.41% |
| 6 | 0.30 | 300 | 512 | 55.68% | 29.75% | 60.08% |
| 7 | 0.25 | 200 | 1024 | 55.38% | 24.77% | 61.51% |
| 8 | 0.25 | 100 | 1024 | 55.01% | 24.65% | 61.26% |
| 9 | 0.30 | 100 | 512 | 55.00% | 29.46% | 59.66% |
| 10 | 0.25 | 300 | 1024 | 54.74% | 24.68% | 61.04% |
