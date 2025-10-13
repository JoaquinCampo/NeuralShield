# Hyperparameter Search Results

**Date**: 2025-10-12 14:09:40

## Search Configuration

- **Total configurations**: 96
- **FPR constraint**: 0.05
- **Search space**:
  - `contamination`: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
  - `n_estimators`: [100, 200, 300, 500]
  - `max_samples`: ['auto', 256, 512, 1024]

## Pareto Frontier

**Found 4 non-dominated solutions:**

| Rank | Contamination | n_estimators | max_samples | Recall | FPR | F1 | Precision |
|------|---------------|--------------|-------------|--------|-----|----|-----------|
| 1 | 0.05 | 500 | 1024 | 20.63% | 4.83% | 32.90% | 81.07% |
| 2 | 0.05 | 200 | 1024 | 20.60% | 4.76% | 32.87% | 81.28% |
| 3 | 0.05 | 300 | 1024 | 20.57% | 4.71% | 32.84% | 81.41% |
| 4 | 0.05 | 200 | 512 | 18.28% | 4.62% | 29.76% | 79.86% |

## Best Model (Highest Recall)

**Configuration:**
- Contamination: `0.05`
- n_estimators: `500`
- max_samples: `1024`

**Performance:**
- Recall: `20.63%`
- Precision: `81.07%`
- F1-Score: `32.90%`
- FPR: `4.83%`
- Accuracy: `57.85%`

## Top 10 Models by Recall

| Rank | Contamination | n_estimators | max_samples | Recall | FPR | F1 |
|------|---------------|--------------|-------------|--------|-----|----||
| 1 | 0.30 | 500 | 1024 | 61.06% | 24.89% | 65.70% |
| 2 | 0.30 | 300 | 1024 | 59.51% | 24.63% | 64.66% |
| 3 | 0.30 | 500 | 512 | 59.47% | 24.49% | 64.68% |
| 4 | 0.30 | 200 | 1024 | 59.31% | 24.35% | 64.61% |
| 5 | 0.30 | 200 | 512 | 59.15% | 23.94% | 64.64% |
| 6 | 0.30 | 100 | 1024 | 58.79% | 24.39% | 64.21% |
| 7 | 0.30 | 500 | auto | 58.66% | 24.93% | 63.92% |
| 8 | 0.30 | 500 | 256 | 58.66% | 24.93% | 63.92% |
| 9 | 0.30 | 300 | 512 | 58.59% | 24.06% | 64.18% |
| 10 | 0.30 | 200 | auto | 58.34% | 24.86% | 63.71% |
