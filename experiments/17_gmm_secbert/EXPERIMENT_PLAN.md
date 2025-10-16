# Experiment Plan: GMM + SecBERT

## Background

Mahalanobis distance assumes normal data follows a single Gaussian distribution. If legitimate traffic actually has multiple distinct patterns or clusters, GMM could model this better.

## Research Question

Does modeling normal traffic as a mixture of Gaussians improve anomaly detection over single-distribution Mahalanobis?

## Methodology

### Data

- Use precomputed SecBERT embeddings (768-dim)
- Training: Normal CSIC traffic only
- Test: Mixed normal and attack traffic

### Models

- Baseline: Mahalanobis (single Gaussian) from experiment 03
- Test: GMM with 1-10 components

### Evaluation

- Primary metric: F1 score
- Secondary: Recall, BIC/AIC (model selection)
- Constraint: FPR ≤ 5%

## Hypotheses

### H1: GMM outperforms Mahalanobis

If normal traffic is naturally multi-modal, GMM should achieve higher recall/F1.

### H2: Optimal n_components > 1

BIC/AIC should favor multiple components if clustering exists.

### H3: Diminishing returns

Too many components will overfit and degrade test performance.

## Success Criteria

**GMM is worthwhile if:**

- F1 score improves by ≥2% over Mahalanobis
- Improvement is robust across different FPR thresholds
- Optimal n_components is clearly identified

**Otherwise:**

- Stick with simpler Mahalanobis (fewer hyperparameters)
- Extra complexity not justified by performance gain

## Implementation

1. Component sweep: Test n_components from 1 to 10
2. Use BIC/AIC for model selection
3. Use "full" covariance (most flexible)
4. Compare best GMM vs Mahalanobis baseline

## Expected Timeline

- Component comparison: ~10 min (1-10 components)
- Analysis and visualization: 5 min
- Total: ~15 min
