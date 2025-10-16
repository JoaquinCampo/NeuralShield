# Experiment 14: Preprocessing Cross-Dataset Impact

**Goal:** Determine if preprocessing helps or hurts cross-dataset generalization

---

## Structure

```
14_preprocessing_cross_dataset/
├── EXPERIMENT_PLAN.md              # Detailed methodology
├── README.md                       # This file
└── tfidf_pca_mahalanobis/          # TF-IDF+PCA+Mahalanobis experiment
    ├── run_experiment.py           # Run all 4 variants
    ├── README.md                   # Model-specific docs
    ├── srbh_to_csic_with_prep/     # Results
    ├── srbh_to_csic_without_prep/  # Results
    ├── csic_to_srbh_with_prep/     # Results
    └── csic_to_srbh_without_prep/  # Results
```

---

## Run TF-IDF Experiment

```bash
cd experiments/14_preprocessing_cross_dataset/tfidf_pca_mahalanobis
uv run python run_experiment.py
```

---

## Future Models

Add new model experiments as sibling directories:

```
14_preprocessing_cross_dataset/
├── tfidf_pca_mahalanobis/
├── bge_small/                      # TODO: BGE-small cross-dataset
├── byt5/                           # TODO: ByT5 cross-dataset
└── secbert/                        # TODO: SecBERT replication
```

Each model folder should have:

- `run_experiment.py` - Self-contained experiment script
- `README.md` - Model-specific documentation
- 4 result directories (with/without prep, both directions)

---

## Key Questions

1. Does preprocessing help cross-dataset generalization?
2. Is the effect direction-dependent (small→large vs large→small)?
3. Which models benefit most from preprocessing?
