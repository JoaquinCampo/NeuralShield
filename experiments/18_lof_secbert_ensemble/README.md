# LOF + SecBERT Ensemble Experiment

This experiment trains a Local Outlier Factor detector on TF-IDF+PCA embeddings
and a Mahalanobis detector on SecBERT embeddings, then evaluates:

- Each detector individually
- A union ensemble (flag when either detector fires)
- A score-level fusion ensemble with z-score normalised detector scores

## Running

```bash
uv run python experiments/18_lof_secbert_ensemble/run_experiment.py
```

Artifacts (detectors, PCA copy, metrics) are written to
`experiments/18_lof_secbert_ensemble/artifacts/`.

The default configuration relies on existing embedding dumps:

- `embeddings/tfidf_dump.npz` (train)
- `experiments/15_lof_comparison/tfidf_pca_150/csic_test_embeddings.npz`
- `experiments/03_secbert_comparison/secbert_with_preprocessing/csic_*_embeddings_converted.npz`
