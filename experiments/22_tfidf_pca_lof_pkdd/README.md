# Experiment: TF-IDF + 175D PCA + LOF on PKDD

Goal:
- Generate TF-IDF features from the PKDD dataset.
- Reduce dimensionality to 175 components via PCA.
- Train a LOF detector (default k=100) on the training split.
- Evaluate on the test split and save metrics/models for comparison.

Outputs (default):
- `experiments/22_tfidf_pca_lof_pkdd/lof_tfidf_pca175_k100.joblib`
- `experiments/22_tfidf_pca_lof_pkdd/pkdd_test_embeddings.npz`
- `experiments/22_tfidf_pca_lof_pkdd/model_metrics.json`

Run:
```bash
uv run python experiments/22_tfidf_pca_lof_pkdd/train_lof_tfidf_pca.py
```

Use `--help` for configurable options (n_neighbors, PCA dimension, output dir, etc.).
