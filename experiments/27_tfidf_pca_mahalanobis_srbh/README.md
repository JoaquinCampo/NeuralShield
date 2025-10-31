# Experimento: TF-IDF + PCA + Mahalanobis en SRBH

Objetivo:
- Vectorizar las peticiones SRBH con TF-IDF (1-3 n-gramas, 5k características).
- Reducir la dimensionalidad a 175 componentes (o la varianza objetivo indicada).
- Ajustar un detector Mahalanobis (covarianza empírica) con tráfico normal.
- Calibrar el umbral para una tasa máxima de falsos positivos y evaluar sobre test.

Ejecución:

```bash
uv run experiments/27_tfidf_pca_mahalanobis_srbh/train_mahalanobis_tfidf_pca.py
```

Opciones útiles (`--help` para más):
- `--target-variance` para fijar la cantidad de varianza retenida por PCA.
- `--max-fpr` para ajustar la tasa falsa deseada.
- `--no-preprocess` para usar peticiones sin pipeline HTTP.

Artefactos esperados:
- `experiments/27_tfidf_pca_mahalanobis_srbh/mahalanobis_tfidf_pca{n}.joblib`
- `experiments/27_tfidf_pca_mahalanobis_srbh/srbh_test_embeddings.npz`
- `experiments/27_tfidf_pca_mahalanobis_srbh/model_metrics.json`
