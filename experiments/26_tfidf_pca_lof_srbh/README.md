# Experimento: TF-IDF + PCA + LOF en SRBH

Objetivo:
- Generar representaciones TF-IDF del dataset SRBH 2020.
- Reducir dimensionalidad con PCA (175 componentes por defecto o varianza objetivo).
- Entrenar un detector LOF calibrado para una tasa máxima de falsos positivos configurable.
- Guardar artefactos (modelo, métricas y embeddings de test) para análisis posteriores.

Ejecución:

```bash
uv run experiments/26_tfidf_pca_lof_srbh/train_lof_tfidf_pca.py
```

Parámetros útiles (`--help` para ver todos):
- `--train-path` y `--test-path` para apuntar a splits alternativos.
- `--pca-components` o `--target-variance` para controlar PCA.
- `--n-neighbors`, `--contamination` y `--max-fpr` para ajustar LOF.
- `--no-preprocess` para trabajar con datos crudos.

Salidas esperadas:
- `experiments/26_tfidf_pca_lof_srbh/lof_tfidf_pca{n}_k{neighbors}.joblib`
- `experiments/26_tfidf_pca_lof_srbh/srbh_test_embeddings.npz`
- `experiments/26_tfidf_pca_lof_srbh/model_metrics.json`
