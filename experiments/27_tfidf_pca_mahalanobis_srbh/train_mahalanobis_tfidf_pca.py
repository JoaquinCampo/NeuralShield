from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import typer
from loguru import logger
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from neuralshield.encoding.data.jsonl import JSONLRequestReader
from neuralshield.preprocessing.pipeline import preprocess
from neuralshield.preprocessing.steps.exceptions import MalformedHttpRequestError

app = typer.Typer(
    help="Entrena Mahalanobis sobre SRBH usando embeddings TF-IDF reducidos con PCA."
)


def load_dataset(
    path: Path,
    batch_size: int,
    apply_preprocess: bool,
) -> tuple[list[str], list[str]]:
    reader = JSONLRequestReader(path, use_pipeline=False)
    texts: list[str] = []
    labels: list[str] = []
    skipped = 0

    for batch, batch_labels in reader.iter_batches(batch_size=batch_size):
        for idx, text in enumerate(batch):
            processed = text
            if apply_preprocess:
                try:
                    processed = preprocess(text)
                except MalformedHttpRequestError:
                    skipped += 1
                    continue
            texts.append(processed)
            if batch_labels:
                labels.append(str(batch_labels[idx]))

    if not labels:
        labels = ["valid"] * len(texts)
    if skipped:
        logger.debug(
            "Se descartaron {count} peticiones malformadas de {path}",
            count=skipped,
            path=str(path),
        )
    return texts, labels


@app.command()
def main(
    train_path: Path = typer.Option(
        Path("src/neuralshield/data/SR_BH_2020/train.jsonl"),
        help="Split de entrenamiento (debería contener solo tráfico normal).",
    ),
    test_path: Path = typer.Option(
        Path("src/neuralshield/data/SR_BH_2020/test.jsonl"),
        help="Split de evaluación con ataques.",
    ),
    output_dir: Path = typer.Option(
        Path("experiments/27_tfidf_pca_mahalanobis_srbh"),
        help="Directorio para artefactos del experimento.",
    ),
    batch_size: int = typer.Option(1000, help="Tamaño de lote del lector JSONL."),
    max_features: int = typer.Option(5000, help="Número máximo de features TF-IDF."),
    ngram_min: int = typer.Option(1, help="Longitud mínima de n-gramas."),
    ngram_max: int = typer.Option(3, help="Longitud máxima de n-gramas."),
    min_df: int = typer.Option(2, help="Mínimo DF para TF-IDF."),
    pca_components: int = typer.Option(175, help="Número de componentes PCA."),
    target_variance: float | None = typer.Option(
        None,
        help=(
            "Si se define en (0,1], PCA retiene el mínimo número de componentes "
            "que preserven al menos esa fracción de varianza. Sobrescribe --pca-components."
        ),
    ),
    max_fpr: float = typer.Option(0.05, help="Tasa máxima de falsos positivos."),
    preprocess_data: bool = typer.Option(
        True,
        "--preprocess/--no-preprocess",
        help="Aplicar el pipeline HTTP antes de vectorizar.",
    ),
) -> None:
    if target_variance is not None and not 0.0 < target_variance <= 1.0:
        raise ValueError("target_variance debe estar en (0, 1].")

    output_dir.mkdir(parents=True, exist_ok=True)
    prep_state = "ACTIVO" if preprocess_data else "INACTIVO"

    if target_variance is not None:
        pca_descriptor = f"variance>={target_variance:.0%}"
    else:
        pca_descriptor = str(pca_components)

    logger.info("=" * 80)
    logger.info(
        "TF-IDF + PCA({components}) + Mahalanobis en SRBH (preprocess={preprocess})",
        components=pca_descriptor,
        preprocess=prep_state.lower(),
    )
    logger.info("=" * 80)

    logger.info(
        "Cargando entrenamiento {path} (preprocess={preprocess})",
        path=str(train_path),
        preprocess=prep_state.lower(),
    )
    train_texts, train_labels = load_dataset(
        train_path, batch_size, apply_preprocess=preprocess_data
    )
    logger.info(
        "Cargadas {count} muestras de entrenamiento (labels={labels})",
        count=len(train_texts),
        labels=set(train_labels),
    )

    logger.info(
        "Cargando test {path} (preprocess={preprocess})",
        path=str(test_path),
        preprocess=prep_state.lower(),
    )
    test_texts, test_labels = load_dataset(
        test_path, batch_size, apply_preprocess=preprocess_data
    )
    logger.info(
        "Cargadas {count} muestras de test dist_labels={dist}",
        count=len(test_texts),
        dist={label: test_labels.count(label) for label in sorted(set(test_labels))},
    )

    logger.info(
        "Entrenando TF-IDF (max_features={max_features}, ngram_range=({min_n}, {max_n}), min_df={min_df})",
        max_features=max_features,
        min_n=ngram_min,
        max_n=ngram_max,
        min_df=min_df,
    )
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)
    logger.info("TF-IDF shapes train=%s test=%s", train_tfidf.shape, test_tfidf.shape)

    logger.info("Aplicando PCA")
    pca_kwargs: dict[str, float | int] = {"random_state": 42}
    if target_variance is not None:
        logger.info("PCA con target_variance {variance:.0%}", variance=target_variance)
        pca_kwargs["n_components"] = target_variance
    else:
        logger.info("PCA con {components} componentes", components=pca_components)
        pca_kwargs["n_components"] = pca_components

    pca = PCA(**pca_kwargs)
    train_embeddings = pca.fit_transform(train_tfidf.toarray())
    test_embeddings = pca.transform(test_tfidf.toarray())
    explained = float(pca.explained_variance_ratio_.sum())
    effective_components = int(train_embeddings.shape[1])
    logger.info(
        "PCA explained variance {variance:.2%} (components={components})",
        variance=explained,
        components=effective_components,
    )

    logger.info("Ajustando EmpiricalCovariance (Mahalanobis)")
    detector = EmpiricalCovariance()
    detector.fit(train_embeddings.astype(np.float32))

    logger.info("Calculando distancias Mahalanobis sobre test")
    test_scores = detector.mahalanobis(test_embeddings.astype(np.float32))

    labels_binary = np.array(
        [1 if label == "attack" else 0 for label in test_labels], dtype=np.int32
    )
    normal_mask = labels_binary == 0
    normal_scores = test_scores[normal_mask]

    threshold = float(np.percentile(normal_scores, 100 * (1 - max_fpr)))
    actual_fpr = float(np.mean(normal_scores > threshold))
    logger.info(
        "Umbral={threshold:.4f} max_fpr={target:.2%} actual_fpr={actual:.2%}",
        threshold=threshold,
        target=max_fpr,
        actual=actual_fpr,
    )

    predictions = (test_scores > threshold).astype(int)
    attack_scores = test_scores[~normal_mask]
    recall = float(np.mean(attack_scores > threshold))

    tp = int(np.sum((predictions == 1) & (labels_binary == 1)))
    fp = int(np.sum((predictions == 1) & (labels_binary == 0)))
    tn = int(np.sum((predictions == 0) & (labels_binary == 0)))
    fn = int(np.sum((predictions == 0) & (labels_binary == 1)))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    accuracy = float((tp + tn) / len(labels_binary))
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )

    model_payload = {
        "name": f"Mahalanobis_TFIDF_PCA{effective_components}",
        "detector": detector,
        "vectorizer": vectorizer,
        "pca": pca,
        "threshold": threshold,
        "n_components": effective_components,
        "explained_variance": explained,
        "max_fpr": max_fpr,
        "preprocess": preprocess_data,
        "target_variance": target_variance,
    }
    model_path = output_dir / f"mahalanobis_tfidf_pca{effective_components}.joblib"
    joblib.dump(model_payload, model_path)
    logger.info("Modelo guardado en {path}", path=str(model_path))

    embeddings_path = output_dir / "srbh_test_embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        embeddings=test_embeddings.astype(np.float32),
        labels=np.array(test_labels),
    )
    logger.info("Embeddings de test guardados en {path}", path=str(embeddings_path))

    metrics = {
        "model": model_payload["name"],
        "threshold": threshold,
        "max_fpr": max_fpr,
        "actual_fpr": actual_fpr,
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "explained_variance": explained,
        "n_components": effective_components,
        "preprocess": preprocess_data,
        "target_variance": target_variance,
    }
    metrics_path = output_dir / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Métricas guardadas en {path}", path=str(metrics_path))

    logger.info("=" * 80)
    logger.info(
        "Resultados: recall={recall:.2%} precision={precision:.2%} "
        "f1={f1:.2%} accuracy={accuracy:.2%}",
        recall=recall,
        precision=precision,
        f1=f1,
        accuracy=accuracy,
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    app()
