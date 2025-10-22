from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import typer
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from neuralshield.anomaly import LOFDetector
from neuralshield.encoding.data.jsonl import JSONLRequestReader
from neuralshield.preprocessing.pipeline import preprocess
from neuralshield.preprocessing.steps.exceptions import MalformedHttpRequestError

app = typer.Typer(help="Train LOF on PKDD using TF-IDF + PCA embeddings.")


def load_dataset(
    path: Path,
    batch_size: int,
) -> tuple[list[str], list[str]]:
    reader = JSONLRequestReader(path, use_pipeline=False)
    texts: list[str] = []
    labels: list[str] = []
    skipped = 0

    for batch, batch_labels in reader.iter_batches(batch_size=batch_size):
        for idx, text in enumerate(batch):
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
            "Skipped {count} malformed requests from {path}",
            count=skipped,
            path=str(path),
        )
    return texts, labels


@app.command()
def main(
    train_path: Path = typer.Option(
        Path("src/neuralshield/data/PKDD/train.jsonl"),
        help="PKDD training split (expected to be normal traffic).",
    ),
    test_path: Path = typer.Option(
        Path("src/neuralshield/data/PKDD/test.jsonl"),
        help="PKDD test split.",
    ),
    output_dir: Path = typer.Option(
        Path("experiments/22_tfidf_pca_lof_pkdd"),
        help="Directory to store artifacts.",
    ),
    batch_size: int = typer.Option(1000, help="Batch size for JSONL reader."),
    max_features: int = typer.Option(5000, help="TF-IDF max features."),
    ngram_min: int = typer.Option(1, help="Minimum n-gram length for TF-IDF."),
    ngram_max: int = typer.Option(3, help="Maximum n-gram length for TF-IDF."),
    min_df: int = typer.Option(2, help="Minimum document frequency for TF-IDF."),
    pca_components: int = typer.Option(175, help="Number of PCA components."),
    n_neighbors: int = typer.Option(100, help="Number of neighbors for LOF."),
    max_fpr: float = typer.Option(0.05, help="Target false positive rate."),
    contamination: float = typer.Option(
        0.05, help="LOF contamination parameter."
    ),
    target_variance: float | None = typer.Option(
        None,
        help=(
            "If set to a value in (0, 1], PCA retains the minimum number "
            "of components that preserve at least this fraction of total "
            "variance. Overrides --pca-components."
        ),
    ),
) -> None:
    if target_variance is not None and not 0.0 < target_variance <= 1.0:
        raise ValueError("target_variance must be in (0, 1].")

    if target_variance is not None:
        pca_descriptor = f"variance>={target_variance:.0%}"
    else:
        pca_descriptor = str(pca_components)

    logger.info("=" * 80)
    logger.info(
        "TF-IDF + PCA({components}) + LOF (k={neighbors}) on PKDD",
        components=pca_descriptor,
        neighbors=n_neighbors,
    )
    logger.info("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training data from {path}", path=str(train_path))
    train_texts, train_labels = load_dataset(train_path, batch_size)
    logger.info(
        "Loaded {count} training samples (labels={labels})",
        count=len(train_texts),
        labels=set(train_labels),
    )

    logger.info("Loading test data from {path}", path=str(test_path))
    test_texts, test_labels = load_dataset(test_path, batch_size)
    logger.info(
        "Loaded {count} test samples label_dist={dist}",
        count=len(test_texts),
        dist={
            label: test_labels.count(label) for label in sorted(set(test_labels))
        },
    )

    logger.info(
        "Fitting TF-IDF (max_features={max_features}, "
        "ngram_range=({min_n}, {max_n}), min_df={min_df})",
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

    pca_kwargs: dict[str, float | int] = {"random_state": 42}
    if target_variance is not None:
        logger.info(
            "Applying PCA to retain >= {variance:.0%} variance",
            variance=target_variance,
        )
        pca_kwargs["n_components"] = target_variance
    else:
        logger.info(
            "Applying PCA to {components} components",
            components=pca_components,
        )
        pca_kwargs["n_components"] = pca_components

    pca = PCA(**pca_kwargs)
    train_embeddings = pca.fit_transform(train_tfidf.toarray())
    test_embeddings = pca.transform(test_tfidf.toarray())
    explained = float(pca.explained_variance_ratio_.sum())
    logger.info(
        "PCA explained variance {variance:.2%} (components={components})",
        variance=explained,
        components=train_embeddings.shape[1],
    )

    logger.info(
        "Training LOF detector (n_neighbors={neighbors})",
        neighbors=n_neighbors,
    )
    detector = LOFDetector(
        n_neighbors=n_neighbors,
        contamination=contamination,
    )
    detector.fit(train_embeddings.astype(np.float32))

    logger.info("Scoring test embeddings and calibrating threshold")
    test_scores = detector.scores(test_embeddings.astype(np.float32))

    labels_binary = np.array(
        [1 if label == "attack" else 0 for label in test_labels], dtype=np.int32
    )
    normal_mask = labels_binary == 0
    normal_scores = test_scores[normal_mask]

    threshold = float(np.percentile(normal_scores, 100 * (1 - max_fpr)))
    detector._threshold = threshold
    actual_fpr = float(np.mean(normal_scores > threshold))
    logger.info(
        "Threshold={threshold:.4f} target_fpr={target:.2%} actual_fpr={actual:.2%}",
        threshold=threshold,
        target=max_fpr,
        actual=actual_fpr,
    )

    predictions = detector.predict(test_embeddings.astype(np.float32))
    anomalous_scores = test_scores[~normal_mask]
    recall = float(np.mean(anomalous_scores > threshold))

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

    effective_components = train_embeddings.shape[1]

    model_payload = {
        "name": f"LOF_TFIDF_PCA{effective_components}_k{n_neighbors}",
        "detector": detector,
        "vectorizer": vectorizer,
        "pca": pca,
        "threshold": threshold,
        "n_neighbors": n_neighbors,
        "n_components": effective_components,
        "explained_variance": explained,
        "contamination": contamination,
        "target_variance": target_variance,
    }
    model_path = (
        output_dir
        / f"lof_tfidf_pca{effective_components}_k{n_neighbors}.joblib"
    )
    joblib.dump(model_payload, model_path)
    logger.info("Saved model to {path}", path=str(model_path))

    embeddings_path = output_dir / "pkdd_test_embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        embeddings=test_embeddings.astype(np.float32),
        labels=np.array(test_labels),
    )
    logger.info("Saved test embeddings to {path}", path=str(embeddings_path))

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
        "n_neighbors": n_neighbors,
        "n_components": effective_components,
        "target_variance": target_variance,
    }
    metrics_path = output_dir / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved metrics to {path}", path=str(metrics_path))

    logger.info("=" * 80)
    logger.info("Results: recall={recall:.2%} precision={precision:.2%} "
                "f1={f1:.2%} accuracy={accuracy:.2%}",
                recall=recall, precision=precision, f1=f1, accuracy=accuracy)
    logger.info("=" * 80)


if __name__ == "__main__":
    app()
