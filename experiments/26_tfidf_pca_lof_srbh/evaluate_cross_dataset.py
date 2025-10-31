from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import typer
from loguru import logger

from neuralshield.encoding.data.jsonl import JSONLRequestReader
from neuralshield.preprocessing.pipeline import preprocess
from neuralshield.preprocessing.steps.exceptions import MalformedHttpRequestError

app = typer.Typer(help="Evalúa un modelo LOF TF-IDF+PCA sobre un dataset externo.")


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
    model_path: Path = typer.Argument(
        Path("experiments/26_tfidf_pca_lof_srbh/lof_tfidf_pca175_k100.joblib"),
        help="Modelo LOF entrenado sobre SRBH.",
    ),
    dataset_path: Path = typer.Argument(
        Path("src/neuralshield/data/PKDD/test.jsonl"),
        help="Dataset JSONL a evaluar.",
    ),
    batch_size: int = typer.Option(1000, help="Batch size para el lector JSONL."),
    output_path: Path | None = typer.Option(
        None,
        help="Ruta opcional para guardar métricas en JSON.",
    ),
) -> None:
    payload = joblib.load(model_path)
    detector = payload["detector"]
    vectorizer = payload["vectorizer"]
    pca = payload["pca"]
    threshold = payload["threshold"]
    preprocess_flag = bool(payload.get("preprocess", True))

    logger.info("Evaluando modelo {name} sobre {dataset}", name=payload["name"], dataset=str(dataset_path))
    logger.info("Preprocess esperado por el modelo: {}", "habilitado" if preprocess_flag else "deshabilitado")

    texts, labels = load_dataset(dataset_path, batch_size, apply_preprocess=preprocess_flag)
    logger.info("Cargadas {count} peticiones (etiquetas únicas={labels})", count=len(texts), labels=set(labels))

    embeddings = vectorizer.transform(texts)
    embeddings = pca.transform(embeddings.toarray()).astype(np.float32)

    if detector._model is None:
        raise RuntimeError("El detector cargado no tiene modelo interno.")

    scores = detector._model.score_samples(embeddings)
    predictions = (scores >= threshold).astype(int)
    labels_binary = np.array([1 if label == "attack" else 0 for label in labels], dtype=np.int32)

    attack_scores = scores[labels_binary == 1]
    normal_scores = scores[labels_binary == 0]

    recall = float(np.mean(attack_scores >= threshold)) if attack_scores.size else float("nan")
    actual_fpr = float(np.mean(normal_scores >= threshold)) if normal_scores.size else float("nan")

    tp = int(np.sum((predictions == 1) & (labels_binary == 1)))
    fp = int(np.sum((predictions == 1) & (labels_binary == 0)))
    tn = int(np.sum((predictions == 0) & (labels_binary == 0)))
    fn = int(np.sum((predictions == 0) & (labels_binary == 1)))

    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    accuracy = float((tp + tn) / len(labels_binary)) if len(labels_binary) else float("nan")
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    metrics = {
        "model": payload["name"],
        "dataset": str(dataset_path),
        "threshold": threshold,
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "accuracy": accuracy,
        "actual_fpr": actual_fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "preprocess": preprocess_flag,
    }

    logger.info(
        "Resultados: recall={recall:.2%} precision={precision:.2%} "
        "f1={f1_score:.2%} accuracy={accuracy:.2%} fpr={actual_fpr:.2%}",
        **metrics,
    )

    if output_path:
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info("Métricas guardadas en {path}", path=str(output_path))


if __name__ == "__main__":
    app()
