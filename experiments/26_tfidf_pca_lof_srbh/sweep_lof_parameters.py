from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import typer
from loguru import logger

from neuralshield.anomaly import LOFDetector

app = typer.Typer(help="Barrer parámetros de LOF sobre embeddings SRBH precomputados.")


def load_embeddings(
    train_path: Path,
    test_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if not train_path.exists():
        raise FileNotFoundError(f"No se encontró embeddings de entrenamiento {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"No se encontró embeddings de test {test_path}")

    train_data = np.load(train_path)
    test_data = np.load(test_path)
    train_embeddings = train_data["embeddings"].astype(np.float32)
    test_embeddings = test_data["embeddings"].astype(np.float32)
    labels = test_data["labels"]
    label_binary = np.array([1 if label == "attack" else 0 for label in labels], dtype=np.int32)
    return train_embeddings, test_embeddings, label_binary


def evaluate_configuration(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    *,
    n_neighbors: int,
    max_fpr: float,
    contamination: float,
) -> dict[str, float | int]:
    detector = LOFDetector(n_neighbors=n_neighbors, contamination=contamination)
    detector.fit(train_embeddings)

    if detector._model is None:
        raise RuntimeError("LOFDetector no quedó inicializado correctamente.")

    scores = detector._model.score_samples(test_embeddings)
    normal_mask = test_labels == 0
    normal_scores = scores[normal_mask]

    threshold = float(np.percentile(normal_scores, 100 * (1 - max_fpr)))
    predictions = (scores >= threshold).astype(int)

    attack_scores = scores[~normal_mask]
    actual_fpr = float(np.mean(normal_scores >= threshold))
    recall = float(np.mean(attack_scores >= threshold))

    tp = int(np.sum((predictions == 1) & (test_labels == 1)))
    fp = int(np.sum((predictions == 1) & (test_labels == 0)))
    tn = int(np.sum((predictions == 0) & (test_labels == 0)))
    fn = int(np.sum((predictions == 0) & (test_labels == 1)))

    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    accuracy = float((tp + tn) / len(test_labels))
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )

    return {
        "n_neighbors": n_neighbors,
        "max_fpr": max_fpr,
        "contamination": contamination,
        "threshold": threshold,
        "actual_fpr": actual_fpr,
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


@app.command()
def main(
    train_embeddings_path: Path = typer.Option(
        Path("experiments/26_tfidf_pca_lof_srbh/srbh_train_embeddings.npz"),
        help="Ruta a embeddings de entrenamiento (normal).",
    ),
    test_embeddings_path: Path = typer.Option(
        Path("experiments/26_tfidf_pca_lof_srbh/srbh_test_embeddings.npz"),
        help="Ruta a embeddings de test con etiquetas.",
    ),
    neighbors: str = typer.Option(
        "50,100,150",
        help="Lista separada por comas de valores n_neighbors.",
    ),
    fprs: str = typer.Option(
        "0.05,0.1,0.2",
        help="Lista separada por comas de tasas objetivo de FP.",
    ),
    contamination: float = typer.Option(
        0.05,
        help="Parámetro de contaminación pasado al LOF.",
    ),
    output_path: Path = typer.Option(
        Path("experiments/26_tfidf_pca_lof_srbh/lof_sweep_results.json"),
        help="Archivo donde persistir resultados agregados.",
    ),
) -> None:
    neighbor_values = [int(value.strip()) for value in neighbors.split(",") if value]
    fpr_values = [float(value.strip()) for value in fprs.split(",") if value]

    logger.info(
        "Cargando embeddings train={train} test={test}",
        train=str(train_embeddings_path),
        test=str(test_embeddings_path),
    )
    train_embeddings, test_embeddings, test_labels = load_embeddings(
        train_embeddings_path, test_embeddings_path
    )
    logger.info(
        "Embeddings cargados: train=%s test=%s ataques=%d normales=%d",
        train_embeddings.shape,
        test_embeddings.shape,
        int(np.sum(test_labels == 1)),
        int(np.sum(test_labels == 0)),
    )

    results: list[dict[str, float | int]] = []
    for n_neighbors, max_fpr in product(neighbor_values, fpr_values):
        logger.info(
            "Evaluando n_neighbors={neighbors} max_fpr={fpr:.0%}",
            neighbors=n_neighbors,
            fpr=max_fpr,
        )
        metrics = evaluate_configuration(
            train_embeddings,
            test_embeddings,
            test_labels,
            n_neighbors=n_neighbors,
            max_fpr=max_fpr,
            contamination=contamination,
        )
        results.append(metrics)
        logger.info(
            "Resultados: recall={recall:.2%} precision={precision:.2%} "
            "f1={f1_score:.2%} actual_fpr={actual_fpr:.2%}",
            **metrics,
        )

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Resultados guardados en {path}", path=str(output_path))


if __name__ == "__main__":
    app()
