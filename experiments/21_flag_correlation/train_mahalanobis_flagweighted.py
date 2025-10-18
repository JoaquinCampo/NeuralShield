#!/usr/bin/env python3
"""Train and evaluate a Mahalanobis detector on flag-weighted SecBERT embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import typer
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import wandb
from neuralshield.anomaly.mahalanobis import MahalanobisDetector

app = typer.Typer(
    help="Train and evaluate Mahalanobis detector on flag-weighted SecBERT embeddings."
)


def load_embeddings(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels from an NPZ file."""
    logger.info("Loading embeddings from {path}", path=str(npz_path))
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    labels_raw = data["labels"]

    # Convert labels to binary: 0 = normal, 1 = attack
    if labels_raw.dtype.kind in {"U", "S", "O"}:
        labels = np.array(
            [1 if str(label).lower() == "attack" else 0 for label in labels_raw],
            dtype=np.int32,
        )
    else:
        labels = labels_raw.astype(np.int32)

    logger.info(
        "Loaded {n_samples} samples with dimension {dim}",
        n_samples=len(embeddings),
        dim=embeddings.shape[1],
    )
    logger.info(
        "Label distribution: normal={normal} anomalous={anomalous}",
        normal=int(np.sum(labels == 0)),
        anomalous=int(np.sum(labels == 1)),
    )
    return embeddings, labels


def summarize_metrics(
    *,
    y_true: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    """Compute evaluation metrics for Mahalanobis predictions."""
    acc = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "true_positive_rate": float(recall),
        "auc": float(auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


@app.command()
def main(
    train_embeddings: Path = typer.Option(
        Path(
            "experiments/21_flag_correlation/csic_embeddings/train_flagweighted_converted.npz"
        ),
        help="Training embeddings NPZ (expected normal traffic).",
    ),
    test_embeddings: Path = typer.Option(
        Path(
            "experiments/21_flag_correlation/csic_embeddings/test_flagweighted_converted.npz"
        ),
        help="Test embeddings NPZ.",
    ),
    output_dir: Path = typer.Option(
        Path("experiments/21_flag_correlation/csic_embeddings/mahalanobis"),
        help="Directory to store artifacts.",
    ),
    max_fpr: float = typer.Option(
        0.05, help="Desired maximum false positive rate on normal data."
    ),
    save_model: bool = typer.Option(
        True, help="Save trained Mahalanobis model as joblib."
    ),
    use_wandb: bool = typer.Option(True, help="Log metrics to Weights & Biases."),
    wandb_run_name: str = typer.Option(
        "flagweighted-secbert-mahalanobis", help="W&B run name."
    ),
) -> None:
    """Train and evaluate Mahalanobis detector on flag-weighted SecBERT embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_emb, train_labels = load_embeddings(train_embeddings)
    if np.any(train_labels != 0):
        logger.warning(
            "Training embeddings include non-normal labels; "
            "filtering to normal samples only."
        )
        train_mask = train_labels == 0
        train_emb = train_emb[train_mask]

    test_emb, test_labels = load_embeddings(test_embeddings)

    detector = MahalanobisDetector(name="secbert-flag-weighted")
    detector.fit(train_emb)

    threshold = detector.set_threshold(train_emb, max_fpr=max_fpr)
    test_scores = detector.scores(test_emb)
    test_preds = (test_scores > threshold).astype(int)

    metrics = summarize_metrics(
        y_true=test_labels,
        scores=test_scores,
        predictions=test_preds,
    )
    metrics["threshold"] = float(threshold)

    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        logger.info("  {key} = {value}", key=key, value=value)

    if save_model:
        model_path = output_dir / "mahalanobis_flagweighted.joblib"
        detector.save(model_path)

    metrics_path = output_dir / "mahalanobis_flagweighted_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved metrics to {path}", path=str(metrics_path))

    scores_path = output_dir / "mahalanobis_flagweighted_scores.npz"
    np.savez_compressed(
        scores_path,
        scores=test_scores.astype(np.float32),
        predictions=test_preds.astype(np.int32),
        labels=test_labels.astype(np.int32),
    )
    logger.info("Saved scores to {path}", path=str(scores_path))

    if use_wandb:
        wandb_run = wandb.init(
            project="neuralshield",
            name=wandb_run_name,
            config={
                "detector": "mahalanobis",
                "encoder": "secbert-flag-weighted",
                "embedding_dim": int(train_emb.shape[1]),
                "max_fpr": max_fpr,
            },
        )
        wandb_run.log(metrics)
        wandb_run.finish()


if __name__ == "__main__":
    app()
