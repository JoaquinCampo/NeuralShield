#!/usr/bin/env python3
"""Train Isolation Forest on concatenated TF-IDF/PCA + SecBERT embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import typer
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from neuralshield.anomaly import IsolationForestDetector

app = typer.Typer(help="Train IsolationForest on concatenated embeddings.")


def _load_embeddings(path: Path, *, require_labels: bool) -> tuple[np.ndarray, Optional[np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    labels = None
    if require_labels:
        if "labels" not in data.files:
            raise ValueError(f"{path} is missing 'labels'")
        labels = data["labels"].astype(str)
    logger.info("Loaded embeddings from {} with shape {}", path, embeddings.shape)
    return embeddings, labels


def _standardize(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_embeddings.mean(axis=0, dtype=np.float64)
    std = train_embeddings.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    train_scaled = (train_embeddings - mean) / std
    test_scaled = (test_embeddings - mean) / std
    return (
        train_scaled.astype(np.float32),
        test_scaled.astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


@app.command()
def main(
    tfidf_train: Path = typer.Argument(..., help="TF-IDF/PCA train embeddings (.npz)"),
    tfidf_test: Path = typer.Argument(..., help="TF-IDF/PCA test embeddings (.npz)"),
    secbert_train: Path = typer.Argument(..., help="SecBERT train embeddings (.npz)"),
    secbert_test: Path = typer.Argument(..., help="SecBERT test embeddings (.npz)"),
    output_dir: Path = typer.Argument(..., help="Directory for outputs"),
    max_fpr: float = typer.Option(0.05, help="Target false positive rate"),
    contamination: float = typer.Option(0.05, help="IsolationForest contamination"),
    n_estimators: int = typer.Option(200, help="Number of trees"),
    max_samples: str = typer.Option("auto", help="Samples per tree ('auto' or integer)"),
    random_state: int = typer.Option(42, help="Random seed"),
    n_jobs: int = typer.Option(-1, help="Parallel workers"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tfidf_train_embeddings, _ = _load_embeddings(tfidf_train, require_labels=False)
    tfidf_test_embeddings, tfidf_labels = _load_embeddings(tfidf_test, require_labels=True)
    secbert_train_embeddings, secbert_train_labels = _load_embeddings(secbert_train, require_labels=True)
    secbert_test_embeddings, secbert_test_labels = _load_embeddings(secbert_test, require_labels=True)

    if tfidf_train_embeddings.shape[0] != secbert_train_embeddings.shape[0]:
        raise ValueError("Train embedding counts differ between TF-IDF and SecBERT")
    if tfidf_test_embeddings.shape[0] != secbert_test_embeddings.shape[0]:
        raise ValueError("Test embedding counts differ between TF-IDF and SecBERT")
    if tfidf_labels is None or secbert_test_labels is None:
        raise ValueError("Test labels missing from one of the embeddings")
    if not np.array_equal(tfidf_labels, secbert_test_labels):
        raise ValueError("TF-IDF and SecBERT test labels are misaligned")

    logger.info(
        "Concatenating embeddings (TF-IDF dims {} + SecBERT dims {})",
        tfidf_train_embeddings.shape[1],
        secbert_train_embeddings.shape[1],
    )
    train_concat = np.concatenate([tfidf_train_embeddings, secbert_train_embeddings], axis=1)
    test_concat = np.concatenate([tfidf_test_embeddings, secbert_test_embeddings], axis=1)

    logger.info("Standardising concatenated embeddings")
    train_scaled, test_scaled, mean, std = _standardize(train_concat, test_concat)

    max_samples_value: int | str
    if max_samples.lower() == "auto":
        max_samples_value = "auto"
    else:
        try:
            max_samples_value = int(max_samples)
        except ValueError as exc:
            raise typer.BadParameter(
                "max_samples must be 'auto' or an integer"
            ) from exc

    detector = IsolationForestDetector(
        name="iforest_concat",
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples_value,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    detector.fit(train_scaled)

    labels = tfidf_labels
    binary = np.array([1 if label == "attack" else 0 for label in labels], dtype=np.int8)
    normal_mask = binary == 0

    logger.info("Scoring test set")
    scores = detector.scores(test_scaled)
    threshold = float(np.quantile(scores[normal_mask], 1 - max_fpr))

    predictions = scores > threshold

    precision = precision_score(binary, predictions, zero_division=0)
    recall = recall_score(binary, predictions, zero_division=0)
    f1 = f1_score(binary, predictions, zero_division=0)
    accuracy = accuracy_score(binary, predictions)
    roc_auc = roc_auc_score(binary, scores)
    specificity = float(((predictions == 0) & (binary == 0)).sum() / (binary == 0).sum())
    fpr = float(((predictions == 1) & (binary == 0)).sum() / (binary == 0).sum())

    metrics = {
        "model": "IsolationForest_concat_secbert_tfidf",
        "max_fpr": max_fpr,
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
        "specificity": specificity,
        "false_positive_rate": fpr,
        "roc_auc": float(roc_auc),
        "true_positives": int(((predictions == 1) & (binary == 1)).sum()),
        "false_positives": int(((predictions == 1) & (binary == 0)).sum()),
        "true_negatives": int(((predictions == 0) & (binary == 0)).sum()),
        "false_negatives": int(((predictions == 0) & (binary == 1)).sum()),
        "train_shape": train_scaled.shape,
        "test_shape": test_scaled.shape,
        "contamination": contamination,
        "n_estimators": n_estimators,
        "max_samples": max_samples_value,
        "random_state": random_state,
    }

    (output_dir / "metrics_iforest.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Saved metrics to {}", output_dir / "metrics_iforest.json")

    payload = {
        "name": "iforest_concat_secbert_tfidf",
        "detector": detector,
        "threshold": threshold,
        "contamination": contamination,
        "n_estimators": n_estimators,
        "max_samples": max_samples_value,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "scaler_mean": mean,
        "scaler_std": std,
        "feature_dims": {
            "tfidf": tfidf_train_embeddings.shape[1],
            "secbert": secbert_train_embeddings.shape[1],
        },
    }
    joblib.dump(payload, output_dir / "concat_iforest.joblib")
    logger.info("Saved concatenated IsolationForest model to {}", output_dir / "concat_iforest.joblib")


if __name__ == "__main__":
    app()
