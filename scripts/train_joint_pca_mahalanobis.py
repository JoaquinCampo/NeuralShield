#!/usr/bin/env python3
"""Train Mahalanobis detector on joint PCA of raw TF-IDF and SecBERT embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import typer
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from neuralshield.anomaly import MahalanobisDetector

app = typer.Typer(help="Joint PCA fusion of TF-IDF and SecBERT embeddings.")


def _load_npz(path: Path, expect_labels: bool) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    labels = None
    if expect_labels:
        if "labels" not in data.files:
            raise ValueError(f"{path} missing labels array")
        labels = data["labels"].astype(str)
    logger.info("Loaded embeddings from {} with shape {}", path, embeddings.shape)
    return embeddings, labels


def _standardize_block(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64)
    std = train.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    train_std = ((train - mean) / std).astype(np.float32)
    test_std = ((test - mean) / std).astype(np.float32)
    return train_std, test_std, mean.astype(np.float32), std.astype(np.float32)


@app.command()
def main(
    tfidf_train_raw: Path = typer.Argument(..., help="Raw TF-IDF train embeddings (.npz)"),
    tfidf_test_raw: Path = typer.Argument(..., help="Raw TF-IDF test embeddings (.npz)"),
    secbert_train: Path = typer.Argument(..., help="SecBERT train embeddings (.npz)"),
    secbert_test: Path = typer.Argument(..., help="SecBERT test embeddings (.npz)"),
    output_dir: Path = typer.Argument(..., help="Directory for outputs"),
    n_components: int = typer.Option(300, help="Number of PCA components for joint projection"),
    max_fpr: float = typer.Option(0.05, help="Target FPR for threshold calibration"),
    svd_solver: str = typer.Option("randomized", help="PCA SVD solver"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tfidf_train, _ = _load_npz(tfidf_train_raw, expect_labels=False)
    tfidf_test, tfidf_labels = _load_npz(tfidf_test_raw, expect_labels=True)
    secbert_train, secbert_train_labels = _load_npz(secbert_train, expect_labels=True)
    secbert_test, secbert_test_labels = _load_npz(secbert_test, expect_labels=True)

    if tfidf_train.shape[0] != secbert_train.shape[0]:
        raise ValueError("Train sample counts differ between TF-IDF and SecBERT")
    if tfidf_test.shape[0] != secbert_test.shape[0]:
        raise ValueError("Test sample counts differ between TF-IDF and SecBERT")
    if tfidf_labels is None or secbert_test_labels is None:
        raise ValueError("Missing labels for alignment check")
    if not np.array_equal(tfidf_labels, secbert_test_labels):
        raise ValueError("TF-IDF and SecBERT test labels misaligned")

    logger.info("Standardising TF-IDF block")
    tfidf_train_std, tfidf_test_std, tfidf_mean, tfidf_std = _standardize_block(tfidf_train, tfidf_test)

    logger.info("Standardising SecBERT block")
    secbert_train_std, secbert_test_std, secbert_mean, secbert_std = _standardize_block(secbert_train, secbert_test)

    train_concat = np.concatenate([tfidf_train_std, secbert_train_std], axis=1)
    test_concat = np.concatenate([tfidf_test_std, secbert_test_std], axis=1)
    logger.info("Joint feature matrix shapes train=%s test=%s", train_concat.shape, test_concat.shape)

    logger.info("Running PCA to {} components (solver={})", n_components, svd_solver)
    pca = PCA(n_components=n_components, random_state=42, svd_solver=svd_solver)
    train_latent = pca.fit_transform(train_concat).astype(np.float32)
    test_latent = pca.transform(test_concat).astype(np.float32)
    explained = float(pca.explained_variance_ratio_.sum())
    logger.info("Joint PCA explained variance {:.2%}", explained)

    detector = MahalanobisDetector(name="joint_pca_mahalanobis")
    detector.fit(train_latent)

    labels = tfidf_labels
    binary = np.array([1 if label == "attack" else 0 for label in labels], dtype=np.int8)
    normal_mask = binary == 0

    logger.info("Calibrating threshold at FPR {:.1%}", max_fpr)
    threshold = detector.set_threshold(test_latent[normal_mask], max_fpr=max_fpr)
    scores = detector.scores(test_latent)
    predictions = scores > threshold

    precision = precision_score(binary, predictions, zero_division=0)
    recall = recall_score(binary, predictions, zero_division=0)
    f1 = f1_score(binary, predictions, zero_division=0)
    accuracy = accuracy_score(binary, predictions)
    roc_auc = roc_auc_score(binary, scores)
    specificity = float(((predictions == 0) & (binary == 0)).sum() / (binary == 0).sum())
    fpr = float(((predictions == 1) & (binary == 0)).sum() / (binary == 0).sum())

    metrics = {
        "model": "Mahalanobis_joint_pca",
        "n_components": n_components,
        "explained_variance": explained,
        "max_fpr": max_fpr,
        "threshold": float(threshold),
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
        "train_shape": train_latent.shape,
        "test_shape": test_latent.shape,
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Saved metrics to {}", output_dir / "metrics.json")

    payload = {
        "name": "joint_pca_mahalanobis",
        "model": detector._model,
        "threshold": threshold,
        "tfidf_mean": tfidf_mean,
        "tfidf_std": tfidf_std,
        "secbert_mean": secbert_mean,
        "secbert_std": secbert_std,
        "pca": pca,
    }
    joblib.dump(payload, output_dir / "joint_pca_mahalanobis.joblib")
    logger.info("Saved model payload to {}", output_dir / "joint_pca_mahalanobis.joblib")


if __name__ == "__main__":
    app()
