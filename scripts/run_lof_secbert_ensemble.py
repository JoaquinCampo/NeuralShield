#!/usr/bin/env python3
"""Evaluate LOF + SecBERT Mahalanobis ensemble on CSIC embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

app = typer.Typer()


@dataclass
class DetectorPayload:
    scores: np.ndarray
    threshold: float
    name: str


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    return embeddings, labels


def _ensure_labels_match(labels_a: Sequence[str], labels_b: Sequence[str]) -> None:
    if len(labels_a) != len(labels_b):
        raise ValueError("Label arrays have different lengths")
    if not np.all(labels_a == labels_b):
        raise ValueError("Label arrays differ; ensure datasets share ordering")


def _load_lof_scores(
    embeddings_path: Path,
    model_path: Path,
    pca_path: Path | None = None,
) -> DetectorPayload:
    embeddings, labels = _load_npz(embeddings_path)
    payload = joblib.load(model_path)
    detector = payload["detector"]
    threshold = float(payload.get("threshold") or payload.get("_threshold", 0.0))
    # ensure internal threshold populated
    if hasattr(detector, "_threshold") and detector._threshold is None:
        detector._threshold = threshold
    if pca_path:
        pca = joblib.load(pca_path)
        embeddings = pca.transform(embeddings)
    scores = detector.scores(embeddings)
    return DetectorPayload(scores=scores, threshold=threshold, name="lof"), labels


def _load_mahalanobis_scores(
    embeddings_path: Path,
    model_path: Path,
) -> DetectorPayload:
    embeddings, labels = _load_npz(embeddings_path)
    payload = joblib.load(model_path)
    model = payload["model"]
    threshold = float(payload.get("threshold"))
    scores = model.mahalanobis(embeddings)
    return DetectorPayload(scores=scores.astype(np.float32), threshold=threshold, name="mahalanobis"), labels


def _calc_threshold(scores: np.ndarray, normal_mask: np.ndarray, max_fpr: float) -> float:
    normal_scores = scores[normal_mask]
    return float(np.quantile(normal_scores, 1 - max_fpr))


def _metrics_from_predictions(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    label_attack: str = "attack",
    label_normal: str = "valid",
) -> dict[str, float | int]:
    preds = scores > threshold
    binary = np.array([1 if label == label_attack else 0 for label in labels])
    tp = int(((preds == 1) & (binary == 1)).sum())
    fp = int(((preds == 1) & (binary == 0)).sum())
    tn = int(((preds == 0) & (binary == 0)).sum())
    fn = int(((preds == 0) & (binary == 1)).sum())

    precision = precision_score(binary, preds, zero_division=0)
    recall = recall_score(binary, preds, zero_division=0)
    f1 = f1_score(binary, preds, zero_division=0)
    accuracy = accuracy_score(binary, preds)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    roc_auc = roc_auc_score(binary, scores)
    fpr_curve, tpr_curve, thr_curve = roc_curve(binary, scores)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
        "specificity": float(specificity),
        "false_positive_rate": float(fpr),
        "roc_auc": float(roc_auc),
        "threshold": float(threshold),
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "roc_curve": {
            "fpr": fpr_curve.tolist(),
            "tpr": tpr_curve.tolist(),
            "thresholds": thr_curve.tolist(),
        },
    }


def _standardize(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    subset = scores[mask]
    mean = float(subset.mean())
    std = float(subset.std() + 1e-8)
    return (scores - mean) / std


def _build_plot(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    normal_mask = labels == "valid"
    attack_mask = ~normal_mask
    normal_scores = scores[normal_mask]
    attack_scores = scores[attack_mask]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sns.histplot(normal_scores, bins=60, kde=True, ax=axes[0, 0], color="green", label=f"Normal (n={normal_scores.size})", alpha=0.7)
    sns.histplot(attack_scores, bins=60, kde=True, ax=axes[0, 0], color="red", label=f"Attack (n={attack_scores.size})", alpha=0.5)
    axes[0, 0].axvline(threshold, color="black", linestyle="--", linewidth=1.5)
    axes[0, 0].set_title("Score Distribution: Normal vs Attack")
    axes[0, 0].legend()

    axes[0, 1].hist(normal_scores, bins=60, color="green", histtype="stepfilled", alpha=0.5, label="Normal")
    axes[0, 1].hist(attack_scores, bins=60, color="red", histtype="stepfilled", alpha=0.5, label="Attack")
    axes[0, 1].axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold:.3f})")
    axes[0, 1].set_title("Score Distribution (Non-Stacked)")
    axes[0, 1].legend()

    violin_df = pd.DataFrame(
        {"Score": np.concatenate([normal_scores, attack_scores]), "Type": ["Normal"] * normal_scores.size + ["Attack"] * attack_scores.size}
    )
    sns.violinplot(data=violin_df, x="Type", y="Score", ax=axes[1, 0], palette={"Normal": "green", "Attack": "red"})
    axes[1, 0].axhline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold:.3f})")
    axes[1, 0].set_title("Score Distribution (Violin Plot)")
    axes[1, 0].legend()

    sns.boxplot(data=violin_df, x="Type", y="Score", ax=axes[1, 1], palette={"Normal": "green", "Attack": "red"})
    axes[1, 1].axhline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold:.3f})")
    axes[1, 1].set_title("Score Distribution (Box Plot)")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("Anomaly Score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _export_predictions(
    path: Path,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> None:
    data = pd.DataFrame(
        {
            "score": scores,
            "predicted_label": np.where(scores > threshold, "attack", "valid"),
            "true_label": labels,
        }
    )
    data.to_csv(path, index=False)


@app.command()
def main(
    tfidf_embeddings: Path = typer.Argument(..., help="TF-IDF/PCA embeddings (.npz)"),
    tfidf_model: Path = typer.Argument(..., help="LOF detector joblib"),
    tfidf_pca: Path = typer.Option(None, help="Optional PCA joblib for TF-IDF embeddings"),
    secbert_embeddings: Path = typer.Argument(..., help="SecBERT embeddings (.npz)"),
    secbert_model: Path = typer.Argument(..., help="Mahalanobis detector joblib"),
    output_dir: Path = typer.Argument(..., help="Output directory"),
    max_fpr: float = typer.Option(0.05, help="Maximum false positive rate"),
    fusion_weights: list[float] = typer.Option(None, help="Fusion weight(s) for LOF scores"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (lof_payload, lof_labels) = _load_lof_scores(tfidf_embeddings, tfidf_model, tfidf_pca)
    (maha_payload, maha_labels) = _load_mahalanobis_scores(secbert_embeddings, secbert_model)

    _ensure_labels_match(lof_labels, maha_labels)
    labels = lof_labels
    binary = np.array([1 if label == "attack" else 0 for label in labels])
    normal_mask = labels == "valid"

    lof_threshold = _calc_threshold(lof_payload.scores, normal_mask, max_fpr)
    maha_threshold = _calc_threshold(maha_payload.scores, normal_mask, max_fpr)

    lof_metrics = _metrics_from_predictions(lof_payload.scores, labels, lof_threshold)
    maha_metrics = _metrics_from_predictions(maha_payload.scores, labels, maha_threshold)

    fusion_results: list[dict[str, float | int | dict[str, float | list[float]]]] = []

    lof_std = _standardize(lof_payload.scores, normal_mask)
    maha_std = _standardize(maha_payload.scores, normal_mask)

    best_auc = -np.inf
    best_result: dict[str, float | int | dict[str, float | list[float]]] | None = None

    weights = fusion_weights or [0.5]

    for weight in weights:
        weight = float(weight)
        fused_scores = weight * lof_std + (1.0 - weight) * maha_std
        fused_threshold = _calc_threshold(fused_scores, normal_mask, max_fpr)
        fused_metrics = _metrics_from_predictions(fused_scores, labels, fused_threshold)
        fused_metrics["weight"] = weight
        fused_metrics["threshold"] = fused_threshold
        fusion_results.append(fused_metrics)

        if fused_metrics["roc_auc"] > best_auc:
            best_auc = fused_metrics["roc_auc"]
            best_result = fused_metrics
            best_scores = fused_scores
            best_threshold = fused_threshold

    assert best_result is not None

    summary = {
        "max_fpr": max_fpr,
        "lof": lof_metrics,
        "mahalanobis": maha_metrics,
        "fusion": fusion_results,
        "best_fusion": best_result,
    }

    (output_dir / "ensemble_summary.json").write_text(json.dumps(summary, indent=2))

    # Export best fusion artifacts
    roc_df = pd.DataFrame(
        {
            "fpr": best_result["roc_curve"]["fpr"],  # type: ignore[index]
            "tpr": best_result["roc_curve"]["tpr"],  # type: ignore[index]
            "threshold": best_result["roc_curve"]["thresholds"],  # type: ignore[index]
        }
    )
    roc_df.to_csv(output_dir / "roc_curve.csv", index=False)

    _export_predictions(output_dir / "prediction_scores.csv", best_scores, labels, best_threshold)
    _build_plot(best_scores, labels, best_threshold, output_dir / "test_score_distribution.png")

    best_metrics_json = {
        key: value
        for key, value in best_result.items()
        if key not in {"roc_curve"}
    }
    (output_dir / "roc_metrics.json").write_text(json.dumps(best_metrics_json, indent=2))

    logger.info("Best fusion weight = {}", best_result["weight"])
    logger.info("Best fusion ROC AUC = {:.4f}", best_result["roc_auc"])


if __name__ == "__main__":
    app()
