#!/usr/bin/env python3
"""Evaluate LOF + SecBERT Mahalanobis ensemble on CSIC embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
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
    higher_scores_are_anomalous: bool = True


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
    *,
    score_mode: str = "stored",
    train_embeddings_path: Path | None = None,
    max_fpr: float = 0.05,
    score_scale: float = 1.0,
) -> DetectorPayload:
    embeddings, labels = _load_npz(embeddings_path)
    logger.info(
        "Loaded TF-IDF embeddings from {} with shape {}",
        embeddings_path,
        embeddings.shape,
    )
    payload = joblib.load(model_path)
    detector = payload["detector"]
    stored_threshold = float(payload.get("threshold") or payload.get("_threshold", 0.0))
    # ensure internal threshold populated
    if hasattr(detector, "_threshold") and detector._threshold is None:
        detector._threshold = stored_threshold
    if pca_path:
        pca = joblib.load(pca_path)
        embeddings = pca.transform(embeddings)
    use_score_samples = payload.get("score_higher_is_normal", False)
    score_mode = score_mode.lower()
    scores: np.ndarray
    threshold: float

    def _load_train_embeddings() -> np.ndarray:
        if train_embeddings_path is None:
            raise ValueError(
                "train_embeddings_path is required when recalibrating LOF scores."
            )
        data = np.load(train_embeddings_path)
        return data["embeddings"].astype(np.float32)

    logger.info("LOF score mode selected: %s", score_mode)
    if score_mode == "stored":
        if use_score_samples and hasattr(detector, "_model") and detector._model is not None:
            raw_scores = detector._model.score_samples(embeddings.astype(np.float32))
            scores = (-raw_scores).astype(np.float32)
            threshold = float(-stored_threshold)
        else:
            scores = detector.scores(embeddings.astype(np.float32))
            threshold = stored_threshold
    elif score_mode == "score_samples":
        if detector._model is None:
            raise RuntimeError("LOF detector missing internal model.")
        train_scores = detector._model.score_samples(_load_train_embeddings())
        threshold_raw = float(np.percentile(train_scores, 100 * max_fpr))
        raw_scores = detector._model.score_samples(embeddings.astype(np.float32))
        scores = (threshold_raw - raw_scores).astype(np.float32)
        threshold = 0.0
    elif score_mode == "neg_decision":
        train_scores = detector.scores(_load_train_embeddings())
        threshold = float(np.percentile(train_scores, 100 * (1 - max_fpr)))
        scores = detector.scores(embeddings.astype(np.float32))
    else:
        raise ValueError(
            "Invalid --lof-score-mode option. Expected 'stored', 'score_samples', or 'neg_decision'."
        )

    if score_scale != 1.0:
        scores = scores * float(score_scale)
        threshold *= float(score_scale)

    logger.info(
        "Scoring {} samples with LOF detector (k={}, threshold={}): {}",
        embeddings.shape[0],
        getattr(detector, "n_neighbors", "n/a"),
        threshold,
        model_path,
    )
    start = perf_counter()
    logger.info(
        "LOF scoring complete in {:.1f}s (min={:.3e}, max={:.3e})",
        perf_counter() - start,
        float(scores.min()),
        float(scores.max()),
    )
    return (
        DetectorPayload(
            scores=scores,
            threshold=threshold,
            name="lof",
            higher_scores_are_anomalous=True,
        ),
        labels,
    )


def _load_mahalanobis_scores(
    embeddings_path: Path,
    model_path: Path,
) -> DetectorPayload:
    embeddings, labels = _load_npz(embeddings_path)
    logger.info(
        "Loaded SecBERT embeddings from {} with shape {}",
        embeddings_path,
        embeddings.shape,
    )
    payload = joblib.load(model_path)
    model = payload["model"]
    threshold = float(payload.get("threshold"))
    logger.info(
        "Scoring {} samples with Mahalanobis detector (threshold={}): {}",
        embeddings.shape[0],
        threshold,
        model_path,
    )
    start = perf_counter()
    scores = model.mahalanobis(embeddings)
    logger.info(
        "Mahalanobis scoring complete in {:.1f}s (min={:.3e}, max={:.3e})",
        perf_counter() - start,
        float(scores.min()),
        float(scores.max()),
    )
    return DetectorPayload(
        scores=scores.astype(np.float32), threshold=threshold, name="mahalanobis"
    ), labels


def _calc_threshold(
    scores: np.ndarray,
    normal_mask: np.ndarray,
    max_fpr: float,
    *,
    higher_scores_are_anomalous: bool,
) -> float:
    normal_scores = scores[normal_mask]
    if higher_scores_are_anomalous:
        return float(np.quantile(normal_scores, 1 - max_fpr))
    return float(np.quantile(normal_scores, max_fpr))


def _metrics_from_predictions(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    *,
    higher_scores_are_anomalous: bool,
    label_attack: str = "attack",
    label_normal: str = "valid",
) -> dict[str, float | int]:
    if higher_scores_are_anomalous:
        preds = scores > threshold
    else:
        preds = scores < threshold
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
    # For ROC, higher scores must indicate anomaly; transform if needed
    roc_scores = scores if higher_scores_are_anomalous else -scores
    fpr_curve, tpr_curve, thr_curve = roc_curve(binary, roc_scores)

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

    sns.histplot(
        normal_scores,
        bins=60,
        kde=True,
        ax=axes[0, 0],
        color="green",
        label=f"Normal (n={normal_scores.size})",
        alpha=0.7,
    )
    sns.histplot(
        attack_scores,
        bins=60,
        kde=True,
        ax=axes[0, 0],
        color="red",
        label=f"Attack (n={attack_scores.size})",
        alpha=0.5,
    )
    axes[0, 0].axvline(threshold, color="black", linestyle="--", linewidth=1.5)
    axes[0, 0].set_title("Score Distribution: Normal vs Attack")
    axes[0, 0].legend()

    axes[0, 1].hist(
        normal_scores,
        bins=60,
        color="green",
        histtype="stepfilled",
        alpha=0.5,
        label="Normal",
    )
    axes[0, 1].hist(
        attack_scores,
        bins=60,
        color="red",
        histtype="stepfilled",
        alpha=0.5,
        label="Attack",
    )
    axes[0, 1].axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold:.3f})",
    )
    axes[0, 1].set_title("Score Distribution (Non-Stacked)")
    axes[0, 1].legend()

    violin_df = pd.DataFrame(
        {
            "Score": np.concatenate([normal_scores, attack_scores]),
            "Type": ["Normal"] * normal_scores.size + ["Attack"] * attack_scores.size,
        }
    )
    sns.violinplot(
        data=violin_df,
        x="Type",
        y="Score",
        ax=axes[1, 0],
        palette={"Normal": "green", "Attack": "red"},
    )
    axes[1, 0].axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold:.3f})",
    )
    axes[1, 0].set_title("Score Distribution (Violin Plot)")
    axes[1, 0].legend()

    sns.boxplot(
        data=violin_df,
        x="Type",
        y="Score",
        ax=axes[1, 1],
        palette={"Normal": "green", "Attack": "red"},
    )
    axes[1, 1].axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold:.3f})",
    )
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
    *,
    higher_scores_are_anomalous: bool,
) -> None:
    data = pd.DataFrame(
        {
            "score": scores,
            "predicted_label": np.where(
                scores > threshold if higher_scores_are_anomalous else scores < threshold,
                "attack",
                "valid",
            ),
            "true_label": labels,
        }
    )
    data.to_csv(path, index=False)


@app.command()
def main(
    tfidf_embeddings: Path = typer.Argument(..., help="TF-IDF/PCA embeddings (.npz)"),
    tfidf_model: Path = typer.Argument(..., help="LOF detector joblib"),
    tfidf_pca: Path = typer.Option(
        None, help="Optional PCA joblib for TF-IDF embeddings"
    ),
    secbert_embeddings: Path = typer.Argument(..., help="SecBERT embeddings (.npz)"),
    secbert_model: Path = typer.Argument(..., help="Mahalanobis detector joblib"),
    output_dir: Path = typer.Argument(..., help="Output directory"),
    max_fpr: float = typer.Option(0.05, help="Maximum false positive rate"),
    fusion_weights: list[float] = typer.Option(
        None, help="Fusion weight(s) for LOF scores"
    ),
    lof_score_mode: str = typer.Option(
        "stored",
        help="LOF scoring mode: stored (default), score_samples, or neg_decision.",
    ),
    lof_train_embeddings: Path | None = typer.Option(
        None,
        help="Training embeddings for LOF recalibration when score mode != stored.",
    ),
    lof_scale: float = typer.Option(
        1.0,
        help="Scaling factor applied to LOF anomaly scores before fusion.",
    ),
    standardize: bool = typer.Option(
        True,
        "--standardize/--no-standardize",
        help="Z-score standardize detector scores before fusion.",
    ),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (lof_payload, lof_labels) = _load_lof_scores(
        tfidf_embeddings,
        tfidf_model,
        tfidf_pca,
        score_mode=lof_score_mode,
        train_embeddings_path=lof_train_embeddings,
        max_fpr=max_fpr,
        score_scale=lof_scale,
    )
    (maha_payload, maha_labels) = _load_mahalanobis_scores(
        secbert_embeddings, secbert_model
    )

    _ensure_labels_match(lof_labels, maha_labels)
    labels = lof_labels
    binary = np.array([1 if label == "attack" else 0 for label in labels])
    normal_mask = labels == "valid"

    lof_threshold = lof_payload.threshold
    maha_threshold = maha_payload.threshold

    lof_metrics = _metrics_from_predictions(
        lof_payload.scores,
        labels,
        lof_threshold,
        higher_scores_are_anomalous=lof_payload.higher_scores_are_anomalous,
    )
    maha_metrics = _metrics_from_predictions(
        maha_payload.scores,
        labels,
        maha_threshold,
        higher_scores_are_anomalous=maha_payload.higher_scores_are_anomalous,
    )

    fusion_results: list[dict[str, float | int | dict[str, float | list[float]]]] = []

    if standardize:
        lof_fusion_scores = _standardize(lof_payload.scores, normal_mask)
        maha_fusion_scores = _standardize(maha_payload.scores, normal_mask)
    else:
        lof_fusion_scores = lof_payload.scores
        maha_fusion_scores = maha_payload.scores

    best_auc = -np.inf
    best_result: dict[str, float | int | dict[str, float | list[float]]] | None = None

    weights = fusion_weights or [0.5]

    for weight in weights:
        weight = float(weight)
        fused_scores = weight * lof_fusion_scores + (1.0 - weight) * maha_fusion_scores
        fused_threshold = _calc_threshold(
            fused_scores,
            normal_mask,
            max_fpr,
            higher_scores_are_anomalous=True,
        )
        fused_metrics = _metrics_from_predictions(
            fused_scores,
            labels,
            fused_threshold,
            higher_scores_are_anomalous=True,
        )
        fused_metrics["weight"] = weight
        fused_metrics["threshold"] = fused_threshold
        fusion_results.append(fused_metrics)

        if fused_metrics["roc_auc"] > best_auc:
            best_auc = fused_metrics["roc_auc"]
            best_result = fused_metrics
            best_scores = fused_scores
            best_threshold = fused_threshold

    assert best_result is not None

    lof_preds = lof_payload.scores > lof_threshold
    maha_preds = maha_payload.scores > maha_threshold
    or_preds = lof_preds | maha_preds
    or_tp = int(((or_preds == 1) & (binary == 1)).sum())
    or_fp = int(((or_preds == 1) & (binary == 0)).sum())
    or_tn = int(((or_preds == 0) & (binary == 0)).sum())
    or_fn = int(((or_preds == 0) & (binary == 1)).sum())
    or_precision = float(or_tp / (or_tp + or_fp)) if (or_tp + or_fp) else 0.0
    or_recall = float(or_tp / (or_tp + or_fn)) if (or_tp + or_fn) else 0.0
    or_accuracy = float((or_tp + or_tn) / len(binary))
    or_fpr = float(or_fp / (or_fp + or_tn)) if (or_fp + or_tn) else 0.0
    or_f1 = (
        float(2 * or_precision * or_recall / (or_precision + or_recall))
        if (or_precision + or_recall)
        else 0.0
    )
    or_metrics = {
        "precision": or_precision,
        "recall": or_recall,
        "f1_score": or_f1,
        "accuracy": or_accuracy,
        "false_positive_rate": or_fpr,
        "confusion_matrix": {
            "true_positives": or_tp,
            "false_positives": or_fp,
            "true_negatives": or_tn,
            "false_negatives": or_fn,
        },
        "threshold": None,
        "roc_auc": float("nan"),
    }

    summary = {
        "max_fpr": max_fpr,
        "lof": lof_metrics,
        "mahalanobis": maha_metrics,
        "fusion": fusion_results,
        "best_fusion": best_result,
        "logical_or": or_metrics,
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

    _export_predictions(
        output_dir / "prediction_scores.csv",
        best_scores,
        labels,
        best_threshold,
        higher_scores_are_anomalous=True,
    )
    _build_plot(
        best_scores, labels, best_threshold, output_dir / "test_score_distribution.png"
    )

    best_metrics_json = {
        key: value for key, value in best_result.items() if key not in {"roc_curve"}
    }
    (output_dir / "roc_metrics.json").write_text(
        json.dumps(best_metrics_json, indent=2)
    )

    logger.info("Best fusion weight = {}", best_result["weight"])
    logger.info("Best fusion ROC AUC = {:.4f}", best_result["roc_auc"])


if __name__ == "__main__":
    app()
