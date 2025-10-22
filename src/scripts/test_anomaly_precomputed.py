#!/usr/bin/env python3
"""Test anomaly detection model using precomputed embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from sklearn.metrics import roc_auc_score, roc_curve

from neuralshield.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
)

app = typer.Typer()


class SklearnDetectorAdapter:
    """Adapter exposing uniform interface for plain sklearn anomaly models."""

    def __init__(
        self,
        model: Any,
        *,
        transformer: Any | None = None,
        default_threshold: float | None = None,
    ) -> None:
        self.model = model
        self.transformer = transformer
        class_name = type(model).__name__.lower()
        self._is_isolation_forest = "isolationforest" in class_name
        self._is_gaussian_mixture = "gaussianmixture" in class_name
        offset = getattr(model, "offset_", None)
        inferred_threshold = None
        if offset is not None:
            inferred_threshold = (
                -float(offset) if self._is_isolation_forest else float(offset)
            )
        self.threshold_ = (
            float(default_threshold)
            if default_threshold is not None
            else inferred_threshold
        )

        if hasattr(model, "score_samples"):
            self._score_method = "score_samples"
        elif hasattr(model, "decision_function"):
            self._score_method = "decision_function"
        elif hasattr(model, "mahalanobis"):
            self._score_method = "mahalanobis"
        else:
            raise AttributeError(
                f"Model {type(model).__name__} lacks score_samples/decision_function/mahalanobis"
            )

    def _transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self.transformer is None:
            return embeddings
        if hasattr(self.transformer, "transform"):
            return self.transformer.transform(embeddings)
        raise AttributeError(
            f"Transformer {type(self.transformer).__name__} lacks transform()"
        )

    def scores(self, embeddings: np.ndarray) -> np.ndarray:
        transformed = self._transform(embeddings)
        score_fn = getattr(self.model, self._score_method)
        scores = score_fn(transformed)

        if self._score_method == "mahalanobis":
            return np.asarray(scores, dtype=np.float32)

        if self._is_isolation_forest:
            scores = -scores
        elif self._is_gaussian_mixture:
            scores = -scores

        return np.asarray(scores, dtype=np.float32)

    def predict(
        self, embeddings: np.ndarray, *, threshold: float | None = None
    ) -> np.ndarray:
        transformed = self._transform(embeddings)
        if hasattr(self.model, "predict"):
            preds = self.model.predict(transformed)
            preds_arr = np.asarray(preds)
            if preds_arr.dtype.kind in {"i", "f"}:
                unique_vals = np.unique(preds_arr)
                allowed_values = {-1, 0, 1}
                if set(unique_vals.tolist()).issubset(allowed_values):
                    if -1 in unique_vals:
                        return (preds_arr < 0).astype(bool)
                    return preds_arr.astype(bool)
                # Otherwise fall back to threshold-based decision
            elif preds_arr.dtype.kind == "b":
                return preds_arr.astype(bool)

        use_threshold = threshold if threshold is not None else self.threshold_
        if use_threshold is None:
            raise ValueError("Threshold required for prediction with this model")

        scores = self.scores(embeddings)
        tolerance = max(1e-9, abs(use_threshold) * 1e-6)
        return (scores > (use_threshold + tolerance)).astype(bool)


def resolve_detector(payload: Any) -> Tuple[Any, dict]:
    """Extract detector object and metadata from saved payloads."""
    if hasattr(payload, "scores") and hasattr(payload, "predict"):
        return payload, getattr(payload, "metadata", {})

    if isinstance(payload, dict):
        if "detector" in payload:
            detector = payload["detector"]
            metadata = payload.get("metadata", {}) or {}
            threshold = payload.get("threshold")
            if threshold is None:
                threshold = metadata.get("threshold")
            if threshold is not None:
                try:
                    setattr(detector, "threshold_", float(threshold))
                except Exception:
                    pass
                if hasattr(detector, "_threshold"):
                    try:
                        setattr(detector, "_threshold", float(threshold))
                    except Exception:
                        pass
            if not hasattr(detector, "_model") and hasattr(detector, "model"):
                try:
                    setattr(detector, "_model", getattr(detector, "model"))
                except Exception:
                    pass
            # Some legacy pickles miss the fitted flag; assume true
            if not hasattr(detector, "_fitted"):
                try:
                    setattr(detector, "_fitted", True)
                except Exception:
                    pass
            else:
                try:
                    setattr(detector, "_fitted", True)
                except Exception:
                    pass
            metadata = metadata
            return detector, metadata

        if "model" in payload:
            metadata = {
                key: value
                for key, value in payload.items()
                if key not in {"model", "pca", "threshold"}
            }
            transformer = payload.get("pca")
            threshold = payload.get("threshold")
            adapter = SklearnDetectorAdapter(
                payload["model"],
                transformer=transformer,
                default_threshold=threshold,
            )
            return adapter, metadata

    raise KeyError("Could not resolve detector from payload")


@app.command()
def main(
    test_embeddings_path: Path = typer.Argument(..., help="Test embeddings (.npz)"),
    model_path: Path = typer.Argument(..., help="Trained model (.joblib)"),
    wandb_enabled: bool = typer.Option(
        False, "--wandb/--no-wandb", help="Enable wandb logging"
    ),
    wandb_project: str = typer.Option("neuralshield", help="W&B project name"),
    wandb_run_name: str | None = typer.Option(None, help="W&B run name"),
    metrics_out: Path | None = typer.Option(
        None, "--metrics-out", help="Optional JSON output path for metrics"
    ),
    curve_out: Path | None = typer.Option(
        None, "--curve-out", help="Optional CSV output path for ROC curve points"
    ),
    scores_out: Path | None = typer.Option(
        None,
        "--scores-out",
        help="Optional CSV output path for per-sample scores, labels, and predictions",
    ),
    ) -> None:
    """Test anomaly detection model on precomputed embeddings."""

    # Initialize wandb if enabled
    wandb_module = None
    if wandb_enabled:
        import wandb as wandb_module

        wandb_module.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "test_embeddings": str(test_embeddings_path),
                "model": str(model_path),
            },
        )

    # Load model
    logger.info(f"Loading model from: {model_path}")
    model_data = joblib.load(model_path)
    detector, metadata = resolve_detector(model_data)

    logger.info(f"Model metadata: {metadata}")

    # Load test embeddings
    logger.info(f"Loading test embeddings from: {test_embeddings_path}")
    data = np.load(test_embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]

    logger.info(
        f"Loaded {embeddings.shape[0]} samples, embedding dim={embeddings.shape[1]}"
    )

    # Predict
    logger.info("Running predictions...")
    predictions = detector.predict(embeddings)  # Returns list of bools
    scores = detector.scores(embeddings)

    # Normalize labels (handle both "attack"/"anomalous" and "normal"/"valid")
    normalized_labels = []
    for label in labels:
        if label in ["attack", "anomalous"]:
            normalized_labels.append("attack")
        elif label in ["normal", "valid"]:
            normalized_labels.append("valid")
        else:
            raise ValueError(f"Unknown label: {label}")

    # Calculate metrics
    logger.info("Calculating metrics...")
    tp, fp, tn, fn = calculate_confusion_matrix(
        predictions=predictions.tolist()
        if hasattr(predictions, "tolist")
        else predictions,
        labels=normalized_labels,
        positive_label="attack",
        negative_label="valid",
    )

    # Create confusion matrix object manually
    from dataclasses import dataclass

    @dataclass
    class ConfusionMatrix:
        true_positives: int
        false_positives: int
        true_negatives: int
        false_negatives: int

    confusion = ConfusionMatrix(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )

    # Calculate classification metrics manually
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    @dataclass
    class Metrics:
        precision: float
        recall: float
        f1_score: float
        accuracy: float
        specificity: float
        false_positive_rate: float

    metrics = Metrics(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        accuracy=accuracy,
        specificity=specificity,
        false_positive_rate=fpr,
    )

    # Prepare labels for ROC analysis (1 = attack, 0 = valid)
    label_array = np.array([1 if label == "attack" else 0 for label in normalized_labels])
    roc_auc = roc_auc_score(label_array, scores)
    fpr_curve, tpr_curve, threshold_curve = roc_curve(label_array, scores)

    # Interpolate TPR and threshold at 5% FPR
    target_fpr = 0.05
    tpr_at_target = float(np.interp(target_fpr, fpr_curve, tpr_curve))
    threshold_at_target = float(np.interp(target_fpr, fpr_curve, threshold_curve))

    # Print results
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print(f"True Positives:  {confusion.true_positives}")
    print(f"False Positives: {confusion.false_positives}")
    print(f"True Negatives:  {confusion.true_negatives}")
    print(f"False Negatives: {confusion.false_negatives}")

    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    print(f"Precision:    {metrics.precision:.4f}")
    print(f"Recall:       {metrics.recall:.4f}")
    print(f"F1-Score:     {metrics.f1_score:.4f}")
    print(f"Accuracy:     {metrics.accuracy:.4f}")
    print(f"Specificity:  {metrics.specificity:.4f}")
    print(f"FPR:          {metrics.false_positive_rate:.4f}")
    print(f"ROC AUC:      {roc_auc:.4f}")
    print(f"TPR@5% FPR:   {tpr_at_target:.4f}")
    print("=" * 60 + "\n")

    if wandb_module is not None:
        wandb_module.log(
            {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "accuracy": metrics.accuracy,
                "specificity": metrics.specificity,
                "false_positive_rate": metrics.false_positive_rate,
                "roc_auc": roc_auc,
                "tpr_at_5_fpr": tpr_at_target,
            }
        )

    # Persist metrics if requested
    if metrics_out is not None:
        metrics_payload = {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "accuracy": metrics.accuracy,
            "specificity": metrics.specificity,
            "false_positive_rate": metrics.false_positive_rate,
            "roc_auc": roc_auc,
            "tpr_at_5_fpr": tpr_at_target,
            "threshold_at_5_fpr": threshold_at_target,
            "confusion_matrix": {
                "true_positives": confusion.true_positives,
                "false_positives": confusion.false_positives,
                "true_negatives": confusion.true_negatives,
                "false_negatives": confusion.false_negatives,
            },
        }
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(metrics_payload, indent=2))
        logger.info("Saved metrics JSON to {}", metrics_out)

    if curve_out is not None:
        curve_frame = pd.DataFrame(
            {
                "fpr": fpr_curve,
                "tpr": tpr_curve,
                "threshold": threshold_curve,
            }
        )
        curve_out.parent.mkdir(parents=True, exist_ok=True)
        curve_frame.to_csv(curve_out, index=False)
        logger.info("Saved ROC curve CSV to {}", curve_out)

    if scores_out is not None:
        scores_frame = pd.DataFrame(
            {
                "score": scores,
                "predicted_label": ["attack" if p else "valid" for p in predictions],
                "true_label": normalized_labels,
            }
        )
        scores_out.parent.mkdir(parents=True, exist_ok=True)
        scores_frame.to_csv(scores_out, index=False)
        logger.info("Saved per-sample scores CSV to {}", scores_out)

    # Create comprehensive visualization
    logger.info("Creating score distribution visualization")

    # Separate normal and attack scores
    normal_mask = np.array([label in ["normal", "valid"] for label in labels])
    attack_mask = ~normal_mask

    normal_scores = scores[normal_mask]
    attack_scores = scores[attack_mask]

    logger.info(
        f"Normal samples: {len(normal_scores)}, Attack samples: {len(attack_scores)}"
    )

    # Create 2x2 visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Overlapping histograms with KDE
    sns.histplot(
        normal_scores,
        bins=50,
        kde=True,
        ax=axes[0, 0],
        color="green",
        alpha=0.6,
        label=f"Normal (n={len(normal_scores)})",
    )
    sns.histplot(
        attack_scores,
        bins=50,
        kde=True,
        ax=axes[0, 0],
        color="red",
        alpha=0.6,
        label=f"Attack (n={len(attack_scores)})",
    )
    axes[0, 0].axvline(
        detector.threshold_,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({detector.threshold_:.4f})",
    )
    axes[0, 0].set_xlabel("Anomaly Score", fontsize=11)
    axes[0, 0].set_ylabel("Frequency", fontsize=11)
    axes[0, 0].set_title(
        "Score Distribution: Normal vs Attack", fontsize=13, fontweight="bold"
    )
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Separate histograms
    axes[0, 1].hist(
        [normal_scores, attack_scores],
        bins=50,
        label=[f"Normal (n={len(normal_scores)})", f"Attack (n={len(attack_scores)})"],
        color=["green", "red"],
        alpha=0.7,
        stacked=False,
    )
    axes[0, 1].axvline(
        detector.threshold_,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({detector.threshold_:.4f})",
    )
    axes[0, 1].set_xlabel("Anomaly Score", fontsize=11)
    axes[0, 1].set_ylabel("Count", fontsize=11)
    axes[0, 1].set_title(
        "Score Distribution (Non-Stacked)", fontsize=13, fontweight="bold"
    )
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Violin plots
    df_violin = pd.DataFrame(
        {
            "Score": np.concatenate([normal_scores, attack_scores]),
            "Type": ["Normal"] * len(normal_scores) + ["Attack"] * len(attack_scores),
        }
    )
    sns.violinplot(
        data=df_violin,
        x="Type",
        y="Score",
        ax=axes[1, 0],
        palette={"Normal": "green", "Attack": "red"},
        alpha=0.7,
    )
    axes[1, 0].axhline(
        detector.threshold_,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({detector.threshold_:.4f})",
    )
    axes[1, 0].set_ylabel("Anomaly Score", fontsize=11)
    axes[1, 0].set_title(
        "Score Distribution (Violin Plot)", fontsize=13, fontweight="bold"
    )
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # 4. Box plots
    sns.boxplot(
        data=df_violin,
        x="Type",
        y="Score",
        ax=axes[1, 1],
        palette={"Normal": "green", "Attack": "red"},
        showfliers=True,
    )
    axes[1, 1].axhline(
        detector.threshold_,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({detector.threshold_:.4f})",
    )
    axes[1, 1].set_ylabel("Anomaly Score", fontsize=11)
    axes[1, 1].set_title(
        "Score Distribution (Box Plot)", fontsize=13, fontweight="bold"
    )
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save figure
    output_path = model_path.parent / "test_score_distribution.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved visualization to: {output_path}")

    # Log to wandb if enabled
    if wandb_module:
        wandb_module.log(
            {
                "test/precision": metrics.precision,
                "test/recall": metrics.recall,
                "test/f1_score": metrics.f1_score,
                "test/accuracy": metrics.accuracy,
                "test/specificity": metrics.specificity,
                "test/fpr": metrics.false_positive_rate,
                "test/tp": confusion.true_positives,
                "test/fp": confusion.false_positives,
                "test/tn": confusion.true_negatives,
                "test/fn": confusion.false_negatives,
                "test/score_distribution": wandb_module.Image(fig),
            }
        )

    plt.close(fig)

    # Create confusion matrix visualization
    from sklearn.metrics import ConfusionMatrixDisplay

    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=np.array(
            [
                [confusion.true_negatives, confusion.false_positives],
                [confusion.false_negatives, confusion.true_positives],
            ]
        ),
        display_labels=["Normal", "Attack"],
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    cm_display.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    cm_path = model_path.parent / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved confusion matrix to: {cm_path}")

    if wandb_module:
        wandb_module.log({"test/confusion_matrix": wandb_module.Image(fig)})

    plt.close(fig)

    if wandb_module:
        wandb_module.finish()

    logger.info("✓ Testing complete!")


if __name__ == "__main__":
    app()
