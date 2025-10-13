#!/usr/bin/env python3
"""Test anomaly detection model using precomputed embeddings."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger

from neuralshield.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
)

app = typer.Typer()


@app.command()
def main(
    test_embeddings_path: Path = typer.Argument(..., help="Test embeddings (.npz)"),
    model_path: Path = typer.Argument(..., help="Trained model (.joblib)"),
    wandb_enabled: bool = typer.Option(
        False, "--wandb/--no-wandb", help="Enable wandb logging"
    ),
    wandb_project: str = typer.Option("neuralshield", help="W&B project name"),
    wandb_run_name: str | None = typer.Option(None, help="W&B run name"),
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
    detector = model_data["detector"]
    metadata = model_data.get("metadata", {})

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
    print("=" * 60 + "\n")

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
