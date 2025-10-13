#!/usr/bin/env python3
"""Test anomaly detection models using precomputed embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import typer
from loguru import logger

from neuralshield.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
)

app = typer.Typer()


@app.command()
def main(
    embeddings_path: Path = typer.Argument(..., help="Test embeddings file (.npz)"),
    model_path: Path = typer.Argument(..., help="Trained model file (.joblib)"),
    wandb_project: str = typer.Option("", help="W&B project name"),
    wandb_run_name: str = typer.Option("", help="W&B run name"),
) -> None:
    """Test an anomaly detection model using precomputed embeddings."""
    # Setup W&B if requested
    wandb_module: Any = None
    if wandb_project:
        import wandb as wandb_module

        wandb_module.init(
            project=wandb_project,
            name=wandb_run_name or None,
        )

    # Load embeddings
    logger.info("Loading embeddings from: {path}", path=str(embeddings_path))
    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]

    logger.info(
        "Loaded: {samples} samples, {dim} dimensions",
        samples=embeddings.shape[0],
        dim=embeddings.shape[1],
    )

    # Load model
    logger.info("Loading model from: {path}", path=str(model_path))
    model_data = joblib.load(model_path)
    detector = model_data["detector"]
    metadata = model_data.get("metadata", {})

    logger.info("Model metadata: {meta}", meta=metadata)

    # Predict
    logger.info("Running predictions...")
    predictions = detector.predict(embeddings)
    scores = detector.scores(embeddings)

    logger.info("Predictions complete: {count} samples", count=len(predictions))

    # Calculate confusion matrix
    tp, fp, tn, fn = calculate_confusion_matrix(
        predictions=predictions.tolist(),
        labels=labels.tolist(),
        positive_label="attack",
        negative_label="valid",
    )

    # Calculate metrics
    metrics = calculate_classification_metrics(
        predictions=predictions.tolist(),
        labels=labels.tolist(),
        positive_label="attack",
    )

    # Print results
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives:  {tn}")
    print(f"False Negatives: {fn}")

    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1-Score:     {metrics['f1_score']:.4f}")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Specificity:  {metrics['specificity']:.4f}")
    print(f"FPR:          {metrics['fpr']:.4f}")
    print("=" * 60 + "\n")

    # Log to W&B
    if wandb_module:
        wandb_module.log(
            {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "accuracy": metrics["accuracy"],
                "specificity": metrics["specificity"],
                "fpr": metrics["fpr"],
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )

        # Log score distribution
        import matplotlib.pyplot as plt
        import seaborn as sns

        scores_array = np.array(scores)
        labels_array = np.array(labels)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Normal samples
        normal_scores = scores_array[labels_array == "valid"]
        if len(normal_scores) > 0:
            sns.histplot(
                data={"score": normal_scores},
                x="score",
                bins=50,
                alpha=0.5,
                label="Normal",
                ax=ax,
            )

        # Attack samples
        attack_scores = scores_array[labels_array == "attack"]
        if len(attack_scores) > 0:
            sns.histplot(
                data={"score": attack_scores},
                x="score",
                bins=50,
                alpha=0.5,
                label="Attack",
                ax=ax,
            )

        ax.axvline(
            detector.threshold_,
            color="red",
            linestyle="--",
            label=f"Threshold ({detector.threshold_:.4f})",
        )
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Count")
        ax.set_title("Score Distribution")
        ax.legend()

        wandb_module.log({"score_distribution": wandb_module.Image(fig)})
        plt.close(fig)

        # Log confusion matrix
        from sklearn.metrics import ConfusionMatrixDisplay

        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=np.array(
                [
                    [tn, fp],
                    [fn, tp],
                ]
            ),
            display_labels=["Normal", "Attack"],
        )
        fig, ax = plt.subplots(figsize=(8, 8))
        cm_display.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title("Confusion Matrix")

        wandb_module.log({"confusion_matrix": wandb_module.Image(fig)})
        plt.close(fig)

        wandb_module.finish()

    logger.info("✓ Testing complete!")


if __name__ == "__main__":
    app()
