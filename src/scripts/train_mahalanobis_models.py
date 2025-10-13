#!/usr/bin/env python3
"""Train and save Mahalanobis models for model comparison."""

from pathlib import Path

import numpy as np
import typer
from loguru import logger

from neuralshield.anomaly import MahalanobisDetector

app = typer.Typer()


def train_and_save_mahalanobis(
    train_embeddings_path: Path,
    test_embeddings_path: Path,
    output_model_path: Path,
    max_fpr: float = 0.05,
) -> None:
    """Train Mahalanobis detector and save it."""
    logger.info(f"Loading training embeddings from {train_embeddings_path}")
    train_data = np.load(train_embeddings_path, allow_pickle=True)
    train_embeddings = train_data["embeddings"]
    logger.info(f"Loaded {len(train_embeddings)} training samples")

    logger.info(f"Loading test embeddings from {test_embeddings_path}")
    test_data = np.load(test_embeddings_path, allow_pickle=True)
    test_embeddings = test_data["embeddings"]
    test_labels = test_data["labels"]
    logger.info(f"Loaded {len(test_embeddings)} test samples")

    # Get normal test samples for threshold calibration
    is_normal = (test_labels == "valid") | (test_labels == "normal")
    test_normal_embeddings = test_embeddings[is_normal]
    logger.info(f"Found {len(test_normal_embeddings)} normal test samples")

    # Train detector
    logger.info("Training Mahalanobis detector...")
    detector = MahalanobisDetector(name="production")
    detector.fit(train_embeddings)

    # Set threshold
    logger.info(f"Setting threshold for {max_fpr:.1%} FPR...")
    threshold = detector.set_threshold(test_normal_embeddings, max_fpr=max_fpr)
    logger.info(f"Threshold set to {threshold:.4f}")

    # Save model
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(output_model_path)
    logger.info(f"Model saved to {output_model_path}")


@app.command()
def main(
    train_embeddings: Path = typer.Argument(..., help="Training embeddings .npz file"),
    test_embeddings: Path = typer.Argument(..., help="Test embeddings .npz file"),
    output_model: Path = typer.Argument(..., help="Output model .joblib file"),
    max_fpr: float = typer.Option(0.05, help="Maximum false positive rate"),
):
    """Train and save a Mahalanobis detector."""
    train_and_save_mahalanobis(train_embeddings, test_embeddings, output_model, max_fpr)


if __name__ == "__main__":
    app()
