from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import typer
from loguru import logger

import neuralshield.encoding.data.factory as data_factory
from neuralshield.encoding.models.fastembed import FastEmbedEncoder
from neuralshield.encoding.models.tfidf import TFIDFEncoder
from neuralshield.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
)
from neuralshield.preprocessing.pipeline import PreprocessorPipeline, preprocess

app = typer.Typer()


def _resolve_pipeline(
    use_pipeline: bool, pipeline_name: str | None
) -> PreprocessorPipeline | None:
    """Resolve the pipeline to use."""
    if not use_pipeline:
        return None

    selected = pipeline_name if pipeline_name else "preprocess"
    if selected == "preprocess":
        return preprocess

    raise ValueError(
        f"Unsupported pipeline '{selected}'. Only 'preprocess' is available."
    )


@app.command()  # noqa: T201
def main(
    test_dataset: Path = typer.Argument(..., help="Test dataset (JSONL)"),
    model_path: Path = typer.Argument(..., help="Trained model file (.joblib)"),
    encoder_type: str = typer.Option(
        "fastembed", "--encoder", help="Encoder type (fastembed/tfidf)"
    ),
    model_name: str = typer.Option(
        "BAAI/bge-small-en-v1.5",
        "--model-name",
        help="Model name for fastembed",
    ),
    vectorizer_path: Path | None = typer.Option(
        None, "--vectorizer", help="Vectorizer path for TF-IDF"
    ),
    batch_size: int = typer.Option(512, help="Batch size"),
    use_pipeline: bool = typer.Option(
        False, "--use-pipeline", help="Enable preprocessing pipeline"
    ),
    pipeline_name: str = typer.Option("", help="Pipeline name (blank=preprocess)"),
    wandb_project: str = typer.Option("", help="W&B project name"),
    wandb_run_name: str = typer.Option("", help="W&B run name"),
    device: str = typer.Option("cpu", help="Device (cpu/cuda)"),
) -> None:
    """Test an anomaly detection model on a dataset."""
    # Setup W&B if requested
    wandb_module: Any = None
    if wandb_project:
        import wandb as wandb_module

        wandb_module.init(
            project=wandb_project,
            name=wandb_run_name or None,
            config={
                "encoder_type": encoder_type,
                "model_name": model_name if encoder_type == "fastembed" else None,
                "use_pipeline": use_pipeline,
                "batch_size": batch_size,
            },
        )

    logger.info("Loading model from: {path}", path=str(model_path))
    model_data = joblib.load(model_path)
    detector = model_data["detector"]
    metadata = model_data.get("metadata", {})

    logger.info("Model metadata: {meta}", meta=metadata)

    # Setup encoder
    if encoder_type == "fastembed":
        logger.info("Using FastEmbed encoder: {model}", model=model_name)
        encoder = FastEmbedEncoder(model_name=model_name, device=device)
    elif encoder_type == "tfidf":
        if not vectorizer_path:
            raise ValueError("TF-IDF encoder requires --vectorizer path")
        logger.info("Using TF-IDF encoder: {path}", path=str(vectorizer_path))
        encoder = TFIDFEncoder.load(vectorizer_path)
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")

    # Setup pipeline
    resolved_pipeline = _resolve_pipeline(use_pipeline, pipeline_name)
    pipeline_status = "enabled" if use_pipeline else "disabled"
    logger.info("Preprocessing pipeline: {status}", status=pipeline_status)

    # Setup reader
    reader_factory = data_factory.get_reader("jsonl")
    data_reader = reader_factory(
        path=test_dataset,
        pipeline=resolved_pipeline,
        use_pipeline=use_pipeline,
        observer=None,
    )

    # Process and predict
    logger.info("Processing test dataset: {path}", path=str(test_dataset))
    all_predictions: list[bool] = []
    all_labels: list[str] = []
    all_scores: list[float] = []
    all_requests: list[str] = []

    from tqdm import tqdm

    for batch_requests, batch_labels in tqdm(
        data_reader.iter_batches(batch_size),
        desc="Testing",
        unit="batch",
    ):
        # Encode
        embeddings = encoder.encode(batch_requests)

        # Predict
        predictions = detector.predict(embeddings)
        scores = detector.scores(embeddings)

        # Store
        all_predictions.extend(predictions)
        all_labels.extend(batch_labels)
        all_scores.extend(scores.tolist())
        all_requests.extend(batch_requests)

    logger.info("Processed {count} samples", count=len(all_predictions))

    # Calculate metrics
    logger.info("Calculating metrics...")

    binary_labels = [1 if label == "anomalous" else 0 for label in all_labels]
    binary_predictions = [1 if pred else 0 for pred in all_predictions]

    confusion = calculate_confusion_matrix(
        y_true=binary_labels,
        y_pred=binary_predictions,
        positive_label=1,
    )

    metrics = calculate_classification_metrics(confusion)

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

    # Log to W&B
    if wandb_module:
        wandb_module.log(
            {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "accuracy": metrics.accuracy,
                "specificity": metrics.specificity,
                "fpr": metrics.false_positive_rate,
                "tp": confusion.true_positives,
                "fp": confusion.false_positives,
                "tn": confusion.true_negatives,
                "fn": confusion.false_negatives,
            }
        )

        # Log score distribution
        import matplotlib.pyplot as plt
        import seaborn as sns

        scores_array = np.array(all_scores)
        labels_array = np.array(all_labels)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            data={
                "score": scores_array[labels_array == "normal"],
            },
            x="score",
            bins=50,
            alpha=0.5,
            label="Normal",
            ax=ax,
        )
        sns.histplot(
            data={
                "score": scores_array[labels_array == "anomalous"],
            },
            x="score",
            bins=50,
            alpha=0.5,
            label="Anomalous",
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
                    [confusion.true_negatives, confusion.false_positives],
                    [confusion.false_negatives, confusion.true_positives],
                ]
            ),
            display_labels=["Normal", "Anomalous"],
        )
        fig, ax = plt.subplots(figsize=(8, 8))
        cm_display.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title("Confusion Matrix")

        wandb_module.log({"confusion_matrix": wandb_module.Image(fig)})
        plt.close(fig)

        # Log sample requests
        anomalous_indices = [
            i
            for i, pred in enumerate(all_predictions)
            if pred and all_labels[i] == "anomalous"
        ][:10]
        fp_indices = [
            i
            for i, pred in enumerate(all_predictions)
            if pred and all_labels[i] == "normal"
        ][:10]

        if anomalous_indices:
            wandb_module.log(
                {
                    "sample_true_positives": wandb_module.Table(
                        columns=["request", "score"],
                        data=[
                            [all_requests[i], all_scores[i]] for i in anomalous_indices
                        ],
                    )
                }
            )

        if fp_indices:
            wandb_module.log(
                {
                    "sample_false_positives": wandb_module.Table(
                        columns=["request", "score"],
                        data=[[all_requests[i], all_scores[i]] for i in fp_indices],
                    )
                }
            )

        wandb_module.finish()

    logger.info("✓ Testing complete!")


if __name__ == "__main__":
    app()
