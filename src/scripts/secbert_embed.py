from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import typer
from loguru import logger

import neuralshield.encoding.data.factory as data_factory
from neuralshield.encoding.models.secbert import SecBERTEncoder
from neuralshield.encoding.observability import EncodingMetricsTracker, init_wandb_sink
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


@app.command()
def main(
    dataset: Path = typer.Argument(..., help="Input dataset (JSONL)"),
    output: Path = typer.Argument(..., help="Output embeddings file (.npz)"),
    model_name: str = typer.Option(
        "jackaduma/SecBERT",
        "--model",
        help="SecBERT model name",
    ),
    batch_size: int = typer.Option(32, help="Batch size for processing"),
    reader: str = typer.Option("jsonl", help="Dataset reader type"),
    use_pipeline: bool = typer.Option(
        False, "--use-pipeline", help="Enable preprocessing pipeline"
    ),
    pipeline_name: str = typer.Option("", help="Pipeline name (blank=preprocess)"),
    device: str = typer.Option("cpu", help="Device (cpu/cuda/mps)"),
    wandb_enabled: bool = typer.Option(
        False, "--wandb/--no-wandb", help="Enable Weights & Biases logging"
    ),
    wandb_project: str = typer.Option("neuralshield", help="W&B project name"),
    wandb_entity: str | None = typer.Option(None, help="W&B entity/team"),
) -> None:
    """Generate dense embeddings from a dataset using SecBERT."""
    logger.info(
        "Generating SecBERT embeddings: model={model}, device={device}",
        model=model_name,
        device=device,
    )

    # Create output directory
    output.parent.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if enabled
    config = {
        "model": model_name,
        "batch_size": batch_size,
        "device": device,
        "preprocessing": use_pipeline,
        "pipeline": pipeline_name if pipeline_name else "preprocess",
        "dataset": str(dataset),
        "embedding_dim": 768,
    }

    sink, wandb_run = init_wandb_sink(
        wandb_enabled,
        project=wandb_project,
        entity=wandb_entity,
        config=config,
    )

    if wandb_run is not None:
        wandb_run.name = f"secbert-embed-{dataset.stem}"
        wandb_run.tags = [
            "embedding",
            "secbert",
            "with-prep" if use_pipeline else "no-prep",
        ]
        logger.info("Wandb logging enabled: {url}", url=wandb_run.url)

    # Initialize metrics tracker
    metrics_tracker = EncodingMetricsTracker(sink=sink)

    # Initialize encoder
    logger.info("Loading SecBERT model: {model}", model=model_name)
    encoder = SecBERTEncoder(model_name=model_name, device=device)
    logger.info("Model loaded: dimension=768")

    # Setup pipeline
    resolved_pipeline = _resolve_pipeline(use_pipeline, pipeline_name)
    pipeline_status = "enabled" if use_pipeline else "disabled"
    logger.info("Preprocessing pipeline: {status}", status=pipeline_status)

    # Setup reader
    reader_factory = data_factory.get_reader(reader)
    data_reader = reader_factory(
        path=dataset,
        pipeline=resolved_pipeline,
        use_pipeline=use_pipeline,
        observer=None,
    )

    # Process dataset and collect embeddings
    logger.info("Processing dataset: {path}", path=str(dataset))
    all_embeddings = []
    all_labels = []
    total_processed = 0

    from tqdm import tqdm

    for batch_requests, batch_labels in tqdm(
        data_reader.iter_batches(batch_size),
        desc="Encoding",
        unit="batch",
    ):
        # Time encoding
        start_time = time.time()
        embeddings = encoder.encode(batch_requests)
        encode_seconds = time.time() - start_time

        # Store
        all_embeddings.append(embeddings)
        all_labels.extend(batch_labels)
        total_processed += len(batch_requests)

        # Log metrics to wandb
        metrics_tracker.record_batch(
            requests=batch_requests,
            embeddings=embeddings,
            encode_seconds=encode_seconds,
        )

    logger.info("Processed {count} samples", count=total_processed)

    # Concatenate all embeddings
    logger.info("Concatenating embeddings...")
    final_embeddings = np.vstack(all_embeddings)
    logger.info(
        "Final shape: {shape}, dtype: {dtype}",
        shape=final_embeddings.shape,
        dtype=final_embeddings.dtype,
    )

    # Save to npz
    logger.info("Saving embeddings to: {path}", path=str(output))
    np.savez_compressed(
        output,
        embeddings=final_embeddings,
        labels=np.array(all_labels),
        metadata=np.array(
            [
                {
                    "model": model_name,
                    "dimension": final_embeddings.shape[1],
                    "num_samples": final_embeddings.shape[0],
                    "preprocessing": use_pipeline,
                    "pipeline": pipeline_name if pipeline_name else "preprocess",
                }
            ]
        ),
    )

    logger.info("✓ Embeddings saved successfully!")
    logger.info(
        "Summary: {samples} samples, {dim} dimensions, {size:.2f} MB",
        samples=final_embeddings.shape[0],
        dim=final_embeddings.shape[1],
        size=output.stat().st_size / (1024 * 1024),
    )

    # Finalize metrics
    summary_metrics = metrics_tracker.finalize()

    # Log final summary to wandb
    if sink is not None:
        sink.log(
            {
                "summary/total_samples": final_embeddings.shape[0],
                "summary/embedding_dim": final_embeddings.shape[1],
                "summary/file_size_mb": output.stat().st_size / (1024 * 1024),
            }
        )
        logger.info("Logged final metrics to wandb")

    # Finish wandb run
    if wandb_run is not None:
        wandb_run.finish()
        logger.info("Wandb run finished")


if __name__ == "__main__":
    app()
