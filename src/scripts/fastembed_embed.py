from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from loguru import logger

import neuralshield.encoding.data.factory as data_factory
from neuralshield.encoding.models.fastembed import FastEmbedEncoder
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
        "BAAI/bge-small-en-v1.5",
        "--model",
        help="FastEmbed model name",
    ),
    batch_size: int = typer.Option(512, help="Batch size for processing"),
    reader: str = typer.Option("jsonl", help="Dataset reader type"),
    use_pipeline: bool = typer.Option(
        False, "--use-pipeline", help="Enable preprocessing pipeline"
    ),
    pipeline_name: str = typer.Option("", help="Pipeline name (blank=preprocess)"),
    device: str = typer.Option("cpu", help="Device (cpu/cuda)"),
) -> None:
    """Generate dense embeddings from a dataset using FastEmbed."""
    logger.info(
        "Generating FastEmbed embeddings: model={model}, device={device}",
        model=model_name,
        device=device,
    )

    # Create output directory
    output.parent.mkdir(parents=True, exist_ok=True)

    # Initialize encoder
    logger.info("Loading FastEmbed model: {model}", model=model_name)
    encoder = FastEmbedEncoder(model_name=model_name, device=device)
    logger.info(
        "Model loaded: dimension={dim}",
        dim=encoder.embedding_dim if hasattr(encoder, "embedding_dim") else "unknown",
    )

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
        # Encode batch
        embeddings = encoder.encode(batch_requests)

        # Store
        all_embeddings.append(embeddings)
        all_labels.extend(batch_labels)
        total_processed += len(batch_requests)

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


if __name__ == "__main__":
    app()
