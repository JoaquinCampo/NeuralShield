from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Callable, Generator

import typer
from loguru import logger

import neuralshield.encoding.data.factory as data_factory
import neuralshield.encoding.models.factory as models_factory
import wandb
from neuralshield.encoding.config import EmbeddingRunConfig
from neuralshield.encoding.models import EmbeddingBatch
from neuralshield.encoding.observability import (
    EncodingMetricsTracker,
    PipelineObserver,
    WandbSink,
)
from neuralshield.preprocessing.pipeline import PreprocessorPipeline, preprocess

app = typer.Typer(
    help="Run the embedding pipeline over a dataset and emit FastEmbed vectors."
)

READERS_LABEL = ", ".join(sorted(data_factory.available_readers().keys()))
PIPELINES_LABEL = "preprocess"
ENCODERS_LABEL = ", ".join(sorted(models_factory.available_encoders().keys()))


def run_encoder(config: EmbeddingRunConfig) -> Generator[EmbeddingBatch, None, None]:
    """Stream embedding batches according to the provided configuration."""

    sink: WandbSink | None = None
    wandb_run = None
    if config.wandb_enabled:
        wandb_config = config.model_dump(mode="json")
        wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=wandb_config,
        )
        sink = WandbSink()

    reader_factory = data_factory.get_reader(config.reader_name)
    pipeline_runner: PreprocessorPipeline | Callable[[str], str] | None = None
    observer: PipelineObserver | None = None
    if config.use_pipeline:
        pipeline_name = config.pipeline_name or "preprocess"
        if pipeline_name != "preprocess":
            raise ValueError(
                f"Unsupported pipeline '{pipeline_name}'. Only 'preprocess' is available"
            )
        pipeline_runner = preprocess
        logger.info("Pipeline enabled: {name}", name=pipeline_name)

        sample_interval = int(os.getenv("NS_PIPELINE_SAMPLE_INTERVAL", "10"))
        observer = PipelineObserver(sink=sink, sample_interval=sample_interval)

    reader = reader_factory(
        path=config.dataset_path,
        pipeline=pipeline_runner,
        use_pipeline=config.use_pipeline,
        observer=observer,
    )

    encoder_factory = models_factory.get_encoder(config.encoder_name)
    encoder = encoder_factory(
        model_name=config.encoder_model_name, device=config.device
    )

    logger.info(
        "Starting embedding run dataset={dataset} encoder={encoder} model={model} device={device}",
        dataset=str(config.dataset_path),
        encoder=config.encoder_name,
        model=config.encoder_model_name,
        device=config.device,
    )

    metrics_tracker = EncodingMetricsTracker(sink=sink)

    for batch_index, (requests, _labels) in enumerate(
        reader.iter_batches(config.batch_size), start=1
    ):
        logger.debug(
            "Processing batch index={batch_index} size={size}",
            batch_index=batch_index,
            size=len(requests),
        )

        encode_start = perf_counter()
        embeddings = encoder.encode(requests)
        encode_elapsed = perf_counter() - encode_start

        metrics_tracker.record_batch(
            requests=requests,
            embeddings=embeddings,
            encode_seconds=encode_elapsed,
        )

        yield EmbeddingBatch(
            embeddings=embeddings,
            batch_index=batch_index,
            size=len(requests),
            model_name=config.encoder_model_name,
        )

        break

    logger.info(
        "Completed embedding run batches={batches} requests={requests}",
        batches=metrics_tracker.total_batches,
        requests=metrics_tracker.total_requests,
    )

    metrics_tracker.finalize()
    if wandb_run is not None:
        wandb_run.finish()

    encoder.shutdown()


@app.command()
def cli(
    dataset: Path = typer.Argument(..., help="Path to the JSONL dataset file"),
    batch_size: int = typer.Option(16, help="Number of requests per batch"),
    reader: str = typer.Option(
        "jsonl",
        help=f"Dataset reader (available: {READERS_LABEL})",
        show_default=True,
    ),
    use_pipeline: bool = typer.Option(False, help="Enable preprocessing pipeline"),
    pipeline: str | None = typer.Option(
        None,
        help=f"Pipeline name when enabled (available: {PIPELINES_LABEL})",
    ),
    encoder: str = typer.Option(
        "fastembed",
        help=f"Encoder backend (available: {ENCODERS_LABEL})",
        show_default=True,
    ),
    model_name: str = typer.Option(
        "BAAI/bge-small-en-v1.5",
        help="FastEmbed model identifier",
        show_default=True,
    ),
    device: str = typer.Option("mps", help="Encoder device", show_default=True),
    wandb: bool = typer.Option(
        False, "--wandb/--no-wandb", help="Enable Weights & Biases logging"
    ),
    wandb_project: str | None = typer.Option(
        None, help="Weights & Biases project name"
    ),
    wandb_entity: str | None = typer.Option(None, help="Weights & Biases entity/team"),
) -> None:
    """CLI entry point using Typer from tiangolo."""

    config_kwargs: dict[str, object] = {
        "dataset_path": dataset,
        "batch_size": batch_size,
        "reader_name": reader,
        "use_pipeline": use_pipeline,
        "pipeline_name": pipeline,
        "encoder_name": encoder,
        "encoder_model_name": model_name,
        "device": device,
        "wandb_enabled": wandb,
    }

    if wandb_project is not None:
        config_kwargs["wandb_project"] = wandb_project

    if wandb_entity is not None:
        config_kwargs["wandb_entity"] = wandb_entity

    config = EmbeddingRunConfig(**config_kwargs)

    for _batch in run_encoder(config):
        pass


def main() -> None:
    app()


if __name__ == "__main__":
    main()
