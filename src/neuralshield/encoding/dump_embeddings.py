from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Generator, Iterable

import typer
from loguru import logger
from tqdm import tqdm

import neuralshield.encoding.data.factory as data_factory
import neuralshield.encoding.models.factory as models_factory
from neuralshield.encoding.config import EmbeddingDumpConfig
from neuralshield.encoding.observability import (
    EncodingMetricsTracker,
    PipelineObserver,
    init_wandb_sink,
)
from neuralshield.preprocessing.pipeline import PreprocessorPipeline, preprocess

logger.remove()
logger.add(sys.stderr, level="INFO")

app = typer.Typer(help="Dump embeddings from a dataset into an artifact file.")

READERS_LABEL = ", ".join(sorted(data_factory.available_readers().keys()))
PIPELINES_LABEL = "preprocess"
ENCODERS_LABEL = ", ".join(sorted(models_factory.available_encoders().keys()))


def iter_batches(
    reader_factory: Callable[..., Any],
    *,
    dataset: Path,
    batch_size: int,
    pipeline: PreprocessorPipeline | Callable[[str], str] | None,
    use_pipeline: bool,
    observer: PipelineObserver | None,
) -> Generator[tuple[list[str], list[str | None]], None, None]:
    reader = reader_factory(
        path=dataset,
        pipeline=pipeline,
        use_pipeline=use_pipeline,
        observer=observer,
    )
    yield from reader.iter_batches(batch_size)


def payload_stream(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            yield json.loads(raw_line)


def dump_embeddings(config: EmbeddingDumpConfig) -> None:
    sink, wandb_run = init_wandb_sink(
        config.wandb_enabled,
        project=config.wandb_project,
        entity=config.wandb_entity,
        config=config.model_dump(),
    )

    observer = (
        PipelineObserver(sink=sink, sample_interval=config.sample_interval)
        if config.use_pipeline
        else None
    )
    pipeline_runner: PreprocessorPipeline | Callable[[str], str] | None = None
    if config.use_pipeline:
        pipeline_name = config.pipeline_name or "preprocess"
        if pipeline_name != "preprocess":
            raise ValueError(
                "dump_embeddings currently only supports the 'preprocess' pipeline"
            )
        pipeline_runner = preprocess
        logger.info("Pipeline enabled: {name}", name=pipeline_name)

    reader_factory = data_factory.get_reader(config.reader_name)
    encoder_factory = models_factory.get_encoder(config.encoder_name)

    encoder_kwargs: dict[str, object] = {}
    if config.encoder_name.lower() == "secbert-flag-weighted":
        if config.token_weight_paths:
            encoder_kwargs["token_weight_paths"] = config.token_weight_paths

    encoder = encoder_factory(
        model_name=config.encoder_model_name,
        device=config.device,
        **encoder_kwargs,
    )

    logger.info(
        "Dumping embeddings dataset={dataset} encoder={encoder} model={model} "
        "device={device}",
        dataset=str(config.dataset_path),
        encoder=config.encoder_name,
        model=config.encoder_model_name,
        device=config.device,
    )

    metrics = EncodingMetricsTracker(sink=sink)

    output_path = config.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with config.dataset_path.open("r", encoding="utf-8") as source:
        total_requests = sum(1 for line in source if line.strip())

    raw_payloads = payload_stream(config.dataset_path)
    raw_iter = iter(raw_payloads)

    with output_path.open("wb") as f:
        header = {
            "model": config.encoder_model_name,
            "encoder": config.encoder_name,
            "device": config.device,
            "use_pipeline": config.use_pipeline,
            "pipeline_name": config.pipeline_name or "preprocess",
            "batch_size": config.batch_size,
            "total_requests": total_requests,
        }
        f.write((json.dumps(header) + "\n").encode("utf-8"))

        progress = tqdm(
            total=total_requests if total_requests else None,
            desc="Encoding requests",
            unit="req",
        )

        for batch_index, (requests, _labels) in enumerate(
            iter_batches(
                reader_factory,
                dataset=config.dataset_path,
                batch_size=config.batch_size,
                pipeline=pipeline_runner,
                use_pipeline=config.use_pipeline,
                observer=observer,
            ),
            start=1,
        ):
            raw_batch = []
            try:
                for _ in range(len(requests)):
                    raw_batch.append(next(raw_iter))
            except StopIteration as exc:  # pragma: no cover - data mismatch guard
                raise RuntimeError(
                    "Dataset shorter than expected while dumping embeddings"
                ) from exc

            encode_start = perf_counter()
            embeddings = encoder.encode(requests)
            encode_elapsed = perf_counter() - encode_start

            metrics.record_batch(
                requests=requests,
                embeddings=embeddings,
                encode_seconds=encode_elapsed,
            )

            payload = {
                "embeddings": embeddings.tolist(),
            }
            if config.include_requests:
                payload["requests"] = [item.get("request") for item in raw_batch]
            if config.include_labels:
                payload["labels"] = [item.get("label") for item in raw_batch]
            if config.include_ids:
                payload["ids"] = [
                    item.get(config.request_id_field) for item in raw_batch
                ]
            f.write((json.dumps(payload) + "\n").encode("utf-8"))
            progress.update(len(requests))

        progress.close()

    summary = metrics.finalize()
    logger.info("Embedding dump completed", **summary)

    if observer is not None:
        observer.flush_samples()

    encoder.shutdown()

    if wandb_run is not None:
        wandb_run.finish()


@app.command()
def cli(
    dataset: Path = typer.Argument(..., help="Path to the JSONL dataset file"),
    output: Path = typer.Argument(..., help="Destination file for embeddings"),
    batch_size: int = typer.Option(512, help="Requests per batch"),
    reader: str = typer.Option(
        "jsonl",
        help=f"Dataset reader (available: {READERS_LABEL})",
        show_default=True,
    ),
    use_pipeline: bool = typer.Option(True, help="Enable preprocessing pipeline"),
    pipeline: str | None = typer.Option(
        "preprocess",
        help=f"Pipeline name when enabled (available: {PIPELINES_LABEL})",
    ),
    encoder: str = typer.Option(
        "fastembed",
        help=f"Encoder backend (available: {ENCODERS_LABEL})",
        show_default=True,
    ),
    model_name: str = typer.Option(
        "BAAI/bge-small-en-v1.5",
        help="Encoder model identifier",
        show_default=True,
    ),
    device: str = typer.Option("cpu", help="Encoder device"),
    wandb: bool = typer.Option(
        False, "--wandb/--no-wandb", help="Enable Weights & Biases logging"
    ),
    wandb_project: str = typer.Option("neuralshield", help="W&B project name"),
    wandb_entity: str | None = typer.Option(None, help="W&B entity/team"),
    include_requests: bool = typer.Option(
        False, help="Persist the raw requests alongside embeddings"
    ),
    include_labels: bool = typer.Option(
        True, help="Persist labels from the dataset when available"
    ),
    include_ids: bool = typer.Option(
        True, help="Persist request identifiers from the dataset"
    ),
    request_id_field: str = typer.Option(
        "id", help="JSON field to read as request identifier"
    ),
    sample_interval: int = typer.Option(
        10, help="Sample every N requests into the observability table"
    ),
    token_weight: list[Path] = typer.Option(
        [],
        "--token-weight",
        "-tw",
        help="Path to a token-weight JSON (repeatable). "
        "Combine attack boosts and normal downweights as needed.",
    ),
) -> None:
    cfg = EmbeddingDumpConfig(
        dataset_path=dataset,
        output_path=output,
        batch_size=batch_size,
        reader_name=reader,
        use_pipeline=use_pipeline,
        pipeline_name=pipeline,
        encoder_name=encoder,
        encoder_model_name=model_name,
        device=device,
        wandb_enabled=wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        include_requests=include_requests,
        include_labels=include_labels,
        include_ids=include_ids,
        request_id_field=request_id_field,
        sample_interval=sample_interval,
        token_weight_paths=tuple(token_weight) if token_weight else None,
    )

    dump_embeddings(cfg)


if __name__ == "__main__":
    app()
