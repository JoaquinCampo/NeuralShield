from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class EmbeddingRunConfig(BaseSettings):
    """Capture the essential knobs for an embedding run."""

    dataset_path: Path
    batch_size: int = 512
    reader_name: str = "jsonl"
    use_pipeline: bool = False
    pipeline_name: str | None = None

    encoder_name: str = "fastembed"
    encoder_model_name: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"

    wandb_enabled: bool = False
    wandb_project: str = "neuralshield"
    wandb_entity: str | None = None
