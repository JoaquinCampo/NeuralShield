"""Smoke tests for embedding scaffolding registries."""

from pathlib import Path

import pytest

from neuralshield.encoding.config import EmbeddingRunConfig
from neuralshield.encoding.data.factory import available_readers
from neuralshield.encoding.models.factory import available_encoders
from neuralshield.encoding.run_encoder import run_encoder
from neuralshield.preprocessing.pipeline import PreprocessorPipeline, preprocess


def test_default_reader_registered() -> None:
    assert "jsonl" in available_readers()


def test_default_pipeline_available() -> None:
    assert isinstance(preprocess, PreprocessorPipeline)


def test_invalid_pipeline_name_rejected() -> None:
    config = EmbeddingRunConfig(
        dataset_path=Path("/tmp/does-not-matter.jsonl"),
        use_pipeline=True,
        pipeline_name="not-real",
    )

    generator = run_encoder(config)

    with pytest.raises(ValueError):
        next(generator)


def test_default_encoder_registered() -> None:
    assert "fastembed" in available_encoders()


def test_wandb_defaults_apply_project() -> None:
    config = EmbeddingRunConfig(
        dataset_path=Path("/tmp/does-not-matter.jsonl"),
        wandb_enabled=True,
    )

    assert config.wandb_project == "neuralshield"
    assert config.wandb_entity is None
