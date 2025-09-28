"""Embedding workflow package scaffolding."""

# Import subpackages so decorator-based registries populate eagerly.
from neuralshield.encoding import data as _data  # noqa: F401
from neuralshield.encoding import models as _models  # noqa: F401

from .config import EmbeddingRunConfig
from .run_encoder import run_encoder

__all__ = ["EmbeddingRunConfig", "run_encoder"]
