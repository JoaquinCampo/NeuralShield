"""Encoder model interfaces and concrete adapter stubs."""

from . import fastembed  # noqa: F401 - ensure default encoder registration
from .base import EmbeddingBatch, RequestEncoder
from .factory import available_encoders, get_encoder, register_encoder
from .fastembed import FastEmbedEncoder

__all__ = [
    "EmbeddingBatch",
    "RequestEncoder",
    "FastEmbedEncoder",
    "available_encoders",
    "get_encoder",
    "register_encoder",
]
