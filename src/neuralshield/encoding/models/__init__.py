"""Encoder model interfaces and concrete adapter stubs."""

from . import (
    fastembed,  # noqa: F401 - ensure default encoder registration
    secbert,  # noqa: F401 - ensure SecBERT encoder registration
    tfidf,  # noqa: F401 - ensure TF-IDF encoder registration
)
from .base import EmbeddingBatch, RequestEncoder
from .factory import available_encoders, get_encoder, register_encoder
from .fastembed import FastEmbedEncoder
from .secbert import SecBERTEncoder
from .tfidf import TFIDFEncoder

__all__ = [
    "EmbeddingBatch",
    "RequestEncoder",
    "FastEmbedEncoder",
    "SecBERTEncoder",
    "TFIDFEncoder",
    "available_encoders",
    "get_encoder",
    "register_encoder",
]
