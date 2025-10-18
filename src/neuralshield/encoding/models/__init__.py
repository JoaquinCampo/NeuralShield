"""Encoder model interfaces and concrete adapter stubs."""

from . import (
    colbert_muvera,  # noqa: F401 - ensure ColBERT+MUVERA encoder registration
    fastembed,  # noqa: F401 - ensure default encoder registration
    secbert,  # noqa: F401 - ensure SecBERT encoder registration
    secbert_flagweighted,  # noqa: F401 - ensure flag-weighted encoder registration
    tfidf,  # noqa: F401 - ensure TF-IDF encoder registration
)
from .base import EmbeddingBatch, RequestEncoder
from .byt5 import ByT5Encoder
from .colbert_muvera import ColBERTMuveraEncoder
from .factory import available_encoders, get_encoder, register_encoder
from .fastembed import FastEmbedEncoder
from .secbert import SecBERTEncoder
from .secbert_flagweighted import SecBERTFlagWeightedEncoder
from .tfidf import TFIDFEncoder

__all__ = [
    "EmbeddingBatch",
    "RequestEncoder",
    "ByT5Encoder",
    "ColBERTMuveraEncoder",
    "FastEmbedEncoder",
    "SecBERTEncoder",
    "SecBERTFlagWeightedEncoder",
    "TFIDFEncoder",
    "available_encoders",
    "get_encoder",
    "register_encoder",
]
