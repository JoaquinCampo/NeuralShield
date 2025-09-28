from __future__ import annotations

from typing import Sequence

import numpy as np
from fastembed import TextEmbedding
from loguru import logger

from neuralshield.encoding.models.base import RequestEncoder
from neuralshield.encoding.models.factory import register_encoder


@register_encoder("fastembed")
class FastEmbedEncoder(RequestEncoder):
    """Wrap the FastEmbed TextEmbedding interface for request encoding."""

    def __init__(self, *, model_name: str = "default", device: str = "cpu") -> None:
        super().__init__(model_name=model_name, device=device)

        selected = None if model_name == "default" else model_name
        self._embedder = TextEmbedding(model=selected, device=device)
        logger.info(
            "Initialized FastEmbed encoder model={model} device={device}",
            model=model_name,
            device=device,
        )

    def encode(self, batch: Sequence[str]) -> np.ndarray:
        if not batch:
            return np.empty((0, 0), dtype=np.float32)

        logger.debug(
            "Encoding batch size={size} with FastEmbed model={model}",
            size=len(batch),
            model=self.model_name,
        )
        embeddings_iter = self._embedder.embed(batch)
        embeddings = np.asarray(list(embeddings_iter), dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        return embeddings
