from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class EmbeddingBatch(BaseModel):
    embeddings: NDArray[np.float32]
    batch_index: int
    size: int
    model_name: str

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )


class RequestEncoder(ABC):
    """Abstract encoder that converts request strings to vector embeddings."""

    def __init__(self, *, model_name: str = "default", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def encode(self, batch: Sequence[str]) -> NDArray[np.float32]:
        """Return embedding vectors for the provided batch of requests."""
        ...

    def shutdown(self) -> None:
        """Hook for encoders that need explicit cleanup."""
        return None
