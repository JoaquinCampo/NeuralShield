from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detection on embeddings.

    All detectors must implement:
    - fit(): Train on normal embeddings
    - scores(): Compute anomaly scores
    - predict(): Binary anomaly prediction
    - save/load(): Persistence
    """

    def __init__(self, *, name: str = "default") -> None:
        self.name = name
        self._fitted = False

    @abstractmethod
    def fit(self, embeddings: NDArray[np.float32]) -> None:
        """Fit the detector on normal training embeddings.

        Args:
            embeddings: Training embeddings, shape (n_samples, n_features)
                       All samples should be normal (non-anomalous)
        """
        ...

    @abstractmethod
    def scores(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute anomaly scores for embeddings.

        Args:
            embeddings: Input embeddings, shape (n_samples, n_features)

        Returns:
            Anomaly scores, shape (n_samples,)
            Convention: Higher scores = more anomalous (detectors may differ internally)
        """
        ...

    @abstractmethod
    def predict(
        self,
        embeddings: NDArray[np.float32],
        *,
        threshold: float | None = None,
    ) -> NDArray[np.bool_]:
        """Predict if embeddings are anomalies.

        Args:
            embeddings: Input embeddings
            threshold: Optional custom threshold

        Returns:
            Boolean array: True = anomaly, False = normal
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save the trained detector to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        """Load a trained detector from disk."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Check if detector has been fitted."""
        return self._fitted
