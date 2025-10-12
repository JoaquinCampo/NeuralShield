from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from sklearn.covariance import EmpiricalCovariance

from neuralshield.anomaly.base import AnomalyDetector
from neuralshield.anomaly.factory import register_detector


@register_detector("mahalanobis")
class MahalanobisDetector(AnomalyDetector):
    """Mahalanobis distance anomaly detector.

    Uses empirical covariance to compute statistical distance from normal data.
    Accounts for feature correlations (unlike IsolationForest).

    Fast, stable, and interpretable. No hyperparameters to tune.

    Works by:
    1. Computing mean and covariance of normal training data
    2. Measuring distance of test samples from this distribution
    3. Higher distance = more anomalous

    Best for:
    - Dense embeddings with correlated features (e.g., BGE, BERT)
    - Moderate dimensions (<5000)
    - When you want interpretable statistical distance
    """

    def __init__(self, *, name: str = "default") -> None:
        super().__init__(name=name)
        self._model: EmpiricalCovariance | None = None
        self._threshold: float | None = None

    def fit(self, embeddings: NDArray[np.float32]) -> None:
        """Fit Mahalanobis detector on normal training embeddings."""
        logger.info(
            "Fitting Mahalanobis (EmpiricalCovariance)",
            n_samples=len(embeddings),
            n_features=embeddings.shape[1],
        )

        self._model = EmpiricalCovariance()
        self._model.fit(embeddings)
        self._fitted = True

        logger.info("Mahalanobis fitting complete")

    def scores(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute Mahalanobis distances.

        Returns:
            Distances where higher = more anomalous
        """
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Detector has not been fitted yet")

        distances = self._model.mahalanobis(embeddings)
        return distances.astype(np.float32)

    def set_threshold(
        self,
        normal_embeddings: NDArray[np.float32],
        max_fpr: float = 0.05,
    ) -> float:
        """Set decision threshold based on desired FPR on normal data.

        Args:
            normal_embeddings: Known normal samples for calibration
            max_fpr: Maximum false positive rate to allow (default: 5%)

        Returns:
            The threshold value
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before setting threshold")

        distances = self.scores(normal_embeddings)
        threshold = float(np.percentile(distances, 100 * (1 - max_fpr)))
        self._threshold = threshold

        actual_fpr = np.mean(distances > threshold)
        logger.info(
            f"Threshold set to {threshold:.4f} "
            f"(target FPR={max_fpr:.1%}, actual={actual_fpr:.1%})"
        )

        return threshold

    def predict(
        self,
        embeddings: NDArray[np.float32],
        *,
        threshold: float | None = None,
    ) -> NDArray[np.bool_]:
        """Predict anomalies.

        Args:
            embeddings: Input embeddings
            threshold: Custom threshold (default: uses stored threshold)

        Returns:
            Boolean array: True = anomaly, False = normal
        """
        scores = self.scores(embeddings)

        limit = threshold if threshold is not None else self._threshold
        if limit is None:
            raise RuntimeError(
                "No threshold set. Call set_threshold() or provide threshold argument."
            )

        return (scores > limit).astype(bool)

    def save(self, path: str | Path) -> None:
        """Save the trained model to disk."""
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Detector has not been fitted yet")

        payload = {
            "name": self.name,
            "model": self._model,
            "threshold": self._threshold,
        }
        joblib.dump(payload, path)
        logger.info("Saved Mahalanobis model to {path}", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> "MahalanobisDetector":
        """Load a trained model from disk."""
        payload = joblib.load(path)
        detector = cls(name=payload.get("name", "default"))
        detector._model = payload["model"]
        detector._threshold = payload.get("threshold")
        detector._fitted = True
        logger.info("Loaded Mahalanobis model from {path}", path=str(path))
        return detector
