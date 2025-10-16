from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from sklearn.neighbors import LocalOutlierFactor

from neuralshield.anomaly.base import AnomalyDetector
from neuralshield.anomaly.factory import register_detector


@register_detector("lof")
class LOFDetector(AnomalyDetector):
    """Local Outlier Factor anomaly detector.

    Uses local density-based outlier detection to identify anomalies.
    Compares local density of each point to the local densities of its neighbors.

    Key advantage over global methods (like Mahalanobis):
    - Handles multimodal distributions (multiple clusters)
    - Adapts to local density variations
    - Robust to global outliers in training data

    Works by:
    1. For each test point, find k nearest neighbors in training data
    2. Compute local reachability density
    3. Compare to neighbors' densities
    4. Higher ratio = more anomalous (lower local density)

    Best for:
    - Clustered data (e.g., different HTTP endpoint types)
    - When normal data forms multiple distinct groups
    - When local patterns matter more than global distribution
    """

    def __init__(
        self,
        *,
        name: str = "default",
        n_neighbors: int = 100,
        contamination: str | float = "auto",
        metric: str = "euclidean",
    ) -> None:
        super().__init__(name=name)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self._model: LocalOutlierFactor | None = None
        self._threshold: float | None = None

    def fit(self, embeddings: NDArray[np.float32]) -> None:
        """Fit LOF detector on normal training embeddings."""
        logger.info(
            "Fitting LOF (LocalOutlierFactor)",
            n_samples=len(embeddings),
            n_features=embeddings.shape[1],
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
        )

        # novelty=True enables prediction on new data
        # This is critical - without it, LOF can only score training data
        self._model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            metric=self.metric,
            novelty=True,  # Enable prediction on new samples
        )
        self._model.fit(embeddings)
        self._fitted = True

        logger.info("LOF fitting complete")

    def scores(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute LOF anomaly scores.

        Returns:
            Scores where higher = more anomalous
            (Note: LOF internally uses negative scores, we flip them)
        """
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Detector has not been fitted yet")

        # decision_function returns negative outlier factor
        # More negative = more anomalous
        # We flip sign so higher = more anomalous (consistent with other detectors)
        lof_scores = self._model.decision_function(embeddings)
        return (-lof_scores).astype(np.float32)

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

        scores = self.scores(normal_embeddings)
        threshold = float(np.percentile(scores, 100 * (1 - max_fpr)))
        self._threshold = threshold

        actual_fpr = np.mean(scores > threshold)
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
            "n_neighbors": self.n_neighbors,
            "contamination": self.contamination,
            "metric": self.metric,
        }
        joblib.dump(payload, path)
        logger.info("Saved LOF model to {path}", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> "LOFDetector":
        """Load a trained model from disk."""
        payload = joblib.load(path)
        detector = cls(
            name=payload.get("name", "default"),
            n_neighbors=payload.get("n_neighbors", 100),
            contamination=payload.get("contamination", "auto"),
            metric=payload.get("metric", "euclidean"),
        )
        detector._model = payload["model"]
        detector._threshold = payload.get("threshold")
        detector._fitted = True
        logger.info("Loaded LOF model from {path}", path=str(path))
        return detector
