from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest

from neuralshield.anomaly.base import AnomalyDetector
from neuralshield.anomaly.factory import register_detector


@register_detector("isolation-forest")
class IsolationForestDetector(AnomalyDetector):
    """IsolationForest-based anomaly detector.

    Suited for high-dimensional data. Works by isolating anomalies in random tree partitions.
    Treats dimensions independently (doesn't account for correlations).

    Hyperparameters:
        contamination: Expected proportion of anomalies in training data
        n_estimators: Number of trees in the forest
        max_samples: Samples per tree ("auto" or int)
        random_state: Random seed for reproducibility
        n_jobs: Parallel jobs (-1 = all cores)
    """

    def __init__(
        self,
        *,
        name: str = "default",
        contamination: float = 0.01,
        n_estimators: int = 100,
        max_samples: int | str = "auto",
        random_state: int | None = None,
        n_jobs: int = -1,
    ) -> None:
        super().__init__(name=name)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model: IsolationForest | None = None

    def fit(self, embeddings: NDArray[np.float32]) -> None:
        """Fit IsolationForest on normal training embeddings."""
        logger.info(
            "Fitting IsolationForest",
            n_samples=len(embeddings),
            n_features=embeddings.shape[1],
            n_estimators=self.n_estimators,
            contamination=self.contamination,
        )

        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=1,
        )
        self._model.fit(embeddings)
        self._fitted = True

        logger.info("IsolationForest fitting complete")

    def scores(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute anomaly scores.

        Returns:
            Scores where higher = more anomalous (inverted from sklearn convention)
        """
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Detector has not been fitted yet")

        # IsolationForest: higher score = more normal
        # We invert: higher score = more anomalous
        raw_scores = self._model.score_samples(embeddings)
        return (-raw_scores).astype(np.float32)

    def predict(
        self,
        embeddings: NDArray[np.float32],
        *,
        threshold: float | None = None,
    ) -> NDArray[np.bool_]:
        """Predict anomalies.

        Args:
            embeddings: Input embeddings
            threshold: Custom threshold (default: uses model's offset)

        Returns:
            Boolean array: True = anomaly, False = normal
        """
        scores = self.scores(embeddings)

        if threshold is None:
            # Use model's threshold (inverted since we inverted scores)
            if self._model is None:
                raise RuntimeError("Detector not fitted")
            threshold = -float(self._model.offset_)

        return (scores > threshold).astype(bool)

    def save(self, path: str | Path) -> None:
        """Save the trained model to disk."""
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Detector has not been fitted yet")

        payload = {
            "name": self.name,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "model": self._model,
        }
        joblib.dump(payload, path)
        logger.info("Saved IsolationForest model to {path}", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> "IsolationForestDetector":
        """Load a trained model from disk."""
        payload = joblib.load(path)
        detector = cls(
            name=payload.get("name", "default"),
            contamination=float(payload.get("contamination", 0.01)),
            n_estimators=int(payload.get("n_estimators", 100)),
            max_samples=payload.get("max_samples", "auto"),
            random_state=payload.get("random_state"),
            n_jobs=int(payload.get("n_jobs", -1)),
        )
        detector._model = payload["model"]
        detector._fitted = True
        logger.info("Loaded IsolationForest model from {path}", path=str(path))
        return detector
