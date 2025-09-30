from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestDetector:
    """
    Wrapper around sklearn's IsolationForest for anomaly detection.

    Better suited for high-dimensional sparse data like TF-IDF embeddings.
    Much faster than EllipticEnvelope and designed specifically for anomaly detection.
    """

    contamination: float = 0.01
    n_estimators: int = 100
    max_samples: int | str = "auto"
    random_state: int | None = None
    n_jobs: int = -1
    model: IsolationForest | None = None

    def fit(self, embeddings: NDArray[np.float32]) -> None:
        """
        Fit the IsolationForest model on training embeddings.

        Args:
            embeddings: Training embeddings (n_samples, n_features)
        """
        logger.info(
            "Fitting IsolationForest",
            n_samples=len(embeddings),
            n_features=embeddings.shape[1],
            n_estimators=self.n_estimators,
            contamination=self.contamination,
        )

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=1,  # Show progress
        )
        self.model.fit(embeddings)

        logger.info("IsolationForest fitting complete")

    @property
    def threshold_(self) -> float:
        """
        Get the decision threshold.

        For IsolationForest, uses the model's offset_ which is computed
        from the contamination parameter during training.
        Scores > offset_ are normal, scores <= offset_ are anomalies.
        """
        if self.model is None:
            raise RuntimeError("Detector has not been fitted yet")
        return float(self.model.offset_)

    def scores(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Compute anomaly scores for embeddings.

        Args:
            embeddings: Input embeddings (n_samples, n_features)

        Returns:
            Anomaly scores. Higher scores = more normal.
            Scores > 0 are normal, scores < 0 are anomalies.
        """
        if self.model is None:
            raise RuntimeError("Detector has not been fitted yet")
        scores = self.model.score_samples(embeddings)
        return scores.astype(np.float32)

    def predict(
        self,
        embeddings: NDArray[np.float32],
        *,
        threshold: float | None = None,
        score_greater_is_normal: bool = True,
    ) -> NDArray[np.bool_]:
        """
        Predict anomalies for embeddings.

        Args:
            embeddings: Input embeddings
            threshold: Custom threshold (default: 0.0 for IsolationForest)
            score_greater_is_normal: If True, scores > threshold are normal

        Returns:
            Boolean array: True = anomaly detected, False = normal
        """
        scores = self.scores(embeddings)
        limit = self.threshold_ if threshold is None else threshold

        if score_greater_is_normal:
            return (scores < limit).astype(bool)
        return (scores > limit).astype(bool)

    def save(self, path: str) -> None:
        """Save the trained model to disk."""
        if self.model is None:
            raise RuntimeError("Detector has not been fitted yet")

        payload = {
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "model": self.model,
        }
        joblib.dump(payload, path)
        logger.info("Saved IsolationForest model to {path}", path=path)

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        """Load a trained model from disk."""
        payload = joblib.load(path)
        detector = cls(
            contamination=float(payload.get("contamination", 0.01)),
            n_estimators=int(payload.get("n_estimators", 100)),
            max_samples=payload.get("max_samples", "auto"),
            random_state=payload.get("random_state"),
            n_jobs=int(payload.get("n_jobs", -1)),
        )
        detector.model = payload["model"]
        logger.info("Loaded IsolationForest model from {path}", path=path)
        return detector
