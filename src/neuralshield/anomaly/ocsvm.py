from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from sklearn.svm import OneClassSVM

from neuralshield.anomaly.base import AnomalyDetector
from neuralshield.anomaly.factory import register_detector


@register_detector("ocsvm")
class OCSVMDetector(AnomalyDetector):
    """One-Class SVM anomaly detector.

    Uses RBF kernel to learn a non-linear decision boundary around normal data.
    Slower than IsolationForest and Mahalanobis but can capture complex patterns.

    Hyperparameters:
        nu: Upper bound on fraction of outliers (0 < nu <= 1)
            Default: 0.05 (5% contamination tolerance)
        gamma: RBF kernel coefficient
            - "scale": 1 / (n_features * X.var())
            - "auto": 1 / n_features
            - float: explicit value
            Default: "scale"
        kernel: Kernel type (only "rbf" tested)
        cache_size: Kernel cache size in MB (larger = faster but more memory)
        verbose: Print training progress

    Best for:
        - Dense embeddings where linear boundaries fail
        - When you suspect non-linear patterns
        - Moderate sample sizes (<100k)

    Training time:
        - ~20 seconds for 47k samples @ 384 dims (CPU)
        - O(n^2) complexity, slower for large datasets
    """

    def __init__(
        self,
        *,
        name: str = "default",
        nu: float = 0.05,
        gamma: str | float = "scale",
        kernel: str = "rbf",
        cache_size: int = 2000,
        verbose: bool = False,
    ) -> None:
        super().__init__(name=name)
        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel
        self.cache_size = cache_size
        self.verbose = verbose
        self._model: OneClassSVM | None = None
        self._threshold: float | None = None

    def fit(self, embeddings: NDArray[np.float32]) -> None:
        """Fit One-Class SVM on normal training embeddings."""
        logger.info(
            "Fitting One-Class SVM",
            n_samples=len(embeddings),
            n_features=embeddings.shape[1],
            nu=self.nu,
            gamma=self.gamma,
            kernel=self.kernel,
        )

        self._model = OneClassSVM(
            nu=self.nu,
            gamma=self.gamma,
            kernel=self.kernel,
            cache_size=self.cache_size,
            verbose=self.verbose,
        )
        self._model.fit(embeddings)
        self._fitted = True

        logger.info("One-Class SVM fitting complete")

    def scores(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute anomaly scores.

        Returns:
            Scores where higher = more anomalous (inverted from sklearn convention)
        """
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Detector has not been fitted yet")

        # OCSVM: positive score = normal, negative = anomalous
        # We negate: higher score = more anomalous
        raw_scores = self._model.decision_function(embeddings)
        return (-raw_scores).astype(np.float32)

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
            threshold: Custom threshold (default: uses stored threshold or model default)

        Returns:
            Boolean array: True = anomaly, False = normal
        """
        scores = self.scores(embeddings)

        if threshold is not None:
            # Use provided threshold
            return (scores > threshold).astype(bool)
        elif self._threshold is not None:
            # Use stored threshold
            return (scores > self._threshold).astype(bool)
        else:
            # Use model's decision (0 threshold on inverted scores)
            return (scores > 0).astype(bool)

    def save(self, path: str | Path) -> None:
        """Save the trained model to disk."""
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Detector has not been fitted yet")

        payload = {
            "name": self.name,
            "nu": self.nu,
            "gamma": self.gamma,
            "kernel": self.kernel,
            "cache_size": self.cache_size,
            "verbose": self.verbose,
            "model": self._model,
            "threshold": self._threshold,
        }
        joblib.dump(payload, path)
        logger.info("Saved OCSVM model to {path}", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> "OCSVMDetector":
        """Load a trained model from disk."""
        payload = joblib.load(path)
        detector = cls(
            name=payload.get("name", "default"),
            nu=float(payload.get("nu", 0.05)),
            gamma=payload.get("gamma", "scale"),
            kernel=str(payload.get("kernel", "rbf")),
            cache_size=int(payload.get("cache_size", 2000)),
            verbose=bool(payload.get("verbose", False)),
        )
        detector._model = payload["model"]
        detector._threshold = payload.get("threshold")
        detector._fitted = True
        logger.info("Loaded OCSVM model from {path}", path=str(path))
        return detector
