from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture

from neuralshield.anomaly.base import AnomalyDetector
from neuralshield.anomaly.factory import register_detector


@register_detector("gmm")
class GMMDetector(AnomalyDetector):
    """Gaussian Mixture Model anomaly detector.

    Models normal data as a mixture of Gaussians rather than a single distribution.
    Better than Mahalanobis when normal traffic has multiple distinct clusters or modes.

    Uses negative log-likelihood as anomaly score: samples with low probability
    under the learned mixture are considered anomalous.

    Works by:
    1. Fitting a GMM with K components to normal training data
    2. Computing probability density for test samples
    3. Lower probability = more anomalous

    Best for:
    - Dense embeddings with multi-modal distributions (e.g., SecBERT, BGE)
    - When normal traffic naturally clusters into groups
    - Moderate dimensions (<5000)

    Parameters:
        n_components: Number of Gaussian components in the mixture
        covariance_type: {'full', 'tied', 'diag', 'spherical'}
            - 'full': each component has its own covariance matrix
            - 'tied': all components share a single covariance matrix
            - 'diag': diagonal covariance (features independent)
            - 'spherical': single variance per component
    """

    def __init__(
        self,
        *,
        name: str = "default",
        n_components: int = 3,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> None:
        super().__init__(name=name)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self._model: GaussianMixture | None = None
        self._threshold: float | None = None

    def fit(self, embeddings: NDArray[np.float32]) -> None:
        """Fit GMM on normal training embeddings."""
        logger.info(
            "Fitting GMM",
            n_samples=len(embeddings),
            n_features=embeddings.shape[1],
            n_components=self.n_components,
            covariance_type=self.covariance_type,
        )

        self._model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=200,
            n_init=5,
        )
        self._model.fit(embeddings)
        self._fitted = True

        # Log convergence info
        converged = self._model.converged_
        n_iter = self._model.n_iter_
        logger.info(
            "GMM fitting complete",
            converged=converged,
            n_iterations=n_iter,
            bic=self._model.bic(embeddings),
            aic=self._model.aic(embeddings),
        )

    def scores(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute anomaly scores using negative log-likelihood.

        Returns:
            Scores where higher = more anomalous
            (negative of log probability density)
        """
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Detector has not been fitted yet")

        # score_samples returns log-likelihood (higher = more normal)
        # We negate it so higher score = more anomalous
        log_likelihood = self._model.score_samples(embeddings)
        anomaly_scores = -log_likelihood
        return anomaly_scores.astype(np.float32)

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

        scores_val = self.scores(normal_embeddings)
        threshold = float(np.percentile(scores_val, 100 * (1 - max_fpr)))
        self._threshold = threshold

        actual_fpr = np.mean(scores_val > threshold)
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
        scores_pred = self.scores(embeddings)

        limit = threshold if threshold is not None else self._threshold
        if limit is None:
            raise RuntimeError(
                "No threshold set. Call set_threshold() or provide threshold argument."
            )

        return (scores_pred > limit).astype(bool)

    def save(self, path: str | Path) -> None:
        """Save the trained model to disk."""
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Detector has not been fitted yet")

        payload = {
            "name": self.name,
            "model": self._model,
            "threshold": self._threshold,
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "random_state": self.random_state,
        }
        joblib.dump(payload, path)
        logger.info("Saved GMM model to {path}", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> "GMMDetector":
        """Load a trained model from disk."""
        payload = joblib.load(path)
        detector = cls(
            name=payload.get("name", "default"),
            n_components=payload.get("n_components", 3),
            covariance_type=payload.get("covariance_type", "full"),
            random_state=payload.get("random_state", 42),
        )
        detector._model = payload["model"]
        detector._threshold = payload.get("threshold")
        detector._fitted = True
        logger.info("Loaded GMM model from {path}", path=str(path))
        return detector
