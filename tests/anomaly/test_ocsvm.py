"""Tests for OCSVM anomaly detector."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from neuralshield.anomaly import OCSVMDetector, get_detector


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    np.random.seed(42)
    # Normal data: cluster around origin
    normal = np.random.randn(100, 10).astype(np.float32)
    # Anomalous data: far from origin
    anomalous = np.random.randn(20, 10).astype(np.float32) + 5
    return normal, anomalous


def test_ocsvm_init():
    """Test OCSVM detector initialization."""
    detector = OCSVMDetector(nu=0.1, gamma="scale")
    assert detector.nu == 0.1
    assert detector.gamma == "scale"
    assert not detector.is_fitted


def test_ocsvm_fit_scores(sample_embeddings):
    """Test fitting and scoring."""
    normal, anomalous = sample_embeddings

    detector = OCSVMDetector(nu=0.05, gamma="scale")
    detector.fit(normal)

    assert detector.is_fitted

    # Compute scores
    normal_scores = detector.scores(normal)
    anomalous_scores = detector.scores(anomalous)

    # Anomalies should have higher scores
    assert np.mean(anomalous_scores) > np.mean(normal_scores)


def test_ocsvm_predict(sample_embeddings):
    """Test prediction."""
    normal, anomalous = sample_embeddings

    detector = OCSVMDetector(nu=0.05)
    detector.fit(normal)

    # Predict with default threshold
    predictions = detector.predict(np.vstack([normal, anomalous]))
    assert predictions.dtype == bool
    assert len(predictions) == len(normal) + len(anomalous)

    # Custom threshold
    detector.set_threshold(normal, max_fpr=0.1)
    predictions_custom = detector.predict(normal)
    fpr = np.mean(predictions_custom)
    assert fpr <= 0.15  # Allow some tolerance


def test_ocsvm_set_threshold(sample_embeddings):
    """Test threshold setting."""
    normal, _ = sample_embeddings

    detector = OCSVMDetector()
    detector.fit(normal)

    threshold = detector.set_threshold(normal, max_fpr=0.05)
    assert isinstance(threshold, float)
    assert detector._threshold == threshold


def test_ocsvm_save_load(sample_embeddings):
    """Test save/load."""
    normal, anomalous = sample_embeddings

    detector = OCSVMDetector(nu=0.1, gamma="scale")
    detector.fit(normal)
    detector.set_threshold(normal, max_fpr=0.05)

    original_scores = detector.scores(anomalous)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ocsvm_test.joblib"
        detector.save(path)

        loaded = OCSVMDetector.load(path)
        assert loaded.is_fitted
        assert loaded.nu == detector.nu
        assert loaded.gamma == detector.gamma
        assert loaded._threshold == detector._threshold

        loaded_scores = loaded.scores(anomalous)
        np.testing.assert_array_almost_equal(original_scores, loaded_scores, decimal=5)


def test_ocsvm_factory():
    """Test factory registration."""
    DetectorClass = get_detector("ocsvm")
    assert DetectorClass is OCSVMDetector

    detector = DetectorClass(nu=0.05)
    assert isinstance(detector, OCSVMDetector)


def test_ocsvm_not_fitted_error():
    """Test error when using unfitted detector."""
    detector = OCSVMDetector()
    embeddings = np.random.randn(10, 5).astype(np.float32)

    with pytest.raises(RuntimeError, match="not been fitted"):
        detector.scores(embeddings)

    with pytest.raises(RuntimeError, match="not been fitted"):
        detector.predict(embeddings)
