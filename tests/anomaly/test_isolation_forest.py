"""Tests for IsolationForest anomaly detector."""

import numpy as np

from neuralshield.anomaly.model import IsolationForestDetector


def test_isolation_forest_identifies_outlier():
    """Test that IsolationForest can identify clear outliers."""
    rng = np.random.default_rng(0)
    # Normal data clustered around 0
    normal = rng.normal(0.0, 0.2, size=(100, 2)).astype(np.float32)

    detector = IsolationForestDetector(contamination=0.1, random_state=0)
    detector.fit(normal)

    # Point on manifold (should be normal)
    on_manifold = np.array([[0.05, -0.08]], dtype=np.float32)
    # Point far from manifold (should be anomaly)
    off_manifold = np.array([[5.0, -5.0]], dtype=np.float32)

    assert bool(detector.predict(on_manifold)[0]) is False  # Normal
    assert bool(detector.predict(off_manifold)[0]) is True  # Anomaly
    assert detector.is_fitted  # Detector is fitted


def test_isolation_forest_with_sparse_features():
    """Test IsolationForest with sparse, high-dimensional data like TF-IDF."""
    rng = np.random.default_rng(42)

    # Simulate sparse TF-IDF features (mostly zeros)
    n_samples = 200
    n_features = 1000

    # Normal samples: sparse vectors with few non-zero values
    normal = np.zeros((n_samples, n_features), dtype=np.float32)
    for i in range(n_samples):
        # Randomly set 5-10 features to non-zero values
        n_active = rng.integers(5, 10)
        indices = rng.choice(n_features, n_active, replace=False)
        normal[i, indices] = rng.uniform(0.1, 1.0, size=n_active).astype(np.float32)

    # Train detector
    detector = IsolationForestDetector(
        contamination=0.05, n_estimators=50, random_state=42
    )
    detector.fit(normal)

    # Create anomalies: unusual patterns
    anomaly1 = np.zeros(n_features, dtype=np.float32)
    # Anomaly has many more active features
    anomaly1[: n_features // 2] = 0.5

    anomaly2 = np.zeros(n_features, dtype=np.float32)
    # Anomaly has extreme values
    anomaly2[rng.choice(n_features, 5, replace=False)] = 10.0

    anomalies = np.vstack([anomaly1, anomaly2])

    # Predict
    predictions = detector.predict(anomalies)

    # At least one should be detected as anomaly
    assert predictions.sum() > 0


def test_isolation_forest_save_load(tmp_path):
    """Test saving and loading IsolationForest detector."""
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 1.0, size=(50, 3)).astype(np.float32)

    # Train and save
    detector1 = IsolationForestDetector(
        contamination=0.1, n_estimators=20, random_state=0
    )
    detector1.fit(data)

    model_path = tmp_path / "isolation_forest.joblib"
    detector1.save(str(model_path))

    # Load and verify
    detector2 = IsolationForestDetector.load(str(model_path))

    test_sample = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    # Both should produce same results
    score1 = detector1.scores(test_sample)
    score2 = detector2.scores(test_sample)

    assert np.allclose(score1, score2)

    pred1 = detector1.predict(test_sample)
    pred2 = detector2.predict(test_sample)

    assert bool(pred1[0]) == bool(pred2[0])
