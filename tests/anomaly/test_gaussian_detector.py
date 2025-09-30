"""Unit tests for NeuralShield anomaly detection using sklearn backend."""

import json
from pathlib import Path

import numpy as np

from neuralshield.anomaly.config import AnomalyDetectionConfig, AnomalyTrainingConfig
from neuralshield.anomaly.model import EllipticEnvelopeDetector
from neuralshield.anomaly.workflow import detect_anomalies, train_anomaly_detector
from neuralshield.encoding.models.base import RequestEncoder
from neuralshield.encoding.models.factory import available_encoders, register_encoder


def test_elliptic_envelope_detector_identifies_outlier() -> None:
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 0.2, size=(64, 2)).astype(np.float32)
    detector = EllipticEnvelopeDetector(contamination=0.1, random_state=0)
    detector.fit(normal)

    on_manifold = np.array([[0.05, -0.08]], dtype=np.float32)
    off_manifold = np.array([[3.0, -3.0]], dtype=np.float32)

    assert bool(detector.predict(on_manifold)[0]) is False
    assert bool(detector.predict(off_manifold)[0]) is True
    assert detector.threshold_ == detector.threshold_  # sanity: threshold exists


if "dummy-length-encoder" not in available_encoders():

    @register_encoder("dummy-length-encoder")
    class _LengthEncoder(RequestEncoder):
        """Return embeddings equal to simple length statistics."""

        def encode(self, batch):
            features = []
            for request in batch:
                length = float(len(request))
                features.append([length, length % 5])
            return np.asarray(features, dtype=np.float32)


def test_training_and_detection_pipeline(tmp_path) -> None:
    dataset_path = tmp_path / "synthetic_dataset.jsonl"
    samples = [{"request": f"GET /ok/{i}{'x' * i}", "label": "valid"} for i in range(8)]
    samples.extend(
        [
            {"request": "POST /suspect?a=1", "label": "attack"},
            {"request": "DELETE /malicious", "label": "attack"},
        ]
    )
    with dataset_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")

    model_path = tmp_path / "detector.joblib"

    train_config = AnomalyTrainingConfig(
        dataset_path=dataset_path,
        model_path=model_path,
        batch_size=4,
        reader_name="jsonl",
        encoder_name="dummy-length-encoder",
        encoder_model_name="unused",
        device="cpu",
        use_pipeline=False,
        contamination=0.1,
        random_state=0,
    )

    detector = train_anomaly_detector(train_config)
    assert model_path.exists()
    assert detector.threshold_ is not None

    detect_config = AnomalyDetectionConfig(
        dataset_path=dataset_path,
        model_path=model_path,
        batch_size=3,
        reader_name="jsonl",
        encoder_name="dummy-length-encoder",
        encoder_model_name="unused",
        device="cpu",
        use_pipeline=False,
        include_scores=True,
        include_requests=True,
        score_greater_is_normal=True,
    )

    batches = list(detect_anomalies(detect_config))
    scored_labels = [label for batch in batches for label in batch.labels]
    assert scored_labels.count("attack") == 2
