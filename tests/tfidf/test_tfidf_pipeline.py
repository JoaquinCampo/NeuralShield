from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.tfidf import (
    TFIDFAnomalyTrainingConfig,
    TFIDFEncodingConfig,
    dump_tfidf_embeddings,
    train_anomaly_from_embeddings,
)


def _write_dataset(path: Path) -> None:
    samples = [
        {"request": "GET /index HTTP/1.1", "label": "valid"},
        {"request": "POST /login HTTP/1.1", "label": "valid"},
        {"request": "GET /etc/passwd HTTP/1.1", "label": "attack"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")


def test_dump_and_train_pipeline(tmp_path: Path) -> None:
    dataset_path = tmp_path / "requests.jsonl"
    embeddings_path = tmp_path / "embeddings.npz"
    vectorizer_path = tmp_path / "vectorizer.joblib"
    model_path = tmp_path / "detector.joblib"

    _write_dataset(dataset_path)

    dump_config = TFIDFEncodingConfig(
        dataset_path=dataset_path,
        embeddings_path=embeddings_path,
        vectorizer_path=vectorizer_path,
        batch_size=2,
        use_pipeline=False,
    )

    dump_tfidf_embeddings(dump_config)

    assert embeddings_path.exists()
    assert vectorizer_path.exists()

    payload = np.load(embeddings_path, allow_pickle=True)
    embeddings = payload["embeddings"]
    labels = payload["labels"]
    assert embeddings.shape[0] == len(labels) == 3

    train_config = TFIDFAnomalyTrainingConfig(
        embeddings_path=embeddings_path,
        model_path=model_path,
        valid_label="valid",
        contamination=0.1,
        show_progress=False,
    )

    detector = train_anomaly_from_embeddings(train_config)

    assert model_path.exists()
    assert detector.threshold_ is not None
