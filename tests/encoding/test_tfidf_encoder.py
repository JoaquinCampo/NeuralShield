from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralshield.encoding.models import TFIDFEncoder
from neuralshield.encoding.models.factory import available_encoders
from neuralshield.encoding.models.tfidf_settings import TFIDFEncoderConfig


def test_tfidf_encoder_registered() -> None:
    assert "tfidf" in available_encoders()


@pytest.mark.parametrize(
    "corpus",
    [
        ["hello world", "intrusion attempt"],
        ["simple line"],
    ],
)
def test_tfidf_encoder_round_trip(tmp_path: Path, corpus: list[str]) -> None:
    settings = TFIDFEncoderConfig(max_features=8, ngram_range=(1, 2))
    encoder = TFIDFEncoder(settings=settings)

    embeddings = encoder.encode(corpus)
    assert embeddings.dtype == np.float32
    assert embeddings.shape[0] == len(corpus)

    vectorizer_path = tmp_path / "vectorizer.joblib"
    encoder.save(vectorizer_path)
    assert vectorizer_path.exists()

    reloaded = TFIDFEncoder(model_name=str(vectorizer_path), settings=settings)
    transformed = reloaded.encode([corpus[0]])
    assert transformed.dtype == np.float32
    assert transformed.shape[1] == embeddings.shape[1]
