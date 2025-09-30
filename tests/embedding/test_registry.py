"""Tests for registry decorator behavior."""

import numpy as np
from typing import Sequence

from neuralshield.encoding.data.base import DatasetReader
from neuralshield.encoding.data.jsonl import JSONLRequestReader
from neuralshield.encoding.data.factory import get_reader, register_reader
from neuralshield.encoding.models.base import RequestEncoder
from neuralshield.encoding.models.factory import get_encoder, register_encoder


def test_reader_registration_decorator_allows_custom_class():
    @register_reader("custom-jsonl")
    class CustomReader(JSONLRequestReader):
        pass

    factory = get_reader("custom-jsonl")
    assert issubclass(factory, DatasetReader)


def test_encoder_registration_allows_lookup():
    @register_encoder("dummy")
    class DummyEncoder(RequestEncoder):
        def encode(self, batch: Sequence[str]) -> np.ndarray:
            length = len(batch)
            return np.zeros((length, 1), dtype=np.float32)

    factory = get_encoder("dummy")
    enc = factory()
    assert isinstance(enc, DummyEncoder)
