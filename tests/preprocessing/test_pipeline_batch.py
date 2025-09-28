"""Tests for PreprocessorPipeline batching."""

from neuralshield.preprocessing.pipeline import PreprocessorPipeline


class DummyStep:
    def __init__(self, suffix: str) -> None:
        self.suffix = suffix

    def process(self, request: str) -> str:
        return f"{request}{self.suffix}"


def test_pipeline_single() -> None:
    pipeline = PreprocessorPipeline([DummyStep("A"), DummyStep("B")])
    assert pipeline("x") == "xAB"


def test_pipeline_batch_preserves_order() -> None:
    pipeline = PreprocessorPipeline([DummyStep("-"), DummyStep("!")])
    assert pipeline.batch(["foo", "bar"]) == ["foo-!", "bar-!"]


def test_pipeline_batch_empty() -> None:
    pipeline = PreprocessorPipeline([DummyStep("#")])
    assert pipeline.batch([]) == []


def test_pipeline_batch_single() -> None:
    pipeline = PreprocessorPipeline([DummyStep("#")])
    assert pipeline.batch(["solo"]) == ["solo#"]
