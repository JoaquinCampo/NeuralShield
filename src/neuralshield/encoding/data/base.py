from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Iterator

from neuralshield.encoding.observability import PipelineObserver

PipelineFunc = Callable[[str], str]

Batch = list[str]
BatchLabels = list[str | None]
BatchWithLabels = tuple[Batch, BatchLabels]


class DatasetReader(ABC):
    """Abstract base class for iterable request datasets."""

    def __init__(
        self,
        path: str | Path,
        *,
        pipeline: PipelineFunc | None = None,
        use_pipeline: bool = False,
        observer: PipelineObserver | None = None,
    ) -> None:
        self.path = Path(path)
        self._pipeline = pipeline
        self._use_pipeline = use_pipeline and pipeline is not None
        self._observer = observer

    @abstractmethod
    def iter_batches(self, batch_size: int) -> Iterator[BatchWithLabels]:
        """Yield batches from the underlying dataset."""
        ...

    def _maybe_process(self, request: str, *, label: str | None = None) -> str:
        """Apply preprocessing pipeline to request if enabled.

        Args:
            request: The original request string.
            label: Optional label associated with the request.

        Returns:
            The processed request string, or original if pipeline is disabled.
        """
        if not self._use_pipeline:
            return request

        assert self._pipeline is not None

        processed = self._pipeline(request)

        if self._observer is not None:
            self._observer.record(request, processed, label)

        return processed

    def _start_new_batch(self) -> None:
        if self._observer is not None:
            self._observer.start_batch()

    def _finalize_current_batch(self) -> None:
        if self._observer is not None:
            self._observer.finalize_batch()

    def _flush_observer(self) -> None:
        if self._observer is not None:
            self._observer.flush_samples()

    @property
    def uses_pipeline(self) -> bool:
        """Return whether the reader will invoke a preprocessing pipeline."""
        return self._use_pipeline

    @property
    def pipeline(self) -> PipelineFunc | None:
        """Expose the configured pipeline callable, if any."""
        return self._pipeline
