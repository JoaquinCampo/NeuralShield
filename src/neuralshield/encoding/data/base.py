from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Iterator

import wandb

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
    ) -> None:
        self.path = Path(path)
        self._pipeline = pipeline
        self._use_pipeline = use_pipeline and pipeline is not None
        self._wandb_enabled = wandb.run is not None
        self._wandb_table: wandb.Table | None = None

    @abstractmethod
    def iter_batches(self, batch_size: int) -> Iterator[BatchWithLabels]:
        """Yield batches from the underlying dataset."""
        ...

    def _maybe_process(self, request: str, *, label: str | None = None) -> str:
        if not self._use_pipeline:
            return request
        assert self._pipeline is not None
        processed = self._pipeline(request)
        if self._wandb_enabled and request is not None:
            if self._wandb_table is None:
                self._wandb_table = wandb.Table(
                    columns=["original_request", "processed_request", "label"]
                )
            self._wandb_table.add_data(request, processed, label)
        return processed

    def _flush_wandb_table(self) -> None:
        if not self._wandb_enabled or self._wandb_table is None:
            return
        wandb.log({"pipeline/requests": self._wandb_table})
        self._wandb_table = wandb.Table(
            columns=["original_request", "processed_request", "label"]
        )

    @property
    def uses_pipeline(self) -> bool:
        """Return whether the reader will invoke a preprocessing pipeline."""
        return self._use_pipeline

    @property
    def pipeline(self) -> PipelineFunc | None:
        """Expose the configured pipeline callable, if any."""
        return self._pipeline
