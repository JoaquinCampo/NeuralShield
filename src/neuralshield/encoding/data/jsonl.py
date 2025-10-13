from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from loguru import logger

from neuralshield.encoding.data.base import (
    Batch,
    BatchLabels,
    BatchWithLabels,
    DatasetReader,
    PipelineFunc,
)
from neuralshield.encoding.data.factory import register_reader
from neuralshield.encoding.observability import PipelineObserver


@register_reader("jsonl")
class JSONLRequestReader(DatasetReader):
    """Stream batches of HTTP requests from a JSONL file."""

    def __init__(
        self,
        path: Path | str,
        *,
        pipeline: PipelineFunc | None = None,
        use_pipeline: bool = False,
        encoding: str = "utf-8",
        ignore_blank: bool = True,
        observer: PipelineObserver | None = None,
    ) -> None:
        super().__init__(
            path,
            pipeline=pipeline,
            use_pipeline=use_pipeline,
            observer=observer,
        )
        self.encoding = encoding
        self.ignore_blank = ignore_blank

    def iter_batches(self, batch_size: int) -> Iterator[BatchWithLabels]:
        """Yield batches of requests and their labels."""

        requests: Batch = []
        labels: BatchLabels = []

        logger.debug(
            "Reading dataset from {path} batch_size={batch_size}",
            path=str(self.path),
            batch_size=batch_size,
        )

        self._start_new_batch()

        with self.path.open("r", encoding=self.encoding) as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line and self.ignore_blank:
                    continue

                try:
                    payload = json.loads(raw_line)

                    request_text = payload.get("request")
                    label = payload.get("label")
                    processed = self._maybe_process(request_text, label=label)
                    requests.append(processed)
                    labels.append(label)
                except Exception as e:
                    # Skip malformed/empty requests
                    logger.debug(
                        "Skipping malformed request: {error}", error=str(e)[:100]
                    )
                    continue

                if len(requests) >= batch_size:
                    self._finalize_current_batch()
                    self._flush_observer()
                    yield self._yield_payload(requests, labels)
                    self._start_new_batch()
                    requests = []
                    labels = []

        if requests:
            self._finalize_current_batch()
            self._flush_observer()
            yield self._yield_payload(requests, labels)
            self._start_new_batch()

    def _yield_payload(
        self,
        requests: Batch,
        labels: BatchLabels,
    ) -> BatchWithLabels:
        return requests, labels
