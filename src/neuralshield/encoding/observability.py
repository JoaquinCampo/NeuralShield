from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableSequence, Sequence

import numpy as np

import wandb


class WandbSink:
    """Adapter that forwards metrics and tables to Weights & Biases."""

    def log(self, data: Mapping[str, Any]) -> None:
        wandb.log(dict(data))

    def log_table(
        self,
        name: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[Any]],
    ) -> None:
        table = wandb.Table(columns=list(columns))
        for row in rows:
            table.add_data(*row)
        wandb.log({name: table})


@dataclass
class PipelineObserver:
    """Collect preprocessing samples for observability."""

    sink: WandbSink | None = None
    sample_interval: int = 10
    table_name: str = "pipeline/requests"
    table_columns: tuple[str, ...] = (
        "original_request",
        "processed_request",
        "label",
    )

    _request_samples: MutableSequence[Sequence[Any]] = field(default_factory=list)
    _sample_counter: int = 0

    def start_batch(self) -> None:
        """Hook to reset any per-batch state (no-op by default)."""

    def record(
        self,
        original: str | None,
        processed: str,
        label: str | None,
    ) -> None:
        """Record a single preprocessing event."""

        if self.sink is None or original is None:
            return

        interval = self.sample_interval if self.sample_interval > 0 else 1
        if self._sample_counter % interval == 0:
            label_value = "" if label is None else str(label)
            self._request_samples.append((original, processed, label_value))
        self._sample_counter += 1

    def finalize_batch(self) -> None:
        """Hook invoked when a batch completes (no-op by default)."""
        return None

    def flush_samples(self) -> None:
        """Persist queued request samples through the configured sink."""

        if self.sink is None or not self._request_samples:
            return

        self.sink.log_table(
            self.table_name,
            self.table_columns,
            list(self._request_samples),
        )
        self._request_samples.clear()


@dataclass
class EncodingMetricsTracker:
    """Aggregate and emit encoding metrics for batches and the full run."""

    sink: WandbSink | None = None
    total_batches: int = 0
    total_requests: int = 0
    total_encode_seconds: float = 0.0

    def record_batch(
        self,
        *,
        requests: Sequence[str],
        embeddings: np.ndarray,
        encode_seconds: float,
    ) -> Mapping[str, float]:
        batch_size = len(requests)
        request_lengths = np.fromiter(
            (len(request or "") for request in requests), dtype=float, count=batch_size
        )
        embedding_norms = np.linalg.norm(embeddings, axis=1)

        norm_mean = float(embedding_norms.mean()) if embedding_norms.size else 0.0
        norm_std = float(embedding_norms.std()) if embedding_norms.size else 0.0
        length_mean = float(request_lengths.mean()) if request_lengths.size else 0.0
        length_std = float(request_lengths.std()) if request_lengths.size else 0.0

        metrics = {
            "batch/size": float(batch_size),
            "batch/request_length_mean": length_mean,
            "batch/request_length_std": length_std,
            "embedding/norm_mean": norm_mean,
            "embedding/norm_std": norm_std,
            "latency/encode_ms": float(encode_seconds * 1000.0),
        }

        self.total_batches += 1
        self.total_requests += batch_size
        self.total_encode_seconds += encode_seconds

        if self.sink is not None:
            self.sink.log(metrics)

        return metrics

    def finalize(self) -> Mapping[str, float]:
        avg_batch_ms = (
            self.total_encode_seconds / self.total_batches * 1000.0
            if self.total_batches
            else 0.0
        )
        avg_request_ms = (
            self.total_encode_seconds / self.total_requests * 1000.0
            if self.total_requests
            else 0.0
        )

        summary = {
            "run/total_batches": float(self.total_batches),
            "run/total_requests": float(self.total_requests),
            "run/encode_total_ms": float(self.total_encode_seconds * 1000.0),
            "run/encode_avg_batch_ms": float(avg_batch_ms),
            "run/encode_avg_request_ms": float(avg_request_ms),
        }

        if self.sink is not None:
            self.sink.log(summary)

        return summary


def init_wandb_sink(
    enable: bool,
    *,
    project: str,
    entity: str | None,
    config: Mapping[str, Any],
):
    """Return a `(sink, run)` pair based on the requested W&B settings."""

    if not enable:
        return None, None

    run = wandb.init(project=project, entity=entity, config=dict(config))
    return WandbSink(), run
