from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Two-layer projection head used for LOF cluster compaction."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def resolve_device(preferred: str) -> torch.device:
    """Select the best available torch device given a preference string."""

    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        has_mps = getattr(torch.backends, "mps", None)
        if has_mps is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(preferred)
