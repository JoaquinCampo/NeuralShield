from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import typer
from loguru import logger
from projection_head import ProjectionHead, resolve_device

app = typer.Typer(help="Apply the LOF compaction projection head to embeddings.")


def load_config(
    metadata_path: Path | None,
    *,
    hidden_dim: int | None,
    output_dim: int | None,
) -> tuple[int, int]:
    if metadata_path is not None and metadata_path.exists():
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        cfg = data.get("config", {})
        resolved_hidden = int(cfg.get("hidden_dim", hidden_dim or 512))
        resolved_output = int(cfg.get("output_dim", output_dim or 256))
        return resolved_hidden, resolved_output
    if hidden_dim is None or output_dim is None:
        raise ValueError("Provide --hidden-dim/--output-dim when metadata is missing.")
    return hidden_dim, output_dim


@app.command()
def main(
    embeddings_path: Path = typer.Argument(..., help="Input embeddings (.npz)."),
    model_path: Path = typer.Option(
        Path("models/contrastive/lof_compaction_head.pt"),
        help="Path to trained projection head weights.",
    ),
    output_path: Path = typer.Option(
        None,
        help="Optional destination for transformed embeddings (.npz).",
    ),
    metadata_path: Path | None = typer.Option(
        None,
        help="Metadata JSON used to infer projection dimensions.",
    ),
    hidden_dim: int | None = typer.Option(
        None,
        help="Projection hidden dimension when metadata missing.",
    ),
    output_dim: int | None = typer.Option(
        None,
        help="Projection output dimension when metadata missing.",
    ),
    device: str = typer.Option("auto", help="Device to run inference on."),
) -> None:
    embeddings_path = embeddings_path.resolve()
    if output_path is None:
        output_path = embeddings_path.with_name(f"{embeddings_path.stem}_compact.npz")

    logger.info(
        "Applying projection",
        embeddings=str(embeddings_path),
        model=str(model_path),
        output=str(output_path),
    )

    with np.load(embeddings_path, allow_pickle=True) as data:
        embeddings = data["embeddings"].astype(np.float32, copy=False)
        labels = data["labels"]

    hidden_dim_resolved, output_dim_resolved = load_config(
        metadata_path,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    input_dim = embeddings.shape[1]
    device_t = resolve_device(device)

    projector = ProjectionHead(input_dim, hidden_dim_resolved, output_dim_resolved)
    state = torch.load(model_path, map_location=device_t)
    projector.load_state_dict(state)
    projector.to(device_t)
    projector.eval()

    with torch.no_grad():
        tensor = torch.from_numpy(embeddings).to(device_t)
        projected = projector(tensor)
        normalized = F.normalize(projected, dim=1)
        projected_np = normalized.cpu().numpy().astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, embeddings=projected_np, labels=labels)
    logger.info(
        "Saved compacted embeddings",
        path=str(output_path),
        samples=int(projected_np.shape[0]),
        dim=int(projected_np.shape[1]),
    )


if __name__ == "__main__":
    app()
