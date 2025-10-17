from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from loguru import logger
from projection_head import ProjectionHead, resolve_device
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

app = typer.Typer(help="Train projection head for LOF cluster compaction.")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_embeddings(
    path: Path,
    *,
    normal_labels: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        embeddings = data["embeddings"]
        labels = data["labels"]
    mask = np.isin(labels, list(normal_labels))
    filtered = embeddings[mask]
    filtered_labels = labels[mask]
    logger.info(
        "Loaded embeddings",
        path=str(path),
        total=int(labels.shape[0]),
        normals=int(filtered.shape[0]),
    )
    return filtered, filtered_labels


def build_knn_graph(
    embeddings: np.ndarray,
    *,
    neighbors: int,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    if neighbors <= 0:
        raise ValueError("neighbors must be > 0")
    nn_model = NearestNeighbors(n_neighbors=neighbors + 1, metric=metric)
    nn_model.fit(embeddings)
    distances, indices = nn_model.kneighbors(embeddings, return_distance=True)
    # Drop the first neighbour (self)
    neighbour_ids = indices[:, 1:]
    neighbour_distances = distances[:, 1:]
    logger.info(
        "Computed kNN graph",
        samples=int(embeddings.shape[0]),
        neighbours=int(neighbour_ids.shape[1]),
        metric=metric,
        mean_radius=float(neighbour_distances.mean()),
    )
    return neighbour_ids, neighbour_distances


class NeighborPairDataset(Dataset):
    def __init__(
        self,
        embeddings: np.ndarray,
        neighbour_indices: np.ndarray,
        *,
        active_mask: np.ndarray | None = None,
    ) -> None:
        self._embeddings = torch.from_numpy(embeddings).float()
        self._neighbours = neighbour_indices
        self._pairs_per_anchor = neighbour_indices.shape[1]
        if self._pairs_per_anchor == 0:
            raise ValueError("Neighbour set is empty; increase --neighbours.")
        if active_mask is not None:
            active_indices = np.where(active_mask)[0]
            if active_indices.size == 0:
                active_indices = np.arange(neighbour_indices.shape[0])
        else:
            active_indices = np.arange(neighbour_indices.shape[0])
        self._active_indices = active_indices

    def __len__(self) -> int:
        return self._active_indices.size * self._pairs_per_anchor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_pos, neighbour_offset = divmod(idx, self._pairs_per_anchor)
        anchor_idx = int(self._active_indices[anchor_pos])
        anchor = self._embeddings[anchor_idx]
        positive_idx = int(self._neighbours[anchor_idx, neighbour_offset])
        positive = self._embeddings[positive_idx]
        return anchor, positive


def nt_xent_loss(z: torch.Tensor, temperature: float) -> torch.Tensor:
    batch_size = z.shape[0] // 2
    if batch_size * 2 != z.shape[0]:
        raise ValueError("Batch size must be even for NT-Xent loss.")

    z = F.normalize(z, dim=1)
    similarity = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, -1e9)

    positives = torch.arange(batch_size, device=z.device)
    targets = torch.cat((positives + batch_size, positives), dim=0)
    loss = F.cross_entropy(similarity, targets)
    return loss


def compute_mean_radius(
    embeddings: np.ndarray,
    neighbour_indices: np.ndarray,
    *,
    projector: nn.Module | None = None,
    device: torch.device | None = None,
) -> float:
    if embeddings.size == 0:
        return 0.0
    with torch.no_grad():
        tensor = torch.from_numpy(embeddings).float()
        if projector is not None:
            assert device is not None
            tensor = tensor.to(device)
            projected = projector(tensor)
            projected = F.normalize(projected, dim=1)
            tensor = projected.cpu()
        else:
            tensor = F.normalize(tensor, dim=1)
    anchored = tensor.numpy()
    neighbours = anchored[neighbour_indices]
    deltas = neighbours - anchored[:, None, :]
    norms = np.linalg.norm(deltas, axis=-1)
    return float(norms.mean())


@dataclass
class TrainingConfig:
    train_embeddings: Path
    output_path: Path
    neighbours: int = 20
    batch_size: int = 256
    epochs: int = 20
    temperature: float = 0.2
    lr: float = 1e-3
    val_split: float = 0.1
    seed: int = 42
    hidden_dim: int = 512
    output_dim: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    metric: str = "cosine"
    wandb: bool = False
    min_radius_percentile: float = 0.3
    variance_weight: float = 0.05


def select_active_anchors(
    mean_radii: np.ndarray,
    *,
    min_percentile: float,
) -> tuple[np.ndarray, float]:
    if min_percentile <= 0.0:
        mask = np.ones_like(mean_radii, dtype=bool)
        return mask, float(mean_radii.mean())

    percentile = np.clip(min_percentile * 100.0, 0.0, 100.0)
    threshold = float(np.percentile(mean_radii, percentile))
    mask = mean_radii >= threshold
    if not np.any(mask):
        mask = np.ones_like(mean_radii, dtype=bool)
    selected_mean = (
        float(mean_radii[mask].mean()) if np.any(mask) else float(mean_radii.mean())
    )
    return mask, selected_mean


def log_distance_metrics(
    *,
    train_embeddings: np.ndarray,
    train_neighbours: np.ndarray,
    val_embeddings: np.ndarray,
    val_neighbours: np.ndarray,
    projector: nn.Module | None,
    device: torch.device,
    prefix: str,
    log_to_wandb: bool = False,
) -> dict[str, float]:
    train_radius = compute_mean_radius(
        train_embeddings,
        train_neighbours,
        projector=projector,
        device=device,
    )
    val_radius = compute_mean_radius(
        val_embeddings,
        val_neighbours,
        projector=projector,
        device=device,
    )
    metrics = {
        f"{prefix}/train_mean_radius": train_radius,
        f"{prefix}/val_mean_radius": val_radius,
    }
    logger.info(
        "{stage} mean neighbour radius",
        stage=prefix,
        train_radius=train_radius,
        val_radius=val_radius,
    )
    if log_to_wandb:
        try:
            import wandb

            wandb.log(metrics)
        except Exception as exc:  # pragma: no cover - optional logging
            logger.warning("Failed to log metrics to wandb: {exc}", exc=str(exc))
    return metrics


@app.command()
def main(
    train_embeddings: Path = typer.Option(
        Path("embeddings/SecBert/train_embeddings.npz"),
        help="Path to SecBERT training embeddings (.npz).",
    ),
    output_path: Path = typer.Option(
        Path("models/contrastive/lof_compaction_head.pt"),
        help="Destination for the trained projection head.",
    ),
    neighbours: int = typer.Option(20, help="Number of kNN neighbours per anchor."),
    batch_size: int = typer.Option(
        256,
        help="Total samples per NT-Xent step (must be even).",
    ),
    epochs: int = typer.Option(20, help="Training epochs."),
    temperature: float = typer.Option(0.2, help="NT-Xent temperature."),
    lr: float = typer.Option(1e-3, help="Learning rate."),
    val_split: float = typer.Option(
        0.1,
        help="Fraction of normal embeddings reserved for validation.",
    ),
    seed: int = typer.Option(42, help="Random seed."),
    hidden_dim: int = typer.Option(512, help="Projection head hidden dimension."),
    output_dim: int = typer.Option(256, help="Projection head output dimension."),
    metric: str = typer.Option("cosine", help="Distance metric for kNN graph."),
    device: str = typer.Option("auto", help="Device to use (auto, cpu, cuda, mps)."),
    wandb: bool = typer.Option(
        False,
        "--wandb/--no-wandb",
        help="Enable Weights & Biases logging.",
    ),
    min_radius_percentile: float = typer.Option(
        0.3,
        help="Keep anchors whose mean neighbour radius is above this percentile (0-1).",
    ),
    variance_weight: float = typer.Option(
        0.05,
        help="Weight for variance preservation penalty.",
    ),
) -> None:
    config = TrainingConfig(
        train_embeddings=train_embeddings,
        output_path=output_path,
        neighbours=neighbours,
        batch_size=batch_size,
        epochs=epochs,
        temperature=temperature,
        lr=lr,
        val_split=val_split,
        seed=seed,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        metric=metric,
        wandb=wandb,
        device=device,
        min_radius_percentile=min_radius_percentile,
        variance_weight=variance_weight,
    )

    set_global_seed(config.seed)

    resolved_device = resolve_device(config.device)

    logger.info("Using device {device}", device=str(resolved_device))

    normal_labels = {"valid", "normal"}
    embeddings, labels = load_embeddings(
        config.train_embeddings,
        normal_labels=normal_labels,
    )
    if embeddings.shape[0] < 2:
        raise ValueError("Need at least two normal embeddings for training.")

    n_samples = embeddings.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_idx = int(math.floor(n_samples * (1.0 - config.val_split)))
    split_idx = max(split_idx, 1)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    train_embeddings = embeddings[train_idx]
    val_embeddings = embeddings[val_idx] if val_idx.size else embeddings[train_idx]

    train_neighbours, train_distances = build_knn_graph(
        train_embeddings,
        neighbors=config.neighbours,
        metric=config.metric,
    )
    val_neighbours, val_distances = build_knn_graph(
        val_embeddings,
        neighbors=config.neighbours,
        metric=config.metric,
    )

    train_mean_radii = train_distances.mean(axis=1)
    val_mean_radii = val_distances.mean(axis=1)
    active_mask, selected_mean_radius = select_active_anchors(
        train_mean_radii,
        min_percentile=config.min_radius_percentile,
    )
    logger.info(
        "Anchor selection",
        total_anchors=int(train_mean_radii.size),
        active_anchors=int(active_mask.sum()),
        active_fraction=float(active_mask.mean()),
        selected_mean_radius=selected_mean_radius,
    )

    dataset = NeighborPairDataset(
        train_embeddings,
        train_neighbours,
        active_mask=active_mask,
    )
    pairs_per_batch = max(config.batch_size // 2, 1)
    loader = DataLoader(
        dataset,
        batch_size=pairs_per_batch,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    input_dim = train_embeddings.shape[1]
    projector = ProjectionHead(
        input_dim,
        config.hidden_dim,
        config.output_dim,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=config.lr,
        weight_decay=1e-5,
    )

    if config.wandb:
        try:
            import wandb

            wandb.init(
                project="neuralshield",
                name="lof-cluster-compaction",
                config=asdict(config),
            )
        except Exception as exc:  # pragma: no cover - optional logging
            logger.warning("Failed to initialise wandb: {exc}", exc=str(exc))

    log_distance_metrics(
        train_embeddings=train_embeddings,
        train_neighbours=train_neighbours,
        val_embeddings=val_embeddings,
        val_neighbours=val_neighbours,
        projector=None,
        device=resolved_device,
        prefix="baseline",
        log_to_wandb=config.wandb,
    )

    for epoch in range(1, config.epochs + 1):
        projector.train()
        running_loss = 0.0
        total_steps = 0

        for anchors, positives in loader:
            anchors = anchors.to(resolved_device)
            positives = positives.to(resolved_device)
            views = torch.cat([anchors, positives], dim=0)
            projected = projector(views)
            contrastive_loss = nt_xent_loss(projected, config.temperature)

            if config.variance_weight > 0:
                variance_penalty = (
                    (projected.var(dim=0, unbiased=False) - 1.0) ** 2
                ).mean()
                loss = contrastive_loss + config.variance_weight * variance_penalty
            else:
                variance_penalty = torch.tensor(0.0, device=resolved_device)
                loss = contrastive_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

            if config.wandb:
                try:
                    import wandb

                    wandb.log(
                        {
                            "train/contrastive_loss": float(contrastive_loss.item()),
                            "train/variance_penalty": float(variance_penalty.item()),
                        },
                        commit=False,
                    )
                except Exception as exc:  # pragma: no cover
                    logger.warning(
                        "Failed to log step metrics to wandb: {exc}",
                        exc=str(exc),
                    )

            total_steps += 1

        avg_loss = running_loss / max(total_steps, 1)
        logger.info(
            "Epoch {epoch}/{total} loss={loss:.6f}",
            epoch=epoch,
            total=config.epochs,
            loss=avg_loss,
        )

        projector.eval()
        log_distance_metrics(
            train_embeddings=train_embeddings,
            train_neighbours=train_neighbours,
            val_embeddings=val_embeddings,
            val_neighbours=val_neighbours,
            projector=projector,
            device=resolved_device,
            prefix=f"epoch_{epoch}",
            log_to_wandb=config.wandb,
        )
        if config.wandb:
            try:
                import wandb

                wandb.log({"train/loss": avg_loss})
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to log epoch loss to wandb: {exc}", exc=str(exc))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(projector.state_dict(), output_path)
    logger.info("Saved projection head to {path}", path=str(output_path))

    metadata = {
        "config": asdict(config),
        "train_samples": int(train_embeddings.shape[0]),
        "val_samples": int(val_embeddings.shape[0]),
        "train_active_fraction": float(active_mask.mean()),
        "train_mean_radius_selected": selected_mean_radius,
        "train_mean_radius_all": float(train_mean_radii.mean()),
        "val_mean_radius_all": float(val_mean_radii.mean()),
        "final_metrics": log_distance_metrics(
            train_embeddings=train_embeddings,
            train_neighbours=train_neighbours,
            val_embeddings=val_embeddings,
            val_neighbours=val_neighbours,
            projector=projector,
            device=resolved_device,
            prefix="final",
            log_to_wandb=False,
        ),
        "label_distribution": {
            "normal": int(train_embeddings.shape[0] + val_embeddings.shape[0]),
        },
    }

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_payload = json.dumps(metadata, indent=2, default=str)
    metadata_path.write_text(metadata_payload, encoding="utf-8")
    logger.info("Saved metadata to {path}", path=str(metadata_path))

    if config.wandb:
        try:
            import wandb

            wandb.finish()
        except Exception:  # pragma: no cover
            pass


if __name__ == "__main__":
    app()
