#!/usr/bin/env python3
"""Visualize TF-IDF PCA embeddings via t-SNE or UMAP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from sklearn.manifold import TSNE
from umap import UMAP  # type: ignore

ColorMode = Literal["by_label", "density"]

app = typer.Typer(help="Plot 2D projections of TF-IDF PCA embeddings.")


def _load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    labels = data["labels"].astype(str)
    logger.info("Loaded {} samples from {}", embeddings.shape[0], path)
    return embeddings, labels


def _project(
    embeddings: np.ndarray,
    method: Literal["tsne", "umap"],
    random_state: int,
    sample_size: int | None,
) -> np.ndarray:
    if sample_size is not None and embeddings.shape[0] > sample_size:
        logger.info("Subsampling to {} points for visualization", sample_size)
        rng = np.random.default_rng(random_state)
        idx = rng.choice(embeddings.shape[0], sample_size, replace=False)
        embeddings = embeddings[idx]
    else:
        idx = None

    if method == "tsne":
        projector = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=30,
            init="pca",
            learning_rate="auto",
        )
    else:
        projector = UMAP(n_components=2, random_state=random_state)

    coords = projector.fit_transform(embeddings)
    return coords if idx is None else coords, idx


def _scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(8, 6))
    mask_attack = labels == "attack"
    plt.scatter(
        coords[~mask_attack, 0],
        coords[~mask_attack, 1],
        s=5,
        alpha=0.5,
        label="valid",
        color="#1f77b4",
    )
    plt.scatter(
        coords[mask_attack, 0],
        coords[mask_attack, 1],
        s=5,
        alpha=0.5,
        label="attack",
        color="#d62728",
    )
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved figure to {}", output_path)


@app.command()
def main(
    train_embeddings_path: Path = typer.Argument(
        ...,
        help="Path to TF-IDF PCA train embeddings (.npz) without labels.",
    ),
    test_embeddings_path: Path = typer.Argument(
        ...,
        help="Path to TF-IDF PCA test embeddings (.npz) with labels.",
    ),
    output_dir: Path = typer.Argument(
        Path("visualizations/tfidf_pca"),
        help="Directory to store plots.",
    ),
    method: Literal["tsne", "umap"] = typer.Option("tsne", help="Projection method"),
    random_state: int = typer.Option(42, help="Random state for reproducibility"),
    sample_size: int | None = typer.Option(None, help="Optional subsample size"),
) -> None:
    test_embeddings, test_labels = _load_embeddings(test_embeddings_path)
    coords, idx = _project(test_embeddings, method, random_state, sample_size)
    labels = test_labels if idx is None else test_labels[idx]

    plot_path = output_dir / f"csic_tfidf_pca_{method}.png"
    _scatter(coords, labels, plot_path, f"CSIC TF-IDF PCA ({method.upper()})")

    meta = {
        "method": method,
        "random_state": random_state,
        "sample_size": sample_size if sample_size else len(labels),
        "input_embeddings": str(test_embeddings_path),
        "output_image": str(plot_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info("Metadata saved to {}", output_dir / "metadata.json")


if __name__ == "__main__":
    app()
