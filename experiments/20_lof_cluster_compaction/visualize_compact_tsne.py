from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from sklearn.manifold import TSNE

NORMAL_LABELS = {"valid", "normal"}
ATTACK_LABELS = {"attack", "anomaly"}

app = typer.Typer(help="Visualize compacted SecBERT embeddings (train + test) with t-SNE.")


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32, copy=False)
    labels = data["labels"]
    logger.info(
        "Loaded embeddings",
        path=str(path),
        samples=int(embeddings.shape[0]),
        dim=int(embeddings.shape[1]),
    )
    return embeddings, labels


@app.command()
def main(
    train_path: Path = typer.Option(
        Path("embeddings/SecBert/train_embeddings_compact.npz"),
        help="Compacted training embeddings (.npz).",
    ),
    test_path: Path = typer.Option(
        Path("embeddings/SecBert/test_embeddings_compact.npz"),
        help="Compacted evaluation embeddings (.npz).",
    ),
    output_path: Path = typer.Option(
        Path("visualizations/embeddings/secbert_compact_tsne.png"),
        help="Destination image for the visualization.",
    ),
    perplexity: float = typer.Option(
        30.0,
        help="Perplexity parameter for t-SNE.",
    ),
    random_state: int = typer.Option(
        42,
        help="Random seed for reproducibility.",
    ),
    title: str = typer.Option(
        "t-SNE: SecBERT Embeddings Compactados (Train + Test)",
        help="Title for the visualization plot.",
    ),
) -> None:
    train_embeddings, train_labels = load_embeddings(train_path)
    test_embeddings, test_labels = load_embeddings(test_path)

    embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)
    dataset_tags = np.concatenate(
        [np.full(train_labels.shape, "train"), np.full(test_labels.shape, "test")]
    )

    logger.info(
        "Running t-SNE on combined embeddings",
        total_samples=int(embeddings.shape[0]),
        perplexity=perplexity,
    )
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        metric="cosine",
        init="pca",
    )
    coords = tsne.fit_transform(embeddings)
    logger.info("t-SNE complete")

    is_attack = np.isin(labels, list(ATTACK_LABELS))

    fig, ax = plt.subplots(figsize=(12, 10))

    for dataset, marker in [("train", "o"), ("test", "^")]:
        mask_dataset = dataset_tags == dataset
        for label_name, color, is_positive in [
            ("Normal", "#2ecc71", False),
            ("Attack", "#e74c3c", True),
        ]:
            label_mask = is_attack if is_positive else ~is_attack
            mask = mask_dataset & label_mask
            if not np.any(mask):
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                label=f"{label_name} ({dataset}) n={int(mask.sum())}",
                alpha=0.6,
                s=40,
                marker=marker,
                linewidths=0.3,
                edgecolors="white",
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved visualization to {path}", path=str(output_path))


if __name__ == "__main__":
    app()
