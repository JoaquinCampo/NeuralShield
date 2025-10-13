#!/usr/bin/env python3
"""Visualize embeddings to understand model separation of normal vs attack requests."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    embedding_files: list[Path] = typer.Argument(
        ..., help="Paths to embedding .npz files to visualize"
    ),
    output_dir: Path = typer.Option(
        Path("visualizations/embeddings"),
        help="Directory to save visualizations",
    ),
    method: str = typer.Option(
        "umap", help="Dimensionality reduction method: umap, tsne, or pca"
    ),
    n_samples: int = typer.Option(
        5000, help="Number of samples to plot (for performance)"
    ),
    wandb_enabled: bool = typer.Option(
        False, "--wandb/--no-wandb", help="Log to wandb"
    ),
    wandb_project: str = typer.Option("neuralshield", help="Wandb project name"),
    wandb_run_name: str = typer.Option(None, help="Wandb run name (optional)"),
) -> None:
    """Visualize embeddings to see separation between normal and attack requests."""
    # Import dimensionality reduction libraries
    if method == "umap":
        try:
            import umap
        except ImportError:
            logger.error("UMAP not installed. Run: uv pip install umap-learn")
            raise typer.Exit(1)
    elif method == "tsne":
        from sklearn.manifold import TSNE
    elif method == "pca":
        from sklearn.decomposition import PCA
    else:
        logger.error(f"Unknown method: {method}. Use umap, tsne, or pca")
        raise typer.Exit(1)

    # Setup wandb
    if wandb_enabled:
        import wandb

        wandb.init(
            project=wandb_project,
            name=wandb_run_name or f"embedding-viz-{method}",
            job_type="visualization",
        )
        logger.info("Initialized wandb logging")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving visualizations to {output_dir}")

    # Process each embedding file
    for embedding_path in embedding_files:
        logger.info(f"Processing {embedding_path}")

        # Load embeddings
        data = np.load(embedding_path, allow_pickle=True)
        embeddings = data["embeddings"]
        labels = data["labels"]

        logger.info(
            f"Loaded {len(embeddings)} samples with {embeddings.shape[1]} dimensions"
        )

        # Sample for performance
        if len(embeddings) > n_samples:
            logger.info(f"Sampling {n_samples} from {len(embeddings)} samples")
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            embeddings = embeddings[indices]
            labels = labels[indices]

        # Create binary labels for coloring
        # Handle both "normal"/"anomalous" and "valid"/"attack" labels
        is_attack = np.isin(labels, ["attack", "anomalous"])
        label_names = np.where(is_attack, "Attack", "Normal")

        # Count samples
        n_normal = np.sum(~is_attack)
        n_attack = np.sum(is_attack)
        logger.info(f"Normal samples: {n_normal}, Attack samples: {n_attack}")

        # Apply dimensionality reduction
        logger.info(f"Applying {method.upper()} dimensionality reduction...")

        if method == "umap":
            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
            )
            embedding_2d = reducer.fit_transform(embeddings)
        elif method == "tsne":
            reducer = TSNE(
                n_components=2,
                random_state=42,
                perplexity=30,
                metric="cosine",
                init="pca",
            )
            embedding_2d = reducer.fit_transform(embeddings)
        else:  # pca
            reducer = PCA(n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings)

        logger.info("Dimensionality reduction complete")

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot with different colors for normal vs attack
        for label, color, marker in [
            ("Normal", "#2ecc71", "o"),
            ("Attack", "#e74c3c", "^"),
        ]:
            mask = label_names == label
            ax.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=color,
                label=f"{label} (n={np.sum(mask)})",
                alpha=0.6,
                s=50,
                marker=marker,
                edgecolors="white",
                linewidths=0.5,
            )

        # Styling
        model_name = (
            embedding_path.parent.parent.name
        )  # e.g., "02_dense_embeddings_comparison"
        scenario = embedding_path.parent.name  # e.g., "with_preprocessing"
        ax.set_title(
            f"Embedding Visualization: {model_name} ({scenario})\n"
            f"Method: {method.upper()} | Dimensions: {embeddings.shape[1]} → 2",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel(f"{method.upper()} Component 1", fontsize=12)
        ax.set_ylabel(f"{method.upper()} Component 2", fontsize=12)
        ax.legend(loc="best", fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Save figure
        output_filename = f"{embedding_path.stem}_{method}.png"
        output_path = output_dir / output_filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved visualization to {output_path}")

        # Log to wandb
        if wandb_enabled:
            import wandb

            wandb.log(
                {
                    f"embeddings/{model_name}_{scenario}": wandb.Image(fig),
                    f"stats/{model_name}_{scenario}_n_normal": n_normal,
                    f"stats/{model_name}_{scenario}_n_attack": n_attack,
                }
            )
            logger.info("Logged to wandb")

        plt.close(fig)

    logger.info(f"✓ Visualization complete! Saved to {output_dir}")


if __name__ == "__main__":
    app()
