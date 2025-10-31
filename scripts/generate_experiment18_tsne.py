#!/usr/bin/env python3
"""Create t-SNE visualisations for the SecBERT embeddings used in experiment 18."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]

app = typer.Typer(
    help="Generate side-by-side t-SNE plots for experiment 18 embeddings."
)


@dataclass(frozen=True)
class DatasetConfig:
    """Hold candidate locations for a dataset's embeddings."""

    name: str
    with_candidates: tuple[Path, ...]
    without_candidates: tuple[Path, ...]


DEFAULT_DATASETS: dict[str, DatasetConfig] = {
    "pkdd": DatasetConfig(
        name="pkdd",
        with_candidates=(
            ROOT
            / "experiments/18_lof_secbert_ensemble"
            / "pkdd"
            / "with_preprocessing"
            / "secbert_test_embeddings.npz",
            ROOT
            / "experiments/18_lof_secbert_ensemble"
            / "pkdd"
            / "with_preprocessing"
            / "secbert_train_embeddings.npz",
        ),
        without_candidates=(
            ROOT
            / "experiments/18_lof_secbert_ensemble"
            / "pkdd"
            / "without_preprocessing"
            / "secbert_test_embeddings.npz",
            ROOT
            / "experiments/18_lof_secbert_ensemble"
            / "pkdd"
            / "without_preprocessing"
            / "secbert_train_embeddings.npz",
        ),
    ),
    "csic": DatasetConfig(
        name="csic",
        with_candidates=(
            ROOT
            / "experiments/18_lof_secbert_ensemble"
            / "csic_with_preprocessing"
            / "secbert_test_embeddings.npz",
            ROOT
            / "experiments/18_lof_secbert_ensemble"
            / "csic_with_preprocessing"
            / "secbert_train_embeddings.npz",
            ROOT
            / "experiments/03_secbert_comparison"
            / "secbert_with_preprocessing"
            / "csic_test_embeddings_converted.npz",
            ROOT
            / "experiments/03_secbert_comparison"
            / "secbert_with_preprocessing"
            / "csic_train_embeddings.npz",
        ),
        without_candidates=(
            ROOT
            / "experiments/18_lof_secbert_ensemble"
            / "csic_without_preprocessing"
            / "secbert_test_embeddings.npz",
            ROOT
            / "experiments/18_lof_secbert_ensemble"
            / "csic_without_preprocessing"
            / "secbert_train_embeddings.npz",
            ROOT
            / "experiments/03_secbert_comparison"
            / "secbert_without_preprocessing"
            / "csic_test_embeddings_converted.npz",
            ROOT
            / "experiments/03_secbert_comparison"
            / "secbert_without_preprocessing"
            / "csic_train_embeddings.npz",
        ),
    ),
}


def _resolve_path(
    candidates: tuple[Path, ...], dataset: str, variant: str
) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            if candidate != candidates[0]:
                logger.info(
                    "Using fallback embeddings for {} {}: {}",
                    dataset,
                    variant,
                    candidate,
                )
            return candidate
    logger.warning(
        "No embeddings found for {} {}. Checked: {}",
        dataset,
        variant,
        [str(path) for path in candidates],
    )
    return None


def _load_json_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    embeddings_list: list[list[float]] = []
    labels_list: list[str] = []

    with path.open() as handle:
        header = handle.readline()
        if header.strip():
            try:
                json.loads(header)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse metadata in {path}") from exc
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            embeddings_list.extend(payload["embeddings"])
            labels_list.extend(payload["labels"])

    embeddings = np.asarray(embeddings_list, dtype=np.float32)
    labels = np.asarray(labels_list)
    return embeddings, labels


def _load_embeddings(
    path: Path, sample_size: int | None, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        data = np.load(path, allow_pickle=True)
        embeddings = data["embeddings"].astype(np.float32)
        labels = data["labels"]
    except (ValueError, OSError, KeyError, AttributeError, pickle.UnpicklingError):
        logger.info("Falling back to JSON loader for {}", path)
        embeddings, labels = _load_json_embeddings(path)

    if sample_size and embeddings.shape[0] > sample_size:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(embeddings.shape[0], sample_size, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    return embeddings, labels


def _compute_tsne(
    embeddings: np.ndarray, perplexity: float, max_iter: int, random_state: int
) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        init="random",
        random_state=random_state,
        perplexity=perplexity,
        learning_rate="auto",
        max_iter=max_iter,
        metric="cosine",
    )
    return tsne.fit_transform(embeddings)


def _build_plot(
    dataset: str,
    with_coords: np.ndarray,
    with_labels: np.ndarray,
    without_coords: np.ndarray,
    without_labels: np.ndarray,
    output_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    palettes = {"valid": "#2ca02c", "attack": "#d62728"}

    with_df = pd.DataFrame(
        {
            "x": with_coords[:, 0],
            "y": with_coords[:, 1],
            "label": with_labels.astype(str),
        }
    )
    without_df = pd.DataFrame(
        {
            "x": without_coords[:, 0],
            "y": without_coords[:, 1],
            "label": without_labels.astype(str),
        }
    )

    sns.scatterplot(
        data=with_df,
        x="x",
        y="y",
        hue="label",
        palette=palettes,
        hue_order=sorted(with_df["label"].unique()),
        s=10,
        linewidth=0,
        alpha=0.7,
        ax=axes[0],
    )
    axes[0].set_title(f"{dataset.upper()} – With preprocessing")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    sns.scatterplot(
        data=without_df,
        x="x",
        y="y",
        hue="label",
        palette=palettes,
        hue_order=sorted(without_df["label"].unique()),
        s=10,
        linewidth=0,
        alpha=0.7,
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title(f"{dataset.upper()} – Without preprocessing")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles and labels:
        axes[0].legend(handles=handles, labels=labels, title="Label", loc="best")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _process_dataset(
    config: DatasetConfig,
    output_dir: Path,
    sample_size: int | None,
    perplexity: float,
    max_iter: int,
    random_state: int,
) -> bool:
    logger.info("Processing dataset {}", config.name)

    with_path = _resolve_path(config.with_candidates, config.name, "with_preprocessing")
    without_path = _resolve_path(
        config.without_candidates, config.name, "without_preprocessing"
    )

    if with_path is None or without_path is None:
        return False

    with_embeddings, with_labels = _load_embeddings(
        with_path, sample_size, random_state
    )
    without_embeddings, without_labels = _load_embeddings(
        without_path, sample_size, random_state
    )

    logger.debug(
        "{} | with_preprocessing shape = {}, without_preprocessing shape = {}",
        config.name,
        with_embeddings.shape,
        without_embeddings.shape,
    )

    with_coords = _compute_tsne(with_embeddings, perplexity, max_iter, random_state)
    without_coords = _compute_tsne(
        without_embeddings, perplexity, max_iter, random_state
    )

    output_path = output_dir / f"{config.name}_secbert_tsne.png"
    _build_plot(
        dataset=config.name,
        with_coords=with_coords,
        with_labels=with_labels,
        without_coords=without_coords,
        without_labels=without_labels,
        output_path=output_path,
    )

    logger.success("Saved plot to {}", output_path)
    return True


@app.command()
def main(
    output_dir: Path = typer.Option(
        ROOT / "visualizations" / "embeddings" / "experiment_18",
        help="Directory to store the generated figures.",
    ),
    datasets: list[str] = typer.Option(
        ["pkdd", "csic"],
        help="Datasets to visualise. Must be keys from the default configuration.",
    ),
    sample_size: int | None = typer.Option(
        5000,
        min=500,
        help="Randomly sample this many points per dataset to keep t-SNE tractable.",
    ),
    perplexity: float = typer.Option(
        35.0,
        min=5.0,
        help="Perplexity value for t-SNE.",
    ),
    max_iter: int = typer.Option(
        1500,
        min=500,
        help="Number of optimisation iterations for t-SNE.",
    ),
    random_state: int = typer.Option(
        42,
        help="Random seed used for sampling and t-SNE initialisation.",
    ),
) -> None:
    if not datasets:
        logger.error("No datasets selected; nothing to do.")
        raise typer.Exit(code=1)

    for dataset in datasets:
        if dataset not in DEFAULT_DATASETS:
            logger.error(
                "Unknown dataset '{}'. Valid options: {}",
                dataset,
                sorted(DEFAULT_DATASETS),
            )
            raise typer.Exit(code=1)

    generated_any = False
    for dataset_name in datasets:
        config = DEFAULT_DATASETS[dataset_name]
        generated = _process_dataset(
            config=config,
            output_dir=output_dir,
            sample_size=sample_size,
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state,
        )
        generated_any = generated_any or generated

    if not generated_any:
        logger.error(
            "No plots were generated. Check that the embeddings exist locally."
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
