#!/usr/bin/env python3
"""Analyze distance to nearest valid training point for test embeddings.

This script computes the distance from each test embedding to its nearest
valid training point, then plots distributions for valid vs attack test points.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.neighbors import NearestNeighbors


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels from npz file."""
    data = np.load(path)
    embeddings = data["embeddings"]
    labels = data["labels"]
    logger.info(f"Loaded {embeddings.shape[0]} embeddings from {path}")
    return embeddings, labels


def compute_nearest_distances(
    test_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute distance to nearest training point for each test embedding.

    Args:
        test_embeddings: Test embeddings (N_test, dim)
        train_embeddings: Training embeddings (N_train, dim)

    Returns:
        Distances to nearest neighbor for each test point (N_test,)
    """
    logger.info("Building nearest neighbor index...")
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="auto")
    nn.fit(train_embeddings)

    logger.info("Computing distances to nearest neighbors...")
    distances, _ = nn.kneighbors(test_embeddings)

    # distances has shape (N_test, 1), flatten it
    return distances.flatten()


def plot_distance_distributions(
    valid_distances: np.ndarray,
    attack_distances: np.ndarray,
    output_path: Path,
) -> None:
    """Plot histograms of distances for valid and attack points.

    Args:
        valid_distances: Distances for valid test points
        attack_distances: Distances for attack test points
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Compute bins based on combined data
    all_distances = np.concatenate([valid_distances, attack_distances])
    bins = np.linspace(all_distances.min(), all_distances.max(), 50)

    # Plot histograms
    ax.hist(
        valid_distances,
        bins=bins,
        alpha=0.6,
        color="green",
        label=f"Valid (n={len(valid_distances)})",
        edgecolor="black",
        density=True,
    )
    ax.hist(
        attack_distances,
        bins=bins,
        alpha=0.6,
        color="red",
        label=f"Attack (n={len(attack_distances)})",
        edgecolor="black",
        density=True,
    )

    # Add statistics
    valid_mean = valid_distances.mean()
    valid_median = np.median(valid_distances)
    attack_mean = attack_distances.mean()
    attack_median = np.median(attack_distances)

    ax.axvline(
        valid_mean,
        color="darkgreen",
        linestyle="--",
        linewidth=2,
        label=f"Valid mean: {valid_mean:.2f}",
    )
    ax.axvline(
        attack_mean,
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"Attack mean: {attack_mean:.2f}",
    )

    # Labels and formatting
    ax.set_xlabel("Distance to Nearest Valid Training Point", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Distribution of Distances to Nearest Valid Training Point\n"
        "SecBERT Embeddings with Preprocessing",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add text box with statistics
    stats_text = (
        f"Valid:  mean={valid_mean:.2f}, median={valid_median:.2f}\n"
        f"Attack: mean={attack_mean:.2f}, median={attack_median:.2f}\n"
        f"Ratio (attack/valid mean): {attack_mean / valid_mean:.2f}x"
    )
    ax.text(
        0.98,
        0.50,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot to {output_path}")

    # Print summary statistics
    logger.info("=" * 60)
    logger.info("DISTANCE STATISTICS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Valid points (n={len(valid_distances)}):")
    logger.info(f"  Mean:   {valid_mean:.4f}")
    logger.info(f"  Median: {valid_median:.4f}")
    logger.info(f"  Std:    {valid_distances.std():.4f}")
    logger.info(f"  Min:    {valid_distances.min():.4f}")
    logger.info(f"  Max:    {valid_distances.max():.4f}")
    logger.info("")
    logger.info(f"Attack points (n={len(attack_distances)}):")
    logger.info(f"  Mean:   {attack_mean:.4f}")
    logger.info(f"  Median: {attack_median:.4f}")
    logger.info(f"  Std:    {attack_distances.std():.4f}")
    logger.info(f"  Min:    {attack_distances.min():.4f}")
    logger.info(f"  Max:    {attack_distances.max():.4f}")
    logger.info("")
    logger.info(f"Ratio (attack/valid mean): {attack_mean / valid_mean:.4f}x")
    logger.info("=" * 60)


def main() -> None:
    """Main execution function."""
    # Paths
    base_dir = Path("experiments/03_secbert_comparison/secbert_with_preprocessing")
    train_path = base_dir / "csic_train_embeddings_converted.npz"
    test_path = base_dir / "csic_test_embeddings_converted.npz"
    output_path = base_dir / "nearest_neighbor_distance_distribution.png"

    # Load embeddings
    logger.info("Loading training embeddings...")
    train_embeddings, train_labels = load_embeddings(train_path)

    logger.info("Loading test embeddings...")
    test_embeddings, test_labels = load_embeddings(test_path)

    # Filter to valid training points only (label == "valid")
    valid_train_mask = train_labels == "valid"
    valid_train_embeddings = train_embeddings[valid_train_mask]
    logger.info(
        f"Filtered to {len(valid_train_embeddings)} valid training points "
        f"(out of {len(train_embeddings)} total)"
    )

    # Compute distances for all test points
    distances = compute_nearest_distances(test_embeddings, valid_train_embeddings)

    # Split distances by test labels
    valid_test_mask = test_labels == "valid"
    attack_test_mask = test_labels == "attack"

    valid_distances = distances[valid_test_mask]
    attack_distances = distances[attack_test_mask]

    logger.info(
        f"Split into {len(valid_distances)} valid and "
        f"{len(attack_distances)} attack test points"
    )

    # Plot distributions
    plot_distance_distributions(valid_distances, attack_distances, output_path)

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
