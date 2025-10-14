#!/usr/bin/env python3
"""Analyze and visualize MI feature selection results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def plot_mi_distribution(mi_scores: np.ndarray, output_dir: Path):
    """Plot histogram of MI scores."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(mi_scores, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(
        np.median(mi_scores),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(mi_scores):.6f}",
    )
    ax.axvline(
        np.mean(mi_scores),
        color="green",
        linestyle="--",
        label=f"Mean: {np.mean(mi_scores):.6f}",
    )

    ax.set_xlabel("Mutual Information Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of MI Scores Across 768 Dimensions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = output_dir / "mi_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved MI distribution plot to {output_path}")
    plt.close()


def plot_top_dimensions(mi_scores: np.ndarray, top_k: int, output_dir: Path):
    """Plot top K dimensions by MI score."""
    top_dims = np.argsort(mi_scores)[-top_k:]
    top_scores = mi_scores[top_dims]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.barh(range(top_k), top_scores, edgecolor="black")

    # Color gradient
    colors = plt.cm.viridis(np.linspace(0.3, 1, top_k))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"Dim {dim}" for dim in top_dims])
    ax.set_xlabel("Mutual Information Score")
    ax.set_title(f"Top {top_k} Dimensions by MI Score")
    ax.grid(True, axis="x", alpha=0.3)

    output_path = output_dir / f"top_{top_k}_dimensions.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved top dimensions plot to {output_path}")
    plt.close()


def plot_recall_vs_k(results: list[dict], output_dir: Path):
    """Plot recall vs number of dimensions."""
    k_values = [r["k"] for r in results]
    recalls = [r["recall"] for r in results]
    precisions = [r["precision"] for r in results]
    f1_scores = [r["f1_score"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Recall vs K
    ax1.plot(k_values, recalls, marker="o", linewidth=2, markersize=8, label="Recall")
    ax1.axhline(
        recalls[-1], color="red", linestyle="--", alpha=0.5, label="Baseline (768 dims)"
    )
    ax1.set_xlabel("Number of Dimensions (K)")
    ax1.set_ylabel("Recall @ 5% FPR")
    ax1.set_title("Recall vs Number of MI-Selected Dimensions")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Annotate best point
    best_idx = np.argmax(recalls)
    ax1.annotate(
        f"Best: {recalls[best_idx]:.4f}\n(k={k_values[best_idx]})",
        xy=(k_values[best_idx], recalls[best_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    # Plot 2: All metrics
    ax2.plot(k_values, recalls, marker="o", linewidth=2, label="Recall")
    ax2.plot(k_values, precisions, marker="s", linewidth=2, label="Precision")
    ax2.plot(k_values, f1_scores, marker="^", linewidth=2, label="F1-Score")
    ax2.set_xlabel("Number of Dimensions (K)")
    ax2.set_ylabel("Score")
    ax2.set_title("All Metrics vs Number of Dimensions")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    output_path = output_dir / "recall_vs_k.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved recall vs k plot to {output_path}")
    plt.close()


def main():
    """Analyze MI feature selection results."""
    logger.info("Analyzing MI feature selection results...")

    results_dir = Path("experiments/12_mi_feature_selection/results")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load MI scores
    mi_scores_path = results_dir / "mi_scores.npy"
    mi_scores = np.load(mi_scores_path)
    logger.info(f"Loaded MI scores: {mi_scores.shape}")

    # Load results
    results_path = results_dir / "metrics_comparison.json"
    with open(results_path) as f:
        results = json.load(f)
    logger.info(f"Loaded {len(results)} experiment results")

    # Generate plots
    logger.info("\nGenerating visualizations...")
    plot_mi_distribution(mi_scores, plots_dir)
    plot_top_dimensions(mi_scores, 50, plots_dir)
    plot_recall_vs_k(results, plots_dir)

    # Print statistics
    logger.info("\n" + "=" * 80)
    logger.info("MI SCORES STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Min:    {mi_scores.min():.6f}")
    logger.info(f"Max:    {mi_scores.max():.6f}")
    logger.info(f"Mean:   {mi_scores.mean():.6f}")
    logger.info(f"Median: {np.median(mi_scores):.6f}")
    logger.info(f"Std:    {mi_scores.std():.6f}")

    # Top 10 dimensions
    top_10_dims = np.argsort(mi_scores)[-10:][::-1]
    logger.info("\nTop 10 Dimensions:")
    for i, dim in enumerate(top_10_dims, start=1):
        logger.info(f"  {i:2d}. Dimension {dim:3d}: {mi_scores[dim]:.6f}")

    logger.success("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
