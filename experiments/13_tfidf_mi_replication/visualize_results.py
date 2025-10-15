#!/usr/bin/env python3
"""Visualize TF-IDF + MI experiment results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def plot_comparison(results: list[dict], output_dir: Path):
    """Plot TF-IDF+MI performance vs baselines."""
    k_values = [r["k"] for r in results]
    recalls = [r["recall"] for r in results]
    precisions = [r["precision"] for r in results]
    fprs = [r["fpr"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Recall vs K
    ax = axes[0, 0]
    ax.plot(
        k_values, recalls, marker="o", linewidth=2, markersize=8, label="TF-IDF + MI"
    )

    # Baselines
    ax.axhline(
        0.4926, color="green", linestyle="--", linewidth=2, label="SecBERT (49.26%)"
    )
    ax.axhline(
        0.0096, color="red", linestyle="--", linewidth=2, label="Vanilla TF-IDF (0.96%)"
    )

    # Paper's results
    ax.axhline(
        0.9176,
        color="blue",
        linestyle=":",
        linewidth=2,
        alpha=0.5,
        label="Paper Drupal (91.76%)",
    )
    ax.axhline(
        0.7887,
        color="blue",
        linestyle=":",
        linewidth=2,
        alpha=0.3,
        label="Paper SR-BH (78.87%)",
    )

    ax.set_xlabel("Number of Tokens (K)")
    ax.set_ylabel("Recall")
    ax.set_title("Recall vs Number of MI-Selected Tokens")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate best
    best_idx = np.argmax(recalls)
    ax.annotate(
        f"Best: {recalls[best_idx]:.2%}\n(k={k_values[best_idx]})",
        xy=(k_values[best_idx], recalls[best_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->"),
    )

    # Plot 2: Precision vs Recall
    ax = axes[0, 1]
    ax.scatter(
        recalls, precisions, c=k_values, cmap="viridis", s=100, edgecolor="black"
    )

    # Annotate points
    for i, k in enumerate(k_values):
        ax.annotate(
            f"k={k}",
            (recalls[i], precisions[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # Baselines
    ax.scatter(
        [0.4926],
        [0.9081],
        marker="*",
        s=300,
        color="green",
        label="SecBERT",
        edgecolor="black",
    )
    ax.scatter(
        [0.0096],
        [0.72],
        marker="x",
        s=200,
        color="red",
        label="Vanilla TF-IDF",
        linewidths=3,
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: FPR vs K
    ax = axes[1, 0]
    ax.plot(k_values, fprs, marker="s", linewidth=2, markersize=8, color="orange")
    ax.axhline(0.05, color="black", linestyle="--", label="Target FPR (5%)")
    ax.set_xlabel("Number of Tokens (K)")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("FPR vs Number of Tokens")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Bar chart comparison
    ax = axes[1, 1]

    labels_chart = [f"k={k}" for k in k_values] + ["SecBERT", "TF-IDF\n(vanilla)"]
    recall_values = recalls + [0.4926, 0.0096]
    colors_chart = ["#1f77b4"] * len(k_values) + ["green", "red"]

    bars = ax.bar(
        range(len(labels_chart)), recall_values, color=colors_chart, edgecolor="black"
    )
    ax.set_xticks(range(len(labels_chart)))
    ax.set_xticklabels(labels_chart, rotation=45, ha="right")
    ax.set_ylabel("Recall @ ~5% FPR")
    ax.set_title("Performance Comparison")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars, recall_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{val:.2%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    output_path = output_dir / "tfidf_mi_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved comparison plot to {output_path}")
    plt.close()


def plot_mi_token_distribution(
    mi_scores: np.ndarray, feature_names: list[str], output_dir: Path
):
    """Plot MI score distribution for tokens."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(mi_scores, bins=50, edgecolor="black", alpha=0.7)
    ax1.axvline(
        np.median(mi_scores),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(mi_scores):.6f}",
    )
    ax1.axvline(
        np.mean(mi_scores),
        color="green",
        linestyle="--",
        label=f"Mean: {np.mean(mi_scores):.6f}",
    )
    ax1.set_xlabel("MI Score")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of MI Scores (TF-IDF Tokens)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Top 30 tokens
    top_30_idx = np.argsort(mi_scores)[-30:]
    top_30_scores = mi_scores[top_30_idx]
    top_30_names = [feature_names[i] for i in top_30_idx]

    ax2.barh(range(30), top_30_scores, edgecolor="black")
    ax2.set_yticks(range(30))
    ax2.set_yticklabels(top_30_names, fontsize=8)
    ax2.set_xlabel("MI Score")
    ax2.set_title("Top 30 Tokens by MI Score")
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "mi_token_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved token distribution plot to {output_path}")
    plt.close()


def main():
    """Visualize experiment results."""
    logger.info("Visualizing TF-IDF + MI results...")

    results_dir = Path("experiments/13_tfidf_mi_replication/results")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    with open(results_dir / "metrics_comparison.json") as f:
        results = json.load(f)
    logger.info(f"Loaded {len(results)} experiment results")

    # Load MI scores
    mi_scores = np.load(results_dir / "mi_scores_tfidf.npy")
    logger.info(f"Loaded MI scores: {mi_scores.shape}")

    # Load feature names
    with open(results_dir / "feature_names.txt") as f:
        feature_names = [line.strip() for line in f]
    logger.info(f"Loaded {len(feature_names)} feature names")

    # Generate plots
    logger.info("\nGenerating visualizations...")
    plot_comparison(results, plots_dir)
    plot_mi_token_distribution(mi_scores, feature_names, plots_dir)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TF-IDF MI SCORES STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Min:    {mi_scores.min():.6f}")
    logger.info(f"Max:    {mi_scores.max():.6f}")
    logger.info(f"Mean:   {mi_scores.mean():.6f}")
    logger.info(f"Median: {np.median(mi_scores):.6f}")
    logger.info(f"Std:    {mi_scores.std():.6f}")

    # Top 20 tokens
    top_20_idx = np.argsort(mi_scores)[-20:][::-1]
    logger.info("\nTop 20 Tokens:")
    for i, idx in enumerate(top_20_idx, 1):
        logger.info(f"  {i:2d}. '{feature_names[idx]}': {mi_scores[idx]:.6f}")

    logger.success("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
