#!/usr/bin/env python3
"""
Visualization for Experiment 16: Cross-Dataset MI Feature Selection

Generates:
1. MI score distributions (both runs)
2. Top-20 tokens bar charts
3. ROC-style curves (Recall vs K)
4. Token overlap Venn diagram
5. Comparison summary table
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11


def load_results(run_dir: Path) -> dict:
    """Load results from a single run."""
    mi_scores = np.load(run_dir / "mi_scores.npy")

    with open(run_dir / "feature_names.txt") as f:
        feature_names = [line.strip() for line in f]

    with open(run_dir / "metrics_comparison.json") as f:
        metrics = json.load(f)

    return {
        "mi_scores": mi_scores,
        "feature_names": feature_names,
        "metrics": metrics,
    }


def plot_mi_distribution(mi_scores: np.ndarray, title: str, output_path: Path):
    """Plot MI score distribution."""
    plt.figure(figsize=(10, 6))

    plt.hist(mi_scores, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    plt.xlabel("MI Score", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")

    # Add statistics
    stats_text = (
        f"Mean: {mi_scores.mean():.6f}\n"
        f"Median: {np.median(mi_scores):.6f}\n"
        f"Max: {mi_scores.max():.6f}\n"
        f"Min: {mi_scores.min():.6f}"
    )
    plt.text(
        0.95,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {output_path}")


def plot_top_tokens(
    mi_scores: np.ndarray,
    feature_names: list[str],
    title: str,
    output_path: Path,
    top_n: int = 20,
):
    """Plot top-N tokens by MI score."""
    plt.figure(figsize=(10, 8))

    # Get top N
    top_indices = np.argsort(mi_scores)[-top_n:]
    top_scores = mi_scores[top_indices]
    top_tokens = [feature_names[i] for i in top_indices]

    # Plot
    y_pos = np.arange(len(top_tokens))
    plt.barh(y_pos, top_scores, color="coral", edgecolor="black", alpha=0.8)
    plt.yticks(y_pos, top_tokens, fontsize=10)
    plt.xlabel("MI Score", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()  # Highest at top

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {output_path}")


def plot_recall_vs_k(
    run1_metrics: list[dict], run2_metrics: list[dict], output_path: Path
):
    """Plot Recall vs K for both runs."""
    plt.figure(figsize=(10, 6))

    # Extract data
    k_values = [m["k"] for m in run1_metrics]
    run1_recall = [m["recall"] for m in run1_metrics]
    run2_recall = [m["recall"] for m in run2_metrics]

    # Plot
    plt.plot(
        k_values,
        run1_recall,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Run 1: CSIC → SR-BH",
        color="steelblue",
    )
    plt.plot(
        k_values,
        run2_recall,
        marker="s",
        linewidth=2,
        markersize=8,
        label="Run 2: SR-BH → CSIC",
        color="coral",
    )

    # Reference lines
    plt.axhline(
        y=0.75,
        color="green",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Target: 75%",
    )
    plt.axhline(
        y=0.5,
        color="orange",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Acceptable: 50%",
    )

    plt.xlabel("Number of Features (K)", fontsize=12)
    plt.ylabel("Recall (TPR)", fontsize=12)
    plt.title("Recall vs Number of Selected Features", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10, loc="best")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {output_path}")


def plot_fpr_comparison(
    run1_metrics: list[dict], run2_metrics: list[dict], output_path: Path
):
    """Plot FPR comparison."""
    plt.figure(figsize=(10, 6))

    k_values = [m["k"] for m in run1_metrics]
    run1_fpr = [m["fpr"] for m in run1_metrics]
    run2_fpr = [m["fpr"] for m in run2_metrics]

    plt.plot(
        k_values,
        run1_fpr,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Run 1: CSIC → SR-BH",
        color="steelblue",
    )
    plt.plot(
        k_values,
        run2_fpr,
        marker="s",
        linewidth=2,
        markersize=8,
        label="Run 2: SR-BH → CSIC",
        color="coral",
    )

    # Target FPR
    plt.axhline(
        y=0.05, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Target: 5%"
    )

    plt.xlabel("Number of Features (K)", fontsize=12)
    plt.ylabel("False Positive Rate (FPR)", fontsize=12)
    plt.title("FPR vs Number of Selected Features", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10, loc="best")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {output_path}")


def plot_token_overlap(run1_dir: Path, run2_dir: Path, output_path: Path):
    """Plot token overlap Venn diagram."""
    # Load top-100 tokens from both runs
    with open(run1_dir / "selected_tokens_k100.txt") as f:
        run1_tokens = set(line.strip() for line in f)

    with open(run2_dir / "selected_tokens_k100.txt") as f:
        run2_tokens = set(line.strip() for line in f)

    # Calculate overlap
    overlap = run1_tokens & run2_tokens
    only_run1 = run1_tokens - run2_tokens
    only_run2 = run2_tokens - run1_tokens

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw circles (simplified Venn diagram)
    from matplotlib.patches import Circle

    circle1 = Circle((0.3, 0.5), 0.3, color="steelblue", alpha=0.5, label="CSIC tokens")
    circle2 = Circle((0.7, 0.5), 0.3, color="coral", alpha=0.5, label="SR-BH tokens")

    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Add text
    ax.text(
        0.2,
        0.5,
        f"{len(only_run1)}",
        fontsize=20,
        ha="center",
        va="center",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.5,
        f"{len(overlap)}",
        fontsize=20,
        ha="center",
        va="center",
        fontweight="bold",
    )
    ax.text(
        0.8,
        0.5,
        f"{len(only_run2)}",
        fontsize=20,
        ha="center",
        va="center",
        fontweight="bold",
    )

    # Labels
    ax.text(0.2, 0.85, "CSIC only", fontsize=12, ha="center", fontweight="bold")
    ax.text(0.5, 0.85, "Overlap", fontsize=12, ha="center", fontweight="bold")
    ax.text(0.8, 0.85, "SR-BH only", fontsize=12, ha="center", fontweight="bold")

    # Overlap percentage
    overlap_pct = len(overlap) / 100 * 100
    ax.text(
        0.5,
        0.1,
        f"Overlap: {overlap_pct:.1f}% ({len(overlap)}/100 tokens)",
        fontsize=14,
        ha="center",
        fontweight="bold",
        bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.5},
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Token Overlap: Top-100 Features (k=100)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {output_path}")

    # Log overlap tokens
    logger.info(f"\nToken overlap: {len(overlap)}/100 ({overlap_pct:.1f}%)")
    logger.info("Common tokens:")
    for token in sorted(overlap)[:20]:
        logger.info(f"  - '{token}'")


def create_comparison_table(run1_dir: Path, run2_dir: Path, output_path: Path):
    """Create comparison summary table."""
    with open(run1_dir / "metrics_comparison.json") as f:
        run1_metrics = json.load(f)

    with open(run2_dir / "metrics_comparison.json") as f:
        run2_metrics = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")

    # Build table data
    table_data = [["K", "Run 1 Recall", "Run 1 FPR", "Run 2 Recall", "Run 2 FPR"]]

    for r1, r2 in zip(run1_metrics, run2_metrics):
        table_data.append(
            [
                f"{r1['k']}",
                f"{r1['recall']:.2%}",
                f"{r1['fpr']:.2%}",
                f"{r2['recall']:.2%}",
                f"{r2['fpr']:.2%}",
            ]
        )

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.1, 0.2, 0.2, 0.2, 0.2],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor("lightgray")
        cell.set_text_props(weight="bold")

    # Highlight best results
    best_run1_idx = (
        max(range(len(run1_metrics)), key=lambda i: run1_metrics[i]["recall"]) + 1
    )
    best_run2_idx = (
        max(range(len(run2_metrics)), key=lambda i: run2_metrics[i]["recall"]) + 1
    )

    table[(best_run1_idx, 1)].set_facecolor("lightgreen")
    table[(best_run2_idx, 3)].set_facecolor("lightgreen")

    ax.set_title(
        "Comparison Summary: Cross-Dataset MI Feature Selection",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {output_path}")


def main():
    """Generate all visualizations."""
    logger.info("=" * 80)
    logger.info("Generating visualizations for Experiment 16")
    logger.info("=" * 80)

    results_dir = Path("experiments/16_cross_dataset_mi/results")
    run1_dir = results_dir / "run1_csic_to_srbh"
    run2_dir = results_dir / "run2_srbh_to_csic"

    # Create plots directory
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    logger.info("\nLoading results...")
    run1 = load_results(run1_dir)
    run2 = load_results(run2_dir)

    # Generate plots
    logger.info("\n[1/7] MI distributions...")
    plot_mi_distribution(
        run1["mi_scores"],
        "MI Score Distribution: CSIC Features",
        plots_dir / "run1_mi_distribution.png",
    )
    plot_mi_distribution(
        run2["mi_scores"],
        "MI Score Distribution: SR-BH Features",
        plots_dir / "run2_mi_distribution.png",
    )

    logger.info("\n[2/7] Top tokens...")
    plot_top_tokens(
        run1["mi_scores"],
        run1["feature_names"],
        "Top 20 Tokens: CSIC Feature Selection",
        plots_dir / "run1_top_tokens.png",
    )
    plot_top_tokens(
        run2["mi_scores"],
        run2["feature_names"],
        "Top 20 Tokens: SR-BH Feature Selection",
        plots_dir / "run2_top_tokens.png",
    )

    logger.info("\n[3/7] Recall vs K...")
    plot_recall_vs_k(run1["metrics"], run2["metrics"], plots_dir / "recall_vs_k.png")

    logger.info("\n[4/7] FPR comparison...")
    plot_fpr_comparison(run1["metrics"], run2["metrics"], plots_dir / "fpr_vs_k.png")

    logger.info("\n[5/7] Token overlap...")
    plot_token_overlap(run1_dir, run2_dir, plots_dir / "token_overlap.png")

    logger.info("\n[6/7] Comparison table...")
    create_comparison_table(run1_dir, run2_dir, plots_dir / "comparison_table.png")

    logger.info("\n[7/7] Performance comparison...")
    # Create combined plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Recall
    k_values = [m["k"] for m in run1["metrics"]]
    axes[0].plot(
        k_values,
        [m["recall"] for m in run1["metrics"]],
        marker="o",
        label="CSIC → SR-BH",
        linewidth=2,
    )
    axes[0].plot(
        k_values,
        [m["recall"] for m in run2["metrics"]],
        marker="s",
        label="SR-BH → CSIC",
        linewidth=2,
    )
    axes[0].axhline(y=0.75, color="green", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("K (Number of Features)")
    axes[0].set_ylabel("Recall")
    axes[0].set_title("Recall vs K")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # FPR
    axes[1].plot(
        k_values,
        [m["fpr"] for m in run1["metrics"]],
        marker="o",
        label="CSIC → SR-BH",
        linewidth=2,
    )
    axes[1].plot(
        k_values,
        [m["fpr"] for m in run2["metrics"]],
        marker="s",
        label="SR-BH → CSIC",
        linewidth=2,
    )
    axes[1].axhline(y=0.05, color="red", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("K (Number of Features)")
    axes[1].set_ylabel("False Positive Rate")
    axes[1].set_title("FPR vs K")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {plots_dir / 'performance_comparison.png'}")

    logger.info("\n" + "=" * 80)
    logger.info("Visualization complete!")
    logger.info("=" * 80)
    logger.info(f"All plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
