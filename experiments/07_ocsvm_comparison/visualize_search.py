"""Visualize OCSVM hyperparameter search results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger


def plot_heatmap(results: list[dict], output_path: Path):
    """Plot recall heatmap for nu vs gamma."""
    # Get unique nu and gamma values
    nu_values = sorted(set(r["nu"] for r in results))
    gamma_values_raw = sorted(
        set(r["gamma"] for r in results), key=lambda x: (isinstance(x, str), x)
    )

    # Create matrix
    recall_matrix = np.zeros((len(nu_values), len(gamma_values_raw)))

    for result in results:
        nu_idx = nu_values.index(result["nu"])
        gamma_idx = gamma_values_raw.index(result["gamma"])
        recall_matrix[nu_idx, gamma_idx] = result["recall"] * 100

    # Format gamma labels
    gamma_labels = [
        str(g) if isinstance(g, str) else f"{g:.3f}" for g in gamma_values_raw
    ]

    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        recall_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        xticklabels=gamma_labels,
        yticklabels=[f"{nu:.2f}" for nu in nu_values],
        cbar_kws={"label": "Recall (%)"},
        vmin=0,
        vmax=100,
    )
    plt.xlabel("Gamma", fontsize=12, fontweight="bold")
    plt.ylabel("Nu", fontsize=12, fontweight="bold")
    plt.title(
        "OCSVM Recall @ 5% FPR\nHyperparameter Grid Search",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Heatmap saved to {output_path}")
    plt.close()


def plot_nu_effect(results: list[dict], output_path: Path):
    """Plot effect of nu parameter (averaged over gamma)."""
    nu_values = sorted(set(r["nu"] for r in results))

    recalls_by_nu = {nu: [] for nu in nu_values}
    for result in results:
        recalls_by_nu[result["nu"]].append(result["recall"] * 100)

    # Compute mean and std
    mean_recalls = [np.mean(recalls_by_nu[nu]) for nu in nu_values]
    std_recalls = [np.std(recalls_by_nu[nu]) for nu in nu_values]

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        nu_values,
        mean_recalls,
        yerr=std_recalls,
        marker="o",
        linewidth=2,
        capsize=5,
        capthick=2,
    )
    plt.xlabel("Nu (contamination tolerance)", fontsize=12, fontweight="bold")
    plt.ylabel("Mean Recall @ 5% FPR (%)", fontsize=12, fontweight="bold")
    plt.title(
        "Effect of Nu Parameter\n(averaged over gamma)", fontsize=14, fontweight="bold"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Nu effect plot saved to {output_path}")
    plt.close()


def plot_gamma_effect(results: list[dict], output_path: Path):
    """Plot effect of gamma parameter (averaged over nu)."""
    gamma_values_raw = sorted(
        set(r["gamma"] for r in results), key=lambda x: (isinstance(x, str), x)
    )

    recalls_by_gamma = {str(g): [] for g in gamma_values_raw}
    for result in results:
        recalls_by_gamma[str(result["gamma"])].append(result["recall"] * 100)

    # Compute mean and std
    gamma_labels = [str(g) for g in gamma_values_raw]
    mean_recalls = [np.mean(recalls_by_gamma[str(g)]) for g in gamma_values_raw]
    std_recalls = [np.std(recalls_by_gamma[str(g)]) for g in gamma_values_raw]

    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(gamma_labels))
    plt.bar(
        x_pos, mean_recalls, yerr=std_recalls, capsize=5, alpha=0.7, color="steelblue"
    )
    plt.xticks(x_pos, gamma_labels, rotation=45)
    plt.xlabel("Gamma (RBF kernel coefficient)", fontsize=12, fontweight="bold")
    plt.ylabel("Mean Recall @ 5% FPR (%)", fontsize=12, fontweight="bold")
    plt.title(
        "Effect of Gamma Parameter\n(averaged over nu)", fontsize=14, fontweight="bold"
    )
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Gamma effect plot saved to {output_path}")
    plt.close()


def plot_top_configs(results: list[dict], output_path: Path, top_n: int = 10):
    """Plot top N configurations."""
    sorted_results = sorted(results, key=lambda x: x["recall"], reverse=True)[:top_n]

    labels = [f"nu={r['nu']:.2f}\ngamma={r['gamma']}" for r in sorted_results]
    recalls = [r["recall"] * 100 for r in sorted_results]

    plt.figure(figsize=(14, 6))
    x_pos = np.arange(len(labels))
    bars = plt.bar(x_pos, recalls, alpha=0.7)

    # Color best bar differently
    bars[0].set_color("darkgreen")

    plt.xticks(x_pos, labels, rotation=45, ha="right")
    plt.ylabel("Recall @ 5% FPR (%)", fontsize=12, fontweight="bold")
    plt.title(f"Top {top_n} OCSVM Configurations", fontsize=14, fontweight="bold")
    plt.axhline(y=40, color="red", linestyle="--", label="Mahalanobis (40%)")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Top configs plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize hyperparameter search")
    parser.add_argument(
        "results_file", type=Path, help="hyperparameter_results.json file"
    )
    parser.add_argument("output_dir", type=Path, help="Output directory for plots")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    logger.info(f"Loading results from {args.results_file}")
    with open(args.results_file, "r") as f:
        results = json.load(f)

    logger.info(f"Loaded {len(results)} configurations")

    # Generate plots
    logger.info("Generating visualizations...")

    plot_heatmap(results, args.output_dir / "recall_heatmap.png")
    plot_nu_effect(results, args.output_dir / "nu_effect.png")
    plot_gamma_effect(results, args.output_dir / "gamma_effect.png")
    plot_top_configs(results, args.output_dir / "top_configs.png", top_n=10)

    # Print summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    best = max(results, key=lambda x: x["recall"])
    worst = min(results, key=lambda x: x["recall"])

    logger.info(f"Best: nu={best['nu']}, gamma={best['gamma']} → {best['recall']:.2%}")
    logger.info(
        f"Worst: nu={worst['nu']}, gamma={worst['gamma']} → {worst['recall']:.2%}"
    )

    # Compare to baseline
    baseline_recall = 0.2158  # Default config
    improvement = (best["recall"] - baseline_recall) / baseline_recall * 100
    logger.info(
        f"Improvement over baseline (nu=0.05, gamma=scale): {improvement:+.1f}%"
    )

    # Compare to Mahalanobis
    mahalanobis_recall = 0.3996
    if best["recall"] > mahalanobis_recall:
        logger.info(f"OCSVM beats Mahalanobis! {best['recall']:.2%} vs 39.96%")
    else:
        gap = (mahalanobis_recall - best["recall"]) / mahalanobis_recall * 100
        logger.info(f"Still behind Mahalanobis by {gap:.1f}%")

    logger.info("=" * 80)
    logger.info("DONE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
