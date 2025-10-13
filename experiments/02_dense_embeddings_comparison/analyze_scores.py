#!/usr/bin/env python3
"""Analyze training score distributions for different embedding methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer

app = typer.Typer()


@app.command()  # noqa: T201
def main(
    model_paths: list[str] = typer.Argument(
        ..., help="Paths to trained model files (.joblib)"
    ),
    labels: list[str] = typer.Option(
        [], "--label", help="Labels for each model (same order as paths)"
    ),
    output: str = typer.Option(
        "score_distribution_comparison.png",
        "--output",
        "-o",
        help="Output plot filename",
    ),
) -> None:
    """Analyze and compare training score distributions from multiple models."""
    if labels and len(labels) != len(model_paths):
        raise ValueError("Number of labels must match number of model paths")

    if not labels:
        labels = [f"Model {i + 1}" for i in range(len(model_paths))]

    print("\n" + "=" * 80)
    print("TRAINING SCORE DISTRIBUTION ANALYSIS")
    print("=" * 80 + "\n")

    fig, axes = plt.subplots(
        len(model_paths), 1, figsize=(12, 4 * len(model_paths)), squeeze=False
    )

    all_stats: list[dict[str, Any]] = []

    for idx, (model_path, label) in enumerate(zip(model_paths, labels)):
        print(f"Analyzing: {label}")
        print(f"  Path: {model_path}")

        # Load model
        model_data = joblib.load(model_path)
        detector = model_data["detector"]

        # Get training embeddings path (derive from model path)
        embeddings_path = Path(model_path).parent / "embeddings.npz"
        if not embeddings_path.exists():
            print(f"  ⚠️  Embeddings not found at {embeddings_path}, skipping...")
            continue

        # Load embeddings
        data = np.load(embeddings_path, allow_pickle=True)
        embeddings = data["embeddings"]
        labels_arr = data["labels"]

        # Filter normal samples
        normal_mask = labels_arr == "normal"
        normal_embeddings = embeddings[normal_mask]

        # Get scores
        scores = detector.score(normal_embeddings)

        # Calculate statistics
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        median_score = float(np.median(scores))
        q25 = float(np.percentile(scores, 25))
        q75 = float(np.percentile(scores, 75))
        threshold = float(detector.threshold_)

        stats = {
            "label": label,
            "mean": mean_score,
            "std": std_score,
            "min": min_score,
            "max": max_score,
            "median": median_score,
            "q25": q25,
            "q75": q75,
            "threshold": threshold,
            "range": max_score - min_score,
            "num_samples": len(scores),
        }
        all_stats.append(stats)

        # Print stats
        print(f"  Samples:   {stats['num_samples']:,}")
        print(f"  Mean:      {mean_score:.6f}")
        print(f"  Std Dev:   {std_score:.6f}")
        print(f"  Min:       {min_score:.6f}")
        print(f"  Max:       {max_score:.6f}")
        print(f"  Median:    {median_score:.6f}")
        print(f"  Q25:       {q25:.6f}")
        print(f"  Q75:       {q75:.6f}")
        print(f"  Range:     {stats['range']:.6f}")
        print(f"  Threshold: {threshold:.6f}")

        # Check variance quality
        if std_score < 0.001:
            print("  ⚠️  EXTREMELY LOW VARIANCE - Model may not discriminate well!")
        elif std_score < 0.01:
            print("  ⚠️  LOW VARIANCE - Limited discriminative power")
        else:
            print("  ✓ GOOD VARIANCE - Model has discriminative potential")

        print()

        # Plot distribution
        ax = axes[idx, 0]
        sns.histplot(scores, bins=50, kde=True, ax=ax, color="steelblue")
        ax.axvline(
            threshold, color="red", linestyle="--", linewidth=2, label="Threshold"
        )
        ax.axvline(mean_score, color="green", linestyle="--", linewidth=1, label="Mean")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Count")
        title = (
            f"{label}\n"
            f"(μ={mean_score:.4f}, σ={std_score:.4f}, threshold={threshold:.4f})"
        )
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved to: {output}\n")

    # Print comparison table
    if len(all_stats) > 1:
        print("=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80)
        print(f"{'Model':<30} {'Mean':<12} {'StdDev':<12} {'Range':<12}")
        print("-" * 80)
        for stat in all_stats:
            row = (
                f"{stat['label']:<30} {stat['mean']:<12.6f} "
                f"{stat['std']:<12.6f} {stat['range']:<12.6f}"
            )
            print(row)
        print("=" * 80 + "\n")

        # Recommendations
        print("RECOMMENDATIONS")
        print("=" * 80)
        best_variance = max(all_stats, key=lambda x: float(x["std"]))  # type: ignore[arg-type]
        best_std = float(best_variance["std"])  # type: ignore[arg-type]
        print(f"✓ Best variance: {best_variance['label']} (σ={best_std:.6f})")

        if best_std > 0.01:
            print("  → This model has good discriminative potential")
        else:
            print("  ⚠️ Even the best model has low variance. Consider:")
            print("     - Different embedding method")
            print("     - Different anomaly detector")
            print("     - Feature engineering")

        print()


if __name__ == "__main__":
    app()
