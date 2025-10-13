from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from neuralshield.anomaly.model import IsolationForestDetector
from neuralshield.encoding.observability import init_wandb_sink
from neuralshield.evaluation import ClassificationEvaluator, EvaluationConfig

app = typer.Typer()

# Hyperparameter search space
SEARCH_SPACE = {
    "contamination": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    "n_estimators": [100, 200, 300, 500],
    "max_samples": ["auto", 256, 512, 1024],
}


def load_training_embeddings(embeddings_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load precomputed training embeddings.

    Note: Training set contains only valid (non-attack) samples.
    """
    logger.info(f"Loading training embeddings from {embeddings_path}")
    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]

    logger.info(
        f"Loaded {embeddings.shape[0]} training samples, "
        f"embedding dim={embeddings.shape[1]}"
    )

    return embeddings, labels


def load_test_embeddings(embeddings_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load precomputed test embeddings."""
    logger.info(f"Loading test embeddings from {embeddings_path}")
    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"].tolist()

    logger.info(
        f"Loaded {embeddings.shape[0]} test samples, "
        f"embedding dim={embeddings.shape[1]}"
    )

    return embeddings, labels


def train_and_evaluate(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: list[str],
    contamination: float,
    n_estimators: int,
    max_samples: int | str,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train model and evaluate on test set."""
    # Train detector
    detector = IsolationForestDetector(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
    )
    detector.fit(train_embeddings)

    # Predict on test set
    predictions = detector.predict(test_embeddings)

    # Evaluate
    evaluator = ClassificationEvaluator(
        EvaluationConfig(positive_label="attack", negative_label="valid")
    )
    result = evaluator.evaluate(predictions.tolist(), test_labels)

    return {
        "contamination": contamination,
        "n_estimators": n_estimators,
        "max_samples": max_samples,
        "threshold": float(detector.threshold_),
        "precision": result.precision,
        "recall": result.recall,
        "f1_score": result.f1_score,
        "accuracy": result.accuracy,
        "fpr": result.fpr,
        "specificity": result.specificity,
        "tp": result.tp,
        "fp": result.fp,
        "tn": result.tn,
        "fn": result.fn,
    }


def grid_search(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: list[str],
) -> list[dict[str, Any]]:
    """Perform grid search over hyperparameters."""
    results = []

    # Generate all combinations
    configs = []
    for contamination in SEARCH_SPACE["contamination"]:
        for n_estimators in SEARCH_SPACE["n_estimators"]:
            for max_samples in SEARCH_SPACE["max_samples"]:
                configs.append(
                    {
                        "contamination": contamination,
                        "n_estimators": n_estimators,
                        "max_samples": max_samples,
                    }
                )

    total_configs = len(configs)
    logger.info(f"Starting grid search over {total_configs} configurations")

    # Run grid search with progress bar
    for config in tqdm(configs, desc="Grid search progress"):
        result = train_and_evaluate(
            train_embeddings,
            test_embeddings,
            test_labels,
            **config,
        )
        results.append(result)

    return results


def find_pareto_frontier(
    results: list[dict[str, Any]], max_fpr: float = 0.05
) -> list[dict[str, Any]]:
    """Find Pareto frontier: best recall with FPR <= max_fpr."""
    # Filter by FPR constraint
    feasible = [r for r in results if r["fpr"] <= max_fpr]

    if not feasible:
        logger.warning(f"No models with FPR ≤ {max_fpr}!")
        # Relax constraint and show best available
        feasible = sorted(results, key=lambda x: x["fpr"])[:5]
        logger.info(f"Showing top 5 models with lowest FPR instead")

    # Sort by recall descending
    feasible_sorted = sorted(feasible, key=lambda x: x["recall"], reverse=True)

    # Find Pareto frontier (non-dominated solutions)
    pareto = []
    for candidate in feasible_sorted:
        dominated = False
        for other in feasible_sorted:
            if (
                other["recall"] > candidate["recall"]
                and other["fpr"] < candidate["fpr"]
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(candidate)

    return pareto


def plot_results(
    results: list[dict[str, Any]],
    pareto_frontier: list[dict[str, Any]],
    output_path: Path,
    max_fpr: float = 0.05,
) -> None:
    """Create visualization of search results with Pareto frontier."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Recall vs FPR scatter
    ax1 = axes[0]

    # All results
    fprs = [r["fpr"] for r in results]
    recalls = [r["recall"] for r in results]
    f1s = [r["f1_score"] for r in results]

    scatter = ax1.scatter(fprs, recalls, c=f1s, cmap="viridis", alpha=0.6, s=50)

    # Pareto frontier
    pareto_fprs = [r["fpr"] for r in pareto_frontier]
    pareto_recalls = [r["recall"] for r in pareto_frontier]
    ax1.scatter(
        pareto_fprs,
        pareto_recalls,
        color="red",
        s=200,
        marker="*",
        edgecolors="black",
        linewidths=2,
        label="Pareto Frontier",
        zorder=5,
    )

    # FPR constraint line
    ax1.axvline(
        max_fpr, color="red", linestyle="--", linewidth=2, label=f"Max FPR={max_fpr}"
    )

    ax1.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax1.set_ylabel("Recall", fontsize=12)
    ax1.set_title("Recall vs FPR (Pareto Frontier)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label="F1-Score")

    # Plot 2: F1 vs Contamination
    ax2 = axes[1]

    df = pd.DataFrame(results)

    # Group by contamination and plot
    for n_est in sorted(df["n_estimators"].unique()):
        subset = df[df["n_estimators"] == n_est]
        grouped = subset.groupby("contamination")["f1_score"].mean()
        ax2.plot(grouped.index, grouped.values, marker="o", label=f"n_est={n_est}")

    ax2.set_xlabel("Contamination", fontsize=12)
    ax2.set_ylabel("F1-Score (avg over max_samples)", fontsize=12)
    ax2.set_title("F1-Score vs Contamination", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved visualization to {output_path}")


def save_results(
    results: list[dict[str, Any]],
    pareto_frontier: list[dict[str, Any]],
    output_dir: Path,
    max_fpr: float,
) -> None:
    """Save results to CSV and generate summary report."""
    # Save full results
    df = pd.DataFrame(results)
    df = df.sort_values("recall", ascending=False)
    csv_path = output_dir / "hyperparameter_search_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved full results to {csv_path}")

    # Generate summary report
    summary_path = output_dir / "hyperparameter_search_summary.md"

    with open(summary_path, "w") as f:
        f.write("# Hyperparameter Search Results\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Search Configuration\n\n")
        f.write(f"- **Total configurations**: {len(results)}\n")
        f.write(f"- **FPR constraint**: {max_fpr}\n")
        f.write(f"- **Search space**:\n")
        for param, values in SEARCH_SPACE.items():
            f.write(f"  - `{param}`: {values}\n")
        f.write("\n")

        f.write("## Pareto Frontier\n\n")
        f.write(f"**Found {len(pareto_frontier)} non-dominated solutions:**\n\n")

        if pareto_frontier:
            f.write(
                "| Rank | Contamination | n_estimators | max_samples | Recall | FPR | F1 | Precision |\n"
            )
            f.write(
                "|------|---------------|--------------|-------------|--------|-----|----|-----------|\n"
            )
            for i, config in enumerate(pareto_frontier, 1):
                f.write(
                    f"| {i} | {config['contamination']:.2f} | {config['n_estimators']} | "
                    f"{config['max_samples']} | {config['recall']:.2%} | {config['fpr']:.2%} | "
                    f"{config['f1_score']:.2%} | {config['precision']:.2%} |\n"
                )

            f.write("\n## Best Model (Highest Recall)\n\n")
            best = pareto_frontier[0]
            f.write(f"**Configuration:**\n")
            f.write(f"- Contamination: `{best['contamination']}`\n")
            f.write(f"- n_estimators: `{best['n_estimators']}`\n")
            f.write(f"- max_samples: `{best['max_samples']}`\n\n")
            f.write(f"**Performance:**\n")
            f.write(f"- Recall: `{best['recall']:.2%}`\n")
            f.write(f"- Precision: `{best['precision']:.2%}`\n")
            f.write(f"- F1-Score: `{best['f1_score']:.2%}`\n")
            f.write(f"- FPR: `{best['fpr']:.2%}`\n")
            f.write(f"- Accuracy: `{best['accuracy']:.2%}`\n\n")
        else:
            f.write("⚠️ No models found within FPR constraint\n\n")

        f.write("## Top 10 Models by Recall\n\n")
        top10 = sorted(results, key=lambda x: x["recall"], reverse=True)[:10]
        f.write(
            "| Rank | Contamination | n_estimators | max_samples | Recall | FPR | F1 |\n"
        )
        f.write(
            "|------|---------------|--------------|-------------|--------|-----|----||\n"
        )
        for i, config in enumerate(top10, 1):
            f.write(
                f"| {i} | {config['contamination']:.2f} | {config['n_estimators']} | "
                f"{config['max_samples']} | {config['recall']:.2%} | {config['fpr']:.2%} | "
                f"{config['f1_score']:.2%} |\n"
            )

    logger.info(f"Saved summary report to {summary_path}")


def retrain_best_model(
    train_embeddings: np.ndarray,
    best_config: dict[str, Any],
    output_path: Path,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_run_name: str | None,
) -> None:
    """Re-train best model with wandb logging."""
    logger.info("Re-training best model with wandb logging...")

    # Initialize wandb
    config = {
        "contamination": best_config["contamination"],
        "n_estimators": best_config["n_estimators"],
        "max_samples": best_config["max_samples"],
        "embedding_dim": train_embeddings.shape[1],
        "num_training_samples": train_embeddings.shape[0],
    }

    sink, wandb_run = init_wandb_sink(
        wandb_enabled,
        project=wandb_project,
        entity=wandb_entity,
        config=config,
    )

    # Train detector
    detector = IsolationForestDetector(
        contamination=best_config["contamination"],
        n_estimators=best_config["n_estimators"],
        max_samples=best_config["max_samples"],
        random_state=42,
    )
    detector.fit(train_embeddings)

    # Log training metrics
    training_scores = detector.scores(train_embeddings)
    score_stats = {
        "mean": float(np.mean(training_scores)),
        "std": float(np.std(training_scores)),
        "min": float(np.min(training_scores)),
        "max": float(np.max(training_scores)),
        "median": float(np.median(training_scores)),
    }

    if sink:
        sink.log(
            {
                "model/threshold": float(detector.threshold_),
                "training/score_mean": score_stats["mean"],
                "training/score_std": score_stats["std"],
                "training/score_min": score_stats["min"],
                "training/score_max": score_stats["max"],
                "training/score_median": score_stats["median"],
                "search/best_recall": best_config["recall"],
                "search/best_fpr": best_config["fpr"],
                "search/best_f1": best_config["f1_score"],
            }
        )

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "detector": detector,
            "metadata": {
                "contamination": best_config["contamination"],
                "n_estimators": best_config["n_estimators"],
                "max_samples": best_config["max_samples"],
                "random_state": 42,
                "num_training_samples": train_embeddings.shape[0],
                "embedding_dim": train_embeddings.shape[1],
                "threshold": float(detector.threshold_),
                "training_score_stats": score_stats,
                "search_results": best_config,
            },
        },
        output_path,
    )

    logger.info(f"Saved best model to {output_path}")

    if wandb_run is not None:
        wandb_run.finish()


@app.command()
def main(
    train_embeddings_path: Path = typer.Argument(
        ..., help="Path to precomputed training embeddings (.npz)"
    ),
    test_embeddings_path: Path = typer.Argument(
        ..., help="Path to precomputed test embeddings (.npz)"
    ),
    max_fpr: float = typer.Option(0.05, help="Maximum acceptable FPR"),
    output_dir: Path = typer.Option(
        None, help="Output directory (default: same as train embeddings)"
    ),
    wandb_enabled: bool = typer.Option(
        True, "--wandb/--no-wandb", help="Enable wandb for best model"
    ),
    wandb_project: str = typer.Option("neuralshield", help="W&B project name"),
    wandb_entity: str | None = typer.Option(None, help="W&B entity/team"),
    wandb_run_name: str | None = typer.Option(None, help="W&B run name"),
) -> None:
    """Hyperparameter search for anomaly detection models."""
    # Set output directory
    if output_dir is None:
        output_dir = train_embeddings_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("HYPERPARAMETER SEARCH FOR ANOMALY DETECTION")
    logger.info("=" * 80)

    # Load training embeddings
    train_embeddings, _ = load_training_embeddings(train_embeddings_path)

    # Load test embeddings
    test_embeddings, test_labels = load_test_embeddings(test_embeddings_path)

    # Run grid search
    results = grid_search(train_embeddings, test_embeddings, test_labels)

    # Find Pareto frontier
    logger.info(f"Finding Pareto frontier with FPR ≤ {max_fpr}")
    pareto_frontier = find_pareto_frontier(results, max_fpr)

    if pareto_frontier:
        logger.info(f"Found {len(pareto_frontier)} Pareto-optimal configurations")
        best = pareto_frontier[0]
        logger.info(
            f"Best model: contamination={best['contamination']}, "
            f"n_estimators={best['n_estimators']}, "
            f"max_samples={best['max_samples']}"
        )
        logger.info(
            f"Performance: Recall={best['recall']:.2%}, "
            f"FPR={best['fpr']:.2%}, F1={best['f1_score']:.2%}"
        )
    else:
        logger.warning("No Pareto-optimal configurations found within FPR constraint")
        best = sorted(results, key=lambda x: x["recall"], reverse=True)[0]
        logger.info(
            f"Best model by recall: contamination={best['contamination']}, "
            f"n_estimators={best['n_estimators']}, "
            f"max_samples={best['max_samples']}"
        )

    # Save results
    save_results(results, pareto_frontier, output_dir, max_fpr)

    # Create visualization
    plot_path = output_dir / "pareto_frontier_plot.png"
    plot_results(results, pareto_frontier, plot_path, max_fpr)

    # Re-train best model with wandb
    model_path = output_dir / "best_model.joblib"
    retrain_best_model(
        train_embeddings,
        best,
        model_path,
        wandb_enabled,
        wandb_project,
        wandb_entity,
        wandb_run_name,
    )

    logger.info("=" * 80)
    logger.info("HYPERPARAMETER SEARCH COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Best model saved to: {model_path}")


if __name__ == "__main__":
    app()
