from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer
from loguru import logger

from neuralshield.anomaly import IsolationForestDetector
from neuralshield.encoding.observability import init_wandb_sink

app = typer.Typer()


@app.command()
def main(
    embeddings_path: Path = typer.Argument(..., help="Input embeddings file (.npz)"),
    output_path: Path = typer.Argument(..., help="Output model file (.joblib)"),
    contamination: float = typer.Option(0.1, help="Contamination parameter"),
    n_estimators: int = typer.Option(300, help="Number of trees"),
    random_state: int = typer.Option(42, help="Random seed"),
    wandb_enabled: bool = typer.Option(
        False, "--wandb/--no-wandb", help="Enable Weights & Biases logging"
    ),
    wandb_project: str = typer.Option("neuralshield", help="W&B project name"),
    wandb_entity: str | None = typer.Option(None, help="W&B entity/team"),
) -> None:
    """Train an IsolationForest model from precomputed embeddings."""

    # Initialize wandb if enabled
    config = {
        "contamination": contamination,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "embeddings_path": str(embeddings_path),
        "output_path": str(output_path),
    }

    sink, wandb_run = init_wandb_sink(
        wandb_enabled,
        project=wandb_project,
        entity=wandb_entity,
        config=config,
    )

    logger.info("Loading embeddings from: {path}", path=str(embeddings_path))

    # Load embeddings
    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]

    logger.info(
        "Loaded: {samples} samples, {dim} dimensions",
        samples=embeddings.shape[0],
        dim=embeddings.shape[1],
    )

    # Log dataset info to wandb
    if sink:
        sink.log(
            {
                "dataset/total_samples": embeddings.shape[0],
                "dataset/embedding_dim": embeddings.shape[1],
            }
        )

    # Filter for normal samples only (handle both "normal" and "valid" labels)
    normal_mask = (labels == "normal") | (labels == "valid")
    normal_embeddings = embeddings[normal_mask]

    logger.info(
        "Training on {count} normal samples (filtered from {total})",
        count=normal_embeddings.shape[0],
        total=embeddings.shape[0],
    )

    # Log training set info
    if sink:
        sink.log(
            {
                "dataset/normal_samples": normal_embeddings.shape[0],
                "dataset/anomaly_samples": int(np.sum(~normal_mask)),
                "dataset/normal_ratio": float(
                    normal_embeddings.shape[0] / embeddings.shape[0]
                ),
            }
        )

    # Create and train model
    logger.info(
        "Training IsolationForest: contamination={cont}, n_estimators={est}",
        cont=contamination,
        est=n_estimators,
    )

    detector = IsolationForestDetector(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    detector.fit(normal_embeddings)

    logger.info(
        "Training complete. Threshold: {thresh:.6f}",
        thresh=detector.threshold_,
    )

    # Log model metrics
    if sink:
        sink.log(
            {
                "model/threshold": float(detector.threshold_),
            }
        )

    # Analyze training scores
    training_scores = detector.scores(normal_embeddings)
    score_stats = {
        "mean": float(np.mean(training_scores)),
        "std": float(np.std(training_scores)),
        "min": float(np.min(training_scores)),
        "max": float(np.max(training_scores)),
        "median": float(np.median(training_scores)),
        "p25": float(np.percentile(training_scores, 25)),
        "p75": float(np.percentile(training_scores, 75)),
    }

    logger.info(
        "Training scores - mean: {mean:.6f}, std: {std:.6f}, "
        "min: {min:.6f}, max: {max:.6f}",
        **score_stats,
    )

    # Log training score statistics to wandb
    if sink:
        sink.log(
            {
                "training/score_mean": score_stats["mean"],
                "training/score_std": score_stats["std"],
                "training/score_min": score_stats["min"],
                "training/score_max": score_stats["max"],
                "training/score_median": score_stats["median"],
                "training/score_p25": score_stats["p25"],
                "training/score_p75": score_stats["p75"],
            }
        )

    # Create and log score distribution visualization
    logger.info("Creating score distribution visualization")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram with KDE
    sns.histplot(training_scores, bins=50, kde=True, ax=axes[0], color="steelblue")
    axes[0].axvline(
        detector.threshold_,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({detector.threshold_:.4f})",
    )
    axes[0].axvline(
        score_stats["mean"],
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean ({score_stats['mean']:.4f})",
    )
    axes[0].set_xlabel("Anomaly Score", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title("Training Score Distribution", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot with violin overlay
    parts = axes[1].violinplot(
        [training_scores], positions=[0], widths=0.7, showmeans=True, showmedians=True
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.7)
    axes[1].axhline(
        detector.threshold_,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({detector.threshold_:.4f})",
    )
    axes[1].set_ylabel("Anomaly Score", fontsize=11)
    axes[1].set_title(
        "Score Distribution (Violin Plot)", fontsize=13, fontweight="bold"
    )
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["Training Samples"])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Log to wandb
    if sink:
        import wandb

        sink.log({"training/score_distribution": wandb.Image(fig)})
        logger.info("Logged score distribution to wandb")

    plt.close(fig)

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving model to: {path}", path=str(output_path))

    joblib.dump(
        {
            "detector": detector,
            "metadata": {
                "contamination": contamination,
                "n_estimators": n_estimators,
                "random_state": random_state,
                "num_training_samples": normal_embeddings.shape[0],
                "embedding_dim": normal_embeddings.shape[1],
                "threshold": float(detector.threshold_),
                "training_score_stats": score_stats,
            },
        },
        output_path,
    )

    logger.info("✓ Model saved successfully!")

    # Finish wandb run if enabled
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    app()
