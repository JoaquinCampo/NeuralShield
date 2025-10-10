#!/usr/bin/env python3
"""Full end-to-end experiment for ColBERT + MUVERA embeddings.

This script:
1. Generates train/test embeddings (with and without preprocessing)
2. Runs hyperparameter search for both scenarios
3. Retrains best models with wandb logging
4. Tests both models with comprehensive evaluation and wandb logging
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import neuralshield.encoding.data.factory as data_factory
from neuralshield.anomaly.model import IsolationForestDetector
from neuralshield.encoding.models.colbert_muvera import ColBERTMuveraEncoder
from neuralshield.encoding.observability import init_wandb_sink
from neuralshield.evaluation import ClassificationEvaluator, EvaluationConfig
from neuralshield.evaluation.metrics import calculate_confusion_matrix
from neuralshield.preprocessing.pipeline import preprocess

app = typer.Typer()

# Hyperparameter search space
SEARCH_SPACE = {
    "contamination": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    "n_estimators": [100, 200, 300, 500],
    "max_samples": ["auto", 256, 512, 1024],
}


def generate_embeddings(
    dataset_path: Path,
    output_path: Path,
    encoder: ColBERTMuveraEncoder,
    use_pipeline: bool,
    batch_size: int,
) -> None:
    """Generate embeddings from a dataset."""
    logger.info(f"Generating embeddings: {output_path}")
    logger.info(f"  Dataset: {dataset_path}")
    logger.info(f"  Preprocessing: {'enabled' if use_pipeline else 'disabled'}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup reader
    reader_factory = data_factory.get_reader("jsonl")
    data_reader = reader_factory(
        path=dataset_path,
        pipeline=preprocess if use_pipeline else None,
        use_pipeline=use_pipeline,
        observer=None,
    )

    # Process dataset
    all_embeddings = []
    all_labels = []
    total_processed = 0

    for batch_requests, batch_labels in tqdm(
        data_reader.iter_batches(batch_size),
        desc=f"Encoding {dataset_path.name}",
        unit="batch",
    ):
        embeddings = encoder.encode(batch_requests)
        all_embeddings.append(embeddings)
        all_labels.extend(batch_labels)
        total_processed += len(batch_requests)

    # Concatenate and save
    final_embeddings = np.vstack(all_embeddings)
    np.savez_compressed(
        output_path,
        embeddings=final_embeddings,
        labels=np.array(all_labels),
    )

    logger.info(
        f"  Saved: {final_embeddings.shape[0]} samples, "
        f"{final_embeddings.shape[1]} dims, "
        f"{output_path.stat().st_size / (1024**2):.2f} MB"
    )


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings from .npz file."""
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["labels"]


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
    detector = IsolationForestDetector(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
    )
    detector.fit(train_embeddings)

    predictions = detector.predict(test_embeddings)

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

    logger.info(f"Starting grid search over {len(configs)} configurations")

    for config in tqdm(configs, desc="Grid search"):
        result = train_and_evaluate(
            train_embeddings, test_embeddings, test_labels, **config
        )
        results.append(result)

    return results


def find_best_model(
    results: list[dict[str, Any]], max_fpr: float = 0.05
) -> dict[str, Any]:
    """Find best model: highest recall with FPR <= max_fpr."""
    feasible = [r for r in results if r["fpr"] <= max_fpr]

    if not feasible:
        logger.warning(f"No models with FPR ≤ {max_fpr}, relaxing constraint")
        feasible = sorted(results, key=lambda x: x["fpr"])[:10]

    best = sorted(feasible, key=lambda x: x["recall"], reverse=True)[0]
    return best


def retrain_and_log_model(
    train_embeddings: np.ndarray,
    best_config: dict[str, Any],
    output_path: Path,
    scenario_name: str,
    wandb_project: str,
    wandb_entity: str | None,
) -> IsolationForestDetector:
    """Retrain best model with wandb logging."""
    logger.info(f"Retraining best model for: {scenario_name}")

    config = {
        "scenario": scenario_name,
        "contamination": best_config["contamination"],
        "n_estimators": best_config["n_estimators"],
        "max_samples": best_config["max_samples"],
        "embedding_dim": train_embeddings.shape[1],
        "num_training_samples": train_embeddings.shape[0],
    }

    sink, wandb_run = init_wandb_sink(
        True,
        project=wandb_project,
        entity=wandb_entity,
        config=config,
    )

    if wandb_run:
        wandb_run.name = f"colbert-{scenario_name}"

    # Train
    detector = IsolationForestDetector(
        contamination=best_config["contamination"],
        n_estimators=best_config["n_estimators"],
        max_samples=best_config["max_samples"],
        random_state=42,
    )
    detector.fit(train_embeddings)

    # Log training metrics
    training_scores = detector.scores(train_embeddings)
    if sink:
        sink.log(
            {
                "model/threshold": float(detector.threshold_),
                "training/score_mean": float(np.mean(training_scores)),
                "training/score_std": float(np.std(training_scores)),
                "training/score_median": float(np.median(training_scores)),
                "search/best_recall": best_config["recall"],
                "search/best_fpr": best_config["fpr"],
                "search/best_f1": best_config["f1_score"],
            }
        )

        # Training score distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(training_scores, bins=50, alpha=0.7, color="blue", edgecolor="black")
        ax.axvline(
            detector.threshold_,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold: {detector.threshold_:.4f}",
        )
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Training Score Distribution - {scenario_name}")
        ax.legend()
        ax.grid(alpha=0.3)

        import wandb as wandb_module

        sink.log({"training/score_distribution": wandb_module.Image(fig)})
        plt.close(fig)

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "detector": detector,
            "metadata": {
                "scenario": scenario_name,
                "contamination": best_config["contamination"],
                "n_estimators": best_config["n_estimators"],
                "max_samples": best_config["max_samples"],
                "threshold": float(detector.threshold_),
                "search_results": best_config,
            },
        },
        output_path,
    )

    logger.info(f"  Saved model: {output_path}")

    if wandb_run:
        wandb_run.finish()

    return detector


def test_and_log_model(
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    detector: IsolationForestDetector,
    scenario_name: str,
    output_dir: Path,
    wandb_project: str,
    wandb_entity: str | None,
) -> None:
    """Test model and log results to wandb."""
    logger.info(f"Testing model for: {scenario_name}")

    import wandb as wandb_module

    wandb_run = wandb_module.init(
        project=wandb_project,
        entity=wandb_entity,
        name=f"colbert-{scenario_name}-test",
        config={"scenario": scenario_name, "phase": "testing"},
    )

    # Predict
    predictions = detector.predict(test_embeddings)
    scores = detector.scores(test_embeddings)

    # Normalize labels
    normalized_labels = []
    for label in test_labels:
        if label in ["attack", "anomalous"]:
            normalized_labels.append("attack")
        else:
            normalized_labels.append("valid")

    # Calculate metrics
    tp, fp, tn, fn = calculate_confusion_matrix(
        predictions=predictions.tolist()
        if hasattr(predictions, "tolist")
        else predictions,
        labels=normalized_labels,
        positive_label="attack",
        negative_label="valid",
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    logger.info(f"  Recall: {recall:.2%}, FPR: {fpr:.2%}, F1: {f1_score:.2%}")

    # Log metrics
    wandb_module.log(
        {
            "test/precision": precision,
            "test/recall": recall,
            "test/f1_score": f1_score,
            "test/accuracy": accuracy,
            "test/specificity": specificity,
            "test/fpr": fpr,
            "test/tp": tp,
            "test/fp": fp,
            "test/tn": tn,
            "test/fn": fn,
        }
    )

    # Visualizations
    normal_mask = np.array([label in ["normal", "valid"] for label in test_labels])
    attack_mask = ~normal_mask
    normal_scores = scores[normal_mask]
    attack_scores = scores[attack_mask]

    # Score distribution (4-panel)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Overlapping histograms
    sns.histplot(
        normal_scores,
        bins=50,
        kde=True,
        ax=axes[0, 0],
        color="green",
        alpha=0.6,
        label=f"Normal (n={len(normal_scores)})",
    )
    sns.histplot(
        attack_scores,
        bins=50,
        kde=True,
        ax=axes[0, 0],
        color="red",
        alpha=0.6,
        label=f"Attack (n={len(attack_scores)})",
    )
    axes[0, 0].axvline(
        detector.threshold_,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Threshold",
    )
    axes[0, 0].set_xlabel("Anomaly Score")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title(f"Score Distribution - {scenario_name}", fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Separate histograms
    axes[0, 1].hist(
        [normal_scores, attack_scores],
        bins=50,
        label=["Normal", "Attack"],
        color=["green", "red"],
        alpha=0.7,
    )
    axes[0, 1].axvline(detector.threshold_, color="black", linestyle="--", linewidth=2)
    axes[0, 1].set_xlabel("Anomaly Score")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Non-Stacked Histogram", fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Violin plot
    df_violin = pd.DataFrame(
        {
            "Score": np.concatenate([normal_scores, attack_scores]),
            "Type": ["Normal"] * len(normal_scores) + ["Attack"] * len(attack_scores),
        }
    )
    sns.violinplot(
        data=df_violin,
        x="Type",
        y="Score",
        ax=axes[1, 0],
        palette={"Normal": "green", "Attack": "red"},
    )
    axes[1, 0].axhline(detector.threshold_, color="black", linestyle="--", linewidth=2)
    axes[1, 0].set_title("Violin Plot", fontweight="bold")
    axes[1, 0].grid(alpha=0.3, axis="y")

    # Box plot
    sns.boxplot(
        data=df_violin,
        x="Type",
        y="Score",
        ax=axes[1, 1],
        palette={"Normal": "green", "Attack": "red"},
    )
    axes[1, 1].axhline(detector.threshold_, color="black", linestyle="--", linewidth=2)
    axes[1, 1].set_title("Box Plot", fontweight="bold")
    axes[1, 1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    score_plot_path = output_dir / f"test_scores_{scenario_name}.png"
    fig.savefig(score_plot_path, dpi=150, bbox_inches="tight")
    wandb_module.log({"test/score_distribution": wandb_module.Image(fig)})
    plt.close(fig)

    # Confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay

    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=np.array([[tn, fp], [fn, tp]]),
        display_labels=["Normal", "Attack"],
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    cm_display.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix - {scenario_name}", fontsize=14, fontweight="bold")

    cm_path = output_dir / f"confusion_matrix_{scenario_name}.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    wandb_module.log({"test/confusion_matrix": wandb_module.Image(fig)})
    plt.close(fig)

    wandb_run.finish()


@app.command()
def main(
    train_dataset: Path = typer.Argument(
        ..., help="Training dataset path (JSONL)", exists=True
    ),
    test_dataset: Path = typer.Argument(
        ..., help="Test dataset path (JSONL)", exists=True
    ),
    output_dir: Path = typer.Option(
        Path("experiments/04_colbert_muvera_comparison"),
        help="Output directory",
    ),
    model_name: str = typer.Option("colbert-ir/colbertv2.0", help="ColBERT model name"),
    k_sim: int = typer.Option(5, help="MUVERA k_sim parameter (clusters = 2^k_sim)"),
    dim_proj: int = typer.Option(16, help="MUVERA projection dimension"),
    r_reps: int = typer.Option(20, help="MUVERA repetitions"),
    batch_size: int = typer.Option(16, help="Batch size for embedding generation"),
    max_fpr: float = typer.Option(0.05, help="Maximum acceptable FPR"),
    wandb_project: str = typer.Option("neuralshield", help="W&B project name"),
    wandb_entity: str | None = typer.Option(None, help="W&B entity/team"),
    device: str = typer.Option("cpu", help="Device (cpu/cuda/mps)"),
) -> None:
    """Run full ColBERT + MUVERA experiment."""
    logger.info("=" * 80)
    logger.info("COLBERT + MUVERA FULL EXPERIMENT")
    logger.info("=" * 80)

    # Phase 1: Generate embeddings
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: GENERATING EMBEDDINGS")
    logger.info("=" * 80)

    encoder = ColBERTMuveraEncoder(
        model_name=model_name,
        device=device,
        k_sim=k_sim,
        dim_proj=dim_proj,
        r_reps=r_reps,
    )

    output_dim = r_reps * (2**k_sim) * dim_proj
    logger.info(f"MUVERA output dimension: {output_dim}")

    scenarios = {
        "without_preprocessing": False,
        "with_preprocessing": True,
    }

    embedding_paths = {}

    for scenario_name, use_pipeline in scenarios.items():
        scenario_dir = output_dir / scenario_name
        train_emb_path = scenario_dir / "train_embeddings.npz"
        test_emb_path = scenario_dir / "test_embeddings.npz"

        if not train_emb_path.exists():
            generate_embeddings(
                train_dataset, train_emb_path, encoder, use_pipeline, batch_size
            )
        else:
            logger.info(f"Skipping {train_emb_path} (already exists)")

        if not test_emb_path.exists():
            generate_embeddings(
                test_dataset, test_emb_path, encoder, use_pipeline, batch_size
            )
        else:
            logger.info(f"Skipping {test_emb_path} (already exists)")

        embedding_paths[scenario_name] = {
            "train": train_emb_path,
            "test": test_emb_path,
        }

    # Phase 2 & 3: Hyperparameter search + retrain for each scenario
    for scenario_name, use_pipeline in scenarios.items():
        logger.info("\n" + "=" * 80)
        logger.info(f"PHASE 2-3: {scenario_name.upper()}")
        logger.info("=" * 80)

        scenario_dir = output_dir / scenario_name
        train_path = embedding_paths[scenario_name]["train"]
        test_path = embedding_paths[scenario_name]["test"]

        # Load embeddings
        logger.info("Loading embeddings...")
        train_embeddings, _ = load_embeddings(train_path)
        test_embeddings, test_labels = load_embeddings(test_path)

        # Grid search
        logger.info("Running hyperparameter search...")
        results = grid_search(train_embeddings, test_embeddings, test_labels.tolist())

        # Find best
        best = find_best_model(results, max_fpr)
        logger.info(
            f"Best: recall={best['recall']:.2%}, fpr={best['fpr']:.2%}, "
            f"contamination={best['contamination']}, n_estimators={best['n_estimators']}"
        )

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(scenario_dir / "search_results.csv", index=False)

        # Retrain with wandb
        model_path = scenario_dir / "best_model.joblib"
        detector = retrain_and_log_model(
            train_embeddings,
            best,
            model_path,
            scenario_name,
            wandb_project,
            wandb_entity,
        )

        # Phase 4: Test with wandb
        logger.info(f"Testing model for {scenario_name}...")
        test_and_log_model(
            test_embeddings,
            test_labels,
            detector,
            scenario_name,
            scenario_dir,
            wandb_project,
            wandb_entity,
        )

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    app()
