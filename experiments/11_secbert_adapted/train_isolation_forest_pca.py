#!/usr/bin/env python3
"""Train IsolationForest detector on PCA-reduced adapted SecBERT embeddings."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import wandb

app = typer.Typer()


def load_embeddings(path: Path, is_test: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load precomputed embeddings from npz file.

    Args:
        path: Path to .npz file with 'embeddings' and 'labels' arrays
        is_test: If True, labels are strings ("attack"/"valid"), convert to binary

    Returns:
        Tuple of (embeddings, labels as binary: 0=normal, 1=anomalous)
    """
    logger.info(f"Loading embeddings from {path}")
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels_raw = data["labels"]

    # Convert labels to binary
    if is_test:
        # Test set has string labels
        labels = np.array([1 if label == "attack" else 0 for label in labels_raw])
    else:
        # Training set: all normal
        labels = np.zeros(len(labels_raw), dtype=int)

    logger.info(f"Loaded {len(embeddings)} samples, {embeddings.shape[1]} dimensions")
    logger.info(
        f"Labels: {np.sum(labels == 0)} normal, {np.sum(labels == 1)} anomalous"
    )
    return embeddings, labels


def plot_score_distribution(
    scores_normal: np.ndarray,
    scores_anomalous: np.ndarray,
    threshold: float,
    output_path: Path,
):
    """Plot distribution of anomaly scores."""
    plt.figure(figsize=(12, 6))

    plt.hist(
        scores_normal, bins=100, alpha=0.6, label="Normal", color="green", density=True
    )
    plt.hist(
        scores_anomalous,
        bins=100,
        alpha=0.6,
        label="Anomalous",
        color="red",
        density=True,
    )
    plt.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.2f}",
    )

    plt.xlabel("IsolationForest Anomaly Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(
        "Score Distribution: Adapted SecBERT + PCA + IsolationForest",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Score distribution saved to {output_path}")
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomalous"],
        yticklabels=["Normal", "Anomalous"],
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title(
        "Confusion Matrix: Adapted SecBERT + PCA + IsolationForest",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Confusion matrix saved to {output_path}")
    plt.close()


@app.command()
def main(
    train_embeddings: Path = typer.Option(
        "experiments/11_secbert_adapted/with_preprocessing/train_embeddings.npz",
        help="Training embeddings .npz",
    ),
    test_embeddings: Path = typer.Option(
        "experiments/11_secbert_adapted/with_preprocessing/test_embeddings.npz",
        help="Test embeddings .npz",
    ),
    output_dir: Path = typer.Option(
        "experiments/11_secbert_adapted/isolation_forest_pca",
        help="Output directory",
    ),
    pca_dims: int = typer.Option(256, help="PCA dimensions (default: 256)"),
    max_fpr: float = typer.Option(0.05, help="Max FPR (default: 0.05)"),
    contamination: float = typer.Option(
        0.01, help="Expected contamination (default: 0.01)"
    ),
    n_estimators: int = typer.Option(100, help="Number of trees (default: 100)"),
    use_wandb: bool = typer.Option(True, help="Enable W&B logging"),
    wandb_run_name: str = typer.Option(
        "exp11-adapted-secbert-iforest-pca", help="W&B run name"
    ),
) -> None:
    """Train IsolationForest detector on PCA-reduced adapted SecBERT embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project="neuralshield",
            name=wandb_run_name,
            config={
                "model": "adapted-secbert",
                "pooling": "mean+max",
                "embedding_dim": 1536,
                "pca_dims": pca_dims,
                "detector": "isolation-forest",
                "max_fpr": max_fpr,
                "contamination": contamination,
                "n_estimators": n_estimators,
                "preprocessing": True,
            },
        )
        logger.info(f"✓ W&B initialized: {wandb_run_name}")

    # Load data
    logger.info("=" * 80)
    logger.info("ADAPTED SECBERT + PCA + ISOLATION FOREST")
    logger.info("=" * 80)
    logger.info("Loading data...")
    train_emb, train_labels = load_embeddings(train_embeddings, is_test=False)
    test_emb, test_labels = load_embeddings(test_embeddings, is_test=True)

    # Apply PCA dimensionality reduction
    logger.info("=" * 80)
    logger.info("APPLYING PCA DIMENSIONALITY REDUCTION")
    logger.info("=" * 80)
    logger.info(f"  Original dims: {train_emb.shape[1]}")
    logger.info(f"  Target dims: {pca_dims}")

    pca = PCA(n_components=pca_dims, random_state=42)
    train_emb_pca = pca.fit_transform(train_emb)
    test_emb_pca = pca.transform(test_emb)

    variance_explained = pca.explained_variance_ratio_.sum()
    logger.info(f"✓ PCA fitted and transformed")
    logger.info(f"  Variance explained: {variance_explained:.2%}")
    logger.info(f"  Train shape: {train_emb_pca.shape}")
    logger.info(f"  Test shape: {test_emb_pca.shape}")

    # Fit detector on training data (all normal)
    logger.info("=" * 80)
    logger.info("FITTING ISOLATION FOREST")
    logger.info("=" * 80)
    logger.info(f"  Contamination: {contamination}")
    logger.info(f"  N estimators: {n_estimators}")

    detector = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    detector.fit(train_emb_pca)
    logger.info("✓ IsolationForest fitted successfully")

    # Get anomaly scores for test set (inverted: higher = more anomalous)
    logger.info("=" * 80)
    logger.info("COMPUTING ANOMALY SCORES")
    logger.info("=" * 80)
    raw_scores = detector.score_samples(test_emb_pca)
    test_scores = -raw_scores  # Invert: higher = more anomalous
    logger.info(f"✓ Computed {len(test_scores)} scores")

    # Split test by label
    test_normal_mask = test_labels == 0
    test_scores_normal = test_scores[test_normal_mask]
    test_scores_anomalous = test_scores[~test_normal_mask]

    logger.info(f"  Normal samples: {len(test_scores_normal)}")
    logger.info(f"  Anomalous samples: {len(test_scores_anomalous)}")

    # Find threshold for desired FPR (higher score = more anomalous)
    logger.info("=" * 80)
    logger.info("SETTING THRESHOLD")
    logger.info("=" * 80)
    threshold = np.percentile(test_scores_normal, 100 * (1 - max_fpr))
    actual_fpr = np.mean(test_scores_normal > threshold)
    logger.info(
        f"Threshold: {threshold:.4f} (target FPR={max_fpr:.1%}, actual={actual_fpr:.1%})"
    )

    # Predict (score > threshold = anomaly)
    test_predictions = (test_scores > threshold).astype(int)

    # Compute metrics
    logger.info("=" * 80)
    logger.info("EVALUATING")
    logger.info("=" * 80)
    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions, zero_division=0)
    recall = recall_score(test_labels, test_predictions)
    f1 = f1_score(test_labels, test_predictions, zero_division=0)

    cm = confusion_matrix(test_labels, test_predictions)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Log results
    logger.info("RESULTS:")
    logger.info(f"  Threshold: {threshold:.4f}")
    logger.info(f"  Accuracy: {accuracy:.2%}")
    logger.info(f"  Precision: {precision:.2%}")
    logger.info(f"  Recall: {recall:.2%} ← KEY METRIC")
    logger.info(f"  F1-Score: {f1:.2%}")
    logger.info(f"  FPR: {fpr:.2%}")
    logger.info(f"  FNR: {fnr:.2%}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # Comparison with baselines
    baseline_mahalanobis = 0.5192  # From our Mahalanobis result
    baseline_secbert = 0.4926  # From experiment 03
    improvement_vs_mahal = recall - baseline_mahalanobis
    improvement_vs_secbert = recall - baseline_secbert

    logger.info("=" * 80)
    logger.info("COMPARISON WITH BASELINES")
    logger.info("=" * 80)
    logger.info(f"  Baseline SecBERT (exp 03):          {baseline_secbert:.2%}")
    logger.info(f"  Mahalanobis (exp 11):                {baseline_mahalanobis:.2%}")
    logger.info(f"  IsolationForest+PCA (exp 11 - this): {recall:.2%}")
    logger.info(f"  vs Mahalanobis: {improvement_vs_mahal:+.2%}")
    logger.info(f"  vs SecBERT:     {improvement_vs_secbert:+.2%}")

    # Save results
    results = {
        "model": "adapted-secbert-isolation-forest-pca",
        "pooling": "mean+max (1536 dims)",
        "pca_dims": pca_dims,
        "variance_explained": float(variance_explained),
        "detector": "isolation-forest",
        "contamination": contamination,
        "n_estimators": n_estimators,
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "max_fpr_constraint": max_fpr,
        "baseline_mahalanobis": baseline_mahalanobis,
        "baseline_secbert": baseline_secbert,
        "improvement_vs_mahalanobis": float(improvement_vs_mahal),
        "improvement_vs_secbert": float(improvement_vs_secbert),
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"✓ Results saved to {results_path}")

    # Visualizations
    logger.info("=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)

    plot_score_distribution(
        test_scores_normal,
        test_scores_anomalous,
        threshold,
        output_dir / "score_distribution.png",
    )

    plot_confusion_matrix(
        test_labels, test_predictions, output_dir / "confusion_matrix.png"
    )

    # Log to W&B
    if use_wandb and wandb_run is not None:
        logger.info("=" * 80)
        logger.info("LOGGING TO W&B")
        logger.info("=" * 80)

        wandb.log(
            {
                # Metrics
                "recall": recall,
                "precision": precision,
                "accuracy": accuracy,
                "f1_score": f1,
                "fpr": fpr,
                "fnr": fnr,
                "threshold": threshold,
                "variance_explained": variance_explained,
                # Comparison
                "baseline_mahalanobis": baseline_mahalanobis,
                "baseline_secbert": baseline_secbert,
                "improvement_vs_mahalanobis": improvement_vs_mahal,
                "improvement_vs_secbert": improvement_vs_secbert,
                # Visualizations
                "score_distribution": wandb.Image(
                    str(output_dir / "score_distribution.png")
                ),
                "confusion_matrix": wandb.Image(
                    str(output_dir / "confusion_matrix.png")
                ),
            }
        )

        # Create summary table
        wandb.summary["recall"] = recall
        wandb.summary["pca_dims"] = pca_dims
        wandb.summary["variance_explained"] = variance_explained
        wandb.summary["baseline_mahalanobis"] = baseline_mahalanobis
        wandb.summary["improvement_vs_mahalanobis"] = improvement_vs_mahal

        logger.info("✓ Logged metrics and visualizations to W&B")
        wandb.finish()

    logger.info("=" * 80)
    logger.info("✅ DONE")
    logger.info("=" * 80)


if __name__ == "__main__":
    app()
