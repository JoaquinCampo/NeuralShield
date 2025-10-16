"""Test GMM detector with SecBERT embeddings."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from neuralshield.anomaly import GMMDetector

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available, skipping wandb logging")


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

    plt.xlabel("Anomaly Score (Negative Log-Likelihood)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(
        "GMM Score Distribution: Normal vs Anomalous", fontsize=14, fontweight="bold"
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
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Confusion matrix saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Test GMM detector on SecBERT embeddings"
    )
    parser.add_argument("train_embeddings", type=Path, help="Training embeddings .npz")
    parser.add_argument("test_embeddings", type=Path, help="Test embeddings .npz")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--n-components",
        type=int,
        default=3,
        help="Number of Gaussian components (default: 3)",
    )
    parser.add_argument(
        "--covariance-type",
        type=str,
        default="full",
        choices=["full", "tied", "diag", "spherical"],
        help="Covariance type (default: full)",
    )
    parser.add_argument(
        "--max-fpr", type=float, default=0.05, help="Max FPR (default: 0.05)"
    )
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--wandb-run-name", type=str, help="wandb run name")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if args.wandb:
        if not WANDB_AVAILABLE:
            logger.error("wandb requested but not available")
            return
        wandb.init(
            project="neuralshield",
            name=args.wandb_run_name or f"gmm-secbert-{args.n_components}comp",
            config={
                "detector": "gmm",
                "n_components": args.n_components,
                "covariance_type": args.covariance_type,
                "max_fpr": args.max_fpr,
            },
        )

    # Load data
    logger.info("=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    train_embeddings, train_labels = load_embeddings(
        args.train_embeddings, is_test=False
    )
    test_embeddings, test_labels = load_embeddings(args.test_embeddings, is_test=True)

    # Fit GMM detector
    logger.info("=" * 80)
    logger.info("FITTING GMM DETECTOR")
    logger.info("=" * 80)
    detector = GMMDetector(
        n_components=args.n_components,
        covariance_type=args.covariance_type,
    )
    detector.fit(train_embeddings)

    # Set threshold using normal test samples
    logger.info("=" * 80)
    logger.info("SETTING THRESHOLD")
    logger.info("=" * 80)
    test_normal_mask = test_labels == 0
    test_normal_embeddings = test_embeddings[test_normal_mask]
    threshold = detector.set_threshold(test_normal_embeddings, max_fpr=args.max_fpr)

    # Get scores for all test samples
    logger.info("=" * 80)
    logger.info("COMPUTING ANOMALY SCORES")
    logger.info("=" * 80)
    test_scores = detector.scores(test_embeddings)
    test_scores_normal = test_scores[test_normal_mask]
    test_scores_anomalous = test_scores[~test_normal_mask]

    # Predict
    test_predictions = detector.predict(test_embeddings)

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
    logger.info(f"  N Components: {args.n_components}")
    logger.info(f"  Covariance Type: {args.covariance_type}")
    logger.info(f"  Threshold: {threshold:.4f}")
    logger.info(f"  Accuracy: {accuracy:.2%}")
    logger.info(f"  Precision: {precision:.2%}")
    logger.info(f"  Recall: {recall:.2%}")
    logger.info(f"  F1-Score: {f1:.2%}")
    logger.info(f"  FPR: {fpr:.2%}")
    logger.info(f"  FNR: {fnr:.2%}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # Save results
    results = {
        "detector": "gmm",
        "n_components": args.n_components,
        "covariance_type": args.covariance_type,
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
        "max_fpr_constraint": args.max_fpr,
    }

    results_path = args.output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Save model
    model_path = args.output_dir / "gmm_detector.joblib"
    detector.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Visualizations
    logger.info("=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)

    plot_score_distribution(
        test_scores_normal,
        test_scores_anomalous,
        threshold,
        args.output_dir / "score_distribution.png",
    )

    plot_confusion_matrix(
        test_labels, test_predictions, args.output_dir / "confusion_matrix.png"
    )

    # Log to wandb
    if args.wandb:
        wandb.log(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "fpr": fpr,
                "fnr": fnr,
                "threshold": threshold,
                "n_components": args.n_components,
                "score_distribution": wandb.Image(
                    str(args.output_dir / "score_distribution.png")
                ),
                "confusion_matrix": wandb.Image(
                    str(args.output_dir / "confusion_matrix.png")
                ),
            }
        )
        wandb.finish()

    logger.info("=" * 80)
    logger.info("DONE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
