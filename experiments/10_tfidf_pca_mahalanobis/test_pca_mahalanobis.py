import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

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

    plt.xlabel("Mahalanobis Distance", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Score Distribution: Normal vs Anomalous", fontsize=14, fontweight="bold")
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
        description="Test Mahalanobis on TF-IDF with PCA reduction"
    )
    parser.add_argument("train_embeddings", type=Path, help="Training embeddings .npz")
    parser.add_argument("test_embeddings", type=Path, help="Test embeddings .npz")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--n-components", type=int, default=300, help="PCA components (default: 300)"
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
            name=args.wandb_run_name or f"tfidf-pca-{args.n_components}-mahalanobis",
            config={
                "detector": "pca_mahalanobis",
                "n_components": args.n_components,
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

    # Apply PCA
    logger.info("=" * 80)
    logger.info(f"APPLYING PCA REDUCTION TO {args.n_components} COMPONENTS")
    logger.info("=" * 80)
    pca = PCA(n_components=args.n_components, random_state=42)
    train_embeddings_pca = pca.fit_transform(train_embeddings)
    test_embeddings_pca = pca.transform(test_embeddings)

    explained_variance = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA explained variance: {explained_variance:.2%}")
    logger.info(
        f"Reduced dimensions: {train_embeddings.shape[1]} -> {train_embeddings_pca.shape[1]}"
    )

    # Fit detector on training data (all normal)
    logger.info("=" * 80)
    logger.info("FITTING MAHALANOBIS (EMPIRICAL COVARIANCE)")
    logger.info("=" * 80)
    detector = EmpiricalCovariance()
    detector.fit(train_embeddings_pca)
    logger.info("Covariance fitted successfully")

    # Get Mahalanobis distances for test set
    logger.info("=" * 80)
    logger.info("COMPUTING MAHALANOBIS DISTANCES")
    logger.info("=" * 80)
    test_scores = detector.mahalanobis(test_embeddings_pca)

    # Split test by label
    test_normal_mask = test_labels == 0
    test_scores_normal = test_scores[test_normal_mask]
    test_scores_anomalous = test_scores[~test_normal_mask]

    # Find threshold for desired FPR (higher Mahalanobis distance = more anomalous)
    logger.info("=" * 80)
    logger.info("SETTING THRESHOLD")
    logger.info("=" * 80)
    threshold = np.percentile(test_scores_normal, 100 * (1 - args.max_fpr))
    actual_fpr = np.mean(test_scores_normal > threshold)
    logger.info(
        f"Threshold: {threshold:.4f} (target FPR={args.max_fpr:.1%}, actual={actual_fpr:.1%})"
    )

    # Predict (distance > threshold = anomaly)
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
    logger.info(f"  PCA Components: {args.n_components}")
    logger.info(f"  Explained Variance: {explained_variance:.2%}")
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
        "n_components": args.n_components,
        "explained_variance": float(explained_variance),
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
                "n_components": args.n_components,
                "explained_variance": explained_variance,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "fpr": fpr,
                "fnr": fnr,
                "threshold": threshold,
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
