"""GPU-accelerated Mahalanobis distance detector using PyTorch."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from loguru import logger
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
    logger.warning("wandb not available")


class GPUMahalanobisDetector:
    """GPU-accelerated Mahalanobis distance detector using PyTorch."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.mean = None
        self.inv_cov = None
        self.threshold = None

    def fit(self, embeddings: np.ndarray) -> None:
        """Fit on normal training embeddings."""
        logger.info(f"Fitting on {len(embeddings)} samples, {embeddings.shape[1]} dims")

        # Move to GPU
        X = torch.from_numpy(embeddings).float().to(self.device)

        # Compute mean
        self.mean = X.mean(dim=0)
        logger.info("Mean computed")

        # Compute covariance (on GPU)
        X_centered = X - self.mean
        cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
        logger.info("Covariance computed")

        # Regularize
        reg = 1e-6 * torch.eye(cov.shape[0], device=self.device)
        cov_reg = cov + reg

        # Invert (on GPU)
        self.inv_cov = torch.linalg.inv(cov_reg)
        logger.info("Covariance inverted")

        # Check condition number
        cond = torch.linalg.cond(cov_reg).item()
        logger.info(f"Condition number: {cond:.2e}")

    def scores(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances on GPU."""
        # Move to GPU
        X = torch.from_numpy(embeddings).float().to(self.device)

        # Compute distances in batches to manage memory
        batch_size = 10000
        distances = []

        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            delta = batch - self.mean
            # Mahalanobis: sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
            mahal_sq = (delta @ self.inv_cov * delta).sum(dim=1)
            batch_distances = torch.sqrt(torch.clamp(mahal_sq, min=0))
            distances.append(batch_distances.cpu())

        return torch.cat(distances).numpy()

    def set_threshold(
        self, normal_embeddings: np.ndarray, max_fpr: float = 0.05
    ) -> float:
        """Set threshold based on desired FPR."""
        distances = self.scores(normal_embeddings)
        threshold = float(np.percentile(distances, 100 * (1 - max_fpr)))
        self.threshold = threshold

        actual_fpr = np.mean(distances > threshold)
        logger.info(
            f"Threshold: {threshold:.4f} (target FPR={max_fpr:.1%}, actual={actual_fpr:.1%})"
        )

        return threshold


def load_embeddings(path: Path, is_test: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings from npz file."""
    logger.info(f"Loading embeddings from {path}")
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels_raw = data["labels"]

    if is_test:
        labels = np.array([1 if label == "attack" else 0 for label in labels_raw])
    else:
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
    """Plot score distribution."""
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
    parser = argparse.ArgumentParser(description="GPU-accelerated Mahalanobis test")
    parser.add_argument("train_embeddings", type=Path, help="Training embeddings .npz")
    parser.add_argument("test_embeddings", type=Path, help="Test embeddings .npz")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--max-fpr", type=float, default=0.05, help="Max FPR (default: 0.05)"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
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
            name=args.wandb_run_name or "mahalanobis-gpu-test",
            config={
                "detector": "mahalanobis_gpu",
                "max_fpr": args.max_fpr,
                "device": args.device,
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

    # Fit detector
    logger.info("=" * 80)
    logger.info("FITTING MAHALANOBIS (GPU-ACCELERATED)")
    logger.info("=" * 80)
    detector = GPUMahalanobisDetector(device=args.device)
    detector.fit(train_embeddings)

    # Compute scores
    logger.info("=" * 80)
    logger.info("COMPUTING MAHALANOBIS DISTANCES")
    logger.info("=" * 80)
    test_scores = detector.scores(test_embeddings)

    # Split by label
    test_normal_mask = test_labels == 0
    test_scores_normal = test_scores[test_normal_mask]
    test_scores_anomalous = test_scores[~test_normal_mask]

    # Set threshold
    logger.info("=" * 80)
    logger.info("SETTING THRESHOLD")
    logger.info("=" * 80)
    threshold = detector.set_threshold(
        test_embeddings[test_normal_mask], max_fpr=args.max_fpr
    )

    # Predict
    test_predictions = (test_scores > threshold).astype(int)

    # Evaluate
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
    logger.info(f"  Recall: {recall:.2%}")
    logger.info(f"  F1-Score: {f1:.2%}")
    logger.info(f"  FPR: {fpr:.2%}")
    logger.info(f"  FNR: {fnr:.2%}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # Save results
    results = {
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
