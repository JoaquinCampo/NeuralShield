"""GPU-accelerated OCSVM using cuML (RAPIDS) for Colab.

Installation on Colab:
    !pip install cuml-cu11

Usage:
    python test_ocsvm_cuml.py \
        train_embeddings.npz \
        test_embeddings.npz \
        output_dir \
        --nu 0.05 \
        --gamma scale \
        --wandb \
        --wandb-run-name "ocsvm-bge-with-prep"
"""

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

try:
    from cuml.svm import OneClassSVM

    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    logger.warning("cuML not available - install with: pip install cuml-cu11")

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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

    plt.xlabel("OCSVM Decision Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(
        "Score Distribution: Normal vs Anomalous (OCSVM)",
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
    plt.title("Confusion Matrix (OCSVM)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Confusion matrix saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated OCSVM test (cuML)")
    parser.add_argument("train_embeddings", type=Path, help="Training embeddings .npz")
    parser.add_argument("test_embeddings", type=Path, help="Test embeddings .npz")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--nu", type=float, default=0.05, help="Nu parameter (default: 0.05)"
    )
    parser.add_argument(
        "--gamma", type=str, default="scale", help="Gamma (scale/auto or float)"
    )
    parser.add_argument(
        "--max-fpr", type=float, default=0.05, help="Max FPR target (default: 0.05)"
    )
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--wandb-run-name", type=str, help="wandb run name")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check cuML availability
    if not CUML_AVAILABLE:
        logger.error(
            "cuML not available. Install with:\n"
            "  pip install cuml-cu11  (for CUDA 11)\n"
            "  pip install cuml-cu12  (for CUDA 12)"
        )
        return

    # Initialize wandb
    if args.wandb:
        if not WANDB_AVAILABLE:
            logger.error("wandb not available")
            return
        wandb.init(
            project="neuralshield",
            name=args.wandb_run_name or "ocsvm-cuml-test",
            config={
                "detector": "ocsvm_cuml",
                "nu": args.nu,
                "gamma": args.gamma,
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

    # Fit OCSVM on GPU
    logger.info("=" * 80)
    logger.info("FITTING ONE-CLASS SVM (GPU-ACCELERATED)")
    logger.info("=" * 80)
    logger.info(
        f"Training on {len(train_embeddings)} samples, {train_embeddings.shape[1]} dims"
    )
    logger.info(f"Parameters: nu={args.nu}, gamma={args.gamma}, kernel=rbf")

    # Parse gamma
    if args.gamma in ["scale", "auto"]:
        gamma_val = args.gamma
    else:
        gamma_val = float(args.gamma)

    detector = OneClassSVM(
        kernel="rbf",
        nu=args.nu,
        gamma=gamma_val,
    )

    detector.fit(train_embeddings)
    logger.info("OCSVM training complete")

    # Compute scores
    logger.info("=" * 80)
    logger.info("COMPUTING DECISION SCORES")
    logger.info("=" * 80)

    # cuML returns scores as cuDF, convert to numpy
    test_scores = detector.decision_function(test_embeddings)
    if hasattr(test_scores, "to_numpy"):
        test_scores = test_scores.to_numpy().ravel()
    else:
        test_scores = np.asarray(test_scores).ravel()

    logger.info(f"Computed scores for {len(test_scores)} samples")

    # Split by label
    test_normal_mask = test_labels == 0
    test_scores_normal = test_scores[test_normal_mask]
    test_scores_anomalous = test_scores[~test_normal_mask]

    # Set threshold based on desired FPR
    # OCSVM: negative scores = anomalies, positive = normal
    # Higher score = more normal
    logger.info("=" * 80)
    logger.info("SETTING THRESHOLD")
    logger.info("=" * 80)
    threshold = float(np.percentile(test_scores_normal, args.max_fpr * 100))
    actual_fpr = np.mean(test_scores_normal < threshold)
    logger.info(
        f"Threshold: {threshold:.4f} (target FPR={args.max_fpr:.1%}, actual={actual_fpr:.1%})"
    )

    # Predict (score < threshold = anomaly)
    test_predictions = (test_scores < threshold).astype(int)

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
        "nu": args.nu,
        "gamma": args.gamma,
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
