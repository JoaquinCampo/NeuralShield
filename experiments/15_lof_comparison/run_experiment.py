#!/usr/bin/env python3
"""
Experiment 15: Local Outlier Factor (LOF) Comparison

Runs LOF detector on SecBERT embeddings and compares to Mahalanobis baseline.

Tests:
1. Same-dataset performance (SR_BH, CSIC)
2. With/without preprocessing
3. Direct comparison to Mahalanobis
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.metrics import confusion_matrix

from neuralshield.anomaly import LOFDetector, MahalanobisDetector


def compute_metrics_at_fpr(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float = 0.05,
) -> dict:
    """Compute metrics at a specific FPR threshold."""
    # Convert labels to binary if they're strings
    if isinstance(labels[0], str):
        binary_labels = np.array([1 if label == "attack" else 0 for label in labels])
    else:
        binary_labels = labels

    # Separate normal and attack scores
    normal_mask = binary_labels == 0
    attack_mask = binary_labels == 1

    normal_scores = scores[normal_mask]
    attack_scores = scores[attack_mask]

    # Find threshold at target FPR
    threshold = float(np.percentile(normal_scores, 100 * (1 - target_fpr)))

    # Compute predictions
    predictions = scores > threshold

    # Compute metrics
    tp = np.sum((predictions == 1) & (binary_labels == 1))
    fp = np.sum((predictions == 1) & (binary_labels == 0))
    tn = np.sum((predictions == 0) & (binary_labels == 0))
    fn = np.sum((predictions == 0) & (binary_labels == 1))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    actual_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0

    return {
        "threshold": float(threshold),
        "recall": float(recall),
        "precision": float(precision),
        "f1_score": float(f1_score),
        "fpr": float(actual_fpr),
        "accuracy": float(accuracy),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def load_embeddings(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels from NPZ file."""
    logger.info(f"Loading embeddings from {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    logger.info(f"Loaded {len(embeddings):,} samples, {embeddings.shape[1]} dimensions")
    return embeddings, labels


def plot_confusion_matrix(y_true, y_pred, output_path: Path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xticklabels(["Normal", "Attack"])
    ax.set_yticklabels(["Normal", "Attack"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved confusion matrix to {output_path}")
    plt.close()


def plot_score_distribution(
    scores_normal, scores_anomalous, threshold, output_path: Path
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

    plt.xlabel("Anomaly Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Score Distribution", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved score distribution to {output_path}")
    plt.close()


def run_variant(
    detector_name: str,
    detector_class,
    detector_kwargs: dict,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    output_dir: Path,
    target_fpr: float = 0.05,
) -> dict:
    """Run a single detector variant and save results."""
    logger.info("=" * 80)
    logger.info(f"Running {detector_name}")
    logger.info("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Train detector
    logger.info("Training detector...")
    start_time = time.time()
    detector = detector_class(**detector_kwargs)
    detector.fit(train_embeddings)
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f}s")

    # Compute scores
    logger.info("Computing scores...")
    start_time = time.time()
    test_scores = detector.scores(test_embeddings)
    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.2f}s")

    # Compute metrics
    metrics = compute_metrics_at_fpr(test_scores, test_labels, target_fpr)
    metrics["train_time_seconds"] = train_time
    metrics["inference_time_seconds"] = inference_time
    metrics["detector"] = detector_name
    metrics["n_train_samples"] = len(train_embeddings)
    metrics["n_test_samples"] = len(test_embeddings)

    # Log results
    logger.info(f"Recall @ {target_fpr:.0%} FPR: {metrics['recall']:.2%}")
    logger.info(f"Precision: {metrics['precision']:.2%}")
    logger.info(f"F1-Score: {metrics['f1_score']:.2%}")
    logger.info(f"Accuracy: {metrics['accuracy']:.2%}")

    # Save metrics
    metrics_path = output_dir / "results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Convert labels to binary for plotting
    if isinstance(test_labels[0], str):
        binary_labels = np.array(
            [1 if label == "attack" else 0 for label in test_labels]
        )
    else:
        binary_labels = test_labels

    # Plot confusion matrix
    predictions = test_scores > metrics["threshold"]
    plot_confusion_matrix(
        binary_labels, predictions, output_dir / "confusion_matrix.png"
    )

    # Plot score distribution
    normal_mask = binary_labels == 0
    plot_score_distribution(
        test_scores[normal_mask],
        test_scores[~normal_mask],
        metrics["threshold"],
        output_dir / "score_distribution.png",
    )

    return metrics


def main():
    logger.info("=" * 80)
    logger.info("EXPERIMENT 15: LOF COMPARISON")
    logger.info("=" * 80)

    # Paths to SecBERT embeddings
    base_dir = Path(__file__).parent
    repo_root = base_dir.parent.parent

    # Define experiments
    experiments = [
        {
            "name": "CSIC with preprocessing",
            "train_path": repo_root
            / "experiments/03_secbert_comparison/secbert_with_preprocessing/csic_train_embeddings_converted.npz",
            "test_path": repo_root
            / "experiments/03_secbert_comparison/secbert_with_preprocessing/csic_test_embeddings_converted.npz",
            "output_dir": base_dir / "csic_with_preprocessing",
        },
        {
            "name": "CSIC without preprocessing",
            "train_path": repo_root
            / "experiments/03_secbert_comparison/secbert_without_preprocessing/csic_train_embeddings_converted.npz",
            "test_path": repo_root
            / "experiments/03_secbert_comparison/secbert_without_preprocessing/csic_test_embeddings_converted.npz",
            "output_dir": base_dir / "csic_without_preprocessing",
        },
        {
            "name": "SR_BH with preprocessing",
            "train_path": repo_root
            / "experiments/08_secbert_srbh/with_preprocessing/train_embeddings.npz",
            "test_path": repo_root
            / "experiments/08_secbert_srbh/with_preprocessing/test_embeddings.npz",
            "output_dir": base_dir / "srbh_with_preprocessing",
        },
        {
            "name": "SR_BH without preprocessing",
            "train_path": repo_root
            / "experiments/08_secbert_srbh/without_preprocessing/train_embeddings.npz",
            "test_path": repo_root
            / "experiments/08_secbert_srbh/without_preprocessing/test_embeddings.npz",
            "output_dir": base_dir / "srbh_without_preprocessing",
        },
    ]

    all_results = []

    for exp in experiments:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"EXPERIMENT: {exp['name']}")
        logger.info("=" * 80)

        # Check if files exist
        if not exp["train_path"].exists():
            logger.warning(f"Train file not found: {exp['train_path']}")
            logger.warning("Skipping this experiment")
            continue

        if not exp["test_path"].exists():
            logger.warning(f"Test file not found: {exp['test_path']}")
            logger.warning("Skipping this experiment")
            continue

        # Load data
        train_embeddings, _ = load_embeddings(exp["train_path"])
        test_embeddings, test_labels = load_embeddings(exp["test_path"])

        # Run LOF
        lof_results = run_variant(
            detector_name="LOF",
            detector_class=LOFDetector,
            detector_kwargs={"n_neighbors": 100},
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
            output_dir=exp["output_dir"] / "lof",
        )
        lof_results["experiment"] = exp["name"]

        # Run Mahalanobis (baseline)
        maha_results = run_variant(
            detector_name="Mahalanobis",
            detector_class=MahalanobisDetector,
            detector_kwargs={},
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
            output_dir=exp["output_dir"] / "mahalanobis",
        )
        maha_results["experiment"] = exp["name"]

        # Compare
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"COMPARISON: {exp['name']}")
        logger.info("=" * 80)
        logger.info(f"LOF Recall:         {lof_results['recall']:.2%}")
        logger.info(f"Mahalanobis Recall: {maha_results['recall']:.2%}")
        improvement = (
            (lof_results["recall"] - maha_results["recall"])
            / maha_results["recall"]
            * 100
        )
        logger.info(f"Improvement:        {improvement:+.1f}%")

        all_results.append({"lof": lof_results, "mahalanobis": maha_results})

    # Save summary
    summary_path = base_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved comparison summary to {summary_path}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
