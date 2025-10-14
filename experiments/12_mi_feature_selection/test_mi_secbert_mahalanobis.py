#!/usr/bin/env python3
"""
Experiment 12: MI-Based Feature Selection for SecBERT + Mahalanobis

Tests if selecting dimensions via mutual information improves detection performance.
"""

import json
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.feature_selection import mutual_info_classif

from neuralshield.anomaly.mahalanobis import MahalanobisDetector


def compute_metrics_at_fpr(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float = 0.05,
) -> dict:
    """Compute metrics at a specific FPR threshold.

    Args:
        scores: Anomaly scores (higher = more anomalous)
        labels: Ground truth labels (0=normal, 1=attack or string labels)
        target_fpr: Target false positive rate

    Returns:
        Dictionary with metrics: recall, precision, f1_score, fpr, threshold
    """
    # Convert labels to binary if they're strings
    if isinstance(labels[0], str):
        binary_labels = np.array(
            [1 if label in ["attack", "anomalous"] else 0 for label in labels]
        )
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
    data = np.load(npz_path, allow_pickle=True)
    return data["embeddings"], data["labels"]


def load_srbh_attacks_only(npz_path: Path) -> np.ndarray:
    """Load SR_BH test embeddings, filter to attacks only."""
    embeddings, labels = load_embeddings(npz_path)

    # Filter to attack only
    attack_mask = labels == "attack"
    logger.info(f"SR_BH: {attack_mask.sum():,} attacks out of {len(labels):,} total")

    return embeddings[attack_mask]


def compute_mi_scores(
    normal_embeddings: np.ndarray, attack_embeddings: np.ndarray
) -> np.ndarray:
    """Compute mutual information scores for each dimension."""
    logger.info(
        f"Computing MI: {len(normal_embeddings):,} normal + {len(attack_embeddings):,} attacks"
    )

    # Combine embeddings
    X = np.vstack([normal_embeddings, attack_embeddings])
    y = np.array([0] * len(normal_embeddings) + [1] * len(attack_embeddings))

    logger.info(f"Combined shape: {X.shape}, Label distribution: {np.bincount(y)}")

    # Compute MI scores
    logger.info("Computing mutual information scores...")
    mi_scores = mutual_info_classif(X, y, random_state=42)

    logger.info(
        f"MI scores computed. Range: [{mi_scores.min():.6f}, {mi_scores.max():.6f}]"
    )

    return mi_scores


def test_k_dimensions(
    k: int,
    mi_scores: np.ndarray,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    target_fpr: float = 0.05,
) -> dict:
    """Test detector with top-k MI-selected dimensions."""
    logger.info(f"\nTesting k={k} dimensions...")

    # Select top k dimensions
    if k == 768:
        # Baseline: all dimensions
        top_dims = np.arange(768)
    else:
        top_dims = np.argsort(mi_scores)[-k:]

    # Filter embeddings to selected dimensions
    train_selected = train_embeddings[:, top_dims]
    test_selected = test_embeddings[:, top_dims]

    # Train detector
    detector = MahalanobisDetector()
    detector.fit(train_selected)

    # Get anomaly scores
    scores = detector.scores(test_selected)

    # Compute metrics
    metrics = compute_metrics_at_fpr(
        scores=scores,
        labels=test_labels,
        target_fpr=target_fpr,
    )

    logger.info(
        f"k={k:3d}: Recall={metrics['recall']:.4f}, "
        f"Precision={metrics['precision']:.4f}, "
        f"F1={metrics['f1_score']:.4f}"
    )

    return {
        "k": k,
        "selected_dims": top_dims.tolist()
        if k <= 200
        else None,  # Save only for small k
        **metrics,
    }


def main():
    """Run MI feature selection experiment."""
    logger.info("=" * 80)
    logger.info("Experiment 12: MI-Based Feature Selection")
    logger.info("=" * 80)

    # Paths
    csic_train_path = Path(
        "experiments/03_secbert_comparison/secbert_with_preprocessing/csic_train_embeddings_converted.npz"
    )
    csic_test_path = Path(
        "experiments/03_secbert_comparison/secbert_with_preprocessing/csic_test_embeddings_converted.npz"
    )
    srbh_test_path = Path(
        "experiments/08_secbert_srbh/with_preprocessing/test_embeddings.npz"
    )

    results_dir = Path("experiments/12_mi_feature_selection/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    logger.info("\n[1/4] Loading embeddings...")
    csic_train_emb, csic_train_labels = load_embeddings(csic_train_path)
    csic_test_emb, csic_test_labels = load_embeddings(csic_test_path)
    srbh_attacks_emb = load_srbh_attacks_only(srbh_test_path)

    logger.info(f"CSIC train: {csic_train_emb.shape}")
    logger.info(f"CSIC test: {csic_test_emb.shape}")
    logger.info(f"SR_BH attacks: {srbh_attacks_emb.shape}")

    # Compute MI scores
    logger.info("\n[2/4] Computing MI scores...")
    mi_scores = compute_mi_scores(csic_train_emb, srbh_attacks_emb)

    # Save MI scores
    mi_scores_path = results_dir / "mi_scores.npy"
    np.save(mi_scores_path, mi_scores)
    logger.info(f"Saved MI scores to {mi_scores_path}")

    # Test different K values
    logger.info("\n[3/4] Testing different K values...")
    k_values = [50, 100, 200, 300, 400, 768]  # 768 = baseline (all dims)

    results = []
    for k in k_values:
        metrics = test_k_dimensions(
            k=k,
            mi_scores=mi_scores,
            train_embeddings=csic_train_emb,
            test_embeddings=csic_test_emb,
            test_labels=csic_test_labels,
            target_fpr=0.05,
        )
        results.append(metrics)

        # Save selected dimensions for small k
        if k <= 200:
            dims_path = results_dir / f"selected_dims_k{k}.npy"
            top_dims = np.argsort(mi_scores)[-k:]
            np.save(dims_path, top_dims)

    # Save results
    logger.info("\n[4/4] Saving results...")
    results_path = results_dir / "metrics_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    baseline = [r for r in results if r["k"] == 768][0]
    best = max(results, key=lambda r: r["recall"])

    logger.info(f"Baseline (all 768 dims): {baseline['recall']:.4f} recall @ 5% FPR")
    logger.info(f"Best (k={best['k']}): {best['recall']:.4f} recall @ 5% FPR")

    improvement = (best["recall"] - baseline["recall"]) / baseline["recall"] * 100
    if improvement > 0:
        logger.success(f"✅ MI selection improves by +{improvement:.1f}%!")
    else:
        logger.warning(f"❌ MI selection decreases by {improvement:.1f}%")

    logger.info("\nFull results:")
    for r in results:
        logger.info(
            f"  k={r['k']:3d}: {r['recall']:.4f} recall, {r['precision']:.4f} precision, {r['f1_score']:.4f} F1"
        )


if __name__ == "__main__":
    main()
