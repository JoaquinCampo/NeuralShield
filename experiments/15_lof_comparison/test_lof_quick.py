#!/usr/bin/env python3
"""Quick test of LOF detector on a small subset of CSIC data."""

from pathlib import Path

import numpy as np
from loguru import logger

from neuralshield.anomaly import LOFDetector, MahalanobisDetector


def load_embeddings(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels from NPZ file."""
    logger.info(f"Loading embeddings from {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    logger.info(f"Loaded {len(embeddings):,} samples, {embeddings.shape[1]} dimensions")
    return embeddings, labels


def main():
    logger.info("=" * 80)
    logger.info("QUICK LOF TEST")
    logger.info("=" * 80)

    # Use CSIC data (smaller dataset for quick test)
    repo_root = Path(__file__).parent.parent.parent
    train_path = (
        repo_root
        / "experiments/03_secbert_comparison/secbert_with_preprocessing/csic_train_embeddings_converted.npz"
    )
    test_path = (
        repo_root
        / "experiments/03_secbert_comparison/secbert_with_preprocessing/csic_test_embeddings_converted.npz"
    )

    # Load data
    train_embeddings, _ = load_embeddings(train_path)
    test_embeddings, test_labels = load_embeddings(test_path)

    # Use subset for quick test
    train_subset = train_embeddings[:5000]
    test_subset = test_embeddings[:5000]
    test_labels_subset = test_labels[:5000]

    logger.info(f"Using {len(train_subset):,} training samples")
    logger.info(f"Using {len(test_subset):,} test samples")

    # Test LOF
    logger.info("")
    logger.info("=" * 80)
    logger.info("Testing LOF Detector")
    logger.info("=" * 80)

    lof = LOFDetector(n_neighbors=100)
    logger.info("Fitting LOF...")
    lof.fit(train_subset)
    logger.info("Computing scores...")
    lof_scores = lof.scores(test_subset)
    logger.info(
        f"LOF scores - min: {lof_scores.min():.4f}, max: {lof_scores.max():.4f}"
    )

    # Test Mahalanobis
    logger.info("")
    logger.info("=" * 80)
    logger.info("Testing Mahalanobis Detector (baseline)")
    logger.info("=" * 80)

    maha = MahalanobisDetector()
    logger.info("Fitting Mahalanobis...")
    maha.fit(train_subset)
    logger.info("Computing scores...")
    maha_scores = maha.scores(test_subset)
    logger.info(
        f"Mahalanobis scores - min: {maha_scores.min():.4f}, max: {maha_scores.max():.4f}"
    )

    # Quick metrics
    logger.info("")
    logger.info("=" * 80)
    logger.info("Quick Metrics @ 5% FPR")
    logger.info("=" * 80)

    # Convert labels to binary
    binary_labels = np.array(
        [1 if label == "attack" else 0 for label in test_labels_subset]
    )
    normal_mask = binary_labels == 0
    attack_mask = binary_labels == 1

    # LOF metrics
    lof_threshold = float(np.percentile(lof_scores[normal_mask], 95))
    lof_recall = np.mean(lof_scores[attack_mask] > lof_threshold)
    logger.info(f"LOF Recall: {lof_recall:.2%}")

    # Mahalanobis metrics
    maha_threshold = float(np.percentile(maha_scores[normal_mask], 95))
    maha_recall = np.mean(maha_scores[attack_mask] > maha_threshold)
    logger.info(f"Mahalanobis Recall: {maha_recall:.2%}")

    improvement = (lof_recall - maha_recall) / maha_recall * 100
    logger.info(f"LOF vs Mahalanobis: {improvement:+.1f}%")

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ LOF DETECTOR WORKING!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
