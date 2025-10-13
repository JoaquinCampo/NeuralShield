"""Hyperparameter search for OCSVM detector.

Searches over:
- nu: [0.01, 0.03, 0.05, 0.07, 0.1]
- gamma: ['scale', 'auto', 0.001, 0.01, 0.1]

Evaluates recall @ 5% FPR.
"""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from loguru import logger

from neuralshield.anomaly import OCSVMDetector

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
    return embeddings, labels


def evaluate_config(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    nu: float,
    gamma: str | float,
    max_fpr: float = 0.05,
) -> dict:
    """Evaluate a single OCSVM configuration."""
    logger.info(f"Testing nu={nu}, gamma={gamma}")

    # Train detector
    detector = OCSVMDetector(nu=nu, gamma=gamma, verbose=False)
    detector.fit(train_embeddings)

    # Compute scores
    test_scores = detector.scores(test_embeddings)

    # Split by label
    test_normal_mask = test_labels == 0
    test_scores_normal = test_scores[test_normal_mask]
    test_scores_anomalous = test_scores[~test_normal_mask]

    # Set threshold based on FPR
    threshold = float(np.percentile(test_scores_normal, (1 - max_fpr) * 100))
    actual_fpr = np.mean(test_scores_normal > threshold)

    # Compute recall
    recall = np.mean(test_scores_anomalous > threshold)

    # Compute precision
    predictions = (test_scores > threshold).astype(int)
    tp = np.sum((predictions == 1) & (test_labels == 1))
    fp = np.sum((predictions == 1) & (test_labels == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    results = {
        "nu": nu,
        "gamma": gamma,
        "threshold": threshold,
        "recall": float(recall),
        "fpr": float(actual_fpr),
        "precision": float(precision),
        "true_positives": int(tp),
        "false_positives": int(fp),
    }

    logger.info(
        f"  nu={nu}, gamma={gamma} → Recall={recall:.2%} @ FPR={actual_fpr:.2%}, Precision={precision:.2%}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="OCSVM hyperparameter search")
    parser.add_argument("train_embeddings", type=Path, help="Training embeddings .npz")
    parser.add_argument("test_embeddings", type=Path, help="Test embeddings .npz")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--max-fpr", type=float, default=0.05, help="Max FPR target (default: 0.05)"
    )
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--wandb-project", type=str, default="neuralshield")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if args.wandb:
        if not WANDB_AVAILABLE:
            logger.error("wandb not available")
            return
        wandb.init(
            project=args.wandb_project,
            name="ocsvm-hyperparameter-search",
            config={"max_fpr": args.max_fpr},
        )

    # Load data
    logger.info("=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    train_embeddings, _ = load_embeddings(args.train_embeddings, is_test=False)
    test_embeddings, test_labels = load_embeddings(args.test_embeddings, is_test=True)

    # Define search space
    nu_values = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    gamma_values = ["scale", "auto", 0.001, 0.01, 0.1, 1.0]

    logger.info("=" * 80)
    logger.info("HYPERPARAMETER SEARCH")
    logger.info("=" * 80)
    logger.info(f"Nu values: {nu_values}")
    logger.info(f"Gamma values: {gamma_values}")
    logger.info(f"Total configurations: {len(nu_values) * len(gamma_values)}")

    # Grid search
    all_results = []
    best_recall = 0.0
    best_config = None

    for nu, gamma in product(nu_values, gamma_values):
        try:
            results = evaluate_config(
                train_embeddings,
                test_embeddings,
                test_labels,
                nu,
                gamma,
                args.max_fpr,
            )
            all_results.append(results)

            # Log to wandb
            if args.wandb:
                wandb.log(results)

            # Track best
            if results["recall"] > best_recall:
                best_recall = results["recall"]
                best_config = results

        except Exception as e:
            logger.error(f"Failed for nu={nu}, gamma={gamma}: {e}")
            continue

    # Save all results
    logger.info("=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    results_path = args.output_dir / "hyperparameter_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"All results saved to {results_path}")

    # Save best config
    best_config_path = args.output_dir / "best_config.json"
    with open(best_config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    logger.info(f"Best config saved to {best_config_path}")

    # Summary
    logger.info("=" * 80)
    logger.info("BEST CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Nu: {best_config['nu']}")
    logger.info(f"Gamma: {best_config['gamma']}")
    logger.info(f"Recall: {best_config['recall']:.2%} @ FPR={best_config['fpr']:.2%}")
    logger.info(f"Precision: {best_config['precision']:.2%}")
    logger.info(f"Threshold: {best_config['threshold']:.4f}")

    # Find top 5
    sorted_results = sorted(all_results, key=lambda x: x["recall"], reverse=True)
    logger.info("=" * 80)
    logger.info("TOP 5 CONFIGURATIONS")
    logger.info("=" * 80)
    for i, result in enumerate(sorted_results[:5], 1):
        logger.info(
            f"{i}. nu={result['nu']}, gamma={result['gamma']} → "
            f"Recall={result['recall']:.2%} @ FPR={result['fpr']:.2%}"
        )

    if args.wandb:
        wandb.log({"best_recall": best_recall, "best_nu": best_config["nu"]})
        wandb.finish()

    logger.info("=" * 80)
    logger.info("DONE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
