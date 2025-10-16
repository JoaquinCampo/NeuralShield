"""Hyperparameter search for LOF detector.

Searches over:
- n_neighbors: [50, 100, 150, 200, 300, 500]

Evaluates recall @ 5% FPR.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger

from neuralshield.anomaly import LOFDetector

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
    n_neighbors: int,
    max_fpr: float = 0.05,
) -> dict:
    """Evaluate a single LOF configuration."""
    logger.info(f"Testing n_neighbors={n_neighbors}")

    # Train detector
    detector = LOFDetector(n_neighbors=n_neighbors)
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

    # Compute F1
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    results = {
        "n_neighbors": n_neighbors,
        "threshold": threshold,
        "recall": float(recall),
        "fpr": float(actual_fpr),
        "precision": float(precision),
        "f1": float(f1),
        "true_positives": int(tp),
        "false_positives": int(fp),
    }

    logger.info(
        f"  n_neighbors={n_neighbors} → Recall={recall:.2%} @ FPR={actual_fpr:.2%}, "
        f"Precision={precision:.2%}, F1={f1:.2%}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="LOF hyperparameter search")
    parser.add_argument("train_embeddings", type=Path, help="Training embeddings .npz")
    parser.add_argument("test_embeddings", type=Path, help="Test embeddings .npz")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--max-fpr", type=float, default=0.05, help="Max FPR target (default: 0.05)"
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200, 300, 500],
        help="n_neighbors values to test",
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
            name="lof-hyperparameter-search",
            config={"max_fpr": args.max_fpr, "n_neighbors_range": args.n_neighbors},
        )

    # Load data
    logger.info("=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    train_embeddings, _ = load_embeddings(args.train_embeddings, is_test=False)
    test_embeddings, test_labels = load_embeddings(args.test_embeddings, is_test=True)

    logger.info("=" * 80)
    logger.info("HYPERPARAMETER SEARCH")
    logger.info("=" * 80)
    logger.info(f"n_neighbors values: {args.n_neighbors}")
    logger.info(f"Total configurations: {len(args.n_neighbors)}")

    # Search
    all_results = []
    best_recall = 0.0
    best_config = None

    for n_neighbors in args.n_neighbors:
        try:
            results = evaluate_config(
                train_embeddings,
                test_embeddings,
                test_labels,
                n_neighbors,
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
            logger.error(f"Failed for n_neighbors={n_neighbors}: {e}")
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
    logger.info(f"n_neighbors: {best_config['n_neighbors']}")
    logger.info(f"Recall: {best_config['recall']:.2%} @ FPR={best_config['fpr']:.2%}")
    logger.info(f"Precision: {best_config['precision']:.2%}")
    logger.info(f"F1: {best_config['f1']:.2%}")
    logger.info(f"Threshold: {best_config['threshold']:.4f}")

    # Find top 5
    sorted_results = sorted(all_results, key=lambda x: x["recall"], reverse=True)
    logger.info("=" * 80)
    logger.info("TOP 5 CONFIGURATIONS")
    logger.info("=" * 80)
    for i, result in enumerate(sorted_results[:5], 1):
        logger.info(
            f"{i}. n_neighbors={result['n_neighbors']} → "
            f"Recall={result['recall']:.2%} @ FPR={result['fpr']:.2%}, "
            f"F1={result['f1']:.2%}"
        )

    if args.wandb:
        wandb.log(
            {
                "best_recall": best_recall,
                "best_n_neighbors": best_config["n_neighbors"],
                "best_f1": best_config["f1"],
            }
        )
        wandb.finish()

    logger.info("=" * 80)
    logger.info("DONE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
