"""Compare GMM performance with different numbers of components."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from neuralshield.anomaly import GMMDetector


def load_embeddings(path: Path, is_test: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load precomputed embeddings from npz file."""
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


def test_n_components(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    n_components: int,
    max_fpr: float,
) -> dict:
    """Test GMM with specific number of components."""
    logger.info(f"Testing n_components={n_components}")

    # Fit detector
    detector = GMMDetector(n_components=n_components, covariance_type="full")
    detector.fit(train_embeddings)

    # Set threshold on normal test samples
    test_normal_mask = test_labels == 0
    test_normal_embeddings = test_embeddings[test_normal_mask]
    threshold = detector.set_threshold(test_normal_embeddings, max_fpr=max_fpr)

    # Get scores and predictions
    test_scores = detector.scores(test_embeddings)
    test_predictions = detector.predict(test_embeddings)

    # Compute metrics
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions, zero_division=0)
    recall = recall_score(test_labels, test_predictions)
    f1 = f1_score(test_labels, test_predictions, zero_division=0)

    cm = confusion_matrix(test_labels, test_predictions)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Get BIC/AIC
    bic = detector._model.bic(train_embeddings)
    aic = detector._model.aic(train_embeddings)

    return {
        "n_components": n_components,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "fpr": float(fpr),
        "threshold": float(threshold),
        "bic": float(bic),
        "aic": float(aic),
    }


def plot_comparison(results: list[dict], output_dir: Path):
    """Plot comparison of different n_components."""
    n_components_list = [r["n_components"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # F1 Score
    axes[0, 0].plot(
        n_components_list, [r["f1_score"] for r in results], "o-", linewidth=2
    )
    axes[0, 0].set_xlabel("Number of Components")
    axes[0, 0].set_ylabel("F1 Score")
    axes[0, 0].set_title("F1 Score vs Components")
    axes[0, 0].grid(True, alpha=0.3)

    # Recall
    axes[0, 1].plot(
        n_components_list,
        [r["recall"] for r in results],
        "o-",
        linewidth=2,
        color="green",
    )
    axes[0, 1].set_xlabel("Number of Components")
    axes[0, 1].set_ylabel("Recall (TPR)")
    axes[0, 1].set_title("Recall vs Components")
    axes[0, 1].grid(True, alpha=0.3)

    # BIC/AIC
    axes[1, 0].plot(
        n_components_list, [r["bic"] for r in results], "o-", linewidth=2, label="BIC"
    )
    axes[1, 0].plot(
        n_components_list, [r["aic"] for r in results], "s-", linewidth=2, label="AIC"
    )
    axes[1, 0].set_xlabel("Number of Components")
    axes[1, 0].set_ylabel("Information Criterion")
    axes[1, 0].set_title("Model Complexity (lower = better)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # FPR
    axes[1, 1].plot(
        n_components_list, [r["fpr"] for r in results], "o-", linewidth=2, color="red"
    )
    axes[1, 1].axhline(0.05, color="black", linestyle="--", label="Target FPR=5%")
    axes[1, 1].set_xlabel("Number of Components")
    axes[1, 1].set_ylabel("False Positive Rate")
    axes[1, 1].set_title("FPR vs Components")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "component_comparison.png", dpi=150)
    logger.info(f"Comparison plot saved to {output_dir / 'component_comparison.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare GMM with different numbers of components"
    )
    parser.add_argument("train_embeddings", type=Path, help="Training embeddings .npz")
    parser.add_argument("test_embeddings", type=Path, help="Test embeddings .npz")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--min-components",
        type=int,
        default=1,
        help="Minimum number of components (default: 1)",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=10,
        help="Maximum number of components (default: 10)",
    )
    parser.add_argument(
        "--max-fpr", type=float, default=0.05, help="Max FPR (default: 0.05)"
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    train_embeddings, _ = load_embeddings(args.train_embeddings, is_test=False)
    test_embeddings, test_labels = load_embeddings(args.test_embeddings, is_test=True)

    # Test different n_components
    logger.info("=" * 80)
    logger.info("TESTING DIFFERENT N_COMPONENTS")
    logger.info("=" * 80)
    results = []
    for n_components in range(args.min_components, args.max_components + 1):
        result = test_n_components(
            train_embeddings,
            test_embeddings,
            test_labels,
            n_components,
            args.max_fpr,
        )
        results.append(result)
        logger.info(
            f"  n={n_components}: F1={result['f1_score']:.2%}, "
            f"Recall={result['recall']:.2%}, BIC={result['bic']:.0f}"
        )

    # Save results
    results_path = args.output_dir / "component_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Find best n_components
    best_f1 = max(results, key=lambda r: r["f1_score"])
    best_bic = min(results, key=lambda r: r["bic"])
    logger.info("=" * 80)
    logger.info("BEST CONFIGURATIONS")
    logger.info("=" * 80)
    logger.info(
        f"Best F1: n_components={best_f1['n_components']} (F1={best_f1['f1_score']:.2%})"
    )
    logger.info(
        f"Best BIC: n_components={best_bic['n_components']} (BIC={best_bic['bic']:.0f})"
    )

    # Plot comparison
    plot_comparison(results, args.output_dir)

    logger.info("=" * 80)
    logger.info("DONE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
