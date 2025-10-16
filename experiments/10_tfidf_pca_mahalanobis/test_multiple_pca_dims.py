import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def test_pca_dimension(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    n_components: int,
    max_fpr: float = 0.05,
) -> dict:
    """Test a specific PCA dimension."""
    logger.info(f"Testing PCA with {n_components} components")

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    train_pca = pca.fit_transform(train_embeddings)
    test_pca = pca.transform(test_embeddings)

    explained_variance = pca.explained_variance_ratio_.sum()
    logger.info(f"  Explained variance: {explained_variance:.2%}")

    # Fit Mahalanobis
    detector = EmpiricalCovariance()
    detector.fit(train_pca)

    # Compute scores
    test_scores = detector.mahalanobis(test_pca)

    # Split by label
    test_normal_mask = test_labels == 0
    test_scores_normal = test_scores[test_normal_mask]

    # Find threshold
    threshold = np.percentile(test_scores_normal, 100 * (1 - max_fpr))

    # Predict
    test_predictions = (test_scores > threshold).astype(int)

    # Metrics
    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions, zero_division=0)
    recall = recall_score(test_labels, test_predictions)
    f1 = f1_score(test_labels, test_predictions, zero_division=0)

    cm = confusion_matrix(test_labels, test_predictions)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    logger.info(f"  Recall: {recall:.2%}, F1: {f1:.2%}, FPR: {fpr:.2%}")

    return {
        "n_components": n_components,
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
    }


def plot_results(results: list[dict], output_dir: Path):
    """Plot comparison of different PCA dimensions."""
    n_components = [r["n_components"] for r in results]
    recalls = [r["recall"] for r in results]
    f1_scores = [r["f1_score"] for r in results]
    explained_variances = [r["explained_variance"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Recall vs components
    axes[0, 0].plot(n_components, recalls, marker="o", linewidth=2)
    axes[0, 0].set_xlabel("PCA Components")
    axes[0, 0].set_ylabel("Recall")
    axes[0, 0].set_title("Recall vs PCA Dimensions")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale("log")

    # F1-Score vs components
    axes[0, 1].plot(n_components, f1_scores, marker="o", linewidth=2, color="orange")
    axes[0, 1].set_xlabel("PCA Components")
    axes[0, 1].set_ylabel("F1-Score")
    axes[0, 1].set_title("F1-Score vs PCA Dimensions")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale("log")

    # Explained variance vs components
    axes[1, 0].plot(
        n_components, explained_variances, marker="o", linewidth=2, color="green"
    )
    axes[1, 0].set_xlabel("PCA Components")
    axes[1, 0].set_ylabel("Explained Variance")
    axes[1, 0].set_title("Explained Variance vs PCA Dimensions")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale("log")

    # Recall vs explained variance
    axes[1, 1].scatter(explained_variances, recalls, s=100, alpha=0.6)
    for i, n in enumerate(n_components):
        axes[1, 1].annotate(
            f"{n}",
            (explained_variances[i], recalls[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1, 1].set_xlabel("Explained Variance")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_title("Recall vs Explained Variance")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "pca_dimension_comparison.png", dpi=150)
    logger.info(
        f"Comparison plot saved to {output_dir / 'pca_dimension_comparison.png'}"
    )
    plt.close()


def main():
    # Configuration
    train_path = Path(
        "experiments/10_tfidf_pca_mahalanobis/with_preprocessing/train_embeddings.npz"
    )
    test_path = Path(
        "experiments/10_tfidf_pca_mahalanobis/with_preprocessing/test_embeddings.npz"
    )
    output_dir = Path("experiments/10_tfidf_pca_mahalanobis/dimension_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test these dimensions
    dimensions_to_test = [50, 100, 200, 300, 500, 1000, 2000, 3000]
    max_fpr = 0.05

    logger.info("=" * 80)
    logger.info("TESTING MULTIPLE PCA DIMENSIONS")
    logger.info("=" * 80)
    logger.info(f"Dimensions to test: {dimensions_to_test}")
    logger.info(f"Max FPR constraint: {max_fpr}")

    # Load data once
    train_embeddings, train_labels = load_embeddings(train_path, is_test=False)
    test_embeddings, test_labels = load_embeddings(test_path, is_test=True)

    logger.info(f"Original embedding dimensions: {train_embeddings.shape[1]}")

    # Test each dimension
    results = []
    for n_components in dimensions_to_test:
        if n_components >= train_embeddings.shape[1]:
            logger.warning(f"Skipping {n_components} (>= original dimensions)")
            continue

        try:
            result = test_pca_dimension(
                train_embeddings, test_embeddings, test_labels, n_components, max_fpr
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed for {n_components} components: {e}")

    # Save results
    results_path = output_dir / "dimension_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Find best dimension
    best_recall = max(results, key=lambda x: x["recall"])
    best_f1 = max(results, key=lambda x: x["f1_score"])

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"Best recall: {best_recall['recall']:.2%} at {best_recall['n_components']} components"
    )
    logger.info(
        f"Best F1: {best_f1['f1_score']:.2%} at {best_f1['n_components']} components"
    )

    # Print table
    logger.info("\nFull Results Table:")
    logger.info(
        f"{'Components':<12} {'Variance':<10} {'Recall':<10} {'F1-Score':<10} {'Precision':<10}"
    )
    logger.info("-" * 60)
    for r in results:
        logger.info(
            f"{r['n_components']:<12} {r['explained_variance']:<10.2%} "
            f"{r['recall']:<10.2%} {r['f1_score']:<10.2%} {r['precision']:<10.2%}"
        )

    # Generate plots
    plot_results(results, output_dir)

    logger.info("=" * 80)
    logger.info("DONE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
