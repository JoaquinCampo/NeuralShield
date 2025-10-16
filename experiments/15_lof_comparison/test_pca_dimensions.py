"""Test LOF performance across different PCA dimensions.

Tests dimensions: 50, 75, 100, 125, 150, 175, 200, 250, 300
Goal: Find optimal dimensionality for LOF on TF-IDF embeddings
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from neuralshield.anomaly import LOFDetector
from neuralshield.encoding.data.jsonl import JSONLRequestReader


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    """Load dataset without preprocessing."""
    logger.info(f"Loading dataset from {path}")
    dataset = JSONLRequestReader(path, use_pipeline=False)
    texts = []
    labels = []

    for batch, batch_labels in dataset.iter_batches(batch_size=1000):
        for text, label in zip(batch, batch_labels):
            texts.append(text)
            labels.append(label)

    logger.info(f"Loaded {len(texts)} samples")
    return texts, labels


def test_pca_dimension(
    train_tfidf: np.ndarray,
    test_tfidf: np.ndarray,
    test_labels: list[str],
    n_components: int,
) -> dict:
    """Test LOF with specific PCA dimension."""
    logger.info(f"\nTesting n_components={n_components}")

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    train_embeddings = pca.fit_transform(train_tfidf)
    test_embeddings = pca.transform(test_tfidf)

    explained_variance = float(pca.explained_variance_ratio_.sum())
    logger.info(f"  PCA explained variance: {explained_variance:.2%}")

    # Train LOF
    detector = LOFDetector(n_neighbors=100)
    detector.fit(train_embeddings.astype(np.float32))

    # Compute scores
    test_scores = detector.scores(test_embeddings.astype(np.float32))

    # Split by label
    test_labels_binary = np.array(
        [1 if label == "attack" else 0 for label in test_labels]
    )
    test_normal_mask = test_labels_binary == 0
    test_scores_normal = test_scores[test_normal_mask]
    test_scores_anomalous = test_scores[~test_normal_mask]

    # Set threshold at 5% FPR
    threshold = float(np.percentile(test_scores_normal, 95.0))
    actual_fpr = np.mean(test_scores_normal > threshold)

    # Compute metrics
    recall = np.mean(test_scores_anomalous > threshold)

    predictions = (test_scores > threshold).astype(int)
    tp = np.sum((predictions == 1) & (test_labels_binary == 1))
    fp = np.sum((predictions == 1) & (test_labels_binary == 0))
    tn = np.sum((predictions == 0) & (test_labels_binary == 0))
    fn = np.sum((predictions == 0) & (test_labels_binary == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    logger.info(f"  Recall @ 5% FPR: {recall:.2%}")
    logger.info(f"  Precision: {precision:.2%}")
    logger.info(f"  F1-Score: {f1:.2%}")

    return {
        "n_components": n_components,
        "explained_variance": explained_variance,
        "recall": float(recall),
        "precision": float(precision),
        "f1_score": float(f1),
        "fpr": float(actual_fpr),
        "threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def plot_results(results: list[dict], output_dir: Path):
    """Create visualization of results across dimensions."""
    dimensions = [r["n_components"] for r in results]
    recalls = [r["recall"] * 100 for r in results]
    precisions = [r["precision"] * 100 for r in results]
    f1_scores = [r["f1_score"] * 100 for r in results]
    variances = [r["explained_variance"] * 100 for r in results]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "LOF Performance vs PCA Dimensions (TF-IDF, k=100, no preprocessing)",
        fontsize=14,
        fontweight="bold",
    )

    # Plot 1: Recall vs Dimensions
    ax1 = axes[0, 0]
    ax1.plot(
        dimensions, recalls, marker="o", linewidth=2, markersize=8, color="#2E86AB"
    )
    ax1.axhline(
        y=max(recalls), color="red", linestyle="--", alpha=0.5, label="Best recall"
    )
    ax1.set_xlabel("PCA Dimensions", fontsize=11)
    ax1.set_ylabel("Recall @ 5% FPR (%)", fontsize=11)
    ax1.set_title("Recall vs PCA Dimensions", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Find and annotate best
    best_idx = recalls.index(max(recalls))
    ax1.annotate(
        f"Best: {dimensions[best_idx]}D\n{recalls[best_idx]:.2f}%",
        xy=(dimensions[best_idx], recalls[best_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    # Plot 2: Precision and F1
    ax2 = axes[0, 1]
    ax2.plot(
        dimensions, precisions, marker="s", linewidth=2, markersize=8, label="Precision"
    )
    ax2.plot(dimensions, f1_scores, marker="^", linewidth=2, markersize=8, label="F1")
    ax2.set_xlabel("PCA Dimensions", fontsize=11)
    ax2.set_ylabel("Score (%)", fontsize=11)
    ax2.set_title("Precision & F1 vs PCA Dimensions", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Explained Variance
    ax3 = axes[1, 0]
    ax3.plot(
        dimensions, variances, marker="D", linewidth=2, markersize=8, color="#A23B72"
    )
    ax3.axhline(y=90, color="green", linestyle="--", alpha=0.5, label="90% threshold")
    ax3.set_xlabel("PCA Dimensions", fontsize=11)
    ax3.set_ylabel("Explained Variance (%)", fontsize=11)
    ax3.set_title("PCA Explained Variance", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Recall vs Variance (trade-off)
    ax4 = axes[1, 1]
    scatter = ax4.scatter(
        variances,
        recalls,
        c=dimensions,
        s=200,
        cmap="viridis",
        alpha=0.7,
        edgecolors="black",
    )
    ax4.set_xlabel("Explained Variance (%)", fontsize=11)
    ax4.set_ylabel("Recall @ 5% FPR (%)", fontsize=11)
    ax4.set_title("Recall vs Variance Trade-off", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label("Dimensions", fontsize=10)

    # Annotate best point
    ax4.annotate(
        f"{dimensions[best_idx]}D",
        xy=(variances[best_idx], recalls[best_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    plt.tight_layout()
    output_path = output_dir / "pca_dimension_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot to {output_path}")
    plt.close()


def main():
    """Run PCA dimension sweep."""
    logger.info("=" * 80)
    logger.info("LOF PCA DIMENSION OPTIMIZATION")
    logger.info("=" * 80)

    # Paths
    data_dir = Path("src/neuralshield/data")
    train_path = data_dir / "CSIC" / "train.jsonl"
    test_path = data_dir / "CSIC" / "test.jsonl"

    output_dir = Path("experiments/15_lof_comparison/pca_dimension_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("\nLoading datasets")
    train_texts, _ = load_dataset(train_path)
    test_texts, test_labels = load_dataset(test_path)

    # Fit TF-IDF (once)
    logger.info("\nFitting TF-IDF vectorizer")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
    )
    train_tfidf = vectorizer.fit_transform(train_texts).toarray()
    test_tfidf = vectorizer.transform(test_texts).toarray()

    logger.info(f"TF-IDF shape: {train_tfidf.shape}")

    # Test different dimensions
    dimensions_to_test = [50, 75, 100, 125, 150, 175, 200, 250, 300]
    logger.info(f"\nTesting dimensions: {dimensions_to_test}")

    results = []
    for n_components in dimensions_to_test:
        result = test_pca_dimension(train_tfidf, test_tfidf, test_labels, n_components)
        results.append(result)

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Create visualizations
    logger.info("\nCreating visualizations")
    plot_results(results, output_dir)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    # Find best
    best_result = max(results, key=lambda x: x["recall"])
    logger.info(f"\nBest configuration:")
    logger.info(f"  Dimensions: {best_result['n_components']}")
    logger.info(f"  Recall @ 5% FPR: {best_result['recall']:.2%}")
    logger.info(f"  Precision: {best_result['precision']:.2%}")
    logger.info(f"  F1-Score: {best_result['f1_score']:.2%}")
    logger.info(f"  Explained Variance: {best_result['explained_variance']:.2%}")

    # Print table
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS TABLE")
    logger.info("=" * 80)
    logger.info(
        f"{'Dims':>6} | {'Variance':>9} | {'Recall':>7} | {'Precision':>9} | {'F1':>7}"
    )
    logger.info("-" * 80)
    for r in results:
        logger.info(
            f"{r['n_components']:>6} | {r['explained_variance']:>8.2%} | "
            f"{r['recall']:>6.2%} | {r['precision']:>8.2%} | {r['f1_score']:>6.2%}"
        )

    # Analysis
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS")
    logger.info("=" * 80)

    recalls = [r["recall"] for r in results]
    dims = [r["n_components"] for r in results]

    peak_idx = recalls.index(max(recalls))
    logger.info(f"\nPeak performance at {dims[peak_idx]} dimensions")

    if peak_idx > 0:
        improvement_from_prev = (
            (recalls[peak_idx] - recalls[peak_idx - 1]) / recalls[peak_idx - 1] * 100
        )
        logger.info(
            f"  Improvement from {dims[peak_idx - 1]}D: {improvement_from_prev:+.2f}%"
        )

    if peak_idx < len(recalls) - 1:
        degradation_to_next = (
            (recalls[peak_idx] - recalls[peak_idx + 1]) / recalls[peak_idx] * 100
        )
        logger.info(
            f"  Degradation to {dims[peak_idx + 1]}D: {degradation_to_next:.2f}%"
        )

    # Compare to baseline (150D from previous experiments)
    baseline_150_recall = 0.6420
    if 150 in dims:
        idx_150 = dims.index(150)
        actual_150_recall = recalls[idx_150]
        logger.info(f"\n150D performance (baseline): {actual_150_recall:.2%}")
        logger.info(f"Best performance: {max(recalls):.2%}")
        improvement = (max(recalls) - actual_150_recall) / actual_150_recall * 100
        logger.info(f"Improvement over 150D: {improvement:+.2f}%")

    logger.info("\nExperiment complete!")


if __name__ == "__main__":
    main()
