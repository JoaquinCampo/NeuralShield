"""Test LOF on TF-IDF + PCA embeddings.

Generates TF-IDF embeddings with 150-dimensional PCA and tests LOF detector.
Compares against Mahalanobis baseline from experiment 10.
"""

import json
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from neuralshield.anomaly import LOFDetector, MahalanobisDetector
from neuralshield.encoding.data.jsonl import JSONLRequestReader
from neuralshield.preprocessing.pipeline import preprocess


def load_and_embed_tfidf(
    train_path: Path,
    test_path: Path,
    n_components: int = 150,
    with_preprocessing: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load data, apply TF-IDF, and reduce with PCA.

    Returns:
        train_embeddings, test_embeddings, test_labels, explained_variance
    """
    logger.info("Loading training data")
    train_dataset = JSONLRequestReader(train_path, use_pipeline=False)
    train_texts = []

    for batch, _ in train_dataset.iter_batches(batch_size=1000):
        for text in batch:
            if with_preprocessing:
                text = preprocess(text)
            train_texts.append(text)

    logger.info(f"Loaded {len(train_texts)} training samples")

    logger.info("Loading test data")
    test_dataset = JSONLRequestReader(test_path, use_pipeline=False)
    test_texts = []
    test_labels = []

    for batch, labels in test_dataset.iter_batches(batch_size=1000):
        for text, label in zip(batch, labels):
            if with_preprocessing:
                text = preprocess(text)
            test_texts.append(text)
            test_labels.append(1 if label == "attack" else 0)

    logger.info(f"Loaded {len(test_texts)} test samples")

    # Fit TF-IDF
    logger.info("Fitting TF-IDF vectorizer")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)

    logger.info(f"TF-IDF shape: {train_tfidf.shape}")

    # Apply PCA
    logger.info(f"Applying PCA to {n_components} components")
    pca = PCA(n_components=n_components, random_state=42)
    train_embeddings = pca.fit_transform(train_tfidf.toarray())
    test_embeddings = pca.transform(test_tfidf.toarray())

    explained_variance = float(pca.explained_variance_ratio_.sum())
    logger.info(f"PCA explained variance: {explained_variance:.2%}")

    return (
        train_embeddings.astype(np.float32),
        test_embeddings.astype(np.float32),
        np.array(test_labels, dtype=int),
        explained_variance,
    )


def evaluate_detector(
    detector,
    detector_name: str,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    max_fpr: float = 0.05,
) -> dict:
    """Train and evaluate a detector."""
    logger.info(f"Training {detector_name}")
    detector.fit(train_embeddings)

    logger.info(f"Computing scores")
    test_scores = detector.scores(test_embeddings)

    # Split by label
    test_normal_mask = test_labels == 0
    test_scores_normal = test_scores[test_normal_mask]
    test_scores_anomalous = test_scores[~test_normal_mask]

    # Set threshold based on FPR
    threshold = float(np.percentile(test_scores_normal, (1 - max_fpr) * 100))
    actual_fpr = np.mean(test_scores_normal > threshold)

    # Compute metrics
    recall = np.mean(test_scores_anomalous > threshold)

    predictions = (test_scores > threshold).astype(int)
    tp = np.sum((predictions == 1) & (test_labels == 1))
    fp = np.sum((predictions == 1) & (test_labels == 0))
    tn = np.sum((predictions == 0) & (test_labels == 0))
    fn = np.sum((predictions == 0) & (test_labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / len(test_labels)

    logger.info(
        f"{detector_name} → Recall={recall:.2%} @ FPR={actual_fpr:.2%}, "
        f"Precision={precision:.2%}, F1={f1:.2%}"
    )

    return {
        "detector": detector_name,
        "recall": float(recall),
        "precision": float(precision),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
        "fpr": float(actual_fpr),
        "threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def main():
    """Run LOF vs Mahalanobis comparison on TF-IDF + PCA."""
    logger.info("=" * 80)
    logger.info("LOF vs Mahalanobis: TF-IDF + 150D PCA")
    logger.info("=" * 80)

    # Paths
    data_dir = Path("src/neuralshield/data")
    train_path = data_dir / "CSIC" / "train.jsonl"
    test_path = data_dir / "CSIC" / "test.jsonl"

    # Load and embed
    train_embeddings, test_embeddings, test_labels, explained_variance = (
        load_and_embed_tfidf(
            train_path,
            test_path,
            n_components=150,
            with_preprocessing=True,
        )
    )

    logger.info("=" * 80)
    logger.info("EVALUATION")
    logger.info("=" * 80)

    results = {
        "n_components": 150,
        "explained_variance": explained_variance,
        "detectors": [],
    }

    # Test LOF with multiple k values
    for n_neighbors in [5, 10, 20, 30, 50, 100]:
        logger.info(f"\nTesting LOF with n_neighbors={n_neighbors}")
        detector = LOFDetector(n_neighbors=n_neighbors)
        result = evaluate_detector(
            detector,
            f"LOF_k{n_neighbors}",
            train_embeddings,
            test_embeddings,
            test_labels,
        )
        result["n_neighbors"] = n_neighbors
        results["detectors"].append(result)

    # Test Mahalanobis baseline
    logger.info("\nTesting Mahalanobis baseline")
    detector = MahalanobisDetector()
    result = evaluate_detector(
        detector,
        "Mahalanobis",
        train_embeddings,
        test_embeddings,
        test_labels,
    )
    results["detectors"].append(result)

    # Save results
    output_dir = Path("experiments/15_lof_comparison/tfidf_pca_150")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"PCA Explained Variance: {explained_variance:.2%}")
    logger.info(f"\nRecall @ 5% FPR:")

    for result in results["detectors"]:
        name = result["detector"]
        recall = result["recall"]
        f1 = result["f1_score"]
        logger.info(f"  {name:20s} → {recall:6.2%} (F1={f1:.2%})")

    # Find best
    best = max(results["detectors"], key=lambda x: x["recall"])
    logger.info(f"\nBest: {best['detector']} with {best['recall']:.2%} recall")


if __name__ == "__main__":
    main()
