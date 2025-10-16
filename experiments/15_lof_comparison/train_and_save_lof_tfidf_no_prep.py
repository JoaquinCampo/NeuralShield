"""Train LOF on TF-IDF + PCA WITHOUT preprocessing.

This script:
1. Generates TF-IDF embeddings with 150D PCA (NO preprocessing)
2. Trains LOF detector (k=100)
3. Saves model and embeddings for comparison
"""

import json
from pathlib import Path

import joblib
import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from neuralshield.anomaly import LOFDetector
from neuralshield.encoding.data.jsonl import JSONLRequestReader


def main():
    """Train and save LOF + TF-IDF model WITHOUT preprocessing."""
    logger.info("=" * 80)
    logger.info("Training LOF on TF-IDF + 150D PCA (NO PREPROCESSING)")
    logger.info("=" * 80)

    # Paths
    data_dir = Path("src/neuralshield/data")
    train_path = data_dir / "CSIC" / "train.jsonl"
    test_path = data_dir / "CSIC" / "test.jsonl"

    output_dir = Path("experiments/15_lof_comparison/tfidf_pca_150_no_prep")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data (NO preprocessing)
    logger.info("Loading training data (NO preprocessing)")
    train_dataset = JSONLRequestReader(train_path, use_pipeline=False)
    train_texts = []

    for batch, _ in train_dataset.iter_batches(batch_size=1000):
        for text in batch:
            # NO preprocessing applied
            train_texts.append(text)

    logger.info(f"Loaded {len(train_texts)} training samples")

    # Load test data (NO preprocessing)
    logger.info("Loading test data (NO preprocessing)")
    test_dataset = JSONLRequestReader(test_path, use_pipeline=False)
    test_texts = []
    test_labels = []

    for batch, labels in test_dataset.iter_batches(batch_size=1000):
        for text, label in zip(batch, labels):
            # NO preprocessing applied
            test_texts.append(text)
            test_labels.append(label)

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
    logger.info("Applying PCA to 150 components")
    pca = PCA(n_components=150, random_state=42)
    train_embeddings = pca.fit_transform(train_tfidf.toarray())
    test_embeddings = pca.transform(test_tfidf.toarray())

    explained_variance = float(pca.explained_variance_ratio_.sum())
    logger.info(f"PCA explained variance: {explained_variance:.2%}")

    # Train LOF
    logger.info("Training LOF detector (k=100)")
    detector = LOFDetector(n_neighbors=100)
    detector.fit(train_embeddings.astype(np.float32))

    # Compute scores and set threshold
    logger.info("Computing scores and setting threshold")
    test_scores = detector.scores(test_embeddings.astype(np.float32))

    # Split by label for threshold setting
    test_labels_binary = np.array(
        [1 if label == "attack" else 0 for label in test_labels]
    )
    test_normal_mask = test_labels_binary == 0
    test_scores_normal = test_scores[test_normal_mask]

    # Set threshold at 5% FPR
    threshold = float(np.percentile(test_scores_normal, 95.0))
    detector._threshold = threshold

    actual_fpr = np.mean(test_scores_normal > threshold)
    logger.info(f"Threshold set to {threshold:.4f} (actual FPR: {actual_fpr:.2%})")

    # Save model
    model_path = output_dir / "lof_tfidf_pca150_k100_no_prep.joblib"
    logger.info(f"Saving model to {model_path}")

    model_data = {
        "name": "LOF_TF-IDF_PCA150_NoPrep",
        "detector": detector,
        "vectorizer": vectorizer,
        "pca": pca,
        "threshold": threshold,
        "n_neighbors": 100,
        "n_components": 150,
        "explained_variance": explained_variance,
        "preprocessing": False,
    }
    joblib.dump(model_data, model_path)
    logger.info("Model saved")

    # Save test embeddings
    embeddings_path = output_dir / "csic_test_embeddings.npz"
    logger.info(f"Saving test embeddings to {embeddings_path}")

    np.savez(
        embeddings_path,
        embeddings=test_embeddings.astype(np.float32),
        labels=np.array(test_labels),
    )
    logger.info("Embeddings saved")

    # Compute and save metrics
    predictions = detector.predict(test_embeddings.astype(np.float32))
    test_scores_anomalous = test_scores[~test_normal_mask]

    recall = np.mean(test_scores_anomalous > threshold)
    tp = np.sum((predictions == 1) & (test_labels_binary == 1))
    fp = np.sum((predictions == 1) & (test_labels_binary == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics = {
        "model": "LOF_TF-IDF_PCA150_k100_NoPrep",
        "n_neighbors": 100,
        "n_components": 150,
        "explained_variance": explained_variance,
        "preprocessing": False,
        "threshold": threshold,
        "recall": float(recall),
        "precision": float(precision),
        "f1_score": float(f1),
        "fpr": float(actual_fpr),
        "true_positives": int(tp),
        "false_positives": int(fp),
    }

    metrics_path = output_dir / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 80)
    logger.info("SUMMARY (NO PREPROCESSING)")
    logger.info("=" * 80)
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Embeddings saved to: {embeddings_path}")
    logger.info(f"Metrics saved to: {metrics_path}")
    logger.info("")
    logger.info(f"Recall @ 5% FPR: {recall:.2%}")
    logger.info(f"Precision: {precision:.2%}")
    logger.info(f"F1-Score: {f1:.2%}")
    logger.info("")
    logger.info("Compare with preprocessing version:")
    logger.info("  uv run python src/scripts/compare_model_predictions.py \\")
    logger.info(
        "    -m LOF_WithPrep:experiments/15_lof_comparison/tfidf_pca_150/lof_tfidf_pca150_k100.joblib:experiments/15_lof_comparison/tfidf_pca_150/csic_test_embeddings.npz \\"
    )
    logger.info(f"    -m LOF_NoPrep:{model_path}:{embeddings_path}")


if __name__ == "__main__":
    main()
