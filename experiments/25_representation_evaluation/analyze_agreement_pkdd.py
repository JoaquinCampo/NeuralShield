#!/usr/bin/env python3
"""
Análisis de Agreement entre TF-IDF+PCA+LOF y SecBERT+Mahalanobis - PKDD Dataset

Este script analiza el agreement entre ambos modelos en el dataset PKDD.
"""

import json

# Importar funciones compartidas del análisis de CSIC
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from matplotlib_venn import venn2

from neuralshield.anomaly import LOFDetector, MahalanobisDetector
from neuralshield.encoding.data.jsonl import JSONLRequestReader

sys.path.insert(0, str(Path(__file__).parent))
from analyze_agreement import (
    analyze_agreement,
    generate_report,
    plot_agreement_visualizations,
)


def train_tfidf_lof_pkdd(
    train_path: Path,
    test_path: Path,
    test_labels: list[str],
    n_neighbors: int = 100,
    target_fpr: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train TF-IDF + PCA + LOF model for PKDD."""
    logger.info("Training TF-IDF + PCA + LOF for PKDD...")

    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Load train data (only normal for LOF training)
    train_reader = JSONLRequestReader(train_path, use_pipeline=False)
    train_texts = []
    for batch, batch_labels in train_reader.iter_batches(batch_size=1000):
        for text, label in zip(batch, batch_labels):
            if label == "valid":  # Only normal for training
                train_texts.append(text)

    logger.info(f"Loaded {len(train_texts)} train normal samples")

    # Load test texts
    test_reader = JSONLRequestReader(test_path, use_pipeline=False)
    test_texts = []
    for batch, _ in test_reader.iter_batches(batch_size=1000):
        test_texts.extend(batch)

    logger.info(f"Loaded {len(test_texts)} test samples")

    # Train TF-IDF + PCA pipeline from scratch
    # Use same parameters as experiment 22 (1750 components or target variance)
    # Based on model_metrics.json, experiment 18 used 1750 components
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)

    # Use 1750 components as in experiment 18 (based on model_metrics.json)
    n_components_actual = 1750
    pca = PCA(n_components=n_components_actual, random_state=42)
    train_embeddings = pca.fit_transform(train_tfidf.toarray())
    test_embeddings = pca.transform(test_tfidf.toarray())

    explained_variance = float(pca.explained_variance_ratio_.sum())
    logger.info(
        f"PCA explained variance: {explained_variance:.2%}, components: {n_components_actual}"
    )

    # Train LOF on train embeddings
    detector = LOFDetector(n_neighbors=n_neighbors)
    detector.fit(train_embeddings.astype(np.float32))

    # Score test embeddings
    lof_scores = detector.scores(test_embeddings.astype(np.float32))

    # Binary labels
    binary_labels = np.array([1 if label == "attack" else 0 for label in test_labels])
    normal_mask = binary_labels == 0

    # Threshold at target FPR
    threshold = float(np.percentile(lof_scores[normal_mask], 100 * (1 - target_fpr)))
    predictions = (lof_scores > threshold).astype(bool)

    # Metrics
    tp = np.sum((predictions == 1) & (binary_labels == 1))
    fp = np.sum((predictions == 1) & (binary_labels == 0))
    tn = np.sum((predictions == 0) & (binary_labels == 0))
    fn = np.sum((predictions == 0) & (binary_labels == 1))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    actual_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "recall": float(recall),
        "precision": float(precision),
        "f1_score": float(f1),
        "fpr": float(actual_fpr),
        "threshold": float(threshold),
        "explained_variance": explained_variance,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    logger.info(
        f"TF-IDF+LOF - Recall: {recall:.2%}, Precision: {precision:.2%}, FPR: {actual_fpr:.2%}"
    )

    return lof_scores, predictions, metrics


def load_predictions_from_embeddings(
    secbert_embeddings_path: Path,
    secbert_model_path: Path,
    train_path: Path,
    test_path: Path,
    test_labels: list[str],
    target_fpr: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict, list[str]]:
    """Load models and generate predictions from pre-computed embeddings."""
    logger.info("Loading embeddings and models...")

    # Load SecBERT embeddings
    logger.info(f"Loading SecBERT embeddings from {secbert_embeddings_path}")
    secbert_data = np.load(secbert_embeddings_path, allow_pickle=True)
    secbert_embeddings = secbert_data["embeddings"]

    # SecBERT embeddings may have fewer samples due to preprocessing filtering
    # Need to match labels to embeddings
    logger.info(f"SecBERT embeddings shape: {secbert_embeddings.shape}")
    logger.info(f"Original test labels count: {len(test_labels)}")

    # Load test data with preprocessing to get filtered labels
    from neuralshield.preprocessing.pipeline import preprocess
    from neuralshield.preprocessing.steps.exceptions import MalformedHttpRequestError

    test_reader = JSONLRequestReader(test_path, use_pipeline=False)
    test_labels_filtered = []
    filtered_count = 0

    for batch, batch_labels in test_reader.iter_batches(batch_size=1000):
        for text, label in zip(batch, batch_labels):
            try:
                # Try preprocessing to see if it would filter this sample
                _ = preprocess(text)
                test_labels_filtered.append(label)
            except MalformedHttpRequestError:
                # This sample would be filtered
                filtered_count += 1
                continue

    # Ensure we have the right number
    if len(test_labels_filtered) != len(secbert_embeddings):
        logger.warning(
            f"Label count mismatch: filtered={len(test_labels_filtered)}, "
            f"embeddings={len(secbert_embeddings)}. Using embeddings count."
        )
        # Use embeddings count as ground truth
        test_labels_filtered = test_labels_filtered[: len(secbert_embeddings)]

    logger.info(
        f"Filtered labels count: {len(test_labels_filtered)} (filtered out: {filtered_count})"
    )

    # Train TF-IDF + LOF model (will generate predictions for all test samples)
    logger.info(f"Training LOF model from train data")
    lof_scores, lof_predictions, lof_metrics = train_tfidf_lof_pkdd(
        train_path,
        test_path,
        test_labels,  # Use original labels for training
        n_neighbors=100,
        target_fpr=target_fpr,
    )

    # Filter LOF predictions to match SecBERT embeddings
    if len(lof_predictions) != len(test_labels_filtered):
        logger.info(
            f"Filtering LOF predictions: {len(lof_predictions)} -> {len(test_labels_filtered)}"
        )
        # Need to filter LOF predictions the same way SecBERT embeddings were filtered
        # Create a mask for which samples passed preprocessing
        lof_scores_filtered = []
        lof_predictions_filtered = []

        test_reader_2 = JSONLRequestReader(test_path, use_pipeline=False)
        idx = 0
        for batch, batch_labels in test_reader_2.iter_batches(batch_size=1000):
            for text, label in zip(batch, batch_labels):
                try:
                    _ = preprocess(text)
                    # This sample passed preprocessing
                    lof_scores_filtered.append(lof_scores[idx])
                    lof_predictions_filtered.append(lof_predictions[idx])
                except MalformedHttpRequestError:
                    # This sample was filtered
                    pass
                idx += 1

        lof_scores = np.array(lof_scores_filtered)
        lof_predictions = np.array(lof_predictions_filtered)

    # Load SecBERT model
    logger.info(f"Loading SecBERT model from {secbert_model_path}")
    secbert_model_data = joblib.load(secbert_model_path)
    if isinstance(secbert_model_data, dict):
        # Try different possible keys
        secbert_model_raw = secbert_model_data.get("model") or secbert_model_data.get(
            "detector"
        )
        if secbert_model_raw is None:
            raise ValueError(
                f"SecBERT model not found in joblib file. Keys: {list(secbert_model_data.keys())}"
            )
        # If threshold is stored, use it (but we'll recalculate at target FPR)
        stored_threshold = secbert_model_data.get("threshold")
        if stored_threshold is not None:
            logger.info(f"Found stored threshold: {stored_threshold}")

        # If it's a sklearn EmpiricalCovariance, wrap it in MahalanobisDetector
        from sklearn.covariance import EmpiricalCovariance

        if isinstance(secbert_model_raw, EmpiricalCovariance):
            logger.info("Wrapping EmpiricalCovariance in MahalanobisDetector")
            secbert_detector = MahalanobisDetector()
            secbert_detector._model = secbert_model_raw
            secbert_detector._fitted = True
            # Load mean if available
            if hasattr(secbert_model_raw, "location_"):
                secbert_detector._mean = secbert_model_raw.location_
        else:
            secbert_detector = secbert_model_raw
    else:
        # Direct model object
        from sklearn.covariance import EmpiricalCovariance

        if isinstance(secbert_model_data, EmpiricalCovariance):
            logger.info("Wrapping EmpiricalCovariance in MahalanobisDetector")
            secbert_detector = MahalanobisDetector()
            secbert_detector._model = secbert_model_data
            secbert_detector._fitted = True
            if hasattr(secbert_model_data, "location_"):
                secbert_detector._mean = secbert_model_data.location_
        else:
            secbert_detector = secbert_model_data

    # Compute SecBERT scores
    secbert_scores = secbert_detector.scores(secbert_embeddings.astype(np.float32))

    # Binary labels (use filtered labels)
    binary_labels = np.array(
        [1 if label == "attack" else 0 for label in test_labels_filtered]
    )
    normal_mask = binary_labels == 0

    # Thresholds at target FPR
    lof_threshold = float(
        np.percentile(lof_scores[normal_mask], 100 * (1 - target_fpr))
    )
    secbert_threshold = float(
        np.percentile(secbert_scores[normal_mask], 100 * (1 - target_fpr))
    )

    lof_predictions = (lof_scores > lof_threshold).astype(bool)
    secbert_predictions = (secbert_scores > secbert_threshold).astype(bool)

    # Compute metrics
    def compute_metrics(predictions, binary_labels):
        tp = np.sum((predictions == 1) & (binary_labels == 1))
        fp = np.sum((predictions == 1) & (binary_labels == 0))
        tn = np.sum((predictions == 0) & (binary_labels == 0))
        fn = np.sum((predictions == 0) & (binary_labels == 1))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        actual_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            "recall": float(recall),
            "precision": float(precision),
            "f1_score": float(f1),
            "fpr": float(actual_fpr),
            "threshold": float(
                lof_threshold
                if len(lof_scores) == len(binary_labels)
                else secbert_threshold
            ),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }

    lof_metrics = compute_metrics(lof_predictions, binary_labels)
    secbert_metrics = compute_metrics(secbert_predictions, binary_labels)

    logger.info(
        f"TF-IDF+LOF - Recall: {lof_metrics['recall']:.2%}, Precision: {lof_metrics['precision']:.2%}, FPR: {lof_metrics['fpr']:.2%}"
    )
    logger.info(
        f"SecBERT+Mahalanobis - Recall: {secbert_metrics['recall']:.2%}, Precision: {secbert_metrics['precision']:.2%}, FPR: {secbert_metrics['fpr']:.2%}"
    )

    return (
        lof_scores,
        lof_predictions,
        secbert_scores,
        secbert_predictions,
        lof_metrics,
        secbert_metrics,
        test_labels_filtered,
    )


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("ANÁLISIS DE AGREEMENT: TF-IDF+LOF vs SecBERT+Mahalanobis - PKDD")
    logger.info("=" * 80)

    # Paths
    data_dir = Path("src/neuralshield/data/PKDD")
    test_path = data_dir / "test.jsonl"

    # Embeddings paths from experiment 18
    base_dir = Path("experiments/18_lof_secbert_ensemble/pkdd")

    # SecBERT+Mahalanobis (with preprocessing)
    secbert_embeddings_path = (
        base_dir / "with_preprocessing" / "secbert_test_embeddings.npz"
    )
    secbert_model_path = base_dir / "with_preprocessing" / "secbert_mahalanobis.joblib"

    # Validate paths
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not secbert_embeddings_path.exists():
        raise FileNotFoundError(
            f"SecBERT embeddings not found: {secbert_embeddings_path}"
        )
    if not secbert_model_path.exists():
        raise FileNotFoundError(f"SecBERT model not found: {secbert_model_path}")

    output_dir = Path(
        "experiments/25_representation_evaluation/agreement_analysis_pkdd"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test labels
    logger.info(f"Loading test labels from {test_path}")
    test_labels = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            test_labels.append(obj["label"])

    logger.info(f"Loaded {len(test_labels)} test samples")

    # Load train path for LOF training
    train_path = data_dir / "train.jsonl"

    # Load embeddings and generate predictions
    (
        lof_scores,
        lof_predictions,
        secbert_scores,
        secbert_predictions,
        lof_metrics,
        secbert_metrics,
        test_labels_filtered,
    ) = load_predictions_from_embeddings(
        secbert_embeddings_path,
        secbert_model_path,
        train_path,
        test_path,
        test_labels,
        target_fpr=0.05,
    )

    # Align LOF predictions with filtered labels if needed
    if len(lof_predictions) != len(test_labels_filtered):
        logger.warning(
            f"LOF predictions: {len(lof_predictions)}, Filtered labels: {len(test_labels_filtered)}. "
            "Truncating LOF predictions to match."
        )
        lof_predictions = lof_predictions[: len(test_labels_filtered)]

    # Analyze agreement
    agreement_data = analyze_agreement(
        lof_predictions, secbert_predictions, test_labels_filtered
    )

    # Generate visualizations
    plot_agreement_visualizations(
        lof_predictions,
        secbert_predictions,
        test_labels_filtered,
        agreement_data,
        output_dir,
    )

    # Generate report
    generate_report(lof_metrics, secbert_metrics, agreement_data, output_dir)

    # Save JSON results
    results = {
        "dataset": "PKDD",
        "lof_metrics": lof_metrics,
        "secbert_metrics": secbert_metrics,
        "agreement_analysis": agreement_data,
    }

    results_path = output_dir / "agreement_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")
    logger.info("\n" + "=" * 80)
    logger.info("ANÁLISIS COMPLETO - PKDD")
    logger.info("=" * 80)
    logger.info(f"Agreement: {agreement_data['overall_agreement']:.2%}")
    logger.info(f"Recall TF-IDF+LOF: {lof_metrics['recall']:.2%}")
    logger.info(f"Recall SecBERT+Mahalanobis: {secbert_metrics['recall']:.2%}")
    logger.info(
        f"Recall potencial ensemble: {agreement_data['ensemble_potential']['combined_recall']:.2%}"
    )


if __name__ == "__main__":
    main()
