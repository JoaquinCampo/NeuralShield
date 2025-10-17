"""Test LOF on TF-IDF + PCA embeddings.

Generates TF-IDF embeddings with PCA-reduced TF-IDF vectors and evaluates LOF
against a Mahalanobis baseline. This script now accepts command-line arguments
so we can swap datasets, preprocessing pipelines, and detector parameters.
"""

import argparse
import importlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from neuralshield.anomaly import LOFDetector, MahalanobisDetector
from neuralshield.encoding.data.jsonl import JSONLRequestReader


def resolve_preprocess(spec: str) -> Callable[[str], str] | None:
    """Resolve a preprocessing pipeline specifier into a callable."""

    if spec in {"none", "skip", ""}:
        return None

    if spec == "default":
        from neuralshield.preprocessing.pipeline import preprocess

        return preprocess

    if spec == "csic-overfit":
        from neuralshield.preprocessing.pipeline_csic_overfit import (
            preprocess_csic_overfit,
        )

        return preprocess_csic_overfit

    if spec == "srbh-overfit":
        from neuralshield.preprocessing.pipeline_srbh_overfit import (
            preprocess_srbh_overfit,
        )

        return preprocess_srbh_overfit

    if spec == "csic-long-flags":
        from neuralshield.preprocessing.pipeline_csic_long_flags import (
            preprocess_csic_long_flags,
        )

        return preprocess_csic_long_flags

    if spec.startswith("import:"):
        _, target = spec.split(":", 1)
        module_name, attr_name = target.split(":")
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    raise ValueError(
        f"Unsupported preprocess spec '{spec}'. Use 'default', 'csic-overfit', "
        "'srbh-overfit', 'csic-long-flags', 'none', or 'import:module:callable'."
    )


def load_and_embed_tfidf(
    train_path: Path,
    test_path: Path,
    n_components: int = 150,
    preprocess_fn: Callable[[str], str] | None = None,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
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
            if preprocess_fn is not None:
                text = preprocess_fn(text)
            train_texts.append(text)
            if max_train_samples and len(train_texts) >= max_train_samples:
                break
        if max_train_samples and len(train_texts) >= max_train_samples:
            break

    logger.info(f"Loaded {len(train_texts)} training samples")

    logger.info("Loading test data")
    test_dataset = JSONLRequestReader(test_path, use_pipeline=False)
    test_texts = []
    test_labels = []

    for batch, labels in test_dataset.iter_batches(batch_size=1000):
        for text, label in zip(batch, labels):
            if preprocess_fn is not None:
                text = preprocess_fn(text)
            test_texts.append(text)
            test_labels.append(1 if label == "attack" else 0)
            if max_test_samples and len(test_texts) >= max_test_samples:
                break
        if max_test_samples and len(test_texts) >= max_test_samples:
            break

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LOF and Mahalanobis on TF-IDF + PCA embeddings."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("src/neuralshield/data/CSIC/train.jsonl"),
        help="Path to JSONL file containing training requests.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("src/neuralshield/data/CSIC/test.jsonl"),
        help="Path to JSONL file containing test requests.",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="default",
        help=(
            "Preprocessing pipeline spec. "
            "Use 'default', 'csic-overfit', 'srbh-overfit', 'csic-long-flags', "
            "'none', or 'import:module:callable'."
        ),
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=150,
        help="Number of PCA components.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on the number of training samples.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap on the number of test samples.",
    )
    parser.add_argument(
        "--lof-neighbors",
        type=int,
        nargs="+",
        default=[5, 10, 20, 30, 50, 100],
        help="List of neighbor counts to evaluate for LOF.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store results (defaults to experiments/15_lof_comparison/tfidf_pca_<n>).",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    preprocess_fn = resolve_preprocess(args.preprocess)

    logger.info("=" * 80)
    logger.info(
        "LOF vs Mahalanobis: TF-IDF + {}D PCA | preprocess={}",
        args.n_components,
        args.preprocess,
    )
    logger.info("=" * 80)

    train_embeddings, test_embeddings, test_labels, explained_variance = (
        load_and_embed_tfidf(
            args.train_path,
            args.test_path,
            n_components=args.n_components,
            preprocess_fn=preprocess_fn,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
    )

    logger.info("=" * 80)
    logger.info("EVALUATION")
    logger.info("=" * 80)

    results = {
        "n_components": args.n_components,
        "explained_variance": explained_variance,
        "preprocess": args.preprocess,
        "train_path": str(args.train_path),
        "test_path": str(args.test_path),
        "detectors": [],
    }

    for n_neighbors in args.lof_neighbors:
        logger.info("\nTesting LOF with n_neighbors={}", n_neighbors)
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

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path("experiments/15_lof_comparison") / f"tfidf_pca_{args.n_components}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\nResults saved to {}", output_path)

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info("PCA Explained Variance: {:.2f}%", explained_variance * 100)
    logger.info("\nRecall @ 5% FPR:")

    for result in results["detectors"]:
        name = result["detector"]
        recall = result["recall"]
        f1 = result["f1_score"]
        logger.info("  {:<20s} → {:6.2f}% (F1={:.2f}%)", name, recall * 100, f1 * 100)

    best = max(results["detectors"], key=lambda x: x["recall"])
    logger.info(
        "\nBest: {} with {:.2f}% recall", best["detector"], best["recall"] * 100
    )


if __name__ == "__main__":
    main()
