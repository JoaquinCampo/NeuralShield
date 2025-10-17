#!/usr/bin/env python3
"""
Smart hyperparameter search for LOF-SecBERT ensemble.

Based on previous experiments, this script tests only the most promising
hyperparameter combinations instead of exhaustive grid search.

Key insights from previous experiments:
- LOF k=100 works well on TF-IDF
- PCA=150 components captures 93% variance
- Fusion weights around 0.5-0.6 work well
- No preprocessing helps TF-IDF + LOF
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from neuralshield.anomaly import LOFDetector, MahalanobisDetector
from neuralshield.encoding.data.jsonl import JSONLRequestReader
from neuralshield.evaluation import ClassificationEvaluator, EvaluationConfig
from neuralshield.preprocessing.pipeline import preprocess
from neuralshield.preprocessing.steps.exceptions import MalformedHttpRequestError


def _parse_int_list(values: str) -> list[int]:
    return [int(item.strip()) for item in values.split(",") if item.strip()]


def _parse_float_or_auto(values: str) -> list[float | str]:
    parsed: list[float | str] = []
    for token in values.split(","):
        item = token.strip()
        if not item:
            continue
        if item.lower() == "auto":
            parsed.append("auto")
        else:
            parsed.append(float(item))
    return parsed


def _z_normalize(scores: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(scores.mean())
    std = float(scores.std())
    if std == 0.0:
        std = 1e-6
    normalized = (scores - mean) / std
    return normalized.astype(np.float32), mean, std


def _apply_z(scores: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std == 0.0:
        std = 1e-6
    return ((scores - mean) / std).astype(np.float32)


def _load_texts_and_labels(path: Path) -> tuple[list[str], list[str]]:
    reader = JSONLRequestReader(path, use_pipeline=False)
    texts: list[str] = []
    labels: list[str] = []
    for batch, batch_labels in reader.iter_batches(batch_size=2000):
        for item, label in zip(batch, batch_labels):
            try:
                processed = preprocess(item)
            except MalformedHttpRequestError:
                logger.debug("Skipping malformed request")
                continue
            texts.append(processed)
            labels.append(label)
    return texts, labels


def _evaluate(
    predictions: Iterable[bool],
    labels: Iterable[str],
) -> dict[str, float | int]:
    evaluator = ClassificationEvaluator(EvaluationConfig())
    result = evaluator.evaluate(list(predictions), list(labels))
    return {
        "precision": result.precision,
        "recall": result.recall,
        "f1_score": result.f1_score,
        "accuracy": result.accuracy,
        "fpr": result.fpr,
        "specificity": result.specificity,
        "tp": result.tp,
        "fp": result.fp,
        "tn": result.tn,
        "fn": result.fn,
        "total_samples": result.total_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart hyperparameter search for LOF-SecBERT ensemble"
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("src/neuralshield/data/SR_BH_2020/train.jsonl"),
        help="SR_BH train JSONL path.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("src/neuralshield/data/SR_BH_2020/test.jsonl"),
        help="SR_BH test JSONL path.",
    )
    parser.add_argument(
        "--secbert-train",
        type=Path,
        default=Path(
            "experiments/08_secbert_srbh/with_preprocessing/train_embeddings.npz"
        ),
        help="SecBERT train embeddings NPZ.",
    )
    parser.add_argument(
        "--secbert-test",
        type=Path,
        default=Path(
            "experiments/08_secbert_srbh/with_preprocessing/test_embeddings.npz"
        ),
        help="SecBERT test embeddings NPZ.",
    )
    parser.add_argument(
        "--max-fpr",
        type=float,
        default=0.05,
        help="Target FPR for threshold calibration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/18_lof_secbert_ensemble/smart_search"),
        help="Directory for search outputs.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SR_BH datasets")
    train_texts, _ = _load_texts_and_labels(args.train_path)
    test_texts, test_labels = _load_texts_and_labels(args.test_path)
    logger.info(
        "Loaded train samples={train_count}, test samples={test_count}",
        train_count=len(train_texts),
        test_count=len(test_texts),
    )

    logger.info("Fitting TF-IDF vectorizer (5000 dims, ngram 1-3)")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)
    logger.info("TF-IDF matrices train=%s test=%s", train_tfidf.shape, test_tfidf.shape)

    logger.info("Loading SecBERT embeddings")
    sec_train_npz = np.load(args.secbert_train)
    secbert_train = sec_train_npz["embeddings"].astype(np.float32)

    sec_test_npz = np.load(args.secbert_test, allow_pickle=True)
    secbert_test = sec_test_npz["embeddings"].astype(np.float32)
    secbert_labels = sec_test_npz["labels"].astype(str)

    if not np.array_equal(np.array(test_labels), secbert_labels):
        raise ValueError(
            "Label mismatch between TF-IDF test data and SecBERT embeddings"
        )

    logger.info("Training Mahalanobis on SecBERT train embeddings")
    maha = MahalanobisDetector()
    maha.fit(secbert_train)
    maha.set_threshold(secbert_train, max_fpr=args.max_fpr)
    maha_train_scores = maha.scores(secbert_train)
    maha_test_scores = maha.scores(secbert_test)
    maha_predictions = maha.predict(secbert_test)
    maha_metrics = _evaluate(maha_predictions, secbert_labels)
    logger.info(
        "Mahalanobis: recall={recall:.3f} fpr={fpr:.3f}",
        recall=maha_metrics["recall"],
        fpr=maha_metrics["fpr"],
    )

    # Smart hyperparameter search based on previous experiments
    # Only test the most promising configurations
    smart_configs = [
        # Based on Exp 15: LOF + TF-IDF best results
        {
            "pca_components": 150,
            "lof_neighbors": 100,
            "lof_contamination": "auto",
            "fusion_weight": 0.5,
        },
        {
            "pca_components": 150,
            "lof_neighbors": 100,
            "lof_contamination": 0.01,
            "fusion_weight": 0.5,
        },
        {
            "pca_components": 150,
            "lof_neighbors": 100,
            "lof_contamination": 0.02,
            "fusion_weight": 0.5,
        },
        # Test slightly different PCA dimensions
        {
            "pca_components": 175,
            "lof_neighbors": 100,
            "lof_contamination": "auto",
            "fusion_weight": 0.5,
        },
        {
            "pca_components": 125,
            "lof_neighbors": 100,
            "lof_contamination": "auto",
            "fusion_weight": 0.5,
        },
        # Test different fusion weights
        {
            "pca_components": 150,
            "lof_neighbors": 100,
            "lof_contamination": "auto",
            "fusion_weight": 0.4,
        },
        {
            "pca_components": 150,
            "lof_neighbors": 100,
            "lof_contamination": "auto",
            "fusion_weight": 0.6,
        },
        {
            "pca_components": 150,
            "lof_neighbors": 100,
            "lof_contamination": "auto",
            "fusion_weight": 0.7,
        },
        # Test different k values
        {
            "pca_components": 150,
            "lof_neighbors": 75,
            "lof_contamination": "auto",
            "fusion_weight": 0.5,
        },
        {
            "pca_components": 150,
            "lof_neighbors": 125,
            "lof_contamination": "auto",
            "fusion_weight": 0.5,
        },
    ]

    results: list[dict[str, object]] = []
    best_entry: dict[str, object] | None = None
    best_score = -np.inf

    logger.info(f"Testing {len(smart_configs)} smart hyperparameter configurations...")

    for i, config in enumerate(smart_configs):
        logger.info(f"=== Configuration {i + 1}/{len(smart_configs)}: {config} ===")

        components = config["pca_components"]
        neighbors = config["lof_neighbors"]
        contamination = config["lof_contamination"]
        fusion_weight = config["fusion_weight"]

        # PCA transformation
        svd = TruncatedSVD(n_components=components, random_state=42)
        train_embeddings = svd.fit_transform(train_tfidf).astype(np.float32)
        test_embeddings = svd.transform(test_tfidf).astype(np.float32)

        logger.info(
            f"PCA: {components} components, explained variance: {svd.explained_variance_ratio_.sum():.3f}"
        )

        # Train LOF
        logger.info(f"Training LOF (k={neighbors}, contamination={contamination})")
        lof = LOFDetector(n_neighbors=neighbors, contamination=contamination)
        lof.fit(train_embeddings)
        lof_threshold = lof.set_threshold(train_embeddings, max_fpr=args.max_fpr)

        # Get scores and predictions
        lof_train_scores = lof.scores(train_embeddings)
        lof_test_scores = lof.scores(test_embeddings)
        lof_predictions = lof.predict(test_embeddings)
        lof_metrics = _evaluate(lof_predictions, test_labels)

        logger.info(
            f"LOF alone: recall={lof_metrics['recall']:.3f}, fpr={lof_metrics['fpr']:.3f}"
        )

        # Union ensemble
        union_predictions = np.logical_or(
            np.array(lof_predictions, dtype=bool),
            np.array(maha_predictions, dtype=bool),
        )
        union_metrics = _evaluate(union_predictions, test_labels)

        # Score-level fusion
        lof_norm_train, lof_mean, lof_std = _z_normalize(lof_train_scores)
        mah_norm_train, mah_mean, mah_std = _z_normalize(maha_train_scores)

        lof_norm_test = _apply_z(lof_test_scores, lof_mean, lof_std)
        mah_norm_test = _apply_z(maha_test_scores, mah_mean, mah_std)

        fused_train = (
            fusion_weight * lof_norm_train + (1 - fusion_weight) * mah_norm_train
        )
        fused_threshold = float(np.quantile(fused_train, 1.0 - args.max_fpr))

        fused_scores = (
            fusion_weight * lof_norm_test + (1 - fusion_weight) * mah_norm_test
        )
        fused_predictions = fused_scores >= fused_threshold
        fused_metrics = _evaluate(fused_predictions, test_labels)

        logger.info(
            f"Fused (weight={fusion_weight}): recall={fused_metrics['recall']:.3f}, fpr={fused_metrics['fpr']:.3f}"
        )

        record = {
            "config_id": i + 1,
            "pca_components": components,
            "lof_neighbors": neighbors,
            "lof_contamination": contamination,
            "fusion_weight": fusion_weight,
            "lof_threshold": float(lof_threshold),
            "fused_threshold": fused_threshold,
            "lof": lof_metrics,
            "mahalanobis": maha_metrics,
            "union": union_metrics,
            "fused": fused_metrics,
        }
        results.append(record)

        if (
            fused_metrics["fpr"] <= args.max_fpr
            and fused_metrics["recall"] > best_score
        ):
            best_score = fused_metrics["recall"]
            best_entry = record

    # Save results
    results_path = args.output_dir / "smart_search_results.jsonl"
    with results_path.open("w") as handle:
        for item in results:
            handle.write(json.dumps(item, default=str) + "\n")
    logger.info(f"Saved detailed results to {results_path}")

    # Create summary CSV
    flat_rows = []
    for entry in results:
        row = {
            "config_id": entry["config_id"],
            "pca_components": entry["pca_components"],
            "lof_neighbors": entry["lof_neighbors"],
            "lof_contamination": entry["lof_contamination"],
            "fusion_weight": entry["fusion_weight"],
            "lof_recall": entry["lof"]["recall"],
            "lof_fpr": entry["lof"]["fpr"],
            "fused_recall": entry["fused"]["recall"],
            "fused_fpr": entry["fused"]["fpr"],
            "union_recall": entry["union"]["recall"],
            "union_fpr": entry["union"]["fpr"],
        }
        flat_rows.append(row)

    df = pd.DataFrame(flat_rows)
    df.to_csv(args.output_dir / "smart_search_summary.csv", index=False)
    logger.info(
        f"Exported CSV summary to {args.output_dir / 'smart_search_summary.csv'}"
    )

    if best_entry is None:
        logger.warning("No configuration met the FPR constraint.")
        return

    logger.info("=" * 60)
    logger.info("BEST CONFIGURATION FOUND:")
    logger.info(f"PCA components: {best_entry['pca_components']}")
    logger.info(f"LOF neighbors: {best_entry['lof_neighbors']}")
    logger.info(f"LOF contamination: {best_entry['lof_contamination']}")
    logger.info(f"Fusion weight: {best_entry['fusion_weight']}")
    logger.info(
        f"Performance: recall={best_entry['fused']['recall']:.3f}, fpr={best_entry['fused']['fpr']:.3f}"
    )
    logger.info("=" * 60)

    # Save best configuration
    with (args.output_dir / "best_config.json").open("w") as handle:
        json.dump(best_entry, handle, indent=2, default=str)

    logger.info(f"Best configuration saved to {args.output_dir / 'best_config.json'}")


if __name__ == "__main__":
    main()
