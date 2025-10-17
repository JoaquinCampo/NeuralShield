#!/usr/bin/env python3
"""
Quick test of the best known hyperparameter configuration.

Based on consolidated results from experiments 15 and 08:
- TF-IDF (5000 dims) + PCA (150 components) + LOF (k=100, contamination=auto) + no preprocessing
- Fused with SecBERT + Mahalanobis (weight=0.5)

This should give ~64% recall @ 5% FPR in much less time than grid search.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from neuralshield.anomaly import LOFDetector, MahalanobisDetector
from neuralshield.encoding.data.jsonl import JSONLRequestReader
from neuralshield.evaluation import ClassificationEvaluator, EvaluationConfig
from neuralshield.preprocessing.pipeline import preprocess
from neuralshield.preprocessing.steps.exceptions import MalformedHttpRequestError


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
    logger.info("Testing best known hyperparameter configuration...")

    # Load datasets (raw, no preprocessing for TF-IDF)
    logger.info("Loading SR_BH datasets")
    train_texts, _ = _load_texts_and_labels(
        Path("src/neuralshield/data/SR_BH_2020/train.jsonl")
    )
    test_texts, test_labels = _load_texts_and_labels(
        Path("src/neuralshield/data/SR_BH_2020/test.jsonl")
    )
    logger.info(
        f"Loaded train samples={len(train_texts)}, test samples={len(test_texts)}"
    )

    # TF-IDF vectorization (based on Exp 15)
    logger.info("Fitting TF-IDF vectorizer (5000 dims, ngram 1-3)")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)
    logger.info(f"TF-IDF matrices train={train_tfidf.shape} test={test_tfidf.shape}")

    # PCA (based on Exp 15: 150 components)
    logger.info("Applying PCA (150 components)")
    svd = TruncatedSVD(n_components=150, random_state=42)
    train_embeddings = svd.fit_transform(train_tfidf).astype(np.float32)
    test_embeddings = svd.transform(test_tfidf).astype(np.float32)
    logger.info(f"PCA explained variance: {svd.explained_variance_ratio_.sum():.3f}")

    # Load SecBERT embeddings
    logger.info("Loading SecBERT embeddings")
    sec_train_npz = np.load(
        "experiments/08_secbert_srbh/with_preprocessing/train_embeddings.npz"
    )
    secbert_train = sec_train_npz["embeddings"].astype(np.float32)

    sec_test_npz = np.load(
        "experiments/08_secbert_srbh/with_preprocessing/test_embeddings.npz",
        allow_pickle=True,
    )
    secbert_test = sec_test_npz["embeddings"].astype(np.float32)
    secbert_labels = sec_test_npz["labels"].astype(str)

    # Verify label alignment
    if not np.array_equal(np.array(test_labels), secbert_labels):
        raise ValueError(
            "Label mismatch between TF-IDF test data and SecBERT embeddings"
        )

    # Train Mahalanobis on SecBERT (based on Exp 08)
    logger.info("Training Mahalanobis on SecBERT embeddings")
    maha = MahalanobisDetector()
    maha.fit(secbert_train)
    maha.set_threshold(secbert_train, max_fpr=0.05)
    maha_train_scores = maha.scores(secbert_train)
    maha_test_scores = maha.scores(secbert_test)
    maha_predictions = maha.predict(secbert_test)
    maha_metrics = _evaluate(maha_predictions, secbert_labels)
    logger.info(
        f"Mahalanobis: recall={maha_metrics['recall']:.3f} fpr={maha_metrics['fpr']:.3f}"
    )

    # Train LOF on TF-IDF + PCA (based on Exp 15: k=100, contamination=auto)
    logger.info("Training LOF (k=100, contamination=auto)")
    lof = LOFDetector(n_neighbors=100, contamination="auto")
    lof.fit(train_embeddings)
    lof_threshold = lof.set_threshold(train_embeddings, max_fpr=0.05)

    lof_train_scores = lof.scores(train_embeddings)
    lof_test_scores = lof.scores(test_embeddings)
    lof_predictions = lof.predict(test_embeddings)
    lof_metrics = _evaluate(lof_predictions, test_labels)
    logger.info(f"LOF: recall={lof_metrics['recall']:.3f} fpr={lof_metrics['fpr']:.3f}")

    # Union ensemble
    logger.info("Testing union ensemble (LOF OR Mahalanobis)")
    union_predictions = np.logical_or(
        np.array(lof_predictions, dtype=bool),
        np.array(maha_predictions, dtype=bool),
    )
    union_metrics = _evaluate(union_predictions, test_labels)
    logger.info(
        f"Union: recall={union_metrics['recall']:.3f} fpr={union_metrics['fpr']:.3f}"
    )

    # Score-level fusion (based on Exp 15: weight=0.5)
    logger.info("Testing score-level fusion (weight=0.5)")
    lof_norm_train, lof_mean, lof_std = _z_normalize(lof_train_scores)
    mah_norm_train, mah_mean, mah_std = _z_normalize(maha_train_scores)

    lof_norm_test = _apply_z(lof_test_scores, lof_mean, lof_std)
    mah_norm_test = _apply_z(maha_test_scores, mah_mean, mah_std)

    fusion_weight = 0.5
    fused_train = fusion_weight * lof_norm_train + (1 - fusion_weight) * mah_norm_train
    fused_threshold = float(np.quantile(fused_train, 1.0 - 0.05))

    fused_scores = fusion_weight * lof_norm_test + (1 - fusion_weight) * mah_norm_test
    fused_predictions = fused_scores >= fused_threshold
    fused_metrics = _evaluate(fused_predictions, test_labels)
    logger.info(
        f"Fused: recall={fused_metrics['recall']:.3f} fpr={fused_metrics['fpr']:.3f}"
    )

    # Results summary
    results = {
        "configuration": {
            "tfidf": {"max_features": 5000, "ngram_range": [1, 3], "min_df": 2},
            "pca": {
                "n_components": 150,
                "explained_variance": float(svd.explained_variance_ratio_.sum()),
            },
            "lof": {
                "n_neighbors": 100,
                "contamination": "auto",
                "threshold": float(lof_threshold),
            },
            "mahalanobis": {"threshold": float(maha.threshold_)},
            "fusion": {"weight": fusion_weight, "threshold": fused_threshold},
        },
        "metrics": {
            "lof": lof_metrics,
            "mahalanobis": maha_metrics,
            "union": union_metrics,
            "fused": fused_metrics,
        },
    }

    # Save results
    output_dir = Path("experiments/18_lof_secbert_ensemble/quick_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_dir / 'results.json'}")
    logger.info("=" * 60)
    logger.info("FINAL RESULT:")
    logger.info(
        f"Fused ensemble: {fused_metrics['recall']:.1%} recall @ {fused_metrics['fpr']:.1%} FPR"
    )
    logger.info(f"Expected from Exp 15: ~64% recall @ 5% FPR")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
