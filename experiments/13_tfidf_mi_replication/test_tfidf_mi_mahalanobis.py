#!/usr/bin/env python3
"""
Experiment 13 (variant): TF-IDF + MI + Mahalanobis Distance

Matches the paper's preprocessing and feature selection pipeline, but swaps the
One-Class SVM for a Mahalanobis detector to speed up evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.covariance import EmpiricalCovariance
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing import paper_preprocess
from test_tfidf_mi_ocsvm import (
    build_dictionary,
    compute_mi_on_tfidf,
    load_csic_data,
    load_srbh_attacks,
)


def evaluate_with_mahalanobis(
    k: int,
    mi_scores: np.ndarray,
    feature_names: list[str],
    train_requests: list[str],
    test_requests: list[str],
    test_labels: list[str],
) -> dict[str, float | int | None]:
    """Evaluate top-K MI features using Mahalanobis distance."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Mahalanobis with k=%d", k)
    logger.info("=" * 60)

    # Select top-K tokens
    top_indices = np.argsort(mi_scores)[-k:]
    selected_tokens = [feature_names[i] for i in top_indices]

    if k <= 20:
        logger.info("Tokens: %s", selected_tokens)

    bow = CountVectorizer(
        vocabulary=selected_tokens,
        token_pattern=r"\S+",
        lowercase=False,
    )

    X_train = bow.fit_transform(train_requests).toarray().astype(np.float32)
    X_test = bow.transform(test_requests).toarray().astype(np.float32)

    logger.info("Train shape=%s, test shape=%s", X_train.shape, X_test.shape)

    model = EmpiricalCovariance()
    model.fit(X_train)

    train_scores = model.mahalanobis(X_train)
    test_scores = model.mahalanobis(X_test)

    threshold = float(np.quantile(train_scores, 0.95))

    preds = test_scores > threshold
    is_attack_true = np.array([label == "attack" for label in test_labels])
    is_normal_true = ~is_attack_true

    tp = int(np.sum(preds & is_attack_true))
    fp = int(np.sum(preds & is_normal_true))
    tn = int(np.sum(~preds & is_normal_true))
    fn = int(np.sum(~preds & is_attack_true))

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / len(test_labels) if test_labels else 0.0

    logger.info("Recall=%.4f Precision=%.4f FPR=%.4f", recall, precision, fpr)

    return {
        "k": k,
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "fpr": fpr,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "selected_tokens": selected_tokens if k <= 100 else None,
    }


def main() -> None:
    logger.info("=" * 80)
    logger.info("Experiment 13: TF-IDF + MI + Mahalanobis Distance")
    logger.info("=" * 80)

    results_dir = Path("experiments/13_tfidf_mi_replication/results_mahalanobis")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n[1/5] Loading datasets…")
    csic_train, csic_test, csic_test_labels = load_csic_data()
    srbh_attacks = load_srbh_attacks(percentage=0.15)

    logger.info("\n[2/5] Preprocessing with paper pipeline…")
    csic_train_proc = [paper_preprocess(r) for r in csic_train]
    csic_test_proc = [paper_preprocess(r) for r in csic_test]
    srbh_attacks_proc = [paper_preprocess(r) for r in srbh_attacks]

    logger.info("\n[3/5] Building dictionary (Algorithm 1)…")
    dictionary = build_dictionary(
        normal_requests=csic_train_proc,
        attack_requests=srbh_attacks_proc,
        max_features=5000,
    )

    logger.info("\n[4/5] Computing MI scores (Algorithm 2)…")
    mi_scores, feature_names = compute_mi_on_tfidf(
        normal_requests=csic_train_proc,
        attack_requests=srbh_attacks_proc,
        dictionary=dictionary,
    )

    np.save(results_dir / "mi_scores.npy", mi_scores)
    with open(results_dir / "feature_names.txt", "w") as f:
        for token in feature_names:
            f.write(f"{token}\n")

    logger.info("\n[5/5] Evaluating Mahalanobis across K values…")
    k_values = [50, 64, 100, 150, 200]
    results: list[dict[str, float | int | None]] = []

    for k in k_values:
        metrics = evaluate_with_mahalanobis(
            k=k,
            mi_scores=mi_scores,
            feature_names=feature_names,
            train_requests=csic_train_proc,
            test_requests=csic_test_proc,
            test_labels=csic_test_labels,
        )
        results.append(metrics)

        if k <= 100 and metrics["selected_tokens"]:
            with open(results_dir / f"selected_tokens_k{k}.txt", "w") as f:
                for token in metrics["selected_tokens"]:
                    f.write(f"{token}\n")

    with open(results_dir / "metrics_mahalanobis.json", "w") as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda entry: entry["recall"])
    logger.info("Best run: k=%d recall=%.4f fpr=%.4f", best["k"], best["recall"], best["fpr"])


if __name__ == "__main__":
    main()
