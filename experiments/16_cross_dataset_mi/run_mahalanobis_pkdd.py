#!/usr/bin/env python3
"""Evaluate SR-BH → PKDD cross-dataset MI using Mahalanobis distance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.covariance import EmpiricalCovariance

from test_cross_dataset_mi import (
    load_pkdd_data,
    load_srbh_data,
    paper_preprocess,
    build_dictionary,
    compute_mi_scores,
)


def main() -> None:
    results_dir = Path("experiments/16_cross_dataset_mi/results_mahalanobis")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SR-BH and PKDD datasets…")
    (
        srbh_all_normal,
        srbh_all_attacks,
        _,
        _,
        _,
    ) = load_srbh_data()

    (
        pkdd_all_normal,
        pkdd_all_attacks,
        pkdd_train_normal,
        pkdd_test_requests,
        pkdd_test_labels,
    ) = load_pkdd_data()

    logger.info("Preprocessing datasets…")
    srbh_normal_proc = [paper_preprocess(r) for r in srbh_all_normal]
    srbh_attacks_proc = [paper_preprocess(r) for r in srbh_all_attacks]
    pkdd_train_proc = [paper_preprocess(r) for r in pkdd_train_normal]
    pkdd_test_proc = [paper_preprocess(r) for r in pkdd_test_requests]

    logger.info("Building dictionary (max 5000 tokens)…")
    dictionary = build_dictionary(srbh_normal_proc, srbh_attacks_proc, max_features=5000)

    logger.info("Computing mutual information scores…")
    mi_scores, feature_names = compute_mi_scores(
        srbh_normal_proc,
        srbh_attacks_proc,
        dictionary,
    )

    np.save(results_dir / "mi_scores.npy", mi_scores)
    with open(results_dir / "feature_names.txt", "w") as f:
        for token in feature_names:
            f.write(f"{token}\n")

    k_values = [50, 100, 150, 200]

    results: list[dict[str, float | int | None]] = []

    for k in k_values:
        logger.info("=" * 60)
        logger.info("Testing Mahalanobis with k=%d", k)
        top_indices = np.argsort(mi_scores)[-k:]
        selected_tokens = [feature_names[i] for i in top_indices]

        # Vectorise using Bag-of-Words
        from sklearn.feature_extraction.text import CountVectorizer

        bow = CountVectorizer(
            vocabulary=selected_tokens,
            token_pattern=r"\S+",
            lowercase=False,
        )
        X_train = bow.fit_transform(pkdd_train_proc).toarray().astype(np.float32)
        X_test = bow.transform(pkdd_test_proc).toarray().astype(np.float32)

        logger.info("Fitting Empirical Covariance on train set shape=%s", X_train.shape)

        model = EmpiricalCovariance()
        model.fit(X_train)

        train_scores = model.mahalanobis(X_train)
        test_scores = model.mahalanobis(X_test)

        threshold = float(np.quantile(train_scores, 0.95))

        preds = test_scores > threshold
        is_attack_true = np.array([label == "attack" for label in pkdd_test_labels])
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
        accuracy = (tp + tn) / len(pkdd_test_labels) if pkdd_test_labels else 0.0

        logger.info(
            "k=%d results: recall=%.4f precision=%.4f fpr=%.4f accuracy=%.4f",
            k,
            recall,
            precision,
            fpr,
            accuracy,
        )

        results.append(
            {
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
            }
        )

        if k <= 100:
            with open(results_dir / f"selected_tokens_k{k}.txt", "w") as f:
                for token in selected_tokens:
                    f.write(f"{token}\n")

        np.save(results_dir / f"mahalanobis_scores_k{k}.npy", test_scores)

    with open(results_dir / "metrics_mahalanobis.json", "w") as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda entry: entry["recall"])
    logger.info(
        "Best Mahalanobis run: k=%d recall=%.4f fpr=%.4f precision=%.4f",
        best["k"],
        best["recall"],
        best["fpr"],
        best["precision"],
    )


if __name__ == "__main__":
    main()
