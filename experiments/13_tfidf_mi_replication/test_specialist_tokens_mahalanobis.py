#!/usr/bin/env python3
"""Evaluate paper's specialist token list with Mahalanobis detector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.covariance import EmpiricalCovariance
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing import paper_preprocess
from test_tfidf_mi_ocsvm import load_csic_data, load_srbh_attacks


SPECIALIST_TOKENS = [
    "<",
    "../",
    "alert",
    "exec",
    "password",
    "<>",
    "alter",
    "from",
    "path/child",
    "<!--",
    '"',
    "and",
    "href",
    "script",
    "=",
    "(",
    "bash_history",
    "#include",
    "select",
    ">",
    ")",
    "between",
    "insert",
    "shell",
    "|",
    "$",
    "/c",
    "into",
    "table",
    "||",
    "*",
    "cmd",
    "javascript:",
    "union",
    "-",
    "*/",
    "cn=",
    "mail=",
    "upper",
    "->",
    "&",
    "commit",
    "objectclass",
    "url=",
    ";",
    "+",
    "count",
    "onmouseover",
    "user-agent:",
    ":",
    "%00",
    "-craw",
    "or",
    "where",
    "/",
    "%0a",
    "document.cookie",
    "order",
    "winnt",
    "/*",
    "accept:",
    "etc/passwd",
    "passwd",
]


def main() -> None:
    logger.info("Running specialist-token Mahalanobis evaluation…")

    results_dir = Path(
        "experiments/13_tfidf_mi_replication/results_specialist_mahalanobis"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    csic_train, csic_test, csic_test_labels = load_csic_data()
    srbh_attacks = load_srbh_attacks(percentage=0.15)

    csic_train_proc = [paper_preprocess(r) for r in csic_train]
    csic_test_proc = [paper_preprocess(r) for r in csic_test]
    # include attacks so vectorizer sees tokens even if absent in train
    srbh_attacks_proc = [paper_preprocess(r) for r in srbh_attacks]

    tokens = [tok.lower() for tok in SPECIALIST_TOKENS]
    bow = CountVectorizer(
        vocabulary=tokens,
        token_pattern=r"\S+",
        lowercase=False,
    )

    X_train = bow.fit_transform(csic_train_proc).toarray().astype(np.float32)
    if not X_train.any():
        logger.warning("Train matrix is all zeros — specialist tokens absent")
    X_test = bow.transform(csic_test_proc).toarray().astype(np.float32)

    model = EmpiricalCovariance()
    model.fit(X_train)

    train_scores = model.mahalanobis(X_train)
    test_scores = model.mahalanobis(X_test)

    threshold = float(np.quantile(train_scores, 0.95))

    preds = test_scores > threshold
    is_attack_true = np.array([label == "attack" for label in csic_test_labels])
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
    accuracy = (tp + tn) / len(csic_test_labels)

    metrics = {
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "fpr": fpr,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "threshold": threshold,
    }

    logger.info("Metrics: %s", metrics)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
