#!/usr/bin/env python3
"""Cross-dataset TF-IDF + PCA evaluation (Mahalanobis & LOF)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LocalOutlierFactor

import sys
sys.path.insert(0, "experiments/13_tfidf_mi_replication")
from preprocessing import paper_preprocess

DATASETS = {
    "CSIC": {
        "train": "src/neuralshield/data/CSIC/train.jsonl",
        "test": "src/neuralshield/data/CSIC/test.jsonl",
    },
    "SRBH": {
        "train": "src/neuralshield/data/SR_BH_2020/train.jsonl",
        "test": "src/neuralshield/data/SR_BH_2020/test.jsonl",
    },
    "PKDD": {
        "train": "src/neuralshield/data/PKDD/train.jsonl",
        "test": "src/neuralshield/data/PKDD/test.jsonl",
    },
}


def load_dataset(name: str) -> tuple[list[str], list[str], list[str]]:
    cfg = DATASETS[name]
    train_normals: list[str] = []
    test_requests: list[str] = []
    test_labels: list[str] = []

    with open(cfg["train"], encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["label"] == "valid":
                train_normals.append(obj["request"])

    with open(cfg["test"], encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            test_requests.append(obj["request"])
            test_labels.append(obj["label"])

    logger.info(
        "%s: train_normals=%d test=%d",
        name,
        len(train_normals),
        len(test_requests),
    )
    return train_normals, test_requests, test_labels


def build_features(source: str, target: str, max_features: int = 5000, n_components: int = 150):
    train_normals_src, test_src, _ = load_dataset(source)
    _, test_tgt, labels_tgt = load_dataset(target)

    logger.info("Preprocessing source (%s) normals", source)
    train_src_proc = [paper_preprocess(r) for r in train_normals_src]

    logger.info("Preprocessing target (%s) test", target)
    test_tgt_proc = [paper_preprocess(r) for r in test_tgt]

    logger.info("Preprocessing source (%s) test", source)
    test_src_proc = [paper_preprocess(r) for r in test_src]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        token_pattern=r"\S+",
        lowercase=False,
    )
    tfidf_train = vectorizer.fit_transform(train_src_proc)
    tfidf_test_src = vectorizer.transform(test_src_proc)
    tfidf_test_tgt = vectorizer.transform(test_tgt_proc)

    pca = PCA(n_components=n_components, random_state=42)
    train_pca = pca.fit_transform(tfidf_train.toarray())
    test_src_pca = pca.transform(tfidf_test_src.toarray())
    test_tgt_pca = pca.transform(tfidf_test_tgt.toarray())

    logger.info(
        "PCA %.1f%% variance retained",
        pca.explained_variance_ratio_.sum() * 100,
    )

    return train_pca, test_src_pca, test_tgt_pca, labels_tgt


def evaluate_mahalanobis(train_vectors: np.ndarray, test_vectors: np.ndarray, labels: list[str], max_fpr: float = 0.05):
    model = EmpiricalCovariance()
    model.fit(train_vectors)
    train_scores = model.mahalanobis(train_vectors)
    threshold = float(np.quantile(train_scores, 1 - max_fpr))

    test_scores = model.mahalanobis(test_vectors)
    preds = test_scores > threshold

    y_true = np.array([1 if label == "attack" else 0 for label in labels])
    y_pred = preds.astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / len(labels)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "fpr": fpr,
        "accuracy": accuracy,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "threshold": threshold,
    }


def evaluate_lof(train_vectors: np.ndarray, test_vectors: np.ndarray, labels: list[str], n_neighbors: int = 100, max_fpr: float = 0.05):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(train_vectors)

    scores = -lof.decision_function(test_vectors)
    threshold = float(np.quantile(-lof.decision_function(train_vectors), 1 - max_fpr))
    preds = scores > threshold

    y_true = np.array([1 if label == "attack" else 0 for label in labels])
    y_pred = preds.astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / len(labels)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "fpr": fpr,
        "accuracy": accuracy,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "threshold": threshold,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-dataset TF-IDF + PCA evaluation")
    parser.add_argument("source", choices=DATASETS.keys())
    parser.add_argument("target", choices=DATASETS.keys())
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--n-components", type=int, default=150)
    parser.add_argument("--output", type=Path, default=Path("experiments/24_results"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    train_vecs, _, target_vecs, target_labels = build_features(
        source=args.source,
        target=args.target,
        max_features=args.max_features,
        n_components=args.n_components,
    )

    maha_metrics = evaluate_mahalanobis(train_vecs, target_vecs, target_labels)
    logger.info("Mahalanobis metrics: %s", maha_metrics)

    lof_metrics = evaluate_lof(train_vecs, target_vecs, target_labels)
    logger.info("LOF metrics: %s", lof_metrics)

    result_path = args.output / f"{args.source}_to_{args.target}.json"
    with open(result_path, "w") as f:
        json.dump({"mahalanobis": maha_metrics, "lof": lof_metrics}, f, indent=2)

    logger.info("Saved results to %s", result_path)
