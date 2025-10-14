#!/usr/bin/env python3
"""
Experiment 13: TF-IDF + MI + One-Class SVM (Paper Replication)

Replicates the paper's exact approach on CSIC/SR_BH datasets:
1. Paper's 5-step preprocessing
2. TF-IDF vectorization (5000 features)
3. MI-based token selection
4. BoW with selected tokens
5. One-Class SVM training
"""

import json
from pathlib import Path

import numpy as np
from loguru import logger
from preprocessing import paper_preprocess
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import OneClassSVM


def load_csic_data() -> tuple[list[str], list[str], list[str]]:
    """Load CSIC dataset."""
    logger.info("Loading CSIC dataset...")

    # Load from JSONL files
    train_normal = []
    test_data = []
    test_labels = []

    # Train normal
    with open("src/neuralshield/data/CSIC/train.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj["label"] == "valid":
                train_normal.append(obj["request"])

    # Test (both valid and attack)
    with open("src/neuralshield/data/CSIC/test.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            test_data.append(obj["request"])
            test_labels.append(obj["label"])

    logger.info(f"CSIC: {len(train_normal)} train normal, {len(test_data)} test")
    return train_normal, test_data, test_labels


def load_srbh_attacks(percentage: float = 0.15) -> list[str]:
    """
    Load SR_BH attack samples with balanced CAPEC category sampling.

    As per paper Section 3.1.2: "15% of each category was taken"

    Args:
        percentage: Percentage of each CAPEC category to sample (default: 0.15)

    Returns:
        List of attack request strings
    """
    import csv
    import random
    from collections import defaultdict

    logger.info(
        f"Loading SR_BH attacks ({percentage * 100}% from each CAPEC category)..."
    )

    # CAPEC categories (columns 26-38 in CSV, 0-indexed: 25-37)
    capec_columns = [
        "272 - Protocol Manipulation",
        "242 - Code Injection",
        "88 - OS Command Injection",
        "126 - Path Traversal",
        "66 - SQL Injection",
        "16 - Dictionary-based Password Attack",
        "310 - Scanning for Vulnerable Software",
        "153 - Input Data Manipulation",
        "248 - Command Injection",
        "274 - HTTP Verb Tampering",
        "194 - Fake the Source of Data",
        "34 - HTTP Response Splitting",
        "33 - HTTP Request Smuggling",
    ]

    # Group attacks by category
    attacks_by_category: dict[str, list[str]] = defaultdict(list)

    csv_path = "src/neuralshield/data/SR_BH_2020/data_capec_multilabel.csv"

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Check if it's normal (skip normal requests)
            if row["000 - Normal"] == "1":
                continue

            # Build request string (method + URL + body if present)
            method = row["request_http_method"]
            url = row["request_http_request"]
            body = row.get("request_body", "")

            if body:
                request = f"{method} {url} {body}"
            else:
                request = f"{method} {url}"

            # Categorize by CAPEC (a request can belong to multiple categories)
            for capec in capec_columns:
                if row[capec] == "1":
                    attacks_by_category[capec].append(request)

    # Sample 15% from each category
    sampled_attacks: list[str] = []
    random.seed(42)  # For reproducibility

    for capec, attacks in attacks_by_category.items():
        n_sample = max(1, int(len(attacks) * percentage))
        sampled = random.sample(attacks, n_sample)
        sampled_attacks.extend(sampled)
        logger.info(
            f"  {capec}: sampled {n_sample} from {len(attacks)} ({percentage * 100:.0f}%)"
        )

    # Remove duplicates (requests can be in multiple categories)
    sampled_attacks = list(set(sampled_attacks))

    logger.info(
        f"SR_BH: {len(sampled_attacks)} unique attacks loaded (from {len(capec_columns)} CAPEC categories)"
    )
    return sampled_attacks


def build_dictionary(
    normal_requests: list[str],
    attack_requests: list[str],
    max_features: int = 5000,
) -> list[str]:
    """
    Build dictionary using CountVectorizer (Algorithm 1 from paper).

    This creates a fixed vocabulary of tokens from normal + attack data.

    Returns:
        List of tokens (dictionary)
    """
    logger.info(
        f"Building dictionary with CountVectorizer (max {max_features} features)..."
    )

    # Algorithm 1: Use CountVectorizer to create dictionary
    count_vectorizer = CountVectorizer(
        max_features=max_features,
        token_pattern=r"\S+",  # Space-separated tokens
        lowercase=False,  # Already lowercased in preprocessing
    )

    # Combine normal + attacks for dictionary building
    all_requests = normal_requests + attack_requests

    logger.info(
        f"Dictionary samples: {len(all_requests)} ({len(normal_requests)} normal + {len(attack_requests)} attacks)"
    )

    # Fit to get vocabulary
    count_vectorizer.fit(all_requests)

    # Extract dictionary (vocabulary)
    dictionary = count_vectorizer.get_feature_names_out().tolist()

    logger.info(f"Dictionary built: {len(dictionary)} unique tokens")

    return dictionary


def compute_mi_on_tfidf(
    normal_requests: list[str],
    attack_requests: list[str],
    dictionary: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Compute MI scores on TF-IDF features (Algorithm 2 from paper).

    Uses pre-built dictionary as fixed vocabulary.

    Returns:
        mi_scores: MI value for each token
        feature_names: Token names corresponding to scores
    """
    logger.info(f"Computing TF-IDF with fixed vocabulary ({len(dictionary)} tokens)...")

    # Algorithm 2, Line 4: TfidfVectorizer with fixed vocabulary
    vectorizer = TfidfVectorizer(
        vocabulary=dictionary,  # KEY: Use pre-built dictionary!
        token_pattern=r"\S+",
        lowercase=False,
    )

    # Combine normal + attacks for MI computation
    all_requests = normal_requests + attack_requests
    labels = np.array([0] * len(normal_requests) + [1] * len(attack_requests))

    logger.info(
        f"MI computation samples: {len(all_requests)} ({len(normal_requests)} normal + {len(attack_requests)} attacks)"
    )

    # Fit TF-IDF with fixed vocabulary
    X = vectorizer.fit_transform(all_requests)

    logger.info(f"TF-IDF shape: {X.shape}")
    logger.info(f"Sparsity: {(1 - X.nnz / (X.shape[0] * X.shape[1])):.2%}")

    # Compute MI for each feature
    logger.info("Computing mutual information...")
    mi_scores = mutual_info_classif(X, labels, random_state=42)

    feature_names = vectorizer.get_feature_names_out().tolist()

    logger.info(
        f"MI scores computed. Range: [{mi_scores.min():.6f}, {mi_scores.max():.6f}]"
    )

    # Show top 10 tokens
    top_10_idx = np.argsort(mi_scores)[-10:][::-1]
    logger.info("Top 10 tokens by MI:")
    for i, idx in enumerate(top_10_idx, 1):
        logger.info(f"  {i:2d}. '{feature_names[idx]}': {mi_scores[idx]:.6f}")

    return mi_scores, feature_names


def test_with_k_features(
    k: int,
    mi_scores: np.ndarray,
    feature_names: list[str],
    train_requests: list[str],
    test_requests: list[str],
    test_labels: list[str],
    nu: float = 0.05,
    gamma: float = 0.5,
) -> dict:
    """Test OCSVM with top-K MI-selected features (Algorithm 3 from paper)."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Testing k={k} tokens (nu={nu}, gamma={gamma})")
    logger.info(f"{'=' * 60}")

    # Select top K features
    top_k_indices = np.argsort(mi_scores)[-k:]
    selected_tokens = [feature_names[i] for i in top_k_indices]

    logger.info(f"Selected {k} tokens")
    if k <= 20:
        logger.info(f"Tokens: {selected_tokens}")

    # Create BoW with selected tokens only
    bow_vectorizer = CountVectorizer(
        vocabulary=selected_tokens,
        token_pattern=r"\S+",
        lowercase=False,
    )

    # Vectorize
    X_train = bow_vectorizer.fit_transform(train_requests)
    X_test = bow_vectorizer.transform(test_requests)

    logger.info(f"BoW shape: train={X_train.shape}, test={X_test.shape}")

    # Train One-Class SVM (paper's hyperparameters from grid search)
    logger.info("Training One-Class SVM...")
    ocsvm = OneClassSVM(nu=nu, gamma=gamma, kernel="rbf")
    ocsvm.fit(X_train.toarray())

    logger.info("Predicting...")
    # Predict (-1 = anomaly/attack, +1 = normal)
    predictions = ocsvm.predict(X_test.toarray())

    # Convert to boolean
    is_attack_pred = predictions == -1

    # Ground truth
    is_attack_true = np.array([label == "attack" for label in test_labels])
    is_normal_true = ~is_attack_true

    # Confusion matrix
    tp = np.sum(is_attack_pred & is_attack_true)
    fp = np.sum(is_attack_pred & is_normal_true)
    tn = np.sum(~is_attack_pred & is_normal_true)
    fn = np.sum(~is_attack_pred & is_attack_true)

    # Metrics
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    accuracy = (tp + tn) / len(test_labels) if len(test_labels) > 0 else 0

    logger.info(f"\nResults:")
    logger.info(f"  Recall:    {recall:.4f} ({tp}/{tp + fn})")
    logger.info(f"  Precision: {precision:.4f} ({tp}/{tp + fp})")
    logger.info(f"  F1-Score:  {f1:.4f}")
    logger.info(f"  FPR:       {fpr:.4f} ({fp}/{fp + tn})")
    logger.info(f"  Accuracy:  {accuracy:.4f}")

    return {
        "k": k,
        "nu": nu,
        "gamma": gamma,
        "recall": float(recall),
        "precision": float(precision),
        "f1_score": float(f1),
        "fpr": float(fpr),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "selected_tokens": selected_tokens if k <= 100 else None,
    }


def main():
    """Run TF-IDF + MI + OCSVM experiment."""
    logger.info("=" * 80)
    logger.info("Experiment 13: TF-IDF + MI + One-Class SVM (Paper Replication)")
    logger.info("=" * 80)

    results_dir = Path("experiments/13_tfidf_mi_replication/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # [1/5] Load data
    logger.info("\n[1/5] Loading data...")
    csic_train, csic_test, csic_test_labels = load_csic_data()
    srbh_attacks = load_srbh_attacks(percentage=0.15)

    # [2/5] Preprocess with paper's method
    logger.info("\n[2/5] Preprocessing (paper's 5 steps)...")
    logger.info("Processing CSIC train...")
    csic_train_proc = [paper_preprocess(r) for r in csic_train]

    logger.info("Processing CSIC test...")
    csic_test_proc = [paper_preprocess(r) for r in csic_test]

    logger.info("Processing SR_BH attacks...")
    srbh_attacks_proc = [paper_preprocess(r) for r in srbh_attacks]

    logger.info(
        f"Preprocessed {len(csic_train_proc) + len(csic_test_proc) + len(srbh_attacks_proc)} requests"
    )

    # [3/6] Build Dictionary (Algorithm 1)
    logger.info("\n[3/6] Building dictionary (Algorithm 1)...")
    dictionary = build_dictionary(
        normal_requests=csic_train_proc,
        attack_requests=srbh_attacks_proc,
        max_features=5000,
    )

    # [4/6] Compute MI on TF-IDF (Algorithm 2)
    logger.info("\n[4/6] Computing MI on TF-IDF features (Algorithm 2)...")
    mi_scores, feature_names = compute_mi_on_tfidf(
        normal_requests=csic_train_proc,
        attack_requests=srbh_attacks_proc,
        dictionary=dictionary,
    )

    # Save MI scores and feature names
    np.save(results_dir / "mi_scores_tfidf.npy", mi_scores)
    with open(results_dir / "feature_names.txt", "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")

    # [5/6] Test different K values
    logger.info("\n[5/6] Testing different K values...")

    # Paper tests: 50, 64 (expert), 100, 150, 200
    k_values = [50, 64, 100, 150, 200]

    results = []
    for k in k_values:
        metrics = test_with_k_features(
            k=k,
            mi_scores=mi_scores,
            feature_names=feature_names,
            train_requests=csic_train_proc,
            test_requests=csic_test_proc,
            test_labels=csic_test_labels,
            nu=0.05,
            gamma=0.5,
        )
        results.append(metrics)

        # Save selected tokens for small k
        if k <= 100 and metrics["selected_tokens"]:
            with open(results_dir / f"selected_tokens_k{k}.txt", "w") as f:
                for token in metrics["selected_tokens"]:
                    f.write(f"{token}\n")

    # [6/6] Save and summarize
    logger.info("\n[6/6] Saving results...")

    with open(results_dir / "metrics_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_dir / 'metrics_comparison.json'}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    logger.info("\nAll results:")
    for r in results:
        logger.info(
            f"  k={r['k']:3d}: Recall={r['recall']:.4f}, "
            f"Precision={r['precision']:.4f}, FPR={r['fpr']:.4f}, F1={r['f1_score']:.4f}"
        )

    best = max(results, key=lambda r: r["recall"])
    logger.info(f"\nBest configuration:")
    logger.info(f"  k={best['k']}")
    logger.info(f"  Recall:    {best['recall']:.4f}")
    logger.info(f"  Precision: {best['precision']:.4f}")
    logger.info(f"  FPR:       {best['fpr']:.4f}")
    logger.info(f"  F1-Score:  {best['f1_score']:.4f}")

    # Compare to SecBERT baseline
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON TO SECBERT (Experiment 03)")
    logger.info("=" * 80)

    secbert_recall = 0.4926
    secbert_fpr = 0.05

    logger.info(
        f"SecBERT + Mahalanobis:  {secbert_recall:.4f} recall @ {secbert_fpr:.2%} FPR"
    )
    logger.info(
        f"TF-IDF + MI (k={best['k']}):      {best['recall']:.4f} recall @ {best['fpr']:.2%} FPR"
    )
    logger.info(f"Difference:             {(best['recall'] - secbert_recall):+.4f}")

    if best["recall"] < secbert_recall:
        improvement = (1 - best["recall"] / secbert_recall) * 100
        logger.warning(f"❌ TF-IDF + MI is {improvement:.1f}% WORSE than SecBERT")
    else:
        improvement = (best["recall"] / secbert_recall - 1) * 100
        logger.success(f"✅ TF-IDF + MI is {improvement:.1f}% BETTER than SecBERT!")


if __name__ == "__main__":
    main()
