#!/usr/bin/env python3
"""
Experiment 16: Cross-Dataset MI Feature Selection

Tests paper's core claim: Can features from one dataset detect attacks on another?

Run 1: CSIC (feature selection) → SR-BH (train/test)
Run 2: SR-BH (feature selection) → CSIC (train/test)
"""

import json
from pathlib import Path

import numpy as np
from loguru import logger
from preprocessing import paper_preprocess
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import OneClassSVM


def load_csic_data() -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """
    Load CSIC dataset.

    Returns:
        all_normal: All normal requests (for MI computation)
        all_attacks: All attack requests (for MI computation)
        train_normal: Normal training requests (for OCSVM)
        test_requests: All test requests
        test_labels: Labels for test requests ("valid" or "attack")
    """
    logger.info("Loading CSIC dataset...")

    train_normal = []
    all_normal = []
    all_attacks = []
    test_requests = []
    test_labels = []

    # Train data (only normal)
    with open("src/neuralshield/data/CSIC/train.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj["label"] == "valid":
                train_normal.append(obj["request"])
                all_normal.append(obj["request"])

    # Test data (normal + attacks)
    with open("src/neuralshield/data/CSIC/test.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            test_requests.append(obj["request"])
            test_labels.append(obj["label"])

            if obj["label"] == "valid":
                all_normal.append(obj["request"])
            else:
                all_attacks.append(obj["request"])

    logger.info(
        f"CSIC: {len(all_normal)} total normal, {len(all_attacks)} total attacks, "
        f"{len(train_normal)} train normal, {len(test_requests)} test"
    )
    return all_normal, all_attacks, train_normal, test_requests, test_labels


def load_srbh_data() -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """
    Load SR-BH dataset from train/test split.

    Returns:
        all_normal: All normal requests (for MI computation)
        all_attacks: All attack requests (for MI computation)
        train_normal: Normal training requests (for OCSVM)
        test_requests: Test requests
        test_labels: Labels for test requests ("valid" or "attack")
    """
    logger.info("Loading SR-BH dataset...")

    train_normal = []
    all_normal = []
    all_attacks = []
    test_requests = []
    test_labels = []

    # Train data (only normal)
    with open("src/neuralshield/data/SR_BH_2020/train.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj["label"] == "valid":
                train_normal.append(obj["request"])
                all_normal.append(obj["request"])

    # Test data (normal + attacks)
    with open("src/neuralshield/data/SR_BH_2020/test.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            test_requests.append(obj["request"])
            test_labels.append(obj["label"])

            if obj["label"] == "valid":
                all_normal.append(obj["request"])
            else:
                all_attacks.append(obj["request"])

    logger.info(
        f"SR-BH: {len(all_normal)} total normal, {len(all_attacks)} total attacks, "
        f"{len(train_normal)} train normal, {len(test_requests)} test"
    )
    return all_normal, all_attacks, train_normal, test_requests, test_labels


def build_dictionary(
    normal_requests: list[str],
    attack_requests: list[str],
    max_features: int = 5000,
) -> list[str]:
    """Build dictionary using CountVectorizer (Algorithm 1)."""
    logger.info(f"Building dictionary (max {max_features} features)...")

    count_vectorizer = CountVectorizer(
        max_features=max_features,
        token_pattern=r"\S+",  # Space-separated tokens
        lowercase=False,  # Already lowercased
    )

    all_requests = normal_requests + attack_requests
    count_vectorizer.fit(all_requests)
    dictionary = count_vectorizer.get_feature_names_out().tolist()

    logger.info(f"Dictionary: {len(dictionary)} tokens")
    return dictionary


def compute_mi_scores(
    normal_requests: list[str],
    attack_requests: list[str],
    dictionary: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Compute MI scores on TF-IDF features (Algorithm 2)."""
    logger.info(f"Computing MI on {len(dictionary)} tokens...")

    vectorizer = TfidfVectorizer(
        vocabulary=dictionary,
        token_pattern=r"\S+",
        lowercase=False,
    )

    all_requests = normal_requests + attack_requests
    labels = np.array([0] * len(normal_requests) + [1] * len(attack_requests))

    X = vectorizer.fit_transform(all_requests)
    mi_scores = mutual_info_classif(X, labels, random_state=42)
    feature_names = vectorizer.get_feature_names_out().tolist()

    logger.info(f"MI scores: range [{mi_scores.min():.6f}, {mi_scores.max():.6f}]")

    # Show top 10
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
    """Test OCSVM with top-K MI-selected features."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Testing k={k} (nu={nu}, gamma={gamma})")
    logger.info(f"{'=' * 60}")

    # Select top K
    top_k_indices = np.argsort(mi_scores)[-k:]
    selected_tokens = [feature_names[i] for i in top_k_indices]

    logger.info(f"Selected {k} tokens")

    # Create BoW
    bow_vectorizer = CountVectorizer(
        vocabulary=selected_tokens,
        token_pattern=r"\S+",
        lowercase=False,
    )

    X_train = bow_vectorizer.fit_transform(train_requests)
    X_test = bow_vectorizer.transform(test_requests)

    logger.info(f"BoW shape: train={X_train.shape}, test={X_test.shape}")

    # Train OCSVM
    logger.info("Training One-Class SVM...")
    ocsvm = OneClassSVM(nu=nu, gamma=gamma, kernel="rbf")
    ocsvm.fit(X_train.toarray())

    # Predict
    logger.info("Predicting...")
    predictions = ocsvm.predict(X_test.toarray())
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

    logger.info("\nResults:")
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


def run_experiment(
    name: str,
    mi_normal: list[str],
    mi_attacks: list[str],
    train_normal: list[str],
    test_requests: list[str],
    test_labels: list[str],
    output_dir: Path,
    strip_headers_for_mi: bool = False,
) -> list[dict]:
    """Run single experiment: feature selection + training + testing."""
    logger.info("\n" + "=" * 80)
    logger.info(f"EXPERIMENT: {name}")
    logger.info("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # [1/4] Preprocess
    logger.info("\n[1/4] Preprocessing...")
    logger.info(f"  MI normal: {len(mi_normal)} requests")
    mi_normal_proc = [
        paper_preprocess(r, strip_http_headers=strip_headers_for_mi) for r in mi_normal
    ]

    logger.info(f"  MI attacks: {len(mi_attacks)} requests")
    mi_attacks_proc = [
        paper_preprocess(r, strip_http_headers=strip_headers_for_mi) for r in mi_attacks
    ]

    logger.info(f"  Train normal: {len(train_normal)} requests")
    train_normal_proc = [paper_preprocess(r) for r in train_normal]

    logger.info(f"  Test: {len(test_requests)} requests")
    test_requests_proc = [paper_preprocess(r) for r in test_requests]

    # [2/4] Build dictionary
    logger.info("\n[2/4] Building dictionary...")
    dictionary = build_dictionary(mi_normal_proc, mi_attacks_proc, max_features=5000)

    # [3/4] Compute MI
    logger.info("\n[3/4] Computing MI scores...")
    mi_scores, feature_names = compute_mi_scores(
        mi_normal_proc, mi_attacks_proc, dictionary
    )

    # Save MI results
    np.save(output_dir / "mi_scores.npy", mi_scores)
    with open(output_dir / "feature_names.txt", "w") as f:
        for fname in feature_names:
            f.write(f"{fname}\n")

    # [4/4] Test different K values
    logger.info("\n[4/4] Testing K values...")
    k_values = [50, 100, 150, 200]
    results = []

    for k in k_values:
        metrics = test_with_k_features(
            k=k,
            mi_scores=mi_scores,
            feature_names=feature_names,
            train_requests=train_normal_proc,
            test_requests=test_requests_proc,
            test_labels=test_labels,
            nu=0.05,
            gamma=0.5,
        )
        results.append(metrics)

        # Save selected tokens for small k
        if k <= 100 and metrics["selected_tokens"]:
            with open(output_dir / f"selected_tokens_k{k}.txt", "w") as f:
                for token in metrics["selected_tokens"]:
                    f.write(f"{token}\n")

    # Save metrics
    with open(output_dir / "metrics_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")

    return results


def main():
    """Run both cross-dataset experiments."""
    logger.info("=" * 80)
    logger.info("Experiment 16: Cross-Dataset MI Feature Selection")
    logger.info("=" * 80)

    results_dir = Path("experiments/16_cross_dataset_mi/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("\n[SETUP] Loading datasets...")
    (
        csic_all_normal,
        csic_all_attacks,
        csic_train_normal,
        csic_test_requests,
        csic_test_labels,
    ) = load_csic_data()

    (
        srbh_all_normal,
        srbh_all_attacks,
        srbh_train_normal,
        srbh_test_requests,
        srbh_test_labels,
    ) = load_srbh_data()

    # ========== RUN 1: CSIC → SR-BH ==========
    logger.info("\n\n")
    logger.info("▓" * 80)
    logger.info("RUN 1: CSIC (feature selection) → SR-BH (train/test)")
    logger.info("Strip CSIC headers for MI (synthetic/uniform headers)")
    logger.info("▓" * 80)

    run1_results = run_experiment(
        name="CSIC → SR-BH",
        mi_normal=csic_all_normal,  # Use all CSIC normal for MI
        mi_attacks=csic_all_attacks,  # Use all CSIC attacks for MI
        train_normal=srbh_train_normal,  # Train on SR-BH normal
        test_requests=srbh_test_requests,  # Test on SR-BH
        test_labels=srbh_test_labels,
        output_dir=results_dir / "run1_csic_to_srbh",
        strip_headers_for_mi=True,  # CSIC has synthetic headers
    )

    # ========== RUN 2: SR-BH → CSIC ==========
    logger.info("\n\n")
    logger.info("▓" * 80)
    logger.info("RUN 2: SR-BH (feature selection) → CSIC (train/test)")
    logger.info("Keep SR-BH headers for MI (real-world varied headers)")
    logger.info("▓" * 80)

    run2_results = run_experiment(
        name="SR-BH → CSIC",
        mi_normal=srbh_all_normal,  # Use all SR-BH normal for MI
        mi_attacks=srbh_all_attacks,  # Use all SR-BH attacks for MI
        train_normal=csic_train_normal,  # Train on CSIC normal
        test_requests=csic_test_requests,  # Test on CSIC
        test_labels=csic_test_labels,
        output_dir=results_dir / "run2_srbh_to_csic",
        strip_headers_for_mi=False,  # SR-BH has real-world headers
    )

    # ========== SUMMARY ==========
    logger.info("\n\n")
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)

    logger.info("\n" + "─" * 80)
    logger.info("RUN 1: CSIC → SR-BH (Primary Validation)")
    logger.info("─" * 80)
    for r in run1_results:
        logger.info(
            f"  k={r['k']:3d}: Recall={r['recall']:.4f}, "
            f"Precision={r['precision']:.4f}, FPR={r['fpr']:.4f}, "
            f"F1={r['f1_score']:.4f}"
        )

    best1 = max(run1_results, key=lambda r: r["recall"])
    logger.info(
        f"\nBest: k={best1['k']}, Recall={best1['recall']:.4f}, FPR={best1['fpr']:.4f}"
    )

    logger.info("\n" + "─" * 80)
    logger.info("RUN 2: SR-BH → CSIC (Contrast Check)")
    logger.info("─" * 80)
    for r in run2_results:
        logger.info(
            f"  k={r['k']:3d}: Recall={r['recall']:.4f}, "
            f"Precision={r['precision']:.4f}, FPR={r['fpr']:.4f}, "
            f"F1={r['f1_score']:.4f}"
        )

    best2 = max(run2_results, key=lambda r: r["recall"])
    logger.info(
        f"\nBest: k={best2['k']}, Recall={best2['recall']:.4f}, FPR={best2['fpr']:.4f}"
    )

    # Compare to paper and Experiment 13
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON TO BASELINES")
    logger.info("=" * 80)

    logger.info("\nPaper's results:")
    logger.info("  Drupal: 91.76% TPR @ 2.29% FPR (k=100)")
    logger.info("  SR-BH:  78.87% TPR @ 5.18% FPR (k=100)")

    logger.info("\nExperiment 13 (same-dataset):")
    logger.info("  CSIC:   7.80% recall @ 5.21% FPR (k=100)")

    logger.info("\nExperiment 16 (cross-dataset):")
    logger.info(
        f"  Run 1:  {best1['recall']:.2%} recall @ "
        f"{best1['fpr']:.2%} FPR (k={best1['k']})"
    )
    logger.info(
        f"  Run 2:  {best2['recall']:.2%} recall @ "
        f"{best2['fpr']:.2%} FPR (k={best2['k']})"
    )

    # Success evaluation
    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS EVALUATION")
    logger.info("=" * 80)

    if best1["recall"] >= 0.75:
        logger.success(f"✅ Run 1: EXCELLENT ({best1['recall']:.1%} ≥ 75%)")
    elif best1["recall"] >= 0.60:
        logger.success(f"✅ Run 1: GOOD ({best1['recall']:.1%} ≥ 60%)")
    elif best1["recall"] >= 0.50:
        logger.warning(f"⚠️  Run 1: ACCEPTABLE ({best1['recall']:.1%} ≥ 50%)")
    else:
        logger.error(f"❌ Run 1: POOR ({best1['recall']:.1%} < 50%)")

    if best2["recall"] >= 0.85:
        logger.success(f"✅ Run 2: EXCELLENT ({best2['recall']:.1%} ≥ 85%)")
    elif best2["recall"] >= 0.80:
        logger.success(f"✅ Run 2: GOOD ({best2['recall']:.1%} ≥ 80%)")
    else:
        logger.warning(f"⚠️  Run 2: ACCEPTABLE ({best2['recall']:.1%} < 80%)")

    # Cross-dataset improvement
    exp13_recall = 0.0780
    improvement_vs_exp13 = (best1["recall"] / exp13_recall - 1) * 100
    logger.info(f"\nImprovement over Exp 13: {improvement_vs_exp13:+.1f}%")

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
