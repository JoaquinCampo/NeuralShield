"""
TF-IDF + PCA + Mahalanobis Cross-Dataset Experiment

Runs all 4 variants:
  1. SR_BH → CSIC with preprocessing
  2. SR_BH → CSIC without preprocessing
  3. CSIC → SR_BH with preprocessing
  4. CSIC → SR_BH without preprocessing
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from neuralshield.encoding.data.jsonl import JSONLRequestReader


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Attack"])
    ax.set_yticklabels(["Normal", "Attack"])

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=16)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved confusion matrix to {output_path}")
    plt.close()


def plot_score_distribution(scores_normal, scores_anomalous, threshold, output_path):
    """Plot distribution of anomaly scores."""
    plt.figure(figsize=(12, 6))

    plt.hist(
        scores_normal, bins=100, alpha=0.6, label="Normal", color="green", density=True
    )
    plt.hist(
        scores_anomalous,
        bins=100,
        alpha=0.6,
        label="Anomalous",
        color="red",
        density=True,
    )
    plt.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.2f}",
    )

    plt.xlabel("Mahalanobis Distance", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Score Distribution", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved score distribution to {output_path}")
    plt.close()


def run_variant(
    train_path: Path,
    test_path: Path,
    train_size: int | None,
    use_pipeline: bool,
    output_dir: Path,
):
    """Run a single experiment variant."""
    logger.info("=" * 80)
    logger.info(f"Running variant: {output_dir.name}")
    logger.info(
        f"Train: {train_path.name}, Test: {test_path.name}, Preprocessing: {use_pipeline}"
    )
    logger.info("=" * 80)

    # Step 1: Load and vectorize training data
    logger.info("STEP 1: Generate TF-IDF embeddings (training)")
    reader = JSONLRequestReader(train_path, use_pipeline=use_pipeline)
    train_requests = []
    for batch_requests, batch_labels in reader.iter_batches(batch_size=1000):
        for request in batch_requests:
            if train_size and len(train_requests) >= train_size:
                break
            train_requests.append(request)
        if train_size and len(train_requests) >= train_size:
            break

    logger.info(f"Loaded {len(train_requests)} training requests")

    # Fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
    train_embeddings = vectorizer.fit_transform(train_requests).toarray()
    logger.info(f"Training embeddings shape: {train_embeddings.shape}")

    # Save vectorizer
    joblib.dump(vectorizer, output_dir / "vectorizer.joblib")

    # Step 2: Apply PCA
    logger.info("STEP 2: Apply PCA (300 dimensions)")
    pca = PCA(n_components=300)
    train_pca = pca.fit_transform(train_embeddings)
    explained_variance = pca.explained_variance_ratio_.sum()

    logger.info(f"PCA explained variance: {explained_variance:.4%}")
    logger.info(f"PCA embeddings shape: {train_pca.shape}")

    # Save PCA model
    joblib.dump(pca, output_dir / "pca_model.joblib")

    # Step 3: Train Mahalanobis detector
    logger.info("STEP 3: Train Mahalanobis detector")
    detector = EmpiricalCovariance()
    detector.fit(train_pca)

    # Save detector
    joblib.dump(detector, output_dir / "mahalanobis_detector.joblib")

    # Step 4: Load and process test data
    logger.info("STEP 4: Process test data")
    test_reader = JSONLRequestReader(test_path, use_pipeline=use_pipeline)
    test_requests = []
    test_labels = []
    for batch_requests, batch_labels in test_reader.iter_batches(batch_size=1000):
        for request, label in zip(batch_requests, batch_labels):
            test_requests.append(request)
            test_labels.append(1 if label == "attack" else 0)

    logger.info(f"Loaded {len(test_requests)} test requests")
    logger.info(f"Normal: {sum(1 for l in test_labels if l == 0):,}")
    logger.info(f"Attacks: {sum(1 for l in test_labels if l == 1):,}")

    # Transform test data
    test_embeddings = vectorizer.transform(test_requests).toarray()
    test_pca = pca.transform(test_embeddings)

    # Step 5: Compute scores and find threshold
    logger.info("STEP 5: Compute scores and threshold @ 5% FPR")
    scores = detector.mahalanobis(test_pca)
    test_labels = np.array(test_labels)

    normal_scores = scores[test_labels == 0]
    anomalous_scores = scores[test_labels == 1]

    threshold = np.percentile(normal_scores, 95)
    logger.info(f"Threshold: {threshold:.4f}")

    # Make predictions
    predictions = (scores > threshold).astype(int)

    # Compute metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)

    cm = confusion_matrix(test_labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Log results
    logger.info("=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info(f"Recall:    {recall:.4%}")
    logger.info(f"Precision: {precision:.4%}")
    logger.info(f"F1-Score:  {f1:.4%}")
    logger.info(f"FPR:       {fpr:.4%}")
    logger.info(f"TP: {tp:,} | FP: {fp:,} | TN: {tn:,} | FN: {fn:,}")

    # Save results
    results = {
        "n_components": 300,
        "explained_variance": float(explained_variance),
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "max_fpr_constraint": 0.05,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate visualizations
    logger.info("STEP 6: Generate visualizations")
    plot_confusion_matrix(test_labels, predictions, output_dir / "confusion_matrix.png")
    plot_score_distribution(
        normal_scores,
        anomalous_scores,
        threshold,
        output_dir / "score_distribution.png",
    )

    logger.info(f"✓ Variant complete: {output_dir.name}\n")
    return results


def main():
    base_dir = Path("experiments/14_preprocessing_cross_dataset/tfidf_pca_mahalanobis")

    # Dataset paths
    srbh_train = Path("src/neuralshield/data/SR_BH_2020/train.jsonl")
    srbh_test = Path("src/neuralshield/data/SR_BH_2020/test.jsonl")
    csic_train = Path("src/neuralshield/data/CSIC/train.jsonl")
    csic_test = Path("src/neuralshield/data/CSIC/test.jsonl")

    logger.info("=" * 80)
    logger.info("TF-IDF + PCA + Mahalanobis Cross-Dataset Experiment")
    logger.info("=" * 80)
    logger.info("Running 4 variants to test preprocessing impact")
    logger.info("")

    # Variant 1: SR_BH → CSIC with preprocessing
    logger.info("▶ Variant 1/4: SR_BH → CSIC (WITH preprocessing)")
    output_dir = base_dir / "srbh_to_csic_with_prep"
    output_dir.mkdir(exist_ok=True, parents=True)
    results_1 = run_variant(srbh_train, csic_test, 100_000, True, output_dir)

    # Variant 2: SR_BH → CSIC without preprocessing
    logger.info("▶ Variant 2/4: SR_BH → CSIC (WITHOUT preprocessing)")
    output_dir = base_dir / "srbh_to_csic_without_prep"
    output_dir.mkdir(exist_ok=True, parents=True)
    results_2 = run_variant(srbh_train, csic_test, 100_000, False, output_dir)

    # Variant 3: CSIC → SR_BH with preprocessing
    logger.info("▶ Variant 3/4: CSIC → SR_BH (WITH preprocessing)")
    output_dir = base_dir / "csic_to_srbh_with_prep"
    output_dir.mkdir(exist_ok=True, parents=True)
    results_3 = run_variant(csic_train, srbh_test, None, True, output_dir)

    # Variant 4: CSIC → SR_BH without preprocessing
    logger.info("▶ Variant 4/4: CSIC → SR_BH (WITHOUT preprocessing)")
    output_dir = base_dir / "csic_to_srbh_without_prep"
    output_dir.mkdir(exist_ok=True, parents=True)
    results_4 = run_variant(csic_train, srbh_test, None, False, output_dir)

    # Summary
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETE - Summary")
    logger.info("=" * 80)
    logger.info("SR_BH → CSIC:")
    logger.info(
        f"  With prep:    {results_1['recall']:.2%} recall @ {results_1['fpr']:.2%} FPR"
    )
    logger.info(
        f"  Without prep: {results_2['recall']:.2%} recall @ {results_2['fpr']:.2%} FPR"
    )
    logger.info(f"  Δ: {results_1['recall'] - results_2['recall']:+.2%}")
    logger.info("")
    logger.info("CSIC → SR_BH:")
    logger.info(
        f"  With prep:    {results_3['recall']:.2%} recall @ {results_3['fpr']:.2%} FPR"
    )
    logger.info(
        f"  Without prep: {results_4['recall']:.2%} recall @ {results_4['fpr']:.2%} FPR"
    )
    logger.info(f"  Δ: {results_3['recall'] - results_4['recall']:+.2%}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
