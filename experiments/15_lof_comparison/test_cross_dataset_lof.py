"""Test LOF cross-dataset generalization.

Tests:
1. Train on CSIC → Test on SR_BH
2. Train on SR_BH → Test on CSIC

Compares with SecBERT cross-dataset results from Experiment 08.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from neuralshield.anomaly import LOFDetector
from neuralshield.encoding.data.jsonl import JSONLRequestReader


def load_dataset(
    path: Path, max_samples: int | None = None
) -> tuple[list[str], list[str]]:
    """Load dataset without preprocessing."""
    logger.info(f"Loading dataset from {path}")
    dataset = JSONLRequestReader(path, use_pipeline=False)
    texts = []
    labels = []

    for batch, batch_labels in dataset.iter_batches(batch_size=1000):
        for text, label in zip(batch, batch_labels):
            texts.append(text)
            labels.append(label)
            if max_samples and len(texts) >= max_samples:
                break
        if max_samples and len(texts) >= max_samples:
            break

    logger.info(f"Loaded {len(texts)} samples")
    return texts, labels


def train_and_test_lof(
    train_texts: list[str],
    test_texts: list[str],
    test_labels: list[str],
    scenario_name: str,
    output_dir: Path,
) -> dict:
    """Train LOF on training set and test on test set."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Scenario: {scenario_name}")
    logger.info(f"{'=' * 80}")

    # Fit TF-IDF
    logger.info("Fitting TF-IDF vectorizer")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)

    logger.info(f"TF-IDF shape: train={train_tfidf.shape}, test={test_tfidf.shape}")

    # Apply PCA
    logger.info("Applying PCA to 150 components")
    pca = PCA(n_components=150, random_state=42)
    train_embeddings = pca.fit_transform(train_tfidf.toarray())
    test_embeddings = pca.transform(test_tfidf.toarray())

    explained_variance = float(pca.explained_variance_ratio_.sum())
    logger.info(f"PCA explained variance: {explained_variance:.2%}")

    # Train LOF
    logger.info("Training LOF detector (k=100)")
    detector = LOFDetector(n_neighbors=100)
    detector.fit(train_embeddings.astype(np.float32))

    # Compute scores
    logger.info("Computing scores on test set")
    test_scores = detector.scores(test_embeddings.astype(np.float32))

    # Split by label
    test_labels_binary = np.array(
        [1 if label == "attack" else 0 for label in test_labels]
    )
    test_normal_mask = test_labels_binary == 0
    test_scores_normal = test_scores[test_normal_mask]
    test_scores_anomalous = test_scores[~test_normal_mask]

    logger.info(f"Normal samples: {test_normal_mask.sum()}")
    logger.info(f"Attack samples: {(~test_normal_mask).sum()}")

    # Set threshold at 5% FPR
    threshold = float(np.percentile(test_scores_normal, 95.0))
    actual_fpr = np.mean(test_scores_normal > threshold)

    # Compute metrics
    recall = np.mean(test_scores_anomalous > threshold)

    predictions = (test_scores > threshold).astype(int)
    tp = np.sum((predictions == 1) & (test_labels_binary == 1))
    fp = np.sum((predictions == 1) & (test_labels_binary == 0))
    tn = np.sum((predictions == 0) & (test_labels_binary == 0))
    fn = np.sum((predictions == 0) & (test_labels_binary == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / len(test_labels)

    logger.info(f"Threshold: {threshold:.4f} (actual FPR: {actual_fpr:.2%})")
    logger.info(f"Recall @ 5% FPR: {recall:.2%}")
    logger.info(f"Precision: {precision:.2%}")
    logger.info(f"F1-Score: {f1:.2%}")
    logger.info(f"Accuracy: {accuracy:.2%}")

    # Save results
    results = {
        "scenario": scenario_name,
        "n_train": len(train_texts),
        "n_test": len(test_labels),
        "n_test_normal": int(test_normal_mask.sum()),
        "n_test_attack": int((~test_normal_mask).sum()),
        "explained_variance": explained_variance,
        "threshold": threshold,
        "recall": float(recall),
        "precision": float(precision),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
        "fpr": float(actual_fpr),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    return results


def main():
    """Run cross-dataset experiments."""
    logger.info("=" * 80)
    logger.info("LOF CROSS-DATASET GENERALIZATION TEST")
    logger.info("=" * 80)

    data_dir = Path("src/neuralshield/data")
    output_base = Path("experiments/15_lof_comparison/cross_dataset")

    # Load datasets
    logger.info("\n" + "=" * 80)
    logger.info("LOADING DATASETS")
    logger.info("=" * 80)

    csic_train_texts, _ = load_dataset(data_dir / "CSIC" / "train.jsonl")
    csic_test_texts, csic_test_labels = load_dataset(data_dir / "CSIC" / "test.jsonl")

    # Limit SR_BH to match experiment 08 sizes
    srbh_train_texts, _ = load_dataset(
        data_dir / "SR_BH_2020" / "train.jsonl", max_samples=100_000
    )
    srbh_test_texts, srbh_test_labels = load_dataset(
        data_dir / "SR_BH_2020" / "test.jsonl",
        max_samples=None,  # Use full test set
    )

    logger.info("\nDataset sizes:")
    logger.info(f"  CSIC train: {len(csic_train_texts)}")
    logger.info(f"  CSIC test: {len(csic_test_texts)}")
    logger.info(f"  SR_BH train: {len(srbh_train_texts)}")
    logger.info(f"  SR_BH test: {len(srbh_test_texts)}")

    # Experiment 1: CSIC → SR_BH
    results_csic_to_srbh = train_and_test_lof(
        train_texts=csic_train_texts,
        test_texts=srbh_test_texts,
        test_labels=srbh_test_labels,
        scenario_name="CSIC_to_SRBH",
        output_dir=output_base / "csic_to_srbh",
    )

    # Experiment 2: SR_BH → CSIC
    results_srbh_to_csic = train_and_test_lof(
        train_texts=srbh_train_texts,
        test_texts=csic_test_texts,
        test_labels=csic_test_labels,
        scenario_name="SRBH_to_CSIC",
        output_dir=output_base / "srbh_to_csic",
    )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-DATASET SUMMARY")
    logger.info("=" * 80)

    logger.info("\nLOF + TF-IDF + PCA (no preprocessing):")
    logger.info(f"  CSIC → SR_BH: {results_csic_to_srbh['recall']:.2%} recall")
    logger.info(f"  SR_BH → CSIC: {results_srbh_to_csic['recall']:.2%} recall")

    logger.info("\nComparison with SecBERT (Experiment 08):")
    logger.info("  SecBERT + Mahalanobis (with preprocessing):")
    logger.info("    SR_BH → CSIC: 10.26% recall")
    logger.info("    CSIC → SR_BH: 10.76% recall")

    logger.info("\nSame-dataset baselines:")
    logger.info("  LOF on CSIC: 64.20% recall")
    logger.info("  SecBERT on CSIC: 49.26% recall")
    logger.info("  SecBERT on SR_BH: 54.12% recall")

    # Save combined summary
    summary = {
        "lof_cross_dataset": {
            "csic_to_srbh": results_csic_to_srbh,
            "srbh_to_csic": results_srbh_to_csic,
        },
        "secbert_cross_dataset_reference": {
            "srbh_to_csic": {"recall": 0.1026, "precision": 0.6729, "f1": 0.1780},
            "csic_to_srbh": {"recall": 0.1076, "precision": 0.6595, "f1": 0.1850},
        },
        "same_dataset_baselines": {
            "lof_csic": {"recall": 0.6420, "precision": 0.9295, "f1": 0.7595},
            "secbert_csic": {"recall": 0.4926, "precision": 0.9081, "f1": 0.6387},
            "secbert_srbh": {"recall": 0.5412, "precision": 0.9069, "f1": 0.6779},
        },
    }

    summary_path = output_base / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSummary saved to {summary_path}")
    logger.info("\nExperiment complete!")


if __name__ == "__main__":
    main()
