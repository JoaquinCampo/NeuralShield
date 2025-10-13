#!/usr/bin/env python3
"""Generate TF-IDF embeddings for cross-dataset: SRBH train -> CSIC test."""

import json
from pathlib import Path

import joblib
import numpy as np
from loguru import logger

# Paths
SRBH_TRAIN = Path("src/neuralshield/data/SR_BH_2020/train.jsonl")
CSIC_TEST = Path("src/neuralshield/data/CSIC/test.jsonl")
VECTORIZER_WITH_PREP = Path("embeddings/vectorizer.joblib")
OUTPUT_DIR = Path("experiments/10_tfidf_pca_mahalanobis/cross_dataset_srbh_to_csic")


def load_data(path: Path) -> tuple[list[str], list[str]]:
    """Load requests and labels from JSONL file."""
    logger.info(f"Loading data from {path}")
    requests = []
    labels = []

    with open(path) as f:
        for line in f:
            data = json.loads(line)
            requests.append(data["request"])
            labels.append(data["label"])

    logger.info(f"Loaded {len(requests)} samples")
    return requests, labels


def main():
    logger.info("=" * 80)
    logger.info("Cross-Dataset TF-IDF Embedding Generation: SRBH -> CSIC")
    logger.info("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load vectorizer (with preprocessing)
    logger.info(f"Loading vectorizer from {VECTORIZER_WITH_PREP}")
    vectorizer = joblib.load(VECTORIZER_WITH_PREP)

    # Load SRBH train data
    logger.info("\n" + "=" * 80)
    logger.info("SRBH TRAIN DATA")
    logger.info("=" * 80)
    train_requests, train_labels = load_data(SRBH_TRAIN)

    # Transform train data
    logger.info(f"Transforming {len(train_requests)} train requests...")
    train_embeddings_sparse = vectorizer.transform(train_requests)
    train_embeddings_dense = train_embeddings_sparse.toarray().astype(np.float32)
    logger.info(f"Train embeddings shape: {train_embeddings_dense.shape}")

    # Save train embeddings
    train_output_path = OUTPUT_DIR / "srbh_train_embeddings.npz"
    np.savez_compressed(
        train_output_path,
        embeddings=train_embeddings_dense,
        labels=np.array(train_labels),
    )
    logger.info(f"Saved train embeddings to {train_output_path}")

    # Load CSIC test data
    logger.info("\n" + "=" * 80)
    logger.info("CSIC TEST DATA")
    logger.info("=" * 80)
    test_requests, test_labels = load_data(CSIC_TEST)

    # Transform test data
    logger.info(f"Transforming {len(test_requests)} test requests...")
    test_embeddings_sparse = vectorizer.transform(test_requests)
    test_embeddings_dense = test_embeddings_sparse.toarray().astype(np.float32)
    logger.info(f"Test embeddings shape: {test_embeddings_dense.shape}")

    # Save test embeddings
    test_output_path = OUTPUT_DIR / "csic_test_embeddings.npz"
    np.savez_compressed(
        test_output_path,
        embeddings=test_embeddings_dense,
        labels=np.array(test_labels),
    )
    logger.info(f"Saved test embeddings to {test_output_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Embeddings generated successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
