#!/usr/bin/env python3
"""Generate TF-IDF embeddings for PCA-Mahalanobis experiment."""

import json
from pathlib import Path

import joblib
import numpy as np
from loguru import logger

# Paths
TRAIN_DATA = Path("src/neuralshield/data/CSIC/train.jsonl")
TEST_DATA = Path("src/neuralshield/data/CSIC/test.jsonl")
VECTORIZER_WITH_PREP = Path("embeddings/vectorizer.joblib")
VECTORIZER_NO_PREP = Path("embeddings/vectorizer_no_pipeline.joblib")

WITH_PREP_DIR = Path("experiments/10_tfidf_pca_mahalanobis/with_preprocessing")
WITHOUT_PREP_DIR = Path("experiments/10_tfidf_pca_mahalanobis/without_preprocessing")


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


def generate_embeddings(scenario: str, directory: Path, vectorizer_path: Path):
    """Generate embeddings for a given scenario."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Processing {scenario}")
    logger.info(f"{'=' * 80}")

    directory.mkdir(parents=True, exist_ok=True)

    # Load vectorizer
    logger.info(f"Loading vectorizer from {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)

    # Load train data (all valid)
    train_requests, train_labels = load_data(TRAIN_DATA)

    # Transform train data
    logger.info(f"Transforming {len(train_requests)} train requests...")
    train_embeddings_sparse = vectorizer.transform(train_requests)
    train_embeddings_dense = train_embeddings_sparse.toarray().astype(np.float32)
    logger.info(f"Train embeddings shape: {train_embeddings_dense.shape}")

    # Save train embeddings
    train_output_path = directory / "train_embeddings.npz"
    np.savez_compressed(
        train_output_path,
        embeddings=train_embeddings_dense,
        labels=np.array(train_labels),
    )
    logger.info(f"Saved train embeddings to {train_output_path}")

    # Load test data
    test_requests, test_labels = load_data(TEST_DATA)

    # Transform test data
    logger.info(f"Transforming {len(test_requests)} test requests...")
    test_embeddings_sparse = vectorizer.transform(test_requests)
    test_embeddings_dense = test_embeddings_sparse.toarray().astype(np.float32)
    logger.info(f"Test embeddings shape: {test_embeddings_dense.shape}")

    # Save test embeddings
    test_output_path = directory / "test_embeddings.npz"
    np.savez_compressed(
        test_output_path,
        embeddings=test_embeddings_dense,
        labels=np.array(test_labels),
    )
    logger.info(f"Saved test embeddings to {test_output_path}")


def main():
    logger.info("Generating TF-IDF embeddings for PCA-Mahalanobis experiment")

    # With preprocessing
    generate_embeddings("with_preprocessing", WITH_PREP_DIR, VECTORIZER_WITH_PREP)

    # Without preprocessing
    generate_embeddings("without_preprocessing", WITHOUT_PREP_DIR, VECTORIZER_NO_PREP)

    logger.info("\n" + "=" * 80)
    logger.info("All embeddings generated successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
