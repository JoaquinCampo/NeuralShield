#!/usr/bin/env python3
"""Generate TF-IDF test embeddings for model comparison."""

import json
from pathlib import Path

import joblib
import numpy as np
from loguru import logger

# Paths
TEST_DATA = Path("src/neuralshield/data/CSIC/test.jsonl")
WITH_PREP_DIR = Path("experiments/01_tfidf_preprocessing_comparison/with_preprocessing")
WITHOUT_PREP_DIR = Path(
    "experiments/01_tfidf_preprocessing_comparison/without_preprocessing"
)

# Load test data
logger.info(f"Loading test data from {TEST_DATA}")
test_requests = []
test_labels = []

with open(TEST_DATA) as f:
    for line in f:
        data = json.loads(line)
        test_requests.append(data["request"])
        test_labels.append(data["label"])

logger.info(f"Loaded {len(test_requests)} test samples")

# Generate embeddings for both scenarios
for scenario, directory in [
    ("without_preprocessing", WITHOUT_PREP_DIR),
    ("with_preprocessing", WITH_PREP_DIR),
]:
    logger.info(f"\nProcessing {scenario}...")

    # Load vectorizer
    vectorizer_path = directory / "vectorizer.joblib"
    vectorizer = joblib.load(vectorizer_path)
    logger.info(f"Loaded vectorizer from {vectorizer_path}")

    # Transform to TF-IDF
    logger.info(f"Transforming {len(test_requests)} requests...")
    embeddings_sparse = vectorizer.transform(test_requests)
    embeddings_dense = embeddings_sparse.toarray().astype(np.float32)

    logger.info(f"Embeddings shape: {embeddings_dense.shape}")

    # Save to NPZ
    output_path = directory / "test_embeddings.npz"
    np.savez_compressed(
        output_path,
        embeddings=embeddings_dense,
        labels=np.array(test_labels),
    )
    logger.info(f"Saved test embeddings to {output_path}")

logger.info("\nAll TF-IDF test embeddings generated successfully!")
