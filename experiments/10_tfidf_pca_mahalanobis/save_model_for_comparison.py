#!/usr/bin/env python3
"""Save TF-IDF PCA Mahalanobis model for comparison."""

import json
from pathlib import Path

import joblib
import numpy as np
from loguru import logger
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA

# Paths
TRAIN_EMBEDDINGS = Path(
    "experiments/10_tfidf_pca_mahalanobis/with_preprocessing/train_embeddings.npz"
)
RESULTS_JSON = Path(
    "experiments/10_tfidf_pca_mahalanobis/with_preprocessing/results.json"
)
OUTPUT_MODEL = Path(
    "experiments/10_tfidf_pca_mahalanobis/with_preprocessing/tfidf_pca_mahalanobis_model.joblib"
)

# Load results to get threshold
logger.info(f"Loading results from {RESULTS_JSON}")
with open(RESULTS_JSON) as f:
    results = json.load(f)

threshold = results["threshold"]
n_components = results["n_components"]
logger.info(f"Threshold: {threshold:.4f}, PCA components: {n_components}")

# Load training embeddings
logger.info(f"Loading embeddings from {TRAIN_EMBEDDINGS}")
data = np.load(TRAIN_EMBEDDINGS)
train_embeddings = data["embeddings"]
logger.info(f"Train embeddings shape: {train_embeddings.shape}")

# Fit PCA
logger.info(f"Fitting PCA with {n_components} components...")
pca = PCA(n_components=n_components, random_state=42)
train_embeddings_pca = pca.fit_transform(train_embeddings)
logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# Fit Mahalanobis
logger.info("Fitting Mahalanobis (Empirical Covariance)...")
detector = EmpiricalCovariance()
detector.fit(train_embeddings_pca)
logger.info("Covariance fitted successfully")

# Save model with PCA and threshold
model_data = {
    "model": detector,
    "pca": pca,
    "threshold": threshold,
    "name": "tfidf_pca_mahalanobis_with_preprocessing",
}

logger.info(f"Saving model to {OUTPUT_MODEL}")
joblib.dump(model_data, OUTPUT_MODEL)
logger.info("Model saved successfully")
logger.info(f"Threshold: {threshold:.4f}")
