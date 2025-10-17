from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import typer
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from sklearn.decomposition import PCA

from neuralshield.anomaly import MahalanobisDetector
from neuralshield.evaluation import ClassificationEvaluator, EvaluationConfig


class EnsembleConfig(BaseModel):
    """Paths and hyperparameters for the LOF + SecBERT ensemble experiment."""

    tfidf_train_embeddings: Path = Field(
        default=Path("embeddings/tfidf_dump.npz"),
        description="TF-IDF training embeddings (5000 dims, valid only).",
    )
    tfidf_test_embeddings: Path = Field(
        default=Path(
            "experiments/15_lof_comparison/tfidf_pca_150/csic_test_embeddings.npz"
        ),
        description="TF-IDF test embeddings already projected to PCA space (150 dims).",
    )
    tfidf_model_bundle: Path | None = Field(
        default=Path(
            "experiments/15_lof_comparison/tfidf_pca_150/lof_tfidf_pca150_k100.joblib"
        ),
        description="Optional TF-IDF model bundle containing PCA + LOF detector.",
    )
    secbert_train_embeddings: Path = Field(
        default=Path(
            "experiments/03_secbert_comparison/secbert_with_preprocessing/"
            "csic_train_embeddings_converted.npz"
        ),
        description="SecBERT training embeddings (768 dims, valid only).",
    )
    secbert_test_embeddings: Path = Field(
        default=Path(
            "experiments/03_secbert_comparison/secbert_with_preprocessing/"
            "csic_test_embeddings_converted.npz"
        ),
        description="SecBERT test embeddings (768 dims).",
    )
    output_dir: Path = Field(
        default=Path("experiments/18_lof_secbert_ensemble/artifacts"),
        description="Directory to save trained detectors and metrics.",
    )
    tfidf_pca_components: int = Field(
        default=150,
        description="Number of PCA components when training LOF from scratch.",
        ge=10,
        le=2000,
    )
    lof_neighbors: int = Field(
        default=100,
        description="Number of neighbours for LOF when training from scratch.",
        ge=5,
        le=1000,
    )
    max_fpr: float = Field(
        default=0.05,
        description="Target false positive rate for threshold calibration.",
        gt=0.0,
        lt=0.5,
    )
    fusion_weight: float = Field(
        default=0.6,
        description="Weight for LOF scores in fusion (SecBERT uses 1 minus this).",
        gt=0.0,
        lt=1.0,
    )

    model_config = ConfigDict(validate_assignment=True)


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels from a NumPy .npz archive."""

    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    labels = data["labels"].astype(str)
    logger.debug("Loaded embeddings", path=str(path), shape=embeddings.shape)
    return embeddings, labels


class ZScoreNormalizer:
    """Standardise scores using training mean and standard deviation."""

    def __init__(self, reference: np.ndarray) -> None:
        if reference.size == 0:
            raise ValueError("Cannot standardise empty score reference")
        self._mean = float(np.mean(reference))
        self._std = float(np.std(reference))
        if self._std == 0.0:
            self._std = 1e-6

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return np.empty_like(scores, dtype=np.float32)
        standardised = (scores.astype(np.float32) - self._mean) / self._std
        return standardised.astype(np.float32, copy=False)


def evaluate_predictions(
    predictions: np.ndarray, labels: np.ndarray
) -> dict[str, float | int]:
    """Compute classification metrics using the project evaluator."""

    evaluator = ClassificationEvaluator(EvaluationConfig())
    result = evaluator.evaluate(predictions.tolist(), labels.tolist())
    return {
        "precision": result.precision,
        "recall": result.recall,
        "f1_score": result.f1_score,
        "accuracy": result.accuracy,
        "fpr": result.fpr,
        "specificity": result.specificity,
        "tp": result.tp,
        "fp": result.fp,
        "tn": result.tn,
        "fn": result.fn,
        "total_samples": result.total_samples,
    }


def ensure_label_alignment(
    primary: np.ndarray, secondary: np.ndarray, *, source_a: Path, source_b: Path
) -> np.ndarray:
    """Validate that label arrays from two sources align."""

    if primary.shape != secondary.shape or not np.array_equal(primary, secondary):
        raise ValueError(
            f"Label mismatch between embeddings:\n- {source_a}\n- {source_b}"
        )
    return primary


app = typer.Typer(help="Run LOF + SecBERT ensemble training and evaluation.")


@app.command()
def run(
    config_path: Path = typer.Option(
        None,
        "--config",
        help="Optional JSON config overriding default paths and parameters.",
    ),
) -> None:
    """Train LOF and Mahalanobis detectors, then evaluate ensemble variants."""

    if config_path is not None:
        loaded_cfg = json.loads(config_path.read_text())
        config = EnsembleConfig(**loaded_cfg)
        logger.info("Loaded config override from {path}", path=str(config_path))
    else:
        config = EnsembleConfig()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Artifacts will be saved under {path}", path=str(config.output_dir))

    # Load embeddings
    tfidf_train_raw, _ = load_embeddings(config.tfidf_train_embeddings)
    tfidf_test, tfidf_test_labels = load_embeddings(config.tfidf_test_embeddings)

    secbert_train, _ = load_embeddings(config.secbert_train_embeddings)
    secbert_test, secbert_test_labels = load_embeddings(config.secbert_test_embeddings)

    labels = ensure_label_alignment(
        tfidf_test_labels,
        secbert_test_labels,
        source_a=config.tfidf_test_embeddings,
        source_b=config.secbert_test_embeddings,
    )

    # Extract PCA from historical TF-IDF bundle (or load standalone PCA).
    if config.tfidf_model_bundle is not None and config.tfidf_model_bundle.exists():
        bundle = joblib.load(config.tfidf_model_bundle)
        if (
            not isinstance(bundle, dict)
            or "pca" not in bundle
            or "detector" not in bundle
        ):
            raise ValueError(
                "Model bundle at "
                f"{config.tfidf_model_bundle} must contain 'pca' and 'detector'"
            )
        pca: PCA = bundle["pca"]
        lof = bundle["detector"]
    tfidf_train = (
        pca.transform(tfidf_train_raw.astype(np.float32))
        if tfidf_train_raw.shape[1] != pca.n_components_
        else tfidf_train_raw.astype(np.float32, copy=False)
    )
        lof_threshold = float(bundle.get("threshold", getattr(lof, "_threshold", 0.0)))
        logger.info(
            "Loaded pre-trained LOF detector",
            threshold=lof_threshold,
            samples=tfidf_train.shape[0],
            features=tfidf_train.shape[1],
        )
    else:
        logger.info(
            "Training PCA (n_components={components}) and LOF "
            "(k={neighbors}) from scratch",
            components=config.tfidf_pca_components,
            neighbors=config.lof_neighbors,
        )
        pca = PCA(n_components=config.tfidf_pca_components, random_state=42)
        tfidf_train = pca.fit_transform(tfidf_train_raw).astype(np.float32)
        from neuralshield.anomaly import LOFDetector

        lof = LOFDetector(n_neighbors=config.lof_neighbors)
        lof.fit(tfidf_train)
        lof_threshold = lof.set_threshold(tfidf_train, max_fpr=config.max_fpr)
        logger.info(
            "Trained LOF detector from scratch",
            threshold=lof_threshold,
            samples=tfidf_train.shape[0],
            features=tfidf_train.shape[1],
        )

    lof_bundle = {
        "name": "lof_tfidf_pca",
        "detector": lof,
        "pca": pca,
        "threshold": lof_threshold,
        "n_neighbors": getattr(lof, "n_neighbors", config.lof_neighbors),
        "n_components": pca.n_components_,
    }

    joblib.dump(pca, config.output_dir / "tfidf_pca.joblib", compress=3)
    joblib.dump(lof_bundle, config.output_dir / "lof_detector.joblib", compress=3)

    maha = MahalanobisDetector()
    maha.fit(secbert_train)
    maha_threshold = maha.set_threshold(secbert_train, max_fpr=config.max_fpr)
    logger.info("Calibrated Mahalanobis threshold", threshold=maha_threshold)

    # Persist detectors
    maha_path = config.output_dir / "mahalanobis_detector.joblib"
    maha.save(maha_path)

    tfidf_test_projected = (
        pca.transform(tfidf_test.astype(np.float32))
        if tfidf_test.shape[1] != pca.n_components_
        else tfidf_test.astype(np.float32, copy=False)
    )

    # Scores on training (normal only) for fusion calibration
    lof_train_scores = lof.scores(tfidf_train)
    maha_train_scores = maha.scores(secbert_train)
    lof_normalizer = ZScoreNormalizer(lof_train_scores)
    maha_normalizer = ZScoreNormalizer(maha_train_scores)
    lof_train_norm = lof_normalizer.transform(lof_train_scores)
    maha_train_norm = maha_normalizer.transform(maha_train_scores)
    fused_train_scores = (
        config.fusion_weight * lof_train_norm
        + (1.0 - config.fusion_weight) * maha_train_norm
    )
    fused_threshold = float(np.quantile(fused_train_scores, 1.0 - config.max_fpr))
    logger.info(
        "Computed fused threshold from training data",
        threshold=fused_threshold,
        weight_lof=config.fusion_weight,
    )

    # Evaluate on test set
    lof_scores = lof.scores(tfidf_test_projected)
    maha_scores = maha.scores(secbert_test.astype(np.float32))
    lof_predictions = lof.predict(tfidf_test_projected)
    maha_predictions = maha.predict(secbert_test.astype(np.float32))

    lof_metrics = evaluate_predictions(lof_predictions, labels)
    maha_metrics = evaluate_predictions(maha_predictions, labels)

    union_predictions = np.logical_or(lof_predictions, maha_predictions)
    union_metrics = evaluate_predictions(union_predictions, labels)

    lof_test_norm = lof_normalizer.transform(lof_scores)
    maha_test_norm = maha_normalizer.transform(maha_scores)
    fused_scores = (
        config.fusion_weight * lof_test_norm
        + (1.0 - config.fusion_weight) * maha_test_norm
    )
    fused_predictions = fused_scores >= fused_threshold
    fused_metrics = evaluate_predictions(fused_predictions, labels)

    metrics: dict[str, Any] = {
        "config": {
            "max_fpr": config.max_fpr,
            "fusion_weight": config.fusion_weight,
            "tfidf_pca_components": int(tfidf_train.shape[1]),
        },
        "thresholds": {
            "lof": float(lof_threshold),
            "mahalanobis": float(maha_threshold),
            "fused": fused_threshold,
        },
        "lof": lof_metrics,
        "mahalanobis": maha_metrics,
        "union": union_metrics,
        "fused": fused_metrics,
    }

    metrics_path = config.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved metrics to {path}", path=str(metrics_path))

    logger.success(
        "Ensemble evaluation complete",
        lof_recall=lof_metrics["recall"],
        maha_recall=maha_metrics["recall"],
        union_recall=union_metrics["recall"],
        fused_recall=fused_metrics["recall"],
    )


if __name__ == "__main__":
    app()
