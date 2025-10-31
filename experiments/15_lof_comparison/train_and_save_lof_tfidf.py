"""Train LOF on configurable TF-IDF + PCA embeddings and save artifacts."""

import json
from pathlib import Path

import joblib
import numpy as np
import typer
from loguru import logger
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from neuralshield.anomaly import LOFDetector
from neuralshield.encoding.data.jsonl import JSONLRequestReader
from neuralshield.preprocessing.pipeline import preprocess


class DatasetConfig(BaseModel):
    train_path: Path
    test_path: Path
    use_preprocessing: bool = True
    batch_size: int = Field(default=1000, ge=1)


class TFIDFConfig(BaseModel):
    max_features: int = Field(default=5000, ge=1)
    min_df: int = Field(default=2, ge=1)
    ngram_min: int = Field(default=1, ge=1)
    ngram_max: int = Field(default=3, ge=1)

    def build(self) -> TfidfVectorizer:
        if self.ngram_min > self.ngram_max:
            raise ValueError("ngram_min cannot be larger than ngram_max")
        return TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=(self.ngram_min, self.ngram_max),
        )


class PCAConfig(BaseModel):
    n_components: int = Field(default=150, ge=1)
    random_state: int = 42

    def build(self) -> PCA:
        return PCA(n_components=self.n_components, random_state=self.random_state)


class LOFConfig(BaseModel):
    n_neighbors: int = Field(default=100, ge=1)
    target_fpr: float = Field(default=0.05, gt=0, lt=1)


class TrainingConfig(BaseModel):
    dataset: DatasetConfig
    tfidf: TFIDFConfig
    pca: PCAConfig
    lof: LOFConfig
    output_dir: Path
    model_name: str = "LOF_TF-IDF_PCA"


def iter_texts(path: Path, batch_size: int, transform) -> list[str]:
    reader = JSONLRequestReader(path, use_pipeline=False)
    texts: list[str] = []
    for batch, _ in reader.iter_batches(batch_size=batch_size):
        texts.extend(transform(text) for text in batch)
    return texts


def iter_texts_with_labels(path: Path, batch_size: int, transform) -> tuple[list[str], list[str]]:
    reader = JSONLRequestReader(path, use_pipeline=False)
    texts: list[str] = []
    labels: list[str] = []
    for batch, batch_labels in reader.iter_batches(batch_size=batch_size):
        for text, label in zip(batch, batch_labels):
            texts.append(transform(text))
            labels.append(label)
    return texts, labels


def run_training(config: TrainingConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    transform_fn = preprocess if config.dataset.use_preprocessing else (lambda text: text)  # noqa: E731

    logger.info("=" * 80)
    logger.info("Training LOF with TF-IDF + PCA")
    logger.info("=" * 80)

    logger.info("Loading training data")
    train_texts = iter_texts(
        config.dataset.train_path,
        batch_size=config.dataset.batch_size,
        transform=transform_fn,
    )
    logger.info(f"Loaded {len(train_texts)} training samples")

    logger.info("Loading test data")
    test_texts, test_labels = iter_texts_with_labels(
        config.dataset.test_path,
        batch_size=config.dataset.batch_size,
        transform=transform_fn,
    )
    logger.info(f"Loaded {len(test_texts)} test samples")

    logger.info("Fitting TF-IDF vectorizer")
    vectorizer = config.tfidf.build()
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)
    logger.info(f"TF-IDF shape: {train_tfidf.shape}")

    logger.info("Applying PCA")
    pca = config.pca.build()
    train_embeddings = pca.fit_transform(train_tfidf.toarray())
    test_embeddings = pca.transform(test_tfidf.toarray())
    explained_variance = float(pca.explained_variance_ratio_.sum())
    logger.info(f"PCA explained variance: {explained_variance:.2%}")

    logger.info("Training LOF detector")
    detector = LOFDetector(n_neighbors=config.lof.n_neighbors)
    detector.fit(train_embeddings.astype(np.float32))

    logger.info("Computing scores")
    test_scores = detector.scores(test_embeddings.astype(np.float32))
    test_labels_binary = np.array([1 if label == "attack" else 0 for label in test_labels])
    test_normal_mask = test_labels_binary == 0
    test_scores_normal = test_scores[test_normal_mask]

    percentile = (1 - config.lof.target_fpr) * 100
    threshold = float(np.percentile(test_scores_normal, percentile))
    detector._threshold = threshold  # noqa: SLF001
    actual_fpr = np.mean(test_scores_normal > threshold)
    logger.info(f"Threshold set to {threshold:.4f} (actual FPR: {actual_fpr:.2%})")

    model_path = config.output_dir / f"{config.model_name.lower()}_k{config.lof.n_neighbors}.joblib"
    embeddings_path = config.output_dir / "test_embeddings.npz"
    metrics_path = config.output_dir / "model_metrics.json"

    model_data = {
        "name": config.model_name,
        "detector": detector,
        "vectorizer": vectorizer,
        "pca": pca,
        "threshold": threshold,
        "n_neighbors": config.lof.n_neighbors,
        "n_components": config.pca.n_components,
        "explained_variance": explained_variance,
        "preprocessing": config.dataset.use_preprocessing,
        "target_fpr": config.lof.target_fpr,
    }
    joblib.dump(model_data, model_path)

    np.savez(
        embeddings_path,
        embeddings=test_embeddings.astype(np.float32),
        labels=np.array(test_labels),
    )

    predictions = detector.predict(test_embeddings.astype(np.float32))
    test_scores_anomalous = test_scores[~test_normal_mask]
    recall = np.mean(test_scores_anomalous > threshold)
    tp = int(np.sum((predictions == 1) & (test_labels_binary == 1)))
    fp = int(np.sum((predictions == 1) & (test_labels_binary == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "model": config.model_name,
        "n_neighbors": config.lof.n_neighbors,
        "n_components": config.pca.n_components,
        "explained_variance": explained_variance,
        "threshold": threshold,
        "recall": float(recall),
        "precision": float(precision),
        "f1_score": float(f1),
        "fpr": float(actual_fpr),
        "true_positives": tp,
        "false_positives": fp,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Embeddings saved to: {embeddings_path}")
    logger.info(f"Metrics saved to: {metrics_path}")
    logger.info(f"Recall @ {config.lof.target_fpr:.0%} FPR: {recall:.2%}")
    logger.info(f"Precision: {precision:.2%}")
    logger.info(f"F1-Score: {f1:.2%}")


def main(
    train_path: Path = typer.Option(
        Path("src/neuralshield/data/CSIC/train.jsonl"),
        help="Path to training JSONL dataset",
    ),
    test_path: Path = typer.Option(
        Path("src/neuralshield/data/CSIC/test.jsonl"),
        help="Path to test JSONL dataset",
    ),
    output_dir: Path = typer.Option(
        Path("experiments/15_lof_comparison/tfidf_pca_150"),
        help="Directory where artifacts will be stored",
    ),
    model_name: str = typer.Option(
        "LOF_TF-IDF_PCA150",
        help="Identifier embedded in saved artifacts",
    ),
    use_preprocessing: bool = typer.Option(
        True,
        help="Apply request preprocessing pipeline before vectorization",
    ),
    batch_size: int = typer.Option(
        1000,
        help="Batch size for streaming JSONL files",
        min=1,
    ),
    max_features: int = typer.Option(
        5000,
        help="TF-IDF max vocabulary size",
        min=1,
    ),
    min_df: int = typer.Option(
        2,
        help="TF-IDF minimum document frequency",
        min=1,
    ),
    ngram_min: int = typer.Option(
        1,
        help="TF-IDF ngram lower bound",
        min=1,
    ),
    ngram_max: int = typer.Option(
        3,
        help="TF-IDF ngram upper bound",
        min=1,
    ),
    pca_components: int = typer.Option(
        150,
        help="Number of PCA components",
        min=1,
    ),
    n_neighbors: int = typer.Option(
        100,
        help="LOF number of neighbors",
        min=1,
    ),
    target_fpr: float = typer.Option(
        0.05,
        help="Target false positive rate for threshold selection",
        min=1e-6,
        max=0.999999,
    ),
) -> None:
    tfidf_cfg = TFIDFConfig(
        max_features=max_features,
        min_df=min_df,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
    )
    dataset_cfg = DatasetConfig(
        train_path=train_path,
        test_path=test_path,
        use_preprocessing=use_preprocessing,
        batch_size=batch_size,
    )
    pca_cfg = PCAConfig(n_components=pca_components)
    lof_cfg = LOFConfig(n_neighbors=n_neighbors, target_fpr=target_fpr)
    config = TrainingConfig(
        dataset=dataset_cfg,
        tfidf=tfidf_cfg,
        pca=pca_cfg,
        lof=lof_cfg,
        output_dir=output_dir,
        model_name=model_name,
    )
    run_training(config)


if __name__ == "__main__":
    typer.run(main)
