from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import typer
from loguru import logger
from pydantic import BaseModel, field_validator

import neuralshield.encoding.data.factory as data_factory
from neuralshield.encoding.models.tfidf import TFIDFEncoder, TFIDFEncoderConfig
from neuralshield.encoding.observability import init_wandb_sink
from neuralshield.preprocessing.pipeline import PreprocessorPipeline, preprocess

if TYPE_CHECKING:
    from neuralshield.anomaly import MahalanobisDetector


class TFIDFEncodingConfig(BaseModel):
    """Configuration for fitting TF-IDF and dumping embeddings."""

    dataset_path: Path
    embeddings_path: Path
    vectorizer_path: Path
    batch_size: int = 512
    reader_name: str = "jsonl"
    use_pipeline: bool = False
    pipeline_name: str | None = None
    tfidf: TFIDFEncoderConfig = TFIDFEncoderConfig()

    @field_validator("batch_size")
    @classmethod
    def _check_batch_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("batch_size must be positive")
        return value


class TFIDFAnomalyTrainingConfig(BaseModel):
    """Configuration for training an anomaly detector from TF-IDF embeddings."""

    embeddings_path: Path
    model_path: Path
    valid_label: str = "valid"
    contamination: float = 0.01
    random_state: int | None = None
    detector_type: str = "isolation_forest"  # "isolation_forest" or "elliptic_envelope"
    n_estimators: int = 100  # For IsolationForest
    show_progress: bool = True
    wandb_enabled: bool = False
    wandb_project: str = "neuralshield"
    wandb_entity: str | None = None

    @field_validator("contamination")
    @classmethod
    def _check_contamination(cls, value: float) -> float:
        if not 0 < value <= 0.5:
            raise ValueError("contamination must be in (0, 0.5)")
        return value

    @field_validator("detector_type")
    @classmethod
    def _check_detector_type(cls, value: str) -> str:
        allowed = {"isolation_forest", "elliptic_envelope"}
        if value not in allowed:
            raise ValueError(f"detector_type must be one of {allowed}")
        return value


def _resolve_pipeline(
    *,
    use_pipeline: bool,
    pipeline_name: str | None,
) -> PreprocessorPipeline | Callable[[str], str] | None:
    if not use_pipeline:
        return None

    selected = pipeline_name or "preprocess"
    if selected != "preprocess":
        raise ValueError(
            f"Unsupported pipeline '{selected}'. Only 'preprocess' is available."
        )
    return preprocess


def dump_tfidf_embeddings(config: TFIDFEncodingConfig) -> Path:
    """Fit TF-IDF on the dataset and persist embeddings plus vectorizer."""

    reader_factory = data_factory.get_reader(config.reader_name)
    pipeline_runner = _resolve_pipeline(
        use_pipeline=config.use_pipeline, pipeline_name=config.pipeline_name
    )

    reader = reader_factory(
        path=config.dataset_path,
        pipeline=pipeline_runner,
        use_pipeline=config.use_pipeline,
        observer=None,
    )

    requests: list[str] = []
    labels: list[str | None] = []

    for batch_requests, batch_labels in reader.iter_batches(config.batch_size):
        requests.extend(batch_requests)
        labels.extend(batch_labels)

    if not requests:
        raise ValueError("No requests found in dataset; cannot build TF-IDF embeddings")

    encoder = TFIDFEncoder(settings=config.tfidf)
    embeddings = encoder.fit_transform(requests)

    config.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.embeddings_path,
        embeddings=embeddings,
        labels=np.asarray(labels, dtype=object),
    )

    vectorizer_path = encoder.save(config.vectorizer_path)

    logger.info(
        "Dumped TF-IDF embeddings",
        samples=len(requests),
        features=embeddings.shape[1],
        embeddings_path=str(config.embeddings_path),
        vectorizer=str(vectorizer_path),
    )

    return config.embeddings_path


def train_anomaly_from_embeddings(
    config: TFIDFAnomalyTrainingConfig,
) -> "EllipticEnvelopeDetector":
    """Train an anomaly detector using precomputed TF-IDF embeddings."""

    from tqdm.auto import tqdm

    sink, wandb_run = init_wandb_sink(
        config.wandb_enabled,
        project=config.wandb_project,
        entity=config.wandb_entity,
        config=config.model_dump(),
    )

    payload = np.load(config.embeddings_path, allow_pickle=True)
    embeddings = payload["embeddings"].astype(np.float32)
    labels = payload.get("labels")

    if labels is None:
        payload.close()
        if wandb_run is not None:
            wandb_run.finish()
        raise ValueError("Embeddings artifact missing 'labels' array")

    label_array = np.asarray(labels)
    labels_list = label_array.tolist()
    total_count = len(labels_list)
    valid_mask = np.zeros(total_count, dtype=bool)
    valid_label_lower = config.valid_label.lower()

    if config.show_progress:
        with tqdm(
            labels_list,
            desc="Filtering valid samples",
            unit="sample",
            total=total_count,
        ) as progress:
            for idx, label in enumerate(progress):
                normalized = (label or "").lower()
                valid_mask[idx] = normalized == valid_label_lower
    else:
        for idx, label in enumerate(labels_list):
            normalized = (label or "").lower()
            valid_mask[idx] = normalized == valid_label_lower

    valid_embeddings = embeddings[valid_mask]
    valid_count = int(valid_mask.sum())

    if valid_embeddings.size == 0:
        payload.close()
        if wandb_run is not None:
            wandb_run.finish()
        raise RuntimeError(
            "No embeddings matched valid label "
            f"'{config.valid_label}'. Cannot train detector."
        )

    fit_start = perf_counter()

    # Choose detector based on config
    if config.detector_type == "isolation_forest":
        from neuralshield.anomaly import IsolationForestDetector

        detector = IsolationForestDetector(
            contamination=config.contamination,
            n_estimators=config.n_estimators,
            random_state=config.random_state,
        )
        logger.info(
            "Fitting IsolationForest",
            contamination=config.contamination,
            n_estimators=config.n_estimators,
        )
    else:  # mahalanobis (formerly elliptic_envelope)
        from neuralshield.anomaly import MahalanobisDetector

        detector = MahalanobisDetector()
        logger.info("Fitting Mahalanobis distance detector")

    detector.fit(valid_embeddings)
    fit_seconds = perf_counter() - fit_start

    payload.close()

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(str(config.model_path))

    threshold = detector.threshold_
    feature_count = int(valid_embeddings.shape[1]) if valid_embeddings.size else 0

    metrics = {
        "train/total_samples": float(total_count),
        "train/valid_samples": float(valid_count),
        "train/feature_count": float(feature_count),
        "train/fit_seconds": float(fit_seconds),
        "model/contamination": float(config.contamination),
        "model/threshold": float(threshold),
    }

    if sink is not None:
        sink.log(metrics)

    logger.info(
        "Trained anomaly detector from TF-IDF embeddings",
        samples=valid_count,
        features=feature_count,
        model=str(config.model_path),
        fit_seconds=fit_seconds,
        threshold=threshold,
    )

    if wandb_run is not None:
        wandb_run.finish()

    return detector


app = typer.Typer(help="TF-IDF utilities for embedding dumps and anomaly training.")


@app.command()
def dump(
    dataset: Path = typer.Argument(..., help="Path to the labeled dataset"),
    embeddings: Path = typer.Argument(..., help="Output .npz file for embeddings"),
    vectorizer: Path = typer.Argument(..., help="Output .joblib file for vectorizer"),
    batch_size: int = typer.Option(512, help="Requests per reader batch"),
    reader: str = typer.Option("jsonl", help="Dataset reader"),
    use_pipeline: bool = typer.Option(False, help="Enable preprocessing pipeline"),
    pipeline: str = typer.Option(
        "", help="Pipeline name when preprocessing is enabled (blank=none)"
    ),
    max_features: int = typer.Option(
        5000, help="TF-IDF max vocabulary size (-1 disables limit)"
    ),
    min_df: float = typer.Option(1.0, help="Minimum document frequency"),
    max_df: float = typer.Option(1.0, help="Maximum document frequency"),
    ngram_min: int = typer.Option(1, help="Minimum n-gram"),
    ngram_max: int = typer.Option(1, help="Maximum n-gram"),
    lowercase: bool = typer.Option(True, help="Lowercase tokens"),
    stop_words: str = typer.Option(
        "", help="Stop words preset (blank disables, e.g. 'english')"
    ),
    use_idf: bool = typer.Option(True, help="Enable IDF weighting"),
    smooth_idf: bool = typer.Option(True, help="Apply IDF smoothing"),
    sublinear_tf: bool = typer.Option(False, help="Use sublinear TF scaling"),
    binary: bool = typer.Option(False, help="Use binary term frequencies"),
) -> None:
    if ngram_min > ngram_max:
        raise typer.BadParameter("ngram_min cannot exceed ngram_max")

    resolved_pipeline = pipeline or None
    resolved_max_features = None if max_features < 0 else max_features
    resolved_stop_words = stop_words or None
    resolved_min_df = int(min_df) if float(min_df).is_integer() else min_df
    resolved_max_df = int(max_df) if float(max_df).is_integer() else max_df

    tfidf_config = TFIDFEncoderConfig(
        max_features=resolved_max_features,
        min_df=resolved_min_df,
        max_df=resolved_max_df,
        ngram_range=(ngram_min, ngram_max),
        lowercase=lowercase,
        stop_words=resolved_stop_words,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
        binary=binary,
    )

    config = TFIDFEncodingConfig(
        dataset_path=dataset,
        embeddings_path=embeddings,
        vectorizer_path=vectorizer,
        batch_size=batch_size,
        reader_name=reader,
        use_pipeline=use_pipeline,
        pipeline_name=resolved_pipeline,
        tfidf=tfidf_config,
    )

    dump_tfidf_embeddings(config)


@app.command()
def train(
    embeddings: Path = typer.Argument(..., help="Input .npz embedding artifact"),
    model: Path = typer.Argument(..., help="Output path for anomaly detector"),
    valid_label: str = typer.Option("valid", help="Label treated as benign"),
    contamination: float = typer.Option(
        0.01, help="Expected fraction of anomalies in the training set"
    ),
    detector: str = typer.Option(
        "isolation_forest",
        help="Detector: 'isolation_forest' (fast, TF-IDF) or 'elliptic_envelope'",
    ),
    n_estimators: int = typer.Option(
        100, help="Number of trees for IsolationForest (ignored for elliptic_envelope)"
    ),
    random_state: int = typer.Option(-1, help="Random seed (-1 disables determinism)"),
    show_progress: bool = typer.Option(True, help="Display progress bar"),
    wandb: bool = typer.Option(False, "--wandb/--no-wandb", help="Enable W&B logging"),
    wandb_project: str = typer.Option("neuralshield", help="W&B project name"),
    wandb_entity: str = typer.Option("", help="W&B entity/team (empty string skips)"),
) -> None:
    resolved_random_state = None if random_state < 0 else random_state
    resolved_entity = wandb_entity or None

    config = TFIDFAnomalyTrainingConfig(
        embeddings_path=embeddings,
        model_path=model,
        valid_label=valid_label,
        contamination=contamination,
        detector_type=detector,
        n_estimators=n_estimators,
        random_state=resolved_random_state,
        show_progress=show_progress,
        wandb_enabled=wandb,
        wandb_project=wandb_project,
        wandb_entity=resolved_entity,
    )

    train_anomaly_from_embeddings(config)


@app.command()
def test(  # noqa: T201
    dataset: Path = typer.Argument(..., help="Test dataset (JSONL)"),
    vectorizer: Path = typer.Argument(..., help="Trained TF-IDF vectorizer (.joblib)"),
    model: Path = typer.Argument(..., help="Trained anomaly detector (.joblib)"),
    batch_size: int = typer.Option(512, help="Batch size for processing"),
    reader: str = typer.Option("jsonl", help="Dataset reader"),
    use_pipeline: bool = typer.Option(False, help="Enable preprocessing pipeline"),
    pipeline: str = typer.Option("", help="Pipeline name (blank=none)"),
    attack_label: str = typer.Option("attack", help="Label for attacks"),
    valid_label: str = typer.Option("valid", help="Label for normal traffic"),
    output: str = typer.Option("", help="Output JSONL file for predictions (optional)"),
) -> None:
    """Test a trained anomaly detector on a test dataset with evaluation metrics."""
    from neuralshield.anomaly import IsolationForestDetector
    from neuralshield.encoding.models.tfidf import TFIDFEncoder
    from neuralshield.evaluation import ClassificationEvaluator, EvaluationConfig

    logger.info("Loading TF-IDF vectorizer from {path}", path=str(vectorizer))
    encoder = TFIDFEncoder(model_name=str(vectorizer))

    logger.info("Loading anomaly detector from {path}", path=str(model))
    detector = IsolationForestDetector.load(str(model))

    # Setup pipeline
    resolved_pipeline = pipeline if pipeline else None
    pipeline_runner = _resolve_pipeline(
        use_pipeline=use_pipeline, pipeline_name=resolved_pipeline
    )

    # Setup reader
    reader_factory = data_factory.get_reader(reader)
    data_reader = reader_factory(
        path=dataset,
        pipeline=pipeline_runner,
        use_pipeline=use_pipeline,
        observer=None,
    )

    logger.info("Processing test dataset from {path}", path=str(dataset))

    all_predictions = []
    all_labels = []
    all_scores = []
    total_processed = 0

    from tqdm import tqdm

    # Process batches
    for batch_requests, batch_labels in tqdm(
        data_reader.iter_batches(batch_size),
        desc="Testing",
        unit="batch",
    ):
        # Encode requests
        embeddings = encoder.encode(batch_requests)

        # Get predictions and scores
        scores = detector.scores(embeddings)
        predictions = detector.predict(embeddings)

        all_predictions.extend(predictions.tolist())
        all_labels.extend(batch_labels)
        all_scores.extend(scores.tolist())
        total_processed += len(batch_requests)

    logger.info("Processed {count} samples", count=total_processed)

    # Evaluate
    config = EvaluationConfig(positive_label=attack_label, negative_label=valid_label)
    evaluator = ClassificationEvaluator(config)
    result = evaluator.evaluate(all_predictions, all_labels)

    # Display results  # noqa: T201
    print("\n" + "=" * 60)  # noqa: T201
    print("ANOMALY DETECTION TEST RESULTS")  # noqa: T201
    print("=" * 60)  # noqa: T201
    print(f"\nDataset: {dataset}")
    print(f"Total Samples: {result.total_samples}")
    print(f"  Attacks: {result.positive_samples}")
    print(f"  Normal:  {result.negative_samples}")

    print("\nConfusion Matrix:")
    print(f"  True Positives (TP):  {result.tp:6d}  (Attacks detected)")
    print(f"  False Positives (FP): {result.fp:6d}  (False alarms)")
    print(f"  True Negatives (TN):  {result.tn:6d}  (Normal passed)")
    print(f"  False Negatives (FN): {result.fn:6d}  (Attacks missed)")

    print("\nClassification Metrics:")
    print(f"  Precision:  {result.precision:6.2%}  (Detections correct)")
    print(f"  Recall:     {result.recall:6.2%}  (Of attacks, how many caught?)")
    print(f"  F1-Score:   {result.f1_score:6.2%}  (Harmonic mean)")
    print(f"  Accuracy:   {result.accuracy:6.2%}  (Overall correctness)")

    print("\nAnomaly Detection Metrics:")
    print(f"  FPR:         {result.fpr:6.2%}  (False alarm rate)")
    print(f"  Specificity: {result.specificity:6.2%}  (Normal traffic passed)")

    print("\n" + "=" * 60)

    # Save predictions if requested
    if output:
        import json

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            for pred, label, score in zip(all_predictions, all_labels, all_scores):
                f.write(
                    json.dumps(
                        {
                            "prediction": bool(pred),
                            "label": label,
                            "score": float(score),
                        }
                    )
                    + "\n"
                )
        logger.info("Saved predictions to {path}", path=str(output_path))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
