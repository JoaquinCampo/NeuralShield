from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from optuna.samplers import TPESampler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from neuralshield.anomaly import LOFDetector, MahalanobisDetector
from neuralshield.encoding.data.jsonl import JSONLRequestReader
from neuralshield.evaluation import ClassificationEvaluator, EvaluationConfig
from neuralshield.preprocessing.pipeline import preprocess
from neuralshield.preprocessing.steps.exceptions import MalformedHttpRequestError


def _parse_int_list(values: str) -> list[int]:
    return [int(item.strip()) for item in values.split(",") if item.strip()]


def _parse_float_or_auto(values: str) -> list[float | str]:
    parsed: list[float | str] = []
    for token in values.split(","):
        item = token.strip()
        if not item:
            continue
        if item.lower() == "auto":
            parsed.append("auto")
        else:
            parsed.append(float(item))
    return parsed


def _z_normalize(scores: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(scores.mean())
    std = float(scores.std())
    if std == 0.0:
        std = 1e-6
    normalized = (scores - mean) / std
    return normalized.astype(np.float32), mean, std


def _apply_z(scores: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std == 0.0:
        std = 1e-6
    return ((scores - mean) / std).astype(np.float32)


def _load_texts_and_labels(path: Path) -> tuple[list[str], list[str]]:
    reader = JSONLRequestReader(path, use_pipeline=False)
    texts: list[str] = []
    labels: list[str] = []
    for batch, batch_labels in reader.iter_batches(batch_size=2000):
        for item, label in zip(batch, batch_labels):
            try:
                processed = preprocess(item)
            except MalformedHttpRequestError:
                logger.debug("Skipping malformed request")
                continue
            texts.append(processed)
            labels.append(label)
    return texts, labels


def _evaluate(
    predictions: Iterable[bool],
    labels: Iterable[str],
) -> dict[str, float | int]:
    evaluator = ClassificationEvaluator(EvaluationConfig())
    result = evaluator.evaluate(list(predictions), list(labels))
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


def objective(
    trial: optuna.Trial,
    train_tfidf: np.ndarray,
    test_tfidf: np.ndarray,
    maha_predictions: np.ndarray,
    maha_scores: np.ndarray,
    test_labels: list[str],
    max_fpr: float,
) -> float:
    """Optuna objective function for hyperparameter optimization."""

    # Suggest hyperparameters with informed priors based on previous experiments
    pca_components = trial.suggest_categorical("pca_components", [150, 175, 200, 225])
    lof_neighbors = trial.suggest_categorical("lof_neighbors", [75, 100, 125, 150])
    lof_contamination = trial.suggest_categorical(
        "lof_contamination", ["auto", 0.01, 0.02]
    )
    fusion_weight = trial.suggest_float("fusion_weight", 0.3, 0.8, step=0.1)

    # PCA transformation
    svd = TruncatedSVD(n_components=pca_components, random_state=42)
    train_embeddings = svd.fit_transform(train_tfidf).astype(np.float32)
    test_embeddings = svd.transform(test_tfidf).astype(np.float32)

    # Train LOF
    lof = LOFDetector(n_neighbors=lof_neighbors, contamination=lof_contamination)
    lof.fit(train_embeddings)
    lof_threshold = lof.set_threshold(train_embeddings, max_fpr=max_fpr)

    # Get scores and predictions
    lof_test_scores = lof.scores(test_embeddings)
    lof_predictions = lof.predict(test_embeddings)

    # Z-normalize scores for fusion
    lof_norm_train, lof_mean, lof_std = _z_normalize(lof.scores(train_embeddings))
    mah_norm_train, mah_mean, mah_std = _z_normalize(maha_scores)

    lof_norm_test = _apply_z(lof_test_scores, lof_mean, lof_std)
    mah_norm_test = _apply_z(maha_scores, mah_mean, mah_std)

    # Fusion
    fused_scores = fusion_weight * lof_norm_test + (1 - fusion_weight) * mah_norm_test
    fused_threshold = float(np.quantile(fused_scores, 1.0 - max_fpr))
    fused_predictions = fused_scores >= fused_threshold

    # Evaluate
    fused_metrics = _evaluate(fused_predictions, test_labels)

    # Store metrics for later analysis
    trial.set_user_attr("recall", fused_metrics["recall"])
    trial.set_user_attr("fpr", fused_metrics["fpr"])
    trial.set_user_attr("f1", fused_metrics["f1_score"])

    # Objective: maximize recall while constraining FPR
    if fused_metrics["fpr"] <= max_fpr:
        return fused_metrics["recall"]
    else:
        # Penalize FPR violations heavily
        return fused_metrics["recall"] - 2 * (fused_metrics["fpr"] - max_fpr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bayesian hyperparameter optimization for SR_BH TF-IDF -> PCA -> LOF with Mahalanobis fusion."
        )
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("src/neuralshield/data/SR_BH_2020/train.jsonl"),
        help="SR_BH train JSONL path.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("src/neuralshield/data/SR_BH_2020/test.jsonl"),
        help="SR_BH test JSONL path.",
    )
    parser.add_argument(
        "--secbert-train",
        type=Path,
        default=Path(
            "experiments/08_secbert_srbh/with_preprocessing/train_embeddings.npz"
        ),
        help="SecBERT train embeddings NPZ.",
    )
    parser.add_argument(
        "--secbert-test",
        type=Path,
        default=Path(
            "experiments/08_secbert_srbh/with_preprocessing/test_embeddings.npz"
        ),
        help="SecBERT test embeddings NPZ.",
    )
    parser.add_argument(
        "--max-fpr",
        type=float,
        default=0.05,
        help="Target FPR for threshold calibration.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default 50).",
    )
    parser.add_argument(
        "--optimization-mode",
        type=str,
        choices=["recall", "f1"],
        default="recall",
        help="Optimization objective: 'recall' or 'f1'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/18_lof_secbert_ensemble/srbh_optuna"),
        help="Directory for optimization outputs.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SR_BH datasets")
    train_texts, _ = _load_texts_and_labels(args.train_path)
    test_texts, test_labels = _load_texts_and_labels(args.test_path)
    logger.info(
        "Loaded train samples={train_count}, test samples={test_count}",
        train_count=len(train_texts),
        test_count=len(test_texts),
    )

    logger.info("Fitting TF-IDF vectorizer (5000 dims, ngram 1-3)")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)
    logger.info("TF-IDF matrices train=%s test=%s", train_tfidf.shape, test_tfidf.shape)

    logger.info("Loading SecBERT embeddings")
    sec_train_npz = np.load(args.secbert_train)
    secbert_train = sec_train_npz["embeddings"].astype(np.float32)

    sec_test_npz = np.load(args.secbert_test, allow_pickle=True)
    secbert_test = sec_test_npz["embeddings"].astype(np.float32)
    secbert_labels = sec_test_npz["labels"].astype(str)

    if not np.array_equal(np.array(test_labels), secbert_labels):
        raise ValueError(
            "Label mismatch between TF-IDF test data and SecBERT embeddings"
        )

    logger.info("Training Mahalanobis on SecBERT train embeddings")
    maha = MahalanobisDetector()
    maha.fit(secbert_train)
    maha.set_threshold(secbert_train, max_fpr=args.max_fpr)
    maha_train_scores = maha.scores(secbert_train)
    maha_test_scores = maha.scores(secbert_test)
    maha_predictions = maha.predict(secbert_test)
    maha_metrics = _evaluate(maha_predictions, secbert_labels)
    logger.info(
        "Mahalanobis: recall={recall:.3f} fpr={fpr:.3f}",
        recall=maha_metrics["recall"],
        fpr=maha_metrics["fpr"],
    )

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name="lof_secbert_ensemble",
    )

    logger.info(f"Starting Bayesian optimization with {args.n_trials} trials...")

    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial,
            train_tfidf,
            test_tfidf,
            maha_predictions,
            maha_test_scores,
            test_labels,
            args.max_fpr,
        ),
        n_trials=args.n_trials,
        timeout=3600,  # 1 hour timeout
        show_progress_bar=True,
    )

    logger.info("Optimization complete!")

    # Extract results from all trials
    results = []
    for trial in study.trials:
        if trial.state == optuna.TrialState.COMPLETE:
            results.append(
                {
                    "trial": trial.number,
                    "pca_components": trial.params["pca_components"],
                    "lof_neighbors": trial.params["lof_neighbors"],
                    "lof_contamination": trial.params["lof_contamination"],
                    "fusion_weight": trial.params["fusion_weight"],
                    "recall": trial.user_attrs["recall"],
                    "fpr": trial.user_attrs["fpr"],
                    "f1": trial.user_attrs["f1"],
                    "objective_value": trial.value,
                }
            )

    # Find best configuration
    best_trial = study.best_trial
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best objective value: {best_trial.value:.4f}")
    logger.info(f"Best hyperparameters: {best_trial.params}")
    logger.info(
        f"Best metrics: recall={best_trial.user_attrs['recall']:.4f}, "
        f"fpr={best_trial.user_attrs['fpr']:.4f}, "
        f"f1={best_trial.user_attrs['f1']:.4f}"
    )

    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_dir / "optuna_results.csv", index=False)

    # Save study
    with open(args.output_dir / "optuna_study.pkl", "wb") as f:
        joblib.dump(study, f)

    # Retrain best model for deployment
    logger.info("Retraining best model for deployment...")

    best_params = best_trial.params
    svd = TruncatedSVD(n_components=best_params["pca_components"], random_state=42)
    best_train_embeddings = svd.fit_transform(train_tfidf).astype(np.float32)
    best_test_embeddings = svd.transform(test_tfidf).astype(np.float32)

    best_lof = LOFDetector(
        n_neighbors=best_params["lof_neighbors"],
        contamination=best_params["lof_contamination"],
    )
    best_lof.fit(best_train_embeddings)
    best_threshold = best_lof.set_threshold(best_train_embeddings, max_fpr=args.max_fpr)

    # Create ensemble model bundle
    bundle = {
        "vectorizer": vectorizer,
        "pca": svd,
        "lof_detector": best_lof,
        "mahalanobis_detector": maha,
        "fusion_weight": best_params["fusion_weight"],
        "threshold": best_threshold,
        "config": best_params,
        "performance": {
            "recall": best_trial.user_attrs["recall"],
            "fpr": best_trial.user_attrs["fpr"],
            "f1": best_trial.user_attrs["f1"],
        },
    }

    bundle_path = args.output_dir / "best_ensemble_bundle.joblib"
    joblib.dump(bundle, bundle_path)
    logger.info(f"Saved best ensemble bundle to {bundle_path}")

    # Save embeddings for analysis
    train_labels = np.array(["valid"] * best_train_embeddings.shape[0])
    np.savez(
        args.output_dir / "best_train_embeddings.npz",
        embeddings=best_train_embeddings,
        labels=train_labels,
    )
    np.savez(
        args.output_dir / "best_test_embeddings.npz",
        embeddings=best_test_embeddings,
        labels=np.array(test_labels),
    )

    # Save optimization summary
    summary = {
        "optimization": {
            "n_trials": args.n_trials,
            "max_fpr": args.max_fpr,
            "objective": args.optimization_mode,
            "best_trial": best_trial.number,
            "best_value": best_trial.value,
        },
        "best_config": best_params,
        "best_metrics": best_trial.user_attrs,
        "study_stats": {
            "total_trials": len(study.trials),
            "completed_trials": len(
                [t for t in study.trials if t.state == optuna.TrialState.COMPLETE]
            ),
            "pruned_trials": len(
                [t for t in study.trials if t.state == optuna.TrialState.PRUNED]
            ),
        },
    }

    with open(args.output_dir / "optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Optimization summary saved!")
    logger.info(f"Results available in: {args.output_dir}")


if __name__ == "__main__":
    main()
