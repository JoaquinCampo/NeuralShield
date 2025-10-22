#!/usr/bin/env python3
"""Regenerate score-distribution plots for selected experiments."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    embeddings: Path
    model: Path
    output_dir: Path


ROOT = Path(__file__).resolve().parents[1]

CONFIGS: list[ExperimentConfig] = [
    # Experiment 01: TF-IDF + IsolationForest
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/01_tfidf_preprocessing_comparison/without_preprocessing/csic_test_embeddings.npz",
        model=ROOT
        / "experiments/01_tfidf_preprocessing_comparison/without_preprocessing/csic_model.joblib",
        output_dir=ROOT / "experiments/01_tfidf_preprocessing_comparison/without_preprocessing",
    ),
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/01_tfidf_preprocessing_comparison/with_preprocessing/csic_test_embeddings.npz",
        model=ROOT
        / "experiments/01_tfidf_preprocessing_comparison/with_preprocessing/csic_model.joblib",
        output_dir=ROOT / "experiments/01_tfidf_preprocessing_comparison/with_preprocessing",
    ),
    # Experiment 10: TF-IDF+PCA + Mahalanobis
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/10_tfidf_pca_mahalanobis/without_preprocessing/test_embeddings.npz",
        model=ROOT
        / "experiments/10_tfidf_pca_mahalanobis/without_preprocessing/tfidf_pca_mahalanobis_model.joblib",
        output_dir=ROOT / "experiments/10_tfidf_pca_mahalanobis/without_preprocessing",
    ),
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/10_tfidf_pca_mahalanobis/with_preprocessing/test_embeddings.npz",
        model=ROOT
        / "experiments/10_tfidf_pca_mahalanobis/with_preprocessing/tfidf_pca_mahalanobis_model.joblib",
        output_dir=ROOT / "experiments/10_tfidf_pca_mahalanobis/with_preprocessing",
    ),
    # Experiment 15: TF-IDF+PCA + LOF
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/15_lof_comparison/tfidf_pca_150/csic_test_embeddings.npz",
        model=ROOT
        / "experiments/15_lof_comparison/tfidf_pca_150/lof_tfidf_pca150_k100.joblib",
        output_dir=ROOT / "experiments/15_lof_comparison/tfidf_pca_150",
    ),
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/15_lof_comparison/tfidf_pca_150_no_prep/csic_test_embeddings.npz",
        model=ROOT
        / "experiments/15_lof_comparison/tfidf_pca_150_no_prep/lof_tfidf_pca150_k100_no_prep.joblib",
        output_dir=ROOT / "experiments/15_lof_comparison/tfidf_pca_150_no_prep",
    ),
    # Experiment 02: BGE-small + IsolationForest
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/02_dense_embeddings_comparison/dense_with_preprocessing/csic_test_embeddings.npz",
        model=ROOT
        / "experiments/02_dense_embeddings_comparison/dense_with_preprocessing/csic_best_model.joblib",
        output_dir=ROOT / "experiments/02_dense_embeddings_comparison/dense_with_preprocessing",
    ),
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/02_dense_embeddings_comparison/dense_without_preprocessing/csic_test_embeddings.npz",
        model=ROOT
        / "experiments/02_dense_embeddings_comparison/dense_without_preprocessing/csic_best_model.joblib",
        output_dir=ROOT / "experiments/02_dense_embeddings_comparison/dense_without_preprocessing",
    ),
    # Experiment 06: BGE-small + Mahalanobis
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/06_mahalanobis_comparison/with_preprocessing/test_embeddings.npz",
        model=ROOT
        / "experiments/06_mahalanobis_comparison/with_preprocessing/csic_mahalanobis_model.joblib",
        output_dir=ROOT / "experiments/06_mahalanobis_comparison/with_preprocessing",
    ),
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/06_mahalanobis_comparison/without_preprocessing/test_embeddings.npz",
        model=ROOT
        / "experiments/06_mahalanobis_comparison/without_preprocessing/csic_mahalanobis_model.joblib",
        output_dir=ROOT / "experiments/06_mahalanobis_comparison/without_preprocessing",
    ),
    # Experiment 05: ByT5 + IsolationForest
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/05_byt5_comparison/with_preprocessing/csic_test_embeddings.npz",
        model=ROOT
        / "experiments/05_byt5_comparison/with_preprocessing/csic_best_model.joblib",
        output_dir=ROOT / "experiments/05_byt5_comparison/with_preprocessing",
    ),
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/05_byt5_comparison/without_preprocessing/csic_test_embeddings.npz",
        model=ROOT
        / "experiments/05_byt5_comparison/without_preprocessing/csic_best_model.joblib",
        output_dir=ROOT / "experiments/05_byt5_comparison/without_preprocessing",
    ),
    # Experiment 03: SecBERT comparisons
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/03_secbert_comparison/secbert_with_preprocessing/csic_test_embeddings_converted.npz",
        model=ROOT
        / "experiments/03_secbert_comparison/secbert_with_preprocessing/csic_best_model.joblib",
        output_dir=ROOT
        / "experiments/03_secbert_comparison/secbert_with_preprocessing",
    ),
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/03_secbert_comparison/secbert_without_preprocessing/csic_test_embeddings_converted.npz",
        model=ROOT
        / "experiments/03_secbert_comparison/secbert_without_preprocessing/csic_best_model.joblib",
        output_dir=ROOT
        / "experiments/03_secbert_comparison/secbert_without_preprocessing",
    ),
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/03_secbert_comparison/secbert_mahalanobis_with_preprocessing/csic_test_embeddings_converted.npz",
        model=ROOT
        / "experiments/03_secbert_comparison/secbert_mahalanobis_with_preprocessing/csic_mahalanobis_model.joblib",
        output_dir=ROOT
        / "experiments/03_secbert_comparison/secbert_mahalanobis_with_preprocessing",
    ),
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/03_secbert_comparison/secbert_mahalanobis_without_preprocessing/csic_test_embeddings_converted.npz",
        model=ROOT
        / "experiments/03_secbert_comparison/secbert_mahalanobis_without_preprocessing/csic_mahalanobis_model.joblib",
        output_dir=ROOT
        / "experiments/03_secbert_comparison/secbert_mahalanobis_without_preprocessing",
    ),
    # Experiment 17: SecBERT + GMM tuned variants
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/17_gmm_secbert/with_preprocessing_n5_full/test_embeddings.npz",
        model=ROOT
        / "experiments/17_gmm_secbert/with_preprocessing_n5_full/gmm_detector.joblib",
        output_dir=ROOT / "experiments/17_gmm_secbert/with_preprocessing_n5_full",
    ),
    ExperimentConfig(
        embeddings=ROOT
        / "experiments/17_gmm_secbert/without_preprocessing_n5_full/test_embeddings.npz",
        model=ROOT
        / "experiments/17_gmm_secbert/without_preprocessing_n5_full/gmm_detector.joblib",
        output_dir=ROOT / "experiments/17_gmm_secbert/without_preprocessing_n5_full",
    ),
]


def main() -> None:
    mpl_dir = ROOT / ".matplotlib"
    cache_dir = ROOT / ".cache"
    mpl_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(mpl_dir))
    env.setdefault("XDG_CACHE_HOME", str(cache_dir))

    for cfg in CONFIGS:
        embeddings = cfg.embeddings
        model = cfg.model
        output_dir = cfg.output_dir

        if not embeddings.exists():
            print(f"[skip] embeddings missing: {embeddings}", file=sys.stderr)
            continue
        if not model.exists():
            print(f"[skip] model missing: {model}", file=sys.stderr)
            continue

        metrics_out = output_dir / "roc_metrics.json"
        curve_out = output_dir / "roc_curve.csv"
        scores_out = output_dir / "prediction_scores.csv"

        cmd = [
            sys.executable,
            "src/scripts/test_anomaly_precomputed.py",
            str(embeddings),
            str(model),
            "--metrics-out",
            str(metrics_out),
            "--curve-out",
            str(curve_out),
            "--scores-out",
            str(scores_out),
        ]

        print(f"[run] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
