#!/usr/bin/env python3
"""
Experiment 28: Ensemble Optimization with Beautiful WandB Visualizations

Optimizes ensemble of TF-IDF+PCA+LOF and SecBERT+Mahalanobis with:
- Weight optimization sweep
- Beautiful static visualizations
- Time-series logging to WandB
- Comprehensive metrics tracking
"""

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import typer
import wandb
from loguru import logger
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from neuralshield.anomaly import LOFDetector

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from visualizations import (
    create_agreement_venn,
    create_performance_comparison_bars,
    create_roc_comparison_plot,
    create_score_distribution_plot,
    create_weight_optimization_curves,
)

app = typer.Typer()


def load_embeddings(
    npz_path: Path, dataset_path: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings from NPZ file, adding labels from dataset if needed."""
    logger.info(f"Loading embeddings from {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    
    # Try to load labels from NPZ
    if "labels" in data:
        labels = data["labels"]
        logger.info("Labels found in NPZ file")
    elif dataset_path:
        # Load labels from original dataset
        logger.info(f"Loading labels from dataset: {dataset_path}")
        labels = []
        import json
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                labels.append(obj["label"])
        labels = np.array(labels)
        logger.info("Labels loaded from dataset")
    else:
        raise ValueError(
            f"NPZ file {npz_path} has no 'labels' key and no dataset_path provided"
        )
    
    logger.info(
        f"Loaded {embeddings.shape[0]} samples, {embeddings.shape[1]} dimensions"
    )
    return embeddings, labels


def load_lof_model(model_path: Path) -> tuple[LOFDetector, Any]:
    """Load LOF model and metadata."""
    logger.info(f"Loading LOF model from {model_path}")
    model_data = joblib.load(model_path)
    if isinstance(model_data, dict):
        detector = model_data.get("detector") or model_data.get("model")
        metadata = model_data.get("metadata", {})
    else:
        detector = model_data
        metadata = {}
    return detector, metadata


def load_mahalanobis_model(model_path: Path) -> EmpiricalCovariance:
    """Load Mahalanobis detector."""
    logger.info(f"Loading Mahalanobis model from {model_path}")
    model_data = joblib.load(model_path)
    if isinstance(model_data, dict):
        detector = model_data.get("detector") or model_data.get("model")
    else:
        detector = model_data
    return detector


def compute_metrics(
    predictions: np.ndarray, labels: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    """Compute classification metrics."""
    binary_labels = (np.array(labels) == "attack").astype(int)
    binary_preds = predictions.astype(int)

    tp = int(np.sum((binary_preds == 1) & (binary_labels == 1)))
    fp = int(np.sum((binary_preds == 1) & (binary_labels == 0)))
    tn = int(np.sum((binary_preds == 0) & (binary_labels == 0)))
    fn = int(np.sum((binary_preds == 0) & (binary_labels == 1)))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    roc_auc = roc_auc_score(binary_labels, scores)

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
        "roc_auc": roc_auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def standardize_scores(scores: np.ndarray, normal_mask: np.ndarray) -> np.ndarray:
    """Z-score standardize scores using normal samples."""
    normal_scores = scores[normal_mask]
    mean = normal_scores.mean()
    std = normal_scores.std()
    return (scores - mean) / (std + 1e-8)


@app.command()
def main(
    # LOF embeddings and model
    lof_train_embeddings: Path = typer.Option(
        Path("experiments/28_ensemble_optimization/csic_lof_train_embeddings.npz"),
        help="LOF training embeddings (.npz) with labels",
    ),
    lof_test_embeddings: Path = typer.Option(
        Path("experiments/28_ensemble_optimization/csic_lof_test_embeddings.npz"),
        help="LOF test embeddings (.npz) with labels",
    ),
    lof_model: Path = typer.Option(
        Path("experiments/26_tfidf_pca_lof_csic/lof_tfidf_pca200_k15.joblib"),
        help="Trained LOF model (.joblib)",
    ),
    # SecBERT embeddings and model
    secbert_train_embeddings: Path = typer.Option(
        Path("embeddings/SecBert/train_embeddings_compact.npz"),
        help="SecBERT training embeddings (.npz)",
    ),
    secbert_test_embeddings: Path = typer.Option(
        Path("embeddings/SecBert/test_embeddings_compact.npz"),
        help="SecBERT test embeddings (.npz)",
    ),
    secbert_model: Path = typer.Option(
        Path("experiments/03_secbert_comparison/secbert_mahalanobis_with_preprocessing/csic_mahalanobis_model.joblib"),
        help="Trained Mahalanobis model (.joblib)",
    ),
    # Experiment parameters
    max_fpr: float = typer.Option(0.05, help="Target false positive rate"),
    weight_steps: int = typer.Option(21, help="Number of weight values to test"),
    standardize: bool = typer.Option(True, help="Standardize scores before fusion"),
    output_dir: Path = typer.Option(
        Path("experiments/28_ensemble_optimization/results"),
        help="Output directory for results",
    ),
    wandb_project: str = typer.Option(
        "neuralshield-v2", help="WandB project name"
    ),
    wandb_run_name: str = typer.Option(
        None, help="WandB run name (auto-generated if None)"
    ),
) -> None:
    """Run ensemble optimization experiment with beautiful WandB logging."""
    logger.info("=" * 80)
    logger.info("ENSEMBLE OPTIMIZATION EXPERIMENT")
    logger.info("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    wandb.init(
        project=wandb_project,
        name=wandb_run_name or "ensemble-optimization-csic",
        tags=["ensemble", "optimization", "lof", "secbert", "csic"],
        config={
            "max_fpr": max_fpr,
            "weight_steps": weight_steps,
            "standardize": standardize,
            "lof_train": str(lof_train_embeddings),
            "lof_test": str(lof_test_embeddings),
            "secbert_train": str(secbert_train_embeddings),
            "secbert_test": str(secbert_test_embeddings),
        },
    )

    # Load embeddings (with fallback to dataset for labels)
    logger.info("Loading embeddings...")
    csic_train_path = Path("src/neuralshield/data/CSIC/train.jsonl")
    csic_test_path = Path("src/neuralshield/data/CSIC/test.jsonl")
    
    lof_train_emb, lof_train_labels = load_embeddings(
        lof_train_embeddings, csic_train_path
    )
    lof_test_emb, lof_test_labels = load_embeddings(
        lof_test_embeddings, csic_test_path
    )
    secbert_train_emb, secbert_train_labels = load_embeddings(
        secbert_train_embeddings, csic_train_path
    )
    secbert_test_emb, secbert_test_labels = load_embeddings(
        secbert_test_embeddings, csic_test_path
    )

    # Ensure labels match
    assert np.array_equal(lof_test_labels, secbert_test_labels), "Test labels must match!"

    test_labels = lof_test_labels
    normal_mask = np.array(test_labels) == "valid"
    attack_mask = np.array(test_labels) == "attack"

    # Load models and compute scores
    logger.info("Computing component model scores...")
    
    # Load or train LOF detector
    if lof_model.exists():
        lof_detector, _ = load_lof_model(lof_model)
    else:
        logger.warning(f"LOF model not found at {lof_model}, training from embeddings...")
        from neuralshield.anomaly import LOFDetector
        lof_detector = LOFDetector(n_neighbors=100, contamination=0.05)
        lof_detector.fit(lof_train_emb.astype(np.float32))
        logger.info("LOF detector trained successfully")
    
    lof_scores = lof_detector.scores(lof_test_emb.astype(np.float32))
    lof_threshold = np.percentile(lof_scores[normal_mask], 100 * (1 - max_fpr))
    lof_predictions = (lof_scores > lof_threshold).astype(int)

    # Load or train Mahalanobis detector
    if secbert_model.exists():
        maha_detector = load_mahalanobis_model(secbert_model)
        # Check if dimensions match
        if hasattr(maha_detector, "location_") and maha_detector.location_.shape[0] != secbert_test_emb.shape[1]:
            logger.warning(
                f"Mahalanobis model expects {maha_detector.location_.shape[0]} dims, "
                f"but embeddings have {secbert_test_emb.shape[1]} dims. Training new model..."
            )
            from sklearn.covariance import EmpiricalCovariance
            maha_detector = EmpiricalCovariance()
            maha_detector.fit(secbert_train_emb)
            logger.info("Mahalanobis detector trained successfully")
    else:
        logger.warning(f"Mahalanobis model not found at {secbert_model}, training from embeddings...")
        from sklearn.covariance import EmpiricalCovariance
        maha_detector = EmpiricalCovariance()
        maha_detector.fit(secbert_train_emb)
        logger.info("Mahalanobis detector trained successfully")
    
    secbert_scores = maha_detector.mahalanobis(secbert_test_emb)
    secbert_threshold = np.percentile(secbert_scores[normal_mask], 100 * (1 - max_fpr))
    secbert_predictions = (secbert_scores > secbert_threshold).astype(int)

    # Compute component metrics
    lof_metrics = compute_metrics(lof_predictions, test_labels, lof_scores)
    secbert_metrics = compute_metrics(
        secbert_predictions, test_labels, secbert_scores
    )

    logger.info(f"LOF: Recall={lof_metrics['recall']:.2%}, F1={lof_metrics['f1']:.2%}")
    logger.info(
        f"SecBERT: Recall={secbert_metrics['recall']:.2%}, F1={secbert_metrics['f1']:.2%}"
    )

    # Weight optimization sweep
    logger.info("=" * 80)
    logger.info("WEIGHT OPTIMIZATION SWEEP")
    logger.info("=" * 80)

    weights = np.linspace(0.0, 1.0, weight_steps)
    weight_results = []

    # Standardize scores if requested
    if standardize:
        lof_scores_fusion = standardize_scores(lof_scores, normal_mask)
        secbert_scores_fusion = standardize_scores(secbert_scores, normal_mask)
    else:
        lof_scores_fusion = lof_scores
        secbert_scores_fusion = secbert_scores

    best_f1 = -np.inf
    best_weight = None
    best_ensemble_scores = None
    best_ensemble_metrics = None

    for step, weight in enumerate(weights):
        # Fuse scores
        fused_scores = weight * lof_scores_fusion + (1 - weight) * secbert_scores_fusion

        # Find threshold for target FPR
        threshold = np.percentile(fused_scores[normal_mask], 100 * (1 - max_fpr))
        predictions = (fused_scores > threshold).astype(int)

        # Compute metrics
        metrics = compute_metrics(predictions, test_labels, fused_scores)
        metrics["weight"] = weight
        metrics["threshold"] = threshold
        weight_results.append(metrics)

        # Log to WandB (time-series!)
        wandb.log(
            {
                "weight_optimization/step": step,
                "weight_optimization/weight": weight,
                "weight_optimization/recall": metrics["recall"],
                "weight_optimization/precision": metrics["precision"],
                "weight_optimization/f1_score": metrics["f1"],
                "weight_optimization/fpr": metrics["fpr"],
                "weight_optimization/roc_auc": metrics["roc_auc"],
                "weight_optimization/threshold": threshold,
            }
        )

        # Track best
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_weight = weight
            best_ensemble_scores = fused_scores.copy()
            best_ensemble_metrics = metrics.copy()

    logger.info(f"Best weight: {best_weight:.3f} (F1={best_f1:.3f})")

    # Extract arrays for visualization
    weight_recalls = [r["recall"] for r in weight_results]
    weight_precisions = [r["precision"] for r in weight_results]
    weight_f1s = [r["f1"] for r in weight_results]
    weight_fprs = [r["fpr"] for r in weight_results]
    best_weight_idx = np.argmax(weight_f1s)

    # Create visualizations
    logger.info("=" * 80)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("=" * 80)

    # Calculate ensemble threshold
    ensemble_threshold = np.percentile(best_ensemble_scores[normal_mask], 100 * (1 - max_fpr))
    
    # 1. Score distributions
    fig1 = create_score_distribution_plot(
        lof_scores[normal_mask],
        lof_scores[attack_mask],
        secbert_scores[normal_mask],
        secbert_scores[attack_mask],
        best_ensemble_scores[normal_mask],
        best_ensemble_scores[attack_mask],
        lof_threshold,
        secbert_threshold,
        ensemble_threshold,
    )
    wandb.log({"visualizations/score_distributions": wandb.Image(fig1)})
    plt.close(fig1)

    # 2. ROC curves
    fig2 = create_roc_comparison_plot(
        lof_scores, secbert_scores, best_ensemble_scores, test_labels
    )
    wandb.log({"visualizations/roc_curves": wandb.Image(fig2)})
    plt.close(fig2)

    # 3. Agreement Venn
    fig3 = create_agreement_venn(
        lof_predictions, secbert_predictions, test_labels
    )
    wandb.log({"visualizations/agreement_venn": wandb.Image(fig3)})
    plt.close(fig3)

    # 4. Performance bars
    fig4 = create_performance_comparison_bars(
        lof_metrics, secbert_metrics, best_ensemble_metrics
    )
    wandb.log({"visualizations/performance_comparison": wandb.Image(fig4)})
    plt.close(fig4)

    # 5. Weight optimization curves
    fig5 = create_weight_optimization_curves(
        weights, weight_recalls, weight_precisions, weight_f1s, weight_fprs, best_weight_idx
    )
    wandb.log({"visualizations/weight_optimization": wandb.Image(fig5)})
    plt.close(fig5)

    # Log summary metrics
    wandb.summary = {
        "best_weight": best_weight,
        "best_recall": best_ensemble_metrics["recall"],
        "best_precision": best_ensemble_metrics["precision"],
        "best_f1": best_ensemble_metrics["f1"],
        "best_fpr": best_ensemble_metrics["fpr"],
        "best_roc_auc": best_ensemble_metrics["roc_auc"],
        "lof_recall": lof_metrics["recall"],
        "secbert_recall": secbert_metrics["recall"],
        "improvement_vs_best_component": best_ensemble_metrics["recall"] - max(
            lof_metrics["recall"], secbert_metrics["recall"]
        ),
    }

    # Save results
    results = {
        "best_weight": float(best_weight),
        "best_metrics": best_ensemble_metrics,
        "lof_metrics": lof_metrics,
        "secbert_metrics": secbert_metrics,
        "weight_sweep": weight_results,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    wandb.finish()
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    app()

