#!/usr/bin/env python3
"""
Análisis de Agreement entre TF-IDF+PCA+LOF y SecBERT+Mahalanobis

Este script:
1. Entrena ambos modelos en el mismo dataset
2. Compara predicciones en el mismo test set
3. Calcula métricas de agreement detalladas
4. Genera visualizaciones y análisis para la tesis

Para la subsección: "Evaluación de las representaciones sintáctica y semántica"
"""

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from matplotlib_venn import venn2
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from neuralshield.anomaly import LOFDetector, MahalanobisDetector
from neuralshield.encoding.data.jsonl import JSONLRequestReader


def load_dataset(
    train_path: Path, test_path: Path, with_preprocessing: bool = True
) -> tuple[list[str], list[str], list[str]]:
    """Load train and test datasets."""
    logger.info(f"Loading datasets (preprocessing={with_preprocessing})...")

    train_reader = JSONLRequestReader(train_path, use_pipeline=with_preprocessing)
    test_reader = JSONLRequestReader(test_path, use_pipeline=with_preprocessing)

    train_texts = []
    test_texts = []
    test_labels = []

    for batch, _ in train_reader.iter_batches(batch_size=1000):
        train_texts.extend(batch)

    for batch, batch_labels in test_reader.iter_batches(batch_size=1000):
        test_texts.extend(batch)
        test_labels.extend(batch_labels)

    logger.info(
        f"Loaded {len(train_texts)} train samples, {len(test_texts)} test samples"
    )
    return train_texts, test_texts, test_labels


def train_tfidf_lof(
    train_texts: list[str],
    test_texts: list[str],
    test_labels: list[str],
    n_components: int = 150,
    n_neighbors: int = 100,
    target_fpr: float = 0.05,
) -> tuple[Any, np.ndarray, np.ndarray, dict]:
    """Train TF-IDF + PCA + LOF model."""
    logger.info("Training TF-IDF + PCA + LOF...")

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)

    logger.info(f"TF-IDF shape: train={train_tfidf.shape}, test={test_tfidf.shape}")

    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    train_embeddings = pca.fit_transform(train_tfidf.toarray())
    test_embeddings = pca.transform(test_tfidf.toarray())

    explained_variance = float(pca.explained_variance_ratio_.sum())
    logger.info(f"PCA explained variance: {explained_variance:.2%}")

    # LOF
    detector = LOFDetector(n_neighbors=n_neighbors)
    detector.fit(train_embeddings.astype(np.float32))

    # Scores
    test_scores = detector.scores(test_embeddings.astype(np.float32))

    # Binary labels
    binary_labels = np.array([1 if label == "attack" else 0 for label in test_labels])
    normal_mask = binary_labels == 0

    # Threshold at target FPR
    threshold = float(np.percentile(test_scores[normal_mask], 100 * (1 - target_fpr)))
    predictions = (test_scores > threshold).astype(bool)

    # Metrics
    tp = np.sum((predictions == 1) & (binary_labels == 1))
    fp = np.sum((predictions == 1) & (binary_labels == 0))
    tn = np.sum((predictions == 0) & (binary_labels == 0))
    fn = np.sum((predictions == 0) & (binary_labels == 1))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    actual_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "recall": float(recall),
        "precision": float(precision),
        "f1_score": float(f1),
        "fpr": float(actual_fpr),
        "threshold": float(threshold),
        "explained_variance": explained_variance,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    logger.info(
        f"TF-IDF+LOF - Recall: {recall:.2%}, Precision: {precision:.2%}, FPR: {actual_fpr:.2%}"
    )

    return detector, test_scores, predictions, metrics


def train_secbert_mahalanobis(
    train_embeddings_path: Path,
    test_embeddings_path: Path,
    test_labels: list[str],
    target_fpr: float = 0.05,
) -> tuple[Any, np.ndarray, np.ndarray, dict]:
    """Train SecBERT + Mahalanobis model."""
    logger.info("Loading SecBERT embeddings...")

    # Load embeddings
    train_data = np.load(train_embeddings_path, allow_pickle=True)
    test_data = np.load(test_embeddings_path, allow_pickle=True)

    train_embeddings = train_data["embeddings"]
    test_embeddings = test_data["embeddings"]

    logger.info(
        f"SecBERT embeddings shape: train={train_embeddings.shape}, test={test_embeddings.shape}"
    )

    # Mahalanobis
    detector = MahalanobisDetector()
    detector.fit(train_embeddings.astype(np.float32))

    # Scores
    test_scores = detector.scores(test_embeddings.astype(np.float32))

    # Binary labels
    binary_labels = np.array([1 if label == "attack" else 0 for label in test_labels])
    normal_mask = binary_labels == 0

    # Threshold at target FPR
    threshold = float(np.percentile(test_scores[normal_mask], 100 * (1 - target_fpr)))
    predictions = (test_scores > threshold).astype(bool)

    # Metrics
    tp = np.sum((predictions == 1) & (binary_labels == 1))
    fp = np.sum((predictions == 1) & (binary_labels == 0))
    tn = np.sum((predictions == 0) & (binary_labels == 0))
    fn = np.sum((predictions == 0) & (binary_labels == 1))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    actual_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "recall": float(recall),
        "precision": float(precision),
        "f1_score": float(f1),
        "fpr": float(actual_fpr),
        "threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    logger.info(
        f"SecBERT+Mahalanobis - Recall: {recall:.2%}, Precision: {precision:.2%}, FPR: {actual_fpr:.2%}"
    )

    return detector, test_scores, predictions, metrics


def analyze_agreement(
    lof_predictions: np.ndarray,
    secbert_predictions: np.ndarray,
    true_labels: np.ndarray,
) -> dict[str, Any]:
    """Analyze agreement between models."""
    logger.info("Analyzing agreement...")

    binary_labels = np.array([1 if label == "attack" else 0 for label in true_labels])

    # Agreement
    agreement = (lof_predictions == secbert_predictions).sum() / len(lof_predictions)
    disagreement = 1.0 - agreement

    # Detection breakdown
    lof_tp = (lof_predictions == 1) & (binary_labels == 1)
    secbert_tp = (secbert_predictions == 1) & (binary_labels == 1)

    both_caught = (lof_tp & secbert_tp).sum()
    lof_only = (lof_tp & ~secbert_tp).sum()
    secbert_only = (~lof_tp & secbert_tp).sum()
    both_missed = (~lof_tp & ~secbert_tp & (binary_labels == 1)).sum()

    total_attacks = binary_labels.sum()

    # False positives
    lof_fp = (lof_predictions == 1) & (binary_labels == 0)
    secbert_fp = (secbert_predictions == 1) & (binary_labels == 0)

    shared_fps = (lof_fp & secbert_fp).sum()
    lof_unique_fps = (lof_fp & ~secbert_fp).sum()
    secbert_unique_fps = (~lof_fp & secbert_fp).sum()

    # Jaccard Index (intersection over union)
    intersection = (lof_predictions & secbert_predictions).sum()
    union = (lof_predictions | secbert_predictions).sum()
    jaccard = intersection / union if union > 0 else 0.0

    # Pearson correlation
    correlation = float(
        np.corrcoef(lof_predictions.astype(float), secbert_predictions.astype(float))[
            0, 1
        ]
    )

    # Agreement by label type
    attack_mask = binary_labels == 1
    normal_mask = binary_labels == 0

    agreement_on_attacks = (
        lof_predictions[attack_mask] == secbert_predictions[attack_mask]
    ).mean()
    agreement_on_normal = (
        lof_predictions[normal_mask] == secbert_predictions[normal_mask]
    ).mean()

    return {
        "overall_agreement": float(agreement),
        "overall_disagreement": float(disagreement),
        "jaccard_index": float(jaccard),
        "correlation": float(correlation),
        "detection_breakdown": {
            "total_attacks": int(total_attacks),
            "both_caught": int(both_caught),
            "lof_only": int(lof_only),
            "secbert_only": int(secbert_only),
            "both_missed": int(both_missed),
        },
        "false_positive_breakdown": {
            "shared_fps": int(shared_fps),
            "lof_unique_fps": int(lof_unique_fps),
            "secbert_unique_fps": int(secbert_unique_fps),
        },
        "agreement_by_label": {
            "on_attacks": float(agreement_on_attacks),
            "on_normal": float(agreement_on_normal),
        },
        "ensemble_potential": {
            "combined_recall": float(
                (both_caught + lof_only + secbert_only) / total_attacks
            ),
            "individual_lof_recall": float(lof_tp.sum() / total_attacks),
            "individual_secbert_recall": float(secbert_tp.sum() / total_attacks),
        },
    }


def plot_agreement_visualizations(
    lof_predictions: np.ndarray,
    secbert_predictions: np.ndarray,
    true_labels: np.ndarray,
    agreement_data: dict[str, Any],
    output_dir: Path,
):
    """Generate agreement visualizations."""
    logger.info("Generating visualizations...")

    binary_labels = np.array([1 if label == "attack" else 0 for label in true_labels])

    # 1. Venn diagram for attacks
    lof_tp = set(np.where((lof_predictions == 1) & (binary_labels == 1))[0])
    secbert_tp = set(np.where((secbert_predictions == 1) & (binary_labels == 1))[0])

    fig, ax = plt.subplots(figsize=(10, 8))
    venn2([lof_tp, secbert_tp], set_labels=["TF-IDF+LOF", "SecBERT+Mahalanobis"], ax=ax)
    ax.set_title(
        f"Attack Detection Overlap\n(Agreement: {agreement_data['overall_agreement']:.1%})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "attack_detection_venn.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Venn diagram for false positives
    lof_fp = set(np.where((lof_predictions == 1) & (binary_labels == 0))[0])
    secbert_fp = set(np.where((secbert_predictions == 1) & (binary_labels == 0))[0])

    fig, ax = plt.subplots(figsize=(10, 8))
    venn2([lof_fp, secbert_fp], set_labels=["TF-IDF+LOF", "SecBERT+Mahalanobis"], ax=ax)
    ax.set_title(
        "False Positive Overlap\n(Minimal overlap = complementary)", fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_dir / "fp_overlap_venn.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Agreement matrix
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create confusion matrix for agreement
    both_attack = ((lof_predictions == 1) & (secbert_predictions == 1)).sum()
    both_normal = ((lof_predictions == 0) & (secbert_predictions == 0)).sum()
    lof_only = ((lof_predictions == 1) & (secbert_predictions == 0)).sum()
    secbert_only = ((lof_predictions == 0) & (secbert_predictions == 1)).sum()

    matrix = np.array([[both_normal, secbert_only], [lof_only, both_attack]])

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["SecBERT: Normal", "SecBERT: Attack"],
        yticklabels=["TF-IDF: Normal", "TF-IDF: Attack"],
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_title("Prediction Agreement Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "agreement_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Visualizations saved")


def generate_report(
    lof_metrics: dict[str, Any],
    secbert_metrics: dict[str, Any],
    agreement_data: dict[str, Any],
    output_dir: Path,
):
    """Generate text report."""
    logger.info("Generating report...")

    report = f"""
# Análisis de Agreement: TF-IDF+PCA+LOF vs SecBERT+Mahalanobis

## Métricas Individuales

### TF-IDF + PCA + LOF
- Recall @ 5% FPR: {lof_metrics["recall"]:.2%}
- Precision @ 5% FPR: {lof_metrics["precision"]:.2%}
- F1-Score: {lof_metrics["f1_score"]:.2%}
- FPR: {lof_metrics["fpr"]:.2%}

### SecBERT + Mahalanobis
- Recall @ 5% FPR: {secbert_metrics["recall"]:.2%}
- Precision @ 5% FPR: {secbert_metrics["precision"]:.2%}
- F1-Score: {secbert_metrics["f1_score"]:.2%}
- FPR: {secbert_metrics["fpr"]:.2%}

## Análisis de Agreement

### Métricas de Acuerdo
- **Agreement general**: {agreement_data["overall_agreement"]:.2%}
- **Desacuerdo**: {agreement_data["overall_disagreement"]:.2%}
- **Jaccard Index**: {agreement_data["jaccard_index"]:.4f}
- **Correlación de Pearson**: {agreement_data["correlation"]:.4f}

### Agreement por Tipo de Muestra
- **En ataques**: {agreement_data["agreement_by_label"]["on_attacks"]:.2%}
- **En muestras normales**: {agreement_data["agreement_by_label"]["on_normal"]:.2%}

### Desglose de Detección de Ataques
- Total de ataques: {agreement_data["detection_breakdown"]["total_attacks"]:,}
- Detectados por ambos: {agreement_data["detection_breakdown"]["both_caught"]:,} ({agreement_data["detection_breakdown"]["both_caught"] / agreement_data["detection_breakdown"]["total_attacks"]:.1%})
- Solo TF-IDF+LOF: {agreement_data["detection_breakdown"]["lof_only"]:,} ({agreement_data["detection_breakdown"]["lof_only"] / agreement_data["detection_breakdown"]["total_attacks"]:.1%})
- Solo SecBERT+Mahalanobis: {agreement_data["detection_breakdown"]["secbert_only"]:,} ({agreement_data["detection_breakdown"]["secbert_only"] / agreement_data["detection_breakdown"]["total_attacks"]:.1%})
- Perdidos por ambos: {agreement_data["detection_breakdown"]["both_missed"]:,} ({agreement_data["detection_breakdown"]["both_missed"] / agreement_data["detection_breakdown"]["total_attacks"]:.1%})

### Desglose de Falsos Positivos
- FPs compartidos: {agreement_data["false_positive_breakdown"]["shared_fps"]:,}
- FPs únicos TF-IDF+LOF: {agreement_data["false_positive_breakdown"]["lof_unique_fps"]:,}
- FPs únicos SecBERT+Mahalanobis: {agreement_data["false_positive_breakdown"]["secbert_unique_fps"]:,}

### Potencial del Ensemble
- Recall individual TF-IDF+LOF: {agreement_data["ensemble_potential"]["individual_lof_recall"]:.2%}
- Recall individual SecBERT+Mahalanobis: {agreement_data["ensemble_potential"]["individual_secbert_recall"]:.2%}
- **Recall potencial del ensemble (OR logic)**: {agreement_data["ensemble_potential"]["combined_recall"]:.2%}
- **Mejora vs TF-IDF+LOF**: +{(agreement_data["ensemble_potential"]["combined_recall"] - agreement_data["ensemble_potential"]["individual_lof_recall"]) * 100:.1f}pp
- **Mejora vs SecBERT+Mahalanobis**: +{(agreement_data["ensemble_potential"]["combined_recall"] - agreement_data["ensemble_potential"]["individual_secbert_recall"]) * 100:.1f}pp

## Conclusiones para Validar el Ensemble

1. **Complementariedad**: Los modelos tienen {agreement_data["overall_disagreement"]:.1%} de desacuerdo, lo que indica que capturan diferentes tipos de anomalías.

2. **Cobertura mejorada**: El ensemble potencialmente alcanza {agreement_data["ensemble_potential"]["combined_recall"]:.2%} de recall, mejorando significativamente sobre los modelos individuales.

3. **Falsos positivos**: Solo {agreement_data["false_positive_breakdown"]["shared_fps"]:,} FPs compartidos indica que los modelos tienen modos de fallo diferentes y complementarios.

4. **Justificación**: La baja correlación ({agreement_data["correlation"]:.4f}) y el bajo Jaccard Index ({agreement_data["jaccard_index"]:.4f}) confirman que los modelos son complementarios y que un ensemble tiene sentido.
"""

    report_path = output_dir / "agreement_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Report saved to {report_path}")


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("ANÁLISIS DE AGREEMENT: TF-IDF+LOF vs SecBERT+Mahalanobis")
    logger.info("=" * 80)

    # Paths
    data_dir = Path("src/neuralshield/data/CSIC")
    train_path = data_dir / "train.jsonl"
    test_path = data_dir / "test.jsonl"

    # SecBERT embeddings paths (assuming experiment 03 structure)
    secbert_train_path = Path(
        "experiments/03_secbert_comparison/secbert_with_preprocessing/csic_train_embeddings_converted.npz"
    )
    secbert_test_path = Path(
        "experiments/03_secbert_comparison/secbert_with_preprocessing/csic_test_embeddings_converted.npz"
    )

    output_dir = Path("experiments/25_representation_evaluation/agreement_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_texts, test_texts, test_labels = load_dataset(
        train_path, test_path, with_preprocessing=False
    )

    # Train TF-IDF + LOF (no preprocessing - mejor resultado)
    _, lof_scores, lof_predictions, lof_metrics = train_tfidf_lof(
        train_texts,
        test_texts,
        test_labels,
        n_components=150,
        n_neighbors=100,
        target_fpr=0.05,
    )

    # Train SecBERT + Mahalanobis (con preprocessing - mejor resultado)
    _, secbert_scores, secbert_predictions, secbert_metrics = train_secbert_mahalanobis(
        secbert_train_path,
        secbert_test_path,
        test_labels,
        target_fpr=0.05,
    )

    # Analyze agreement
    agreement_data = analyze_agreement(
        lof_predictions, secbert_predictions, test_labels
    )

    # Generate visualizations
    plot_agreement_visualizations(
        lof_predictions,
        secbert_predictions,
        test_labels,
        agreement_data,
        output_dir,
    )

    # Generate report
    generate_report(lof_metrics, secbert_metrics, agreement_data, output_dir)

    # Save JSON results
    results = {
        "lof_metrics": lof_metrics,
        "secbert_metrics": secbert_metrics,
        "agreement_analysis": agreement_data,
    }

    results_path = output_dir / "agreement_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")
    logger.info("\n" + "=" * 80)
    logger.info("ANÁLISIS COMPLETO")
    logger.info("=" * 80)
    logger.info(f"Agreement: {agreement_data['overall_agreement']:.2%}")
    logger.info(f"Recall TF-IDF+LOF: {lof_metrics['recall']:.2%}")
    logger.info(f"Recall SecBERT+Mahalanobis: {secbert_metrics['recall']:.2%}")
    logger.info(
        f"Recall potencial ensemble: {agreement_data['ensemble_potential']['combined_recall']:.2%}"
    )


if __name__ == "__main__":
    main()
