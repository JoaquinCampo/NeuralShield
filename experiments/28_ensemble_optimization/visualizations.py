"""Beautiful visualization functions for ensemble experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib_venn import venn2
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11

# Color palette
COLOR_LOF = "#3498db"  # Blue
COLOR_SECBERT = "#e74c3c"  # Red
COLOR_ENSEMBLE = "#2ecc71"  # Green
COLOR_NORMAL = "#3498db"  # Blue
COLOR_ANOMALOUS = "#e74c3c"  # Red
COLOR_THRESHOLD = "#2ecc71"  # Green


def create_score_distribution_plot(
    lof_normal_scores,
    lof_anomalous_scores,
    secbert_normal_scores,
    secbert_anomalous_scores,
    ensemble_normal_scores,
    ensemble_anomalous_scores,
    lof_threshold,
    secbert_threshold,
    ensemble_threshold,
    output_path: Path | None = None,
):
    """Beautiful dual-model score distribution comparison using violin plots."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: LOF Scores
    ax1 = axes[0]
    
    # Prepare data for violin plot
    lof_data = [lof_normal_scores, lof_anomalous_scores]
    lof_positions = [1, 2]
    lof_labels = [f"Normal\n(n={len(lof_normal_scores):,})", f"Anomalous\n(n={len(lof_anomalous_scores):,})"]
    
    # Create violin plot
    parts = ax1.violinplot(
        lof_data,
        positions=lof_positions,
        widths=0.6,
        showmeans=True,
        showmedians=True,
        showextrema=False,
    )
    
    # Color the violins
    for pc, color in zip(parts["bodies"], [COLOR_NORMAL, COLOR_ANOMALOUS]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor("white")
        pc.set_linewidth(1.5)
    
    # Style the means and medians
    parts["cmeans"].set_color("white")
    parts["cmeans"].set_linewidth(2)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)
    
    # Add threshold line
    ax1.axhline(
        lof_threshold,
        color=COLOR_THRESHOLD,
        linestyle="--",
        linewidth=2.5,
        alpha=0.8,
        label=f"Threshold ({lof_threshold:.3f})",
        zorder=10,
    )
    
    ax1.set_xticks(lof_positions)
    ax1.set_xticklabels(lof_labels, fontsize=11, fontweight="bold")
    ax1.set_ylabel("LOF Anomaly Score", fontsize=12, fontweight="bold")
    ax1.set_title(
        "TF-IDF + LOF Score Distribution", fontsize=13, fontweight="bold", pad=10
    )
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax1.set_ylim(bottom=0)

    # Plot 2: SecBERT Scores
    ax2 = axes[1]
    
    # Prepare data for violin plot
    secbert_data = [secbert_normal_scores, secbert_anomalous_scores]
    secbert_positions = [1, 2]
    secbert_labels = [f"Normal\n(n={len(secbert_normal_scores):,})", f"Anomalous\n(n={len(secbert_anomalous_scores):,})"]
    
    # Create violin plot
    parts2 = ax2.violinplot(
        secbert_data,
        positions=secbert_positions,
        widths=0.6,
        showmeans=True,
        showmedians=True,
        showextrema=False,
    )
    
    # Color the violins
    for pc, color in zip(parts2["bodies"], [COLOR_NORMAL, COLOR_ANOMALOUS]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor("white")
        pc.set_linewidth(1.5)
    
    # Style the means and medians
    parts2["cmeans"].set_color("white")
    parts2["cmeans"].set_linewidth(2)
    parts2["cmedians"].set_color("black")
    parts2["cmedians"].set_linewidth(2)
    
    # Add threshold line
    ax2.axhline(
        secbert_threshold,
        color=COLOR_THRESHOLD,
        linestyle="--",
        linewidth=2.5,
        alpha=0.8,
        label=f"Threshold ({secbert_threshold:.3f})",
        zorder=10,
    )
    
    ax2.set_xticks(secbert_positions)
    ax2.set_xticklabels(secbert_labels, fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mahalanobis Distance", fontsize=12, fontweight="bold")
    ax2.set_title(
        "SecBERT + Mahalanobis Score Distribution",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax2.set_ylim(bottom=0)

    # Plot 3: Ensemble Scores
    ax3 = axes[2]
    
    # Prepare data for ensemble violin plot
    ensemble_data = [ensemble_normal_scores, ensemble_anomalous_scores]
    ensemble_positions = [1, 2]
    ensemble_labels = [f"Normal\n(n={len(ensemble_normal_scores):,})", f"Anomalous\n(n={len(ensemble_anomalous_scores):,})"]
    
    # Create violin plot
    parts3 = ax3.violinplot(
        ensemble_data,
        positions=ensemble_positions,
        widths=0.6,
        showmeans=True,
        showmedians=True,
        showextrema=False,
    )
    
    # Color the violins
    for pc, color in zip(parts3["bodies"], [COLOR_NORMAL, COLOR_ANOMALOUS]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor("white")
        pc.set_linewidth(1.5)
    
    # Style the means and medians
    parts3["cmeans"].set_color("white")
    parts3["cmeans"].set_linewidth(2)
    parts3["cmedians"].set_color("black")
    parts3["cmedians"].set_linewidth(2)
    
    # Add threshold line
    ax3.axhline(
        ensemble_threshold,
        color=COLOR_THRESHOLD,
        linestyle="--",
        linewidth=2.5,
        alpha=0.8,
        label=f"Threshold ({ensemble_threshold:.3f})",
        zorder=10,
    )
    
    ax3.set_xticks(ensemble_positions)
    ax3.set_xticklabels(ensemble_labels, fontsize=11, fontweight="bold")
    ax3.set_ylabel("Ensemble Score", fontsize=12, fontweight="bold")
    ax3.set_title(
        "Ensemble Score Distribution",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax3.set_ylim(bottom=0)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")

    return fig


def create_roc_comparison_plot(
    lof_scores,
    secbert_scores,
    ensemble_scores,
    labels,
    output_path: Path | None = None,
):
    """Beautiful ROC curve comparison."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Convert labels to binary
    binary_labels = (np.array(labels) == "attack").astype(int)

    # Compute ROC curves
    fpr_lof, tpr_lof, _ = roc_curve(binary_labels, lof_scores)
    fpr_secbert, tpr_secbert, _ = roc_curve(binary_labels, secbert_scores)
    fpr_ensemble, tpr_ensemble, _ = roc_curve(binary_labels, ensemble_scores)

    # Compute AUC
    auc_lof = auc(fpr_lof, tpr_lof)
    auc_secbert = auc(fpr_secbert, tpr_secbert)
    auc_ensemble = auc(fpr_ensemble, tpr_ensemble)

    # Plot curves
    ax.plot(
        fpr_lof,
        tpr_lof,
        linewidth=2.5,
        label=f"TF-IDF+LOF (AUC={auc_lof:.3f})",
        color=COLOR_LOF,
        linestyle="-",
    )
    ax.plot(
        fpr_secbert,
        tpr_secbert,
        linewidth=2.5,
        label=f"SecBERT+Mahalanobis (AUC={auc_secbert:.3f})",
        color=COLOR_SECBERT,
        linestyle="-",
    )
    ax.plot(
        fpr_ensemble,
        tpr_ensemble,
        linewidth=3,
        label=f"Ensemble (AUC={auc_ensemble:.3f})",
        color=COLOR_ENSEMBLE,
        linestyle="-",
        marker="o",
        markersize=4,
    )

    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")

    # 5% FPR marker
    ax.axvline(
        0.05,
        color="orange",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label="5% FPR Target",
    )

    ax.set_xlabel("False Positive Rate", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=13, fontweight="bold")
    ax.set_title(
        "ROC Curves: Component Models vs Ensemble",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim([0, 0.2])  # Focus on low FPR region
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")

    return fig


def create_agreement_venn(
    lof_detections,
    secbert_detections,
    labels,
    output_path: Path | None = None,
):
    """Beautiful Venn diagram showing detection overlap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Attack detections
    ax1 = axes[0]
    attack_mask = np.array(labels) == "attack"
    lof_attacks = set(np.where((lof_detections == 1) & attack_mask)[0])
    secbert_attacks = set(np.where((secbert_detections == 1) & attack_mask)[0])

    v1 = venn2(
        [lof_attacks, secbert_attacks],
        set_labels=("TF-IDF+LOF", "SecBERT+Mahalanobis"),
        ax=ax1,
        alpha=0.7,
    )

    # Customize colors
    if v1.get_patch_by_id("10"):
        v1.get_patch_by_id("10").set_color(COLOR_LOF)
    if v1.get_patch_by_id("01"):
        v1.get_patch_by_id("01").set_color(COLOR_SECBERT)
    if v1.get_patch_by_id("11"):
        v1.get_patch_by_id("11").set_color("#9b59b6")

    ax1.set_title("Attack Detection Overlap", fontsize=13, fontweight="bold", pad=15)

    # Add statistics text box
    both = len(lof_attacks & secbert_attacks)
    lof_only = len(lof_attacks - secbert_attacks)
    secbert_only = len(secbert_attacks - lof_attacks)
    total_attacks = attack_mask.sum()

    stats_text = f"Total Attacks: {total_attacks}\n"
    stats_text += f"Both detect: {both} ({both/total_attacks:.1%})\n"
    stats_text += f"LOF only: {lof_only} ({lof_only/total_attacks:.1%})\n"
    stats_text += f"SecBERT only: {secbert_only} ({secbert_only/total_attacks:.1%})"

    ax1.text(
        1.1,
        0.5,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # False positive overlap
    ax2 = axes[1]
    normal_mask = np.array(labels) == "valid"
    lof_fps = set(np.where((lof_detections == 1) & normal_mask)[0])
    secbert_fps = set(np.where((secbert_detections == 1) & normal_mask)[0])

    v2 = venn2(
        [lof_fps, secbert_fps],
        set_labels=("TF-IDF+LOF", "SecBERT+Mahalanobis"),
        ax=ax2,
        alpha=0.7,
    )

    # Customize colors (lighter for FPs)
    if v2.get_patch_by_id("10"):
        v2.get_patch_by_id("10").set_color("#85c1e2")
    if v2.get_patch_by_id("01"):
        v2.get_patch_by_id("01").set_color("#f1948a")
    if v2.get_patch_by_id("11"):
        v2.get_patch_by_id("11").set_color("#bb8fce")

    ax2.set_title("False Positive Overlap", fontsize=13, fontweight="bold", pad=15)

    # FP statistics
    fp_both = len(lof_fps & secbert_fps)
    fp_lof_only = len(lof_fps - secbert_fps)
    fp_secbert_only = len(secbert_fps - lof_fps)
    total_normals = normal_mask.sum()

    fp_stats_text = f"Total Normals: {total_normals}\n"
    fp_stats_text += f"Both flag: {fp_both} ({fp_both/total_normals:.2%})\n"
    fp_stats_text += f"LOF only: {fp_lof_only} ({fp_lof_only/total_normals:.2%})\n"
    fp_stats_text += (
        f"SecBERT only: {fp_secbert_only} ({fp_secbert_only/total_normals:.2%})"
    )

    ax2.text(
        1.1,
        0.5,
        fp_stats_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5),
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")

    return fig


def create_performance_comparison_bars(
    lof_metrics,
    secbert_metrics,
    ensemble_metrics,
    output_path: Path | None = None,
):
    """Beautiful bar chart comparing all metrics."""
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics = ["Recall", "Precision", "F1-Score", "ROC AUC"]
    lof_values = [
        lof_metrics["recall"],
        lof_metrics["precision"],
        lof_metrics["f1"],
        lof_metrics["roc_auc"],
    ]
    secbert_values = [
        secbert_metrics["recall"],
        secbert_metrics["precision"],
        secbert_metrics["f1"],
        secbert_metrics["roc_auc"],
    ]
    ensemble_values = [
        ensemble_metrics["recall"],
        ensemble_metrics["precision"],
        ensemble_metrics["f1"],
        ensemble_metrics["roc_auc"],
    ]

    x = np.arange(len(metrics))
    width = 0.25

    bars1 = ax.bar(
        x - width,
        lof_values,
        width,
        label="TF-IDF+LOF",
        color=COLOR_LOF,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    bars2 = ax.bar(
        x,
        secbert_values,
        width,
        label="SecBERT+Mahalanobis",
        color=COLOR_SECBERT,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    bars3 = ax.bar(
        x + width,
        ensemble_values,
        width,
        label="Ensemble",
        color=COLOR_ENSEMBLE,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Performance Comparison: Component Models vs Ensemble",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_ylim([0, 1.05])

    # Add improvement arrows
    best_recall = max(lof_values[0], secbert_values[0])
    ensemble_recall = ensemble_values[0]
    if ensemble_recall > best_recall:
        ax.annotate(
            "",
            xy=(0 + width, ensemble_recall),
            xytext=(0, best_recall),
            arrowprops=dict(arrowstyle="->", lw=2, color="green"),
        )
        ax.text(
            0.5,
            (best_recall + ensemble_recall) / 2,
            f"+{(ensemble_recall-best_recall)*100:.1f}pp",
            ha="center",
            fontsize=9,
            color="green",
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")

    return fig


def create_weight_optimization_curves(
    weights,
    recalls,
    precisions,
    f1_scores,
    fprs,
    best_weight_idx,
    output_path: Path | None = None,
):
    """Create beautiful weight optimization progress curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        weights,
        recalls,
        "o-",
        linewidth=2.5,
        markersize=6,
        label="Recall",
        color=COLOR_LOF,
    )
    ax.plot(
        weights,
        precisions,
        "s-",
        linewidth=2.5,
        markersize=6,
        label="Precision",
        color=COLOR_SECBERT,
    )
    ax.plot(
        weights,
        f1_scores,
        "^-",
        linewidth=2.5,
        markersize=6,
        label="F1-Score",
        color=COLOR_ENSEMBLE,
    )
    ax.axvline(
        weights[best_weight_idx],
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Best F1 (w={weights[best_weight_idx]:.2f})",
    )
    ax.set_xlabel("LOF Weight", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Weight Optimization Progress", fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")

    return fig

