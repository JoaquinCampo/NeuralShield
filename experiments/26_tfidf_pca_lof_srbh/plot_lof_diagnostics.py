from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def load_artifacts(exp_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    model_path = exp_dir / "lof_tfidf_pca175_k100.joblib"
    embeddings_path = exp_dir / "srbh_test_embeddings.npz"

    if not model_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError("Run the training script before plotting diagnostics.")

    payload = joblib.load(model_path)
    data = np.load(embeddings_path)
    embeddings = data["embeddings"].astype(np.float32)
    labels = data["labels"]
    return payload, embeddings, labels


def plot_score_distributions(
    scores: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    bins: int = 200,
) -> None:
    mask_valid = labels == "valid"
    mask_attack = labels == "attack"

    if isinstance(bins, int):
        unique_values = np.unique(scores)
        if unique_values.size <= 1:
            bin_edges = np.array([unique_values[0] - 0.5, unique_values[0] + 0.5])
        else:
            bin_edges = np.histogram_bin_edges(scores, bins=bins)
    else:
        bin_edges = bins

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        scores[mask_valid],
        bins=bin_edges,
        alpha=0.6,
        label="valid",
        color="#2ecc71",
    )
    ax.hist(
        scores[mask_attack],
        bins=bin_edges,
        alpha=0.6,
        label="attack",
        color="#e74c3c",
    )
    ax.set_title("LOF decision_function distributions")
    ax.set_xlabel("decision_function score (higher ≈ normal)")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_xlim(scores.min(), scores.max())
    fig.tight_layout()
    fig.savefig(output_dir / "lof_score_hist.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(
        [scores[mask_valid], scores[mask_attack]],
        tick_labels=["valid", "attack"],
        vert=True,
        showfliers=False,
    )
    ax.set_title("LOF decision_function boxplot")
    ax.set_ylabel("decision_function score")
    fig.tight_layout()
    fig.savefig(output_dir / "lof_score_boxplot.png", dpi=200)
    plt.close(fig)


def plot_pca_scatter(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    sample_size: int = 20000,
) -> None:
    total = len(labels)
    if total <= sample_size:
        idx = np.arange(total)
    else:
        idx = random.sample(range(total), sample_size)
    sample_embeddings = embeddings[idx, :2]
    sample_labels = labels[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = np.where(sample_labels == "attack", "#e74c3c", "#2ecc71")
    ax.scatter(
        sample_embeddings[:, 0],
        sample_embeddings[:, 1],
        c=colors,
        s=5,
        alpha=0.4,
    )
    ax.set_title("Primeras 2 componentes PCA (muestra)")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_scatter.png", dpi=200)
    plt.close(fig)


def plot_token_presence(
    exp_dir: Path,
    patterns: list[str],
    top_k: int = 10,
) -> None:
    payload_path = exp_dir / "token_presence.json"
    if payload_path.exists():
        logger.info("Token presence data already exists at {}", payload_path)
        return

    # Compute quick statistics to correlate heuristics with labels
    source_path = Path("src/neuralshield/data/SR_BH_2020/test.jsonl")
    stats: dict[str, dict[str, int]] = {
        pattern: {"valid": 0, "attack": 0} for pattern in patterns
    }
    totals = {"valid": 0, "attack": 0}

    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            label = sample["label"]
            totals[label] += 1
            lower_request = sample["request"].lower()
            for pattern in patterns:
                if pattern in lower_request:
                    stats[pattern][label] += 1

    output = {
        "totals": totals,
        "patterns": stats,
    }
    payload_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Saved pattern stats to {}", payload_path)


def plot_sweep_heatmap(results_path: Path, output_dir: Path) -> None:
    with results_path.open(encoding="utf-8") as handle:
        results = json.load(handle)

    if not results:
        logger.warning("Archivo de sweep vacío {}", str(results_path))
        return

    neighbors = sorted({item["n_neighbors"] for item in results})
    fprs = sorted({item["max_fpr"] for item in results})

    recall_matrix = np.zeros((len(neighbors), len(fprs)))
    fpr_matrix = np.zeros_like(recall_matrix)
    for item in results:
        i = neighbors.index(item["n_neighbors"])
        j = fprs.index(item["max_fpr"])
        recall_matrix[i, j] = item["recall"]
        fpr_matrix[i, j] = item["actual_fpr"]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(recall_matrix, cmap="viridis", origin="lower")
    ax.set_xticks(range(len(fprs)))
    ax.set_xticklabels([f"{value:.0%}" for value in fprs])
    ax.set_yticks(range(len(neighbors)))
    ax.set_yticklabels([str(value) for value in neighbors])
    ax.set_xlabel("Max FPR objetivo")
    ax.set_ylabel("n_neighbors")
    ax.set_title("Recall por configuración LOF")
    for i, neighbor in enumerate(neighbors):
        for j, fpr in enumerate(fprs):
            ax.text(
                j,
                i,
                f"{recall_matrix[i, j]:.0%}\n({fpr_matrix[i, j]:.0%})",
                ha="center",
                va="center",
                color="white" if recall_matrix[i, j] < 0.4 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall")
    fig.tight_layout()
    fig.savefig(output_dir / "lof_sweep_heatmap.png", dpi=200)
    plt.close(fig)


def main() -> None:
    exp_dir = Path("experiments/26_tfidf_pca_lof_srbh")
    exp_dir.mkdir(parents=True, exist_ok=True)

    payload, embeddings, labels = load_artifacts(exp_dir)
    detector = payload["detector"]
    threshold = payload["threshold"]
    scores = detector._model.decision_function(embeddings)

    logger.info(
        "Score stats: min={:.2e} max={:.2e} mean={:.2e} threshold={:.2e}",
        scores.min(),
        scores.max(),
        scores.mean(),
        threshold,
    )

    plot_score_distributions(scores, labels, exp_dir)
    plot_pca_scatter(embeddings, labels, exp_dir)
    plot_token_presence(exp_dir, ["union", "sleep", "select", "../"])

    sweep_path = exp_dir / "lof_sweep_results.json"
    if sweep_path.exists():
        plot_sweep_heatmap(sweep_path, exp_dir)


if __name__ == "__main__":
    main()
