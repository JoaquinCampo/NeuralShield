#!/usr/bin/env python3
"""Compare predictions from multiple anomaly detection models."""

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer
from loguru import logger
from matplotlib_venn import venn2, venn3

app = typer.Typer()


def load_model_predictions(
    model_path: Path, embeddings_path: Path, model_name: str
) -> dict[str, Any]:
    """Load model and generate predictions on test set."""
    logger.info(f"[{model_name}] Loading model from {model_path}")

    # Load model
    model_data_loaded = joblib.load(model_path)

    # Extract detector (handle different formats)
    if isinstance(model_data_loaded, dict):
        if "detector" in model_data_loaded:
            # Dense embeddings format (BGE, ByT5)
            model = model_data_loaded["detector"]
            logger.info(
                f"[{model_name}] Model loaded successfully (extracted detector from dict)"
            )
        elif "model" in model_data_loaded:
            # TF-IDF format - wrap in IsolationForestDetector
            from neuralshield.anomaly.model import IsolationForestDetector

            model = IsolationForestDetector(
                contamination=model_data_loaded.get("contamination", 0.01),
                n_estimators=model_data_loaded.get("n_estimators", 100),
                max_samples=model_data_loaded.get("max_samples", "auto"),
                random_state=model_data_loaded.get("random_state"),
                n_jobs=model_data_loaded.get("n_jobs", -1),
            )
            model.model = model_data_loaded["model"]
            logger.info(f"[{model_name}] Model loaded successfully (TF-IDF format)")
        else:
            raise ValueError(
                f"Unknown dict format for model: {model_data_loaded.keys()}"
            )
    else:
        model = model_data_loaded
        logger.info(f"[{model_name}] Model loaded successfully")

    # Load test embeddings
    logger.info(f"[{model_name}] Loading embeddings from {embeddings_path}")
    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    logger.info(
        f"[{model_name}] Loaded {len(embeddings)} embeddings with shape {embeddings.shape}"
    )

    # Generate predictions
    logger.info(f"[{model_name}] Running predictions...")
    is_attack_pred = model.predict(embeddings)  # True = anomaly, False = normal
    scores = model.scores(embeddings)

    # Get ground truth labels
    is_attack_true = (labels == "attack") | (labels == "anomalous")

    n_flagged = is_attack_pred.sum()
    n_total = len(is_attack_pred)
    n_true_attacks = is_attack_true.sum()

    logger.info(
        f"[{model_name}] Predictions complete: {n_flagged}/{n_total} flagged as attacks "
        f"(true attacks: {n_true_attacks})"
    )

    return {
        "name": model_name,
        "predictions": is_attack_pred,
        "scores": scores,
        "true_labels": is_attack_true,
        "n_samples": len(is_attack_pred),
    }


def compute_agreement_matrix(models: list[dict[str, Any]]) -> np.ndarray:
    """Compute pairwise agreement rates between models."""
    n_models = len(models)
    agreement = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                agreement[i, j] = 100.0
            else:
                # % of examples where both models agree
                agree = (models[i]["predictions"] == models[j]["predictions"]).sum()
                agreement[i, j] = 100.0 * agree / models[i]["n_samples"]

    return agreement


def analyze_error_overlap(models: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze overlap in False Positives and False Negatives."""
    n_samples = models[0]["n_samples"]
    true_labels = models[0]["true_labels"]

    # Get FPs and FNs for each model
    fps = []
    fns = []
    tps = []

    for model in models:
        pred = model["predictions"]

        # False Positives: predicted attack, actually normal
        fp = pred & ~true_labels
        fps.append(fp)

        # False Negatives: predicted normal, actually attack
        fn = ~pred & true_labels
        fns.append(fn)

        # True Positives: predicted attack, actually attack
        tp = pred & true_labels
        tps.append(tp)

    # Count overlaps
    def count_overlap(error_sets):
        """Count how many models share each error."""
        overlap_counts = {}
        for n_models in range(1, len(models) + 1):
            overlap_counts[n_models] = 0

        # For each example, count how many models made this error
        error_matrix = np.stack(error_sets)  # (n_models, n_samples)
        errors_per_example = error_matrix.sum(axis=0)

        for n_models in range(1, len(models) + 1):
            overlap_counts[n_models] = (errors_per_example == n_models).sum()

        return overlap_counts

    fp_overlap = count_overlap(fps)
    fn_overlap = count_overlap(fns)

    # Find unique catches (TPs that only one model gets)
    unique_catches = {}
    for i, model in enumerate(models):
        # TPs that this model gets but others don't
        unique_to_this = tps[i].copy()
        for j, other_tp in enumerate(tps):
            if i != j:
                unique_to_this &= ~other_tp
        unique_catches[model["name"]] = unique_to_this.sum()

    # Find universally hard examples (all models miss)
    all_missed = fns[0].copy()
    for fn in fns[1:]:
        all_missed &= fn

    # Find universally easy (all models catch)
    all_caught = tps[0].copy()
    for tp in tps[1:]:
        all_caught &= tp

    return {
        "fp_overlap": fp_overlap,
        "fn_overlap": fn_overlap,
        "unique_catches": unique_catches,
        "universally_missed": all_missed.sum(),
        "universally_caught": all_caught.sum(),
        "total_attacks": true_labels.sum(),
        "total_normal": (~true_labels).sum(),
    }


def plot_agreement_matrix(
    agreement: np.ndarray, model_names: list[str], output_path: Path
):
    """Plot agreement matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        agreement,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=model_names,
        yticklabels=model_names,
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Agreement (%)"},
        ax=ax,
    )

    ax.set_title(
        "Model Agreement Matrix\n(% of examples where models agree)", fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved agreement matrix to {output_path}")

    return fig


def plot_error_venn(models: list[dict[str, Any]], output_dir: Path) -> tuple[Any, Any]:
    """Plot Venn diagrams for error overlap (supports 2-3 models)."""
    if len(models) < 2 or len(models) > 3:
        logger.warning(f"Venn diagrams only support 2-3 models, got {len(models)}")
        return None, None

    true_labels = models[0]["true_labels"]

    # Get FPs and FNs
    fps = []
    fns = []
    for model in models:
        pred = model["predictions"]
        fp = set(np.where(pred & ~true_labels)[0])
        fn = set(np.where(~pred & true_labels)[0])
        fps.append(fp)
        fns.append(fn)

    # Plot FP Venn
    fig_fp, ax_fp = plt.subplots(figsize=(10, 8))
    if len(models) == 2:
        venn2(fps, set_labels=[m["name"] for m in models], ax=ax_fp)
    else:
        venn3(fps, set_labels=[m["name"] for m in models], ax=ax_fp)
    ax_fp.set_title(
        "False Positives Overlap\n(Normal requests flagged as attacks)", fontsize=14
    )
    fig_fp.tight_layout()
    fp_path = output_dir / "fp_overlap_venn.png"
    fig_fp.savefig(fp_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved FP Venn to {fp_path}")

    # Plot FN Venn
    fig_fn, ax_fn = plt.subplots(figsize=(10, 8))
    if len(models) == 2:
        venn2(fns, set_labels=[m["name"] for m in models], ax=ax_fn)
    else:
        venn3(fns, set_labels=[m["name"] for m in models], ax=ax_fn)
    ax_fn.set_title("False Negatives Overlap\n(Attacks missed by models)", fontsize=14)
    fig_fn.tight_layout()
    fn_path = output_dir / "fn_overlap_venn.png"
    fig_fn.savefig(fn_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved FN Venn to {fn_path}")

    return fig_fp, fig_fn


def plot_prediction_heatmap(
    models: list[dict[str, Any]], output_path: Path, n_samples: int = 1000
):
    """Plot heatmap of predictions for a sample of examples."""
    # Sample examples (stratified by true label)
    true_labels = models[0]["true_labels"]
    n_attacks = true_labels.sum()
    n_normal = len(true_labels) - n_attacks

    # Sample 50/50 attacks and normal
    attack_indices = np.where(true_labels)[0]
    normal_indices = np.where(~true_labels)[0]

    sample_size = min(n_samples // 2, min(len(attack_indices), len(normal_indices)))

    sampled_attacks = np.random.choice(attack_indices, sample_size, replace=False)
    sampled_normal = np.random.choice(normal_indices, sample_size, replace=False)
    sampled_indices = np.concatenate([sampled_attacks, sampled_normal])

    # Sort by true label for cleaner visualization
    sampled_indices = sampled_indices[true_labels[sampled_indices].argsort()]

    # Build matrix: rows=examples, cols=models, values=prediction
    pred_matrix = np.zeros((len(sampled_indices), len(models)))
    for i, model in enumerate(models):
        pred_matrix[:, i] = model["predictions"][sampled_indices].astype(int)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 12))

    sns.heatmap(
        pred_matrix,
        cmap=["lightgreen", "lightcoral"],
        cbar_kws={"label": "Prediction", "ticks": [0.25, 0.75]},
        xticklabels=[m["name"] for m in models],
        yticklabels=False,
        ax=ax,
    )

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(["Normal", "Attack"])

    # Add horizontal line to separate normal from attack
    n_attacks_in_sample = true_labels[sampled_indices].sum()
    ax.axhline(
        y=len(sampled_indices) - n_attacks_in_sample,
        color="blue",
        linewidth=2,
        linestyle="--",
    )

    ax.set_title(
        f"Model Predictions on {len(sampled_indices)} Sampled Examples\n"
        f"(Top: Normal, Bottom: Attacks)",
        fontsize=14,
    )
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Test Examples", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved prediction heatmap to {output_path}")

    return fig


@app.command()
def main(
    models: list[str] = typer.Option(
        ...,
        "--model",
        "-m",
        help='Model spec: "NAME:MODEL_PATH:EMBEDDINGS_PATH"',
    ),
    output_dir: Path = typer.Option(
        Path("visualizations/model_comparison"),
        help="Directory to save visualizations",
    ),
    wandb_enabled: bool = typer.Option(False, "--wandb/--no-wandb"),
    wandb_project: str = typer.Option("neuralshield", "--wandb-project"),
    wandb_run_name: str = typer.Option(None, "--wandb-run-name"),
):
    """Compare predictions from multiple anomaly detection models."""

    logger.info("=" * 80)
    logger.info("MODEL COMPARISON ANALYSIS")
    logger.info("=" * 80)

    # Parse model specs
    logger.info(f"Parsing {len(models)} model specifications...")
    model_specs = []
    for spec in models:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid model spec: {spec}. Expected format: NAME:MODEL_PATH:EMBEDDINGS_PATH"
            )
        name, model_path, embeddings_path = parts
        model_specs.append(
            {
                "name": name,
                "model_path": Path(model_path),
                "embeddings_path": Path(embeddings_path),
            }
        )

    # Load all models and predictions
    model_data = []
    for spec in model_specs:
        data = load_model_predictions(
            spec["model_path"],
            spec["embeddings_path"],
            spec["name"],
        )
        model_data.append(data)

    # Verify all models have same test set
    n_samples = model_data[0]["n_samples"]
    for model in model_data[1:]:
        if model["n_samples"] != n_samples:
            raise ValueError("All models must have the same test set size!")

    logger.info(f"All models loaded successfully. Test set size: {n_samples}")

    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup wandb
    if wandb_enabled:
        import wandb

        wandb.init(
            project=wandb_project,
            name=wandb_run_name or "model-comparison",
            job_type="analysis",
        )
        logger.info("Initialized wandb logging")

    # Compute agreement matrix
    logger.info("Computing agreement matrix...")
    agreement = compute_agreement_matrix(model_data)
    model_names = [m["name"] for m in model_data]

    # Print agreement matrix
    print("\n" + "=" * 80)
    print("AGREEMENT MATRIX (% of examples where models agree)")
    print("=" * 80)
    for i, name_i in enumerate(model_names):
        print(f"\n{name_i:20s}", end="")
        for j, name_j in enumerate(model_names):
            print(f"  {agreement[i, j]:5.1f}%", end="")
    print("\n" + "=" * 80 + "\n")

    # Plot agreement matrix
    logger.info("Creating agreement matrix visualization...")
    fig_agreement = plot_agreement_matrix(
        agreement, model_names, output_dir / "agreement_matrix.png"
    )

    # Analyze error overlap
    logger.info("Analyzing error overlap (FPs, FNs, unique catches)...")
    overlap = analyze_error_overlap(model_data)
    logger.info("Error overlap analysis complete")

    # Print overlap analysis
    print("\n" + "=" * 80)
    print("ERROR OVERLAP ANALYSIS")
    print("=" * 80)
    print(f"\nTotal attacks in test set: {overlap['total_attacks']}")
    print(f"Total normal in test set: {overlap['total_normal']}")

    print(f"\nUniversally caught (all models): {overlap['universally_caught']}")
    print(f"Universally missed (all models): {overlap['universally_missed']}")

    print("\n--- Unique Catches (attacks only this model detects) ---")
    for name, count in overlap["unique_catches"].items():
        print(f"  {name:20s}: {count:5d} attacks")

    print("\n--- False Positives Overlap ---")
    for n_models, count in sorted(overlap["fp_overlap"].items(), reverse=True):
        if count > 0:
            print(f"  Shared by {n_models} model(s): {count:5d} examples")

    print("\n--- False Negatives Overlap ---")
    for n_models, count in sorted(overlap["fn_overlap"].items(), reverse=True):
        if count > 0:
            print(f"  Shared by {n_models} model(s): {count:5d} examples")
    print("=" * 80 + "\n")

    # Plot Venn diagrams (if 2-3 models)
    if 2 <= len(model_data) <= 3:
        logger.info("Creating Venn diagrams...")
        fig_fp_venn, fig_fn_venn = plot_error_venn(model_data, output_dir)
    else:
        logger.info(f"Skipping Venn diagrams (need 2-3 models, got {len(model_data)})")
        fig_fp_venn = fig_fn_venn = None

    # Plot prediction heatmap
    logger.info("Creating prediction heatmap...")
    fig_heatmap = plot_prediction_heatmap(
        model_data, output_dir / "prediction_heatmap.png", n_samples=1000
    )

    # Log to wandb
    if wandb_enabled:
        import wandb

        log_data = {
            "agreement_matrix": wandb.Image(fig_agreement),
            "prediction_heatmap": wandb.Image(fig_heatmap),
            "metrics/universally_caught": overlap["universally_caught"],
            "metrics/universally_missed": overlap["universally_missed"],
        }

        if fig_fp_venn is not None:
            log_data["fp_overlap_venn"] = wandb.Image(fig_fp_venn)
        if fig_fn_venn is not None:
            log_data["fn_overlap_venn"] = wandb.Image(fig_fn_venn)

        # Log unique catches
        for name, count in overlap["unique_catches"].items():
            log_data[f"unique_catches/{name}"] = count

        # Log agreement matrix values
        for i, name_i in enumerate(model_names):
            for j, name_j in enumerate(model_names):
                if i != j:
                    log_data[f"agreement/{name_i}_vs_{name_j}"] = agreement[i, j]

        wandb.log(log_data)
        logger.info("Logged all results to wandb")

    logger.info(f"All visualizations saved to {output_dir}")
    logger.info("Analysis complete!")


if __name__ == "__main__":
    app()
