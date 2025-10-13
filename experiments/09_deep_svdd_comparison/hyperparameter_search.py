"""Hyperparameter search for Deep SVDD detector.

Searches over:
- hidden_neurons: [[256, 128], [512, 256], [128, 64], [384, 192]]
- use_ae: [True, False]
- epochs: [50, 100, 150]
- batch_size: [32, 64, 128]
- dropout_rate: [0.1, 0.2, 0.3]
- l2_regularizer: [0.01, 0.1, 0.5]

Evaluates recall @ 5% FPR.
"""

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

import wandb
from neuralshield.anomaly import DeepSVDDDetector

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_embeddings(path: Path, is_test: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings from npz file."""
    logger.info(f"Loading embeddings from {path}")
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels_raw = data["labels"]

    if is_test:
        labels = np.array([1 if label == "attack" else 0 for label in labels_raw])
    else:
        labels = np.zeros(len(labels_raw), dtype=int)

    logger.info(f"Loaded {len(embeddings)} samples, {embeddings.shape[1]} dimensions")
    return embeddings, labels


def evaluate_config(
    train_embeddings: NDArray[np.float32],
    test_embeddings: NDArray[np.float32],
    test_labels: NDArray[np.int32],
    hidden_neurons: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dropout_rate: float,
    weight_decay: float,
    max_fpr: float,
    device: str,
    use_wandb: bool = False,
) -> dict:
    """Evaluate a single Deep SVDD configuration."""
    logger.info(
        f"Testing hidden={hidden_neurons}, epochs={epochs}, batch={batch_size}, "
        f"lr={learning_rate}, dropout={dropout_rate}, weight_decay={weight_decay}"
    )

    # Train detector
    start_time = time.time()
    detector = DeepSVDDDetector(
        hidden_neurons=hidden_neurons,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
        device=device,
        verbose=1,  # Enable to see loss
    )
    detector.fit(train_embeddings)
    train_time = time.time() - start_time

    # Compute scores
    test_scores = detector.scores(test_embeddings)

    # Split by label
    test_normal_mask = test_labels == 0
    test_scores_normal = test_scores[test_normal_mask]
    test_scores_anomalous = test_scores[~test_normal_mask]

    # Set threshold based on FPR
    threshold = float(np.percentile(test_scores_normal, (1 - max_fpr) * 100))
    actual_fpr = np.mean(test_scores_normal > threshold)

    # Compute recall
    recall = np.mean(test_scores_anomalous > threshold)

    # Compute precision
    predictions = (test_scores > threshold).astype(int)
    tp = np.sum((predictions == 1) & (test_labels == 1))
    fp = np.sum((predictions == 1) & (test_labels == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    results = {
        "hidden_neurons": hidden_neurons,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
        "weight_decay": weight_decay,
        "threshold": threshold,
        "recall": float(recall),
        "fpr": float(actual_fpr),
        "precision": float(precision),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "train_time_seconds": float(train_time),
        "final_loss": detector.loss_history[-1] if detector.loss_history else None,
    }

    logger.info(
        f"  → Recall={recall:.2%} @ FPR={actual_fpr:.2%}, Precision={precision:.2%}, "
        f"Time={train_time:.1f}s, Loss={results['final_loss']:.4f}"
    )

    # Log loss curve to wandb
    if use_wandb and detector.loss_history:
        for epoch_idx, loss_val in enumerate(detector.loss_history):
            wandb.log(
                {
                    "epoch": epoch_idx + 1,
                    "train_loss": loss_val,
                    "config_id": f"{hidden_neurons}_{batch_size}_{learning_rate}",
                },
                commit=False,
            )
        # Log final metrics
        wandb.log(results)

    return results


def main():
    parser = argparse.ArgumentParser(description="Deep SVDD hyperparameter search")
    parser.add_argument("train_embeddings", type=Path, help="Training embeddings .npz")
    parser.add_argument("test_embeddings", type=Path, help="Test embeddings .npz")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--max-fpr", type=float, default=0.05, help="Max FPR target (default: 0.05)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu/mps, default: auto-detect)",
    )
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--wandb-project", type=str, default="neuralshield")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: reduced search space for testing",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect device if not specified
    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {args.device}")

    # Initialize wandb
    if args.wandb:
        if not WANDB_AVAILABLE:
            logger.error("wandb not available")
            return
        wandb.init(
            project=args.wandb_project,
            name="deep-svdd-hyperparameter-search",
            config={"max_fpr": args.max_fpr, "device": args.device},
        )

    # Load data
    logger.info("=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    train_embeddings, _ = load_embeddings(args.train_embeddings, is_test=False)
    test_embeddings, test_labels = load_embeddings(args.test_embeddings, is_test=True)

    # Define search space
    if args.quick:
        # Quick mode for testing (includes one large architecture)
        hidden_neurons_values = [[768, 384], [512, 256]]
        epochs_values = [50]
        batch_size_values = [256]
        lr_values = [0.001]
        dropout_values = [0.2]
        weight_decay_values = [1e-6]
    else:
        # Full search with COMPLEX architectures for better learning
        # More layers + wider networks for increased capacity
        hidden_neurons_values = [
            # Very deep networks (4-5 layers)
            [768, 512, 256, 128],  # Deep pyramid
            [1024, 512, 256, 128],  # Deeper + wider
            [768, 384, 192, 96],  # Deep halving
            # Very wide networks (2-3 layers)
            [1536, 768],  # Extra wide (2x input)
            [1024, 512, 256],  # Wide + deep
            [768, 512, 256],  # Wider middle
            # Balanced architectures
            [768, 384],  # Baseline (match input)
            [1024, 512],  # Wide baseline
        ]
        epochs_values = [100]  # More epochs for complex models
        batch_size_values = [4096]  # Optimal range (NOT 4096!)
        lr_values = [0.0001, 0.001]  # Lower LR for complex models
        dropout_values = [0.3, 0.4]  # Higher dropout for regularization
        weight_decay_values = [1e-6, 1e-5]

    logger.info("=" * 80)
    logger.info("HYPERPARAMETER SEARCH")
    logger.info("=" * 80)
    logger.info(f"Hidden neurons: {hidden_neurons_values}")
    logger.info(f"Epochs: {epochs_values}")
    logger.info(f"Batch sizes: {batch_size_values}")
    logger.info(f"Learning rates: {lr_values}")
    logger.info(f"Dropout rates: {dropout_values}")
    logger.info(f"Weight decay: {weight_decay_values}")

    total_configs = (
        len(hidden_neurons_values)
        * len(epochs_values)
        * len(batch_size_values)
        * len(lr_values)
        * len(dropout_values)
        * len(weight_decay_values)
    )
    logger.info(f"Total configurations: {total_configs}")

    # Grid search
    all_results = []
    best_recall = 0.0
    best_config = None

    for hidden, epochs, batch, lr, dropout, weight_decay in product(
        hidden_neurons_values,
        epochs_values,
        batch_size_values,
        lr_values,
        dropout_values,
        weight_decay_values,
    ):
        try:
            results = evaluate_config(
                train_embeddings,
                test_embeddings,
                test_labels,
                hidden,
                epochs,
                batch,
                lr,
                dropout,
                weight_decay,
                args.max_fpr,
                args.device,
                args.wandb,
            )
            all_results.append(results)

            # Track best
            if results["recall"] > best_recall:
                best_recall = results["recall"]
                best_config = results

        except Exception as e:
            logger.error(f"Failed for config: {e}")
            continue

    # Save all results
    logger.info("=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    results_path = args.output_dir / "hyperparameter_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"All results saved to {results_path}")

    # Check if any configs succeeded
    if best_config is None:
        logger.error("=" * 80)
        logger.error("ALL CONFIGURATIONS FAILED")
        logger.error("=" * 80)
        logger.error("No successful configurations. Check errors above.")
        return

    # Save best config
    best_config_path = args.output_dir / "best_config.json"
    with open(best_config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    logger.info(f"Best config saved to {best_config_path}")

    # Summary
    logger.info("=" * 80)
    logger.info("BEST CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Hidden neurons: {best_config['hidden_neurons']}")
    logger.info(f"Epochs: {best_config['epochs']}")
    logger.info(f"Batch size: {best_config['batch_size']}")
    logger.info(f"Learning rate: {best_config['learning_rate']}")
    logger.info(f"Dropout: {best_config['dropout_rate']}")
    logger.info(f"Weight decay: {best_config['weight_decay']}")
    logger.info(f"Recall: {best_config['recall']:.2%} @ FPR={best_config['fpr']:.2%}")
    logger.info(f"Precision: {best_config['precision']:.2%}")
    logger.info(f"Threshold: {best_config['threshold']:.4f}")

    # Find top 5
    sorted_results = sorted(all_results, key=lambda x: x["recall"], reverse=True)
    logger.info("=" * 80)
    logger.info("TOP 5 CONFIGURATIONS")
    logger.info("=" * 80)
    for i, result in enumerate(sorted_results[:5], 1):
        logger.info(
            f"{i}. hidden={result['hidden_neurons']}, epochs={result['epochs']}, "
            f"batch={result['batch_size']}, lr={result['learning_rate']} → "
            f"Recall={result['recall']:.2%} @ FPR={result['fpr']:.2%}"
        )

    # Save summary
    summary_path = args.output_dir / "search_summary.txt"
    with open(summary_path, "w") as f:
        f.write("DEEP SVDD HYPERPARAMETER SEARCH SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total configurations tested: {len(all_results)}\n")
        f.write(f"Device: {args.device}\n\n")
        f.write("BEST CONFIGURATION:\n")
        f.write(f"  Hidden neurons: {best_config['hidden_neurons']}\n")
        f.write(f"  Epochs: {best_config['epochs']}\n")
        f.write(f"  Batch size: {best_config['batch_size']}\n")
        f.write(f"  Learning rate: {best_config['learning_rate']}\n")
        f.write(f"  Dropout: {best_config['dropout_rate']}\n")
        f.write(f"  Weight decay: {best_config['weight_decay']}\n")
        f.write(
            f"  Recall: {best_config['recall']:.2%} @ FPR={best_config['fpr']:.2%}\n"
        )
        f.write(f"  Precision: {best_config['precision']:.2%}\n\n")
        f.write("TOP 5 CONFIGURATIONS:\n")
        for i, result in enumerate(sorted_results[:5], 1):
            f.write(
                f"{i}. hidden={result['hidden_neurons']}, epochs={result['epochs']}, "
                f"batch={result['batch_size']}, lr={result['learning_rate']} → "
                f"Recall={result['recall']:.2%}\n"
            )

    logger.info(f"Summary saved to {summary_path}")

    if args.wandb:
        wandb.log({"best_recall": best_recall})
        wandb.finish()

    logger.info("=" * 80)
    logger.info("DONE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
