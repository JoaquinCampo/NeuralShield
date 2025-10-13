#!/usr/bin/env python3
"""Generate embeddings using the domain-adapted SecBERT model.

This script generates embeddings for the train and test sets using the
SecBERT model that was domain-adapted via MLM on HTTP requests.
"""

from pathlib import Path

import typer
from loguru import logger

from neuralshield.encoding.config import EmbeddingDumpConfig
from neuralshield.encoding.dump_embeddings import dump_embeddings

app = typer.Typer()


def generate_embeddings_for_split(
    split: str,
    dataset_path: Path,
    output_dir: Path,
    device: str = "cpu",
    batch_size: int = 512,
) -> Path:
    """Generate embeddings for a single dataset split."""
    output_path = output_dir / f"{split}_embeddings.jsonl"

    logger.info(f"Generating {split} embeddings...")
    logger.info(f"  Dataset: {dataset_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Device: {device}")

    config = EmbeddingDumpConfig(
        dataset_path=dataset_path,
        output_path=output_path,
        batch_size=batch_size,
        reader_name="jsonl",
        use_pipeline=True,
        pipeline_name="preprocess",
        encoder_name="secbert-adapted",
        encoder_model_name="secbert-http-adapted",
        device=device,
        wandb_enabled=False,
        include_requests=False,
        include_labels=True,
        include_ids=True,
    )

    dump_embeddings(config)
    logger.info(f"✅ {split} embeddings saved to {output_path}")

    return output_path


@app.command()
def main(
    train_dataset: Path = typer.Option(
        "data/csic_2010/train.jsonl",
        help="Path to training dataset",
    ),
    test_dataset: Path = typer.Option(
        "data/csic_2010/test.jsonl",
        help="Path to test dataset",
    ),
    output_dir: Path = typer.Option(
        "experiments/11_secbert_adapted/with_preprocessing",
        help="Output directory for embeddings",
    ),
    device: str = typer.Option(
        "cpu",
        help="Device to use (cpu, cuda, mps)",
    ),
    batch_size: int = typer.Option(
        512,
        help="Batch size for encoding",
    ),
) -> None:
    """Generate embeddings using domain-adapted SecBERT.

    This will generate embeddings for both train and test sets using the
    SecBERT model that was domain-adapted via Masked Language Modeling on
    valid HTTP requests.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("ADAPTED SECBERT EMBEDDING GENERATION")
    logger.info("=" * 80)
    logger.info(f"Model: SecBERT (domain-adapted via MLM)")
    logger.info(f"Preprocessing: Enabled")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)

    # Generate train embeddings
    train_output = generate_embeddings_for_split(
        split="train",
        dataset_path=train_dataset,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
    )

    # Generate test embeddings
    test_output = generate_embeddings_for_split(
        split="test",
        dataset_path=test_dataset,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
    )

    logger.info("=" * 80)
    logger.info("EMBEDDING GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Train: {train_output}")
    logger.info(f"Test: {test_output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    app()
