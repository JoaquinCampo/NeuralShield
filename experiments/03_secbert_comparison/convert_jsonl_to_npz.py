#!/usr/bin/env python3
"""Convert JSONL embeddings to NPZ format for hyperparameter search."""

import json
from pathlib import Path

import numpy as np
import typer
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    jsonl_path: Path = typer.Argument(..., help="Input JSONL file"),
    npz_path: Path = typer.Argument(..., help="Output NPZ file"),
) -> None:
    """Convert JSONL embeddings to NumPy NPZ format."""
    logger.info(f"Converting {jsonl_path} to {npz_path}")

    all_embeddings = []
    all_labels = []

    # Read JSONL file
    with open(jsonl_path, "r", encoding="utf-8") as f:
        # Skip header line
        header = json.loads(f.readline())
        logger.info(f"Header: {header}")

        # Read batches
        for line in f:
            batch = json.loads(line)
            all_embeddings.extend(batch["embeddings"])
            all_labels.extend(batch["labels"])

    # Convert to numpy arrays
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    labels_array = np.array(all_labels)

    logger.info(
        f"Loaded {embeddings_array.shape[0]} samples, "
        f"embedding dim={embeddings_array.shape[1]}"
    )
    logger.info(f"Label distribution: {np.unique(labels_array, return_counts=True)}")

    # Save as NPZ
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, embeddings=embeddings_array, labels=labels_array)

    logger.info(f"✅ Saved to {npz_path}")


if __name__ == "__main__":
    app()
