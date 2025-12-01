#!/usr/bin/env python3
"""
Add labels to embedding files that don't have them.

Loads embeddings from NPZ and original dataset labels, then saves a new NPZ with both.
"""

import json
from pathlib import Path

import numpy as np
import typer
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    embeddings_path: Path = typer.Argument(..., help="Input embeddings file (.npz) without labels"),
    dataset_path: Path = typer.Argument(..., help="Original dataset JSONL file with labels"),
    output_path: Path = typer.Argument(..., help="Output embeddings file (.npz) with labels"),
) -> None:
    """Add labels to embeddings from original dataset."""
    logger.info("=" * 80)
    logger.info("ADDING LABELS TO EMBEDDINGS")
    logger.info("=" * 80)
    
    # Load embeddings
    logger.info(f"Loading embeddings from {embeddings_path}")
    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"]
    logger.info(f"Loaded {embeddings.shape[0]} embeddings, {embeddings.shape[1]} dimensions")
    
    # Load labels from dataset
    logger.info(f"Loading labels from {dataset_path}")
    labels = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            labels.append(obj["label"])
    
    logger.info(f"Loaded {len(labels)} labels")
    
    # Verify counts match
    if len(embeddings) != len(labels):
        logger.error(
            f"Mismatch: {len(embeddings)} embeddings but {len(labels)} labels"
        )
        raise ValueError("Embedding and label counts must match")
    
    # Count label distribution
    unique, counts = np.unique(labels, return_counts=True)
    logger.info(f"Label distribution: {dict(zip(unique, counts))}")
    
    # Save with labels
    logger.info(f"Saving embeddings with labels to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        labels=np.array(labels),
    )
    
    logger.info(f"✅ Saved {len(embeddings)} embeddings with labels to {output_path}")


if __name__ == "__main__":
    app()

