#!/usr/bin/env python3
"""
Create train/test split from CSIC dataset.

- Train: 47,000 randomly sampled valid requests
- Test: Remaining valid requests (25,000) + all attack requests (25,065)
- Ensures no overlap between train and test sets
"""

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load the JSONL dataset into memory."""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def create_train_test_split(samples: List[Dict[str, Any]], train_size: int = 47000):
    """Create train/test split ensuring no overlap."""
    # Separate valid and attack samples
    valid_samples = [s for s in samples if s["label"] == "valid"]
    attack_samples = [s for s in samples if s["label"] == "attack"]

    print(f"Total samples: {len(samples)}")
    print(f"Valid samples: {len(valid_samples)}")
    print(f"Attack samples: {len(attack_samples)}")

    # Randomly sample from valid requests for training
    random.shuffle(valid_samples)
    train_valid = valid_samples[:train_size]
    test_valid = valid_samples[train_size:]

    # Test set = remaining valid + all attacks
    test_samples = test_valid + attack_samples

    print(f"Train samples: {len(train_valid)} valid")
    print(
        f"Test samples: {len(test_valid)} valid + {len(attack_samples)} attack = {len(test_samples)} total"
    )

    return train_valid, test_samples


def save_jsonl(samples: List[Dict[str, Any]], output_path: Path):
    """Save samples to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples to {output_path}")


def main():
    input_file = (
        Path(__file__).parent
        / "src"
        / "neuralshield"
        / "data"
        / "CSIC"
        / "csic_dataset.jsonl"
    )
    output_dir = Path(__file__).parent / "src" / "neuralshield" / "data" / "CSIC"

    print(f"Loading dataset from {input_file}")

    # Load all samples
    samples = load_dataset(input_file)

    # Create train/test split
    train_samples, test_samples = create_train_test_split(samples)

    # Save splits
    save_jsonl(train_samples, output_dir / "train.jsonl")
    save_jsonl(test_samples, output_dir / "test.jsonl")

    print("\nTrain/test split complete!")
    print(f"Train: {len(train_samples)} samples (all valid)")
    print(
        f"Test: {len(test_samples)} samples ({len([s for s in test_samples if s['label'] == 'valid'])} valid + {len([s for s in test_samples if s['label'] == 'attack'])} attack)"
    )


if __name__ == "__main__":
    main()
