#!/usr/bin/env python3
"""
Create train/test split from SR_BH dataset.

- Train: N randomly sampled valid requests (pure anomaly detection)
- Test: Remaining valid requests + all attack requests
- Ensures no overlap between train and test sets

SR_BH has ~154K valid and ~753K attacks, so we have flexibility in train size.
"""

import json
import random
from pathlib import Path

from loguru import logger


def load_dataset(file_path: Path) -> tuple[list[dict], list[dict]]:
    """
    Load JSONL dataset and separate valid/attack samples.

    Returns:
        (valid_samples, attack_samples)
    """
    logger.info("Loading dataset from {path}", path=file_path)

    valid_samples = []
    attack_samples = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                if sample["label"] == "valid":
                    valid_samples.append(sample)
                else:
                    attack_samples.append(sample)

                if line_num % 50000 == 0:
                    logger.info("Loaded {count} samples...", count=line_num)

            except Exception as e:
                logger.warning(
                    "Skipping line {line_num}: {error}",
                    line_num=line_num,
                    error=e,
                )
                continue

    logger.info(
        "Loaded {total} samples: {valid} valid, {attack} attack",
        total=len(valid_samples) + len(attack_samples),
        valid=len(valid_samples),
        attack=len(attack_samples),
    )

    return valid_samples, attack_samples


def create_train_test_split(
    valid_samples: list[dict],
    attack_samples: list[dict],
    *,
    train_size: int = 100000,
    random_seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Create train/test split for anomaly detection.

    Args:
        valid_samples: All valid samples
        attack_samples: All attack samples
        train_size: Number of valid samples for training
        random_seed: Random seed for reproducibility

    Returns:
        (train_samples, test_samples)
    """
    random.seed(random_seed)

    logger.info("Creating train/test split with train_size={size}", size=train_size)

    # Validate train size
    if train_size > len(valid_samples):
        logger.warning(
            "Requested train_size {requested} > available valid samples {available}. "
            "Using {available} instead.",
            requested=train_size,
            available=len(valid_samples),
        )
        train_size = len(valid_samples)

    # Shuffle valid samples
    shuffled_valid = valid_samples.copy()
    random.shuffle(shuffled_valid)

    # Split valid samples
    train_valid = shuffled_valid[:train_size]
    test_valid = shuffled_valid[train_size:]

    # Test set = remaining valid + all attacks
    test_samples = test_valid + attack_samples

    # Shuffle test set
    random.shuffle(test_samples)

    logger.info(
        "Split complete: train={train} valid, test={test_valid} valid + {test_attack} attack = {test_total} total",
        train=len(train_valid),
        test_valid=len(test_valid),
        test_attack=len(attack_samples),
        test_total=len(test_samples),
    )

    return train_valid, test_samples


def save_jsonl(samples: list[dict], output_path: Path) -> None:
    """Save samples to JSONL file."""
    logger.info(
        "Saving {count} samples to {path}", count=len(samples), path=output_path
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info("Saved {count} samples", count=len(samples))


def main():
    """Main entry point."""
    input_file = Path(__file__).parent / "srbh_dataset.jsonl"
    output_dir = Path(__file__).parent

    if not input_file.exists():
        logger.error("Input file not found: {path}", path=input_file)
        logger.error("Please run convert_srbh_to_jsonl.py first")
        return

    # Load dataset
    valid_samples, attack_samples = load_dataset(input_file)

    # Create split - using 100K for training (similar proportion to CSIC)
    train_samples, test_samples = create_train_test_split(
        valid_samples,
        attack_samples,
        train_size=100000,  # ~65% of valid samples
        random_seed=42,
    )

    # Save splits
    save_jsonl(train_samples, output_dir / "train.jsonl")
    save_jsonl(test_samples, output_dir / "test.jsonl")

    logger.info("Train/test split complete!")
    logger.info("Train: {train} samples (all valid)", train=len(train_samples))
    logger.info(
        "Test: {test} samples ({valid} valid + {attack} attack)",
        test=len(test_samples),
        valid=len([s for s in test_samples if s["label"] == "valid"]),
        attack=len([s for s in test_samples if s["label"] == "attack"]),
    )


if __name__ == "__main__":
    main()
