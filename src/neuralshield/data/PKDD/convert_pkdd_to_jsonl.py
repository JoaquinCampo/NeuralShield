#!/usr/bin/env python3
"""
Convert the ECML/PKDD HTTP dataset into the common NeuralShield JSONL format.

Output files:
- pkdd_dataset.jsonl (train + test combined)
- train.jsonl (training split)
- test.jsonl (testing split)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "dataset_ecml_pkdd_train_test"
TRAIN_RAW = RAW_DIR / "xml_train.txt"
TEST_RAW = RAW_DIR / "xml_test.txt"
COMBINED_OUTPUT = DATA_DIR / "pkdd_dataset.jsonl"
TRAIN_OUTPUT = DATA_DIR / "train.jsonl"
TEST_OUTPUT = DATA_DIR / "test.jsonl"


class PKDDSample(BaseModel):
    """NeuralShield-friendly representation of a PKDD HTTP request."""

    request: str
    label: str


def normalize_label(raw_label: str) -> str:
    """Map original PKDD class labels to binary labels."""
    cleaned = raw_label.strip().lower()
    return "valid" if cleaned == "valid" else "attack"


def clean_request_lines(lines: list[str]) -> list[str]:
    """
    Trim placeholder/body markers so the request mirrors wire format.

    The PKDD dumps include a standalone "null" line when bodies are absent.
    We drop that marker and any trailing blank padding.
    """
    # Remove leading blank lines that occasionally precede the request line.
    while lines and not lines[0]:
        lines.pop(0)

    # Drop placeholders for empty bodies.
    filtered = [line for line in lines if line != "null"]

    # Strip trailing blank lines introduced by the export tooling.
    while filtered and not filtered[-1]:
        filtered.pop()

    return filtered


def parse_pkdd_file(file_path: Path) -> tuple[list[PKDDSample], Counter]:
    """Parse a PKDD XML dump into PKDDSample objects."""
    logger.info("Parsing {path}", path=file_path)

    samples: list[PKDDSample] = []
    raw_label_counter: Counter[str] = Counter()

    current_id: str | None = None
    current_label: str | None = None
    request_lines: list[str] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")

            if line.startswith("Start - Id:"):
                current_id = line.split(":", 1)[1].strip()
                current_label = None
                request_lines = []
                continue

            if not current_id:
                continue

            if line.startswith("class:"):
                current_label = line.split(":", 1)[1].strip()
                continue

            if line.startswith("End - Id:"):
                if not current_label:
                    logger.warning(
                        "Skipping record {record_id}: missing class label",
                        record_id=current_id,
                    )
                else:
                    cleaned_lines = clean_request_lines(request_lines.copy())
                    if not cleaned_lines:
                        logger.warning(
                            "Skipping record {record_id}: empty request payload",
                            record_id=current_id,
                        )
                    else:
                        try:
                            sample = PKDDSample(
                                request="\n".join(cleaned_lines),
                                label=normalize_label(current_label),
                            )
                        except ValueError as error:
                            logger.error(
                                "Validation error for record {record_id}: {error}",
                                record_id=current_id,
                                error=error,
                            )
                        else:
                            samples.append(sample)
                            raw_label_counter[current_label] += 1

                current_id = None
                current_label = None
                request_lines = []
                continue

            # Preserve blank lines, but drop whitespace-only padding.
            if current_label is None:
                # Some malformed snippets include data before the class line; skip them.
                continue

            request_lines.append(line)

    logger.info(
        "Parsed {count} samples from {path}",
        count=len(samples),
        path=file_path,
    )
    return samples, raw_label_counter


def write_jsonl(samples: list[PKDDSample], output_path: Path) -> None:
    """Write PKDD samples to disk in JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")

    logger.info("Wrote {count} samples to {path}", count=len(samples), path=output_path)


def convert() -> None:
    """Entrypoint for converting all PKDD splits."""
    train_samples, train_counts = parse_pkdd_file(TRAIN_RAW)
    test_samples, test_counts = parse_pkdd_file(TEST_RAW)

    combined_samples = [*train_samples, *test_samples]
    combined_counts: Counter[str] = Counter()
    combined_counts.update(train_counts)
    combined_counts.update(test_counts)

    write_jsonl(combined_samples, COMBINED_OUTPUT)
    write_jsonl(train_samples, TRAIN_OUTPUT)
    write_jsonl(test_samples, TEST_OUTPUT)

    logger.info(
        "Label distribution (raw classes): {counts}", counts=dict(combined_counts)
    )
    logger.info(
        "Binary distribution: {counts}",
        counts={
            "valid": sum(1 for sample in combined_samples if sample.label == "valid"),
            "attack": sum(1 for sample in combined_samples if sample.label == "attack"),
        },
    )


def main() -> None:
    """CLI wrapper."""
    convert()


if __name__ == "__main__":
    main()
