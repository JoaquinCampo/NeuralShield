from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import typer
from loguru import logger

from neuralshield.preprocessing.pipeline import preprocess
from neuralshield.preprocessing.steps.exceptions import MalformedHttpRequestError

app = typer.Typer(
    help="Compute preprocessing flag correlation statistics for SR_BH 2020."
)

# Fixed flag tokens emitted by the preprocessing pipeline.
FIXED_FLAGS: tuple[str, ...] = (
    "UNUSUAL_METHOD",
    "BADHDRCONT",
    "OBSFOLD",
    "BADCRLF",
    "BADHDRNAME",
    "DUPHDR",
    "HOPBYHOP",
    "HDRMERGE",
    "HDRNORM",
    "WSPAD",
    "ANGLE",
    "QUOTE",
    "SEMICOLON",
    "PAREN",
    "BRACE",
    "PIPE",
    "BACKSLASH",
    "SPACE",
    "NUL",
    "QNUL",
    "MIXEDSCRIPT",
    "HOSTMISMATCH",
    "IDNA",
    "BADHOST",
    "FULLWIDTH",
    "CONTROL",
    "UNICODE_FORMAT",
    "MATH_UNICODE",
    "INVALID_UNICODE",
    "DOUBLEPCT",
    "PCTSLASH",
    "PCTBACKSLASH",
    "PCTSPACE",
    "PCTCONTROL",
    "PCTNULL",
    "PCTSUSPICIOUS",
    "HTMLENT",
    "QSEMISEP",
    "QRAWSEMI",
    "QBARE",
    "QEMPTYVAL",
    "QNONASCII",
    "QLONG",
    "HOME",
    "MULTIPLESLASH",
    "DOTCUR",
    "DOTDOT",
)

# Parameterised flag prefixes (e.g., QARRAY:user, QREPEAT:foo).
PREFIX_FLAGS: tuple[str, ...] = ("QARRAY:", "QREPEAT:")


def extract_flags(processed: str) -> set[str]:
    """Extract flag tokens from a preprocessed request."""

    flags: set[str] = set()
    tokens = processed.replace(",", " ").split()

    for token in tokens:
        if token in FIXED_FLAGS:
            flags.add(token)
            continue
        for prefix in PREFIX_FLAGS:
            if token.startswith(prefix):
                flags.add(prefix)
                break

    return flags


def iter_requests(jsonl_path: Path):
    """Yield (label, request) pairs from a JSONL dataset."""

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            label = obj.get("label", "unknown")
            request = obj.get("request") or ""
            yield label, request


@app.command()
def main(
    dataset_root: Path = typer.Option(
        Path("src/neuralshield/data/SR_BH_2020"),
        help="Directory containing train.jsonl and test.jsonl",
    ),
    output: Path = typer.Option(
        Path("experiments/21_flag_correlation/results.json"),
        help="Destination file for aggregated statistics.",
    ),
) -> None:
    train_path = dataset_root / "train.jsonl"
    test_path = dataset_root / "test.jsonl"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Expected train/test JSONL files under {dataset_root}")

    logger.info(f"Processing dataset {dataset_root}")

    per_flag_counts: dict[str, Counter[str]] = defaultdict(Counter)
    per_flag_presence: dict[str, Counter[str]] = defaultdict(Counter)
    label_totals: Counter[str] = Counter()

    for split_name, path in (("train", train_path), ("test", test_path)):
        for label, request in iter_requests(path):
            label_totals[label] += 1
            try:
                processed = preprocess(request)
            except MalformedHttpRequestError:
                logger.debug("Skipping malformed request in {split}", split=split_name)
                continue
            flags = extract_flags(processed)

            for flag in flags:
                per_flag_presence[flag][label] += 1
            # Count total occurrences (including repeats) via simple membership check.
            for flag in FIXED_FLAGS:
                if flag in flags:
                    per_flag_counts[flag][label] += 1
            for prefix in PREFIX_FLAGS:
                if prefix in flags:
                    per_flag_counts[prefix][label] += 1

    total_requests = sum(label_totals.values())

    stats: dict[str, dict[str, float | int]] = {}
    for flag, label_counts in per_flag_presence.items():
        present_attack = label_counts.get("attack", 0)
        present_valid = label_counts.get("valid", 0)
        total_attack = label_totals.get("attack", 0)
        total_valid = label_totals.get("valid", 0)

        p_attack = present_attack / total_attack if total_attack else 0.0
        p_valid = present_valid / total_valid if total_valid else 0.0

        # Compute odds ratio with smoothing to avoid division by zero.
        eps = 1e-6
        odds_attack = p_attack / max(1.0 - p_attack, eps)
        odds_valid = p_valid / max(1.0 - p_valid, eps)
        odds_ratio = odds_attack / max(odds_valid, eps)

        stats[flag] = {
            "present_in_attack": present_attack,
            "present_in_valid": present_valid,
            "p_flag_given_attack": p_attack,
            "p_flag_given_valid": p_valid,
            "odds_ratio": odds_ratio,
            "total_occurrences_attack": per_flag_counts.get(flag, {}).get("attack", 0),
            "total_occurrences_valid": per_flag_counts.get(flag, {}).get("valid", 0),
        }

    summary = {
        "label_totals": label_totals,
        "total_requests": total_requests,
        "flags": stats,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved flag statistics to {path}", path=str(output))


if __name__ == "__main__":
    app()
