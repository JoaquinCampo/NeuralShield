from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import typer
from loguru import logger

app = typer.Typer(help="Summarize flag token occurrences and compute weighting hints.")


@app.command()
def main(
    samples_path: Path = typer.Option(
        Path("experiments/21_flag_correlation/token_alignment/samples.jsonl"),
        help="JSONL file produced by extract_flag_tokens.py",
    ),
    per_flag_output: Path = typer.Option(
        Path("experiments/21_flag_correlation/token_alignment/token_weights.json"),
        help="Destination JSON file with per-flag token frequencies.",
    ),
    token_lookup_output: Path = typer.Option(
        Path(
            "experiments/21_flag_correlation/token_alignment/token_weight_lookup.json"
        ),
        help="Destination JSON file with aggregated token weights.",
    ),
    token_stats_output: Path = typer.Option(
        Path("experiments/21_flag_correlation/token_alignment/token_log_odds.json"),
        help=(
            "Destination JSON file with per-token label statistics and log-odds scores."
        ),
    ),
    top_k: int = typer.Option(10, help="Number of tokens to keep per flag."),
    min_support: float = typer.Option(
        0.01,
        help="Minimum support for a token to appear in the aggregated lookup.",
    ),
    smoothing: float = typer.Option(
        0.5,
        help=(
            "Additive smoothing applied when computing log-odds "
            "(larger = more conservative)."
        ),
    ),
) -> None:
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    per_flag_tokens: dict[str, Counter[str]] = defaultdict(Counter)
    token_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    total_by_label: Counter[str] = Counter()

    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            flag = record["flag"]
            tokens = record.get("tokens", [])
            label = str(record.get("label", "unknown")).lower()
            for token in tokens:
                per_flag_tokens[flag][token] += 1
                token_label_counts[token][label] += 1
                total_by_label[label] += 1

    summary: dict[str, dict[str, object]] = {}
    token_max_support: dict[str, float] = {}
    for flag, counter in per_flag_tokens.items():
        total = sum(counter.values())
        most_common = counter.most_common(top_k)
        summary[flag] = {
            "total_occurrences": total,
            "tokens": [
                {
                    "token": token,
                    "count": count,
                    "support": count / total if total else 0.0,
                }
                for token, count in most_common
            ],
        }
        for token, count in counter.items():
            support = count / total if total else 0.0
            token_max_support[token] = max(token_max_support.get(token, 0.0), support)

    per_flag_output.parent.mkdir(parents=True, exist_ok=True)
    per_flag_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    token_lookup = {
        token: 1.0 + support
        for token, support in token_max_support.items()
        if support >= min_support
    }
    token_lookup_output.write_text(
        json.dumps(token_lookup, indent=2),
        encoding="utf-8",
    )

    if not total_by_label:
        raise RuntimeError(
            "No token label counts were collected; cannot compute log-odds."
        )

    attack_total = total_by_label.get("attack", 0)
    valid_total = total_by_label.get("valid", 0)
    if attack_total == 0 or valid_total == 0:
        logger.warning(
            "Samples are missing attack ({attack}) or valid ({valid}) labels. "
            "Log-odds will be degenerate.",
            attack=attack_total,
            valid=valid_total,
        )

    token_stats: dict[str, dict[str, float | int]] = {}
    for token, label_counts in token_label_counts.items():
        attack_count = label_counts.get("attack", 0)
        valid_count = label_counts.get("valid", 0)

        attack_prob = (attack_count + smoothing) / (attack_total + smoothing * 2)
        valid_prob = (valid_count + smoothing) / (valid_total + smoothing * 2)

        log_odds = float(math.log(attack_prob / valid_prob))

        token_stats[token] = {
            "attack_count": int(attack_count),
            "valid_count": int(valid_count),
            "support": int(sum(label_counts.values())),
            "log_odds": log_odds,
        }

    token_stats_output.parent.mkdir(parents=True, exist_ok=True)
    token_stats_output.write_text(
        json.dumps(token_stats, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info(
        (
            "Wrote per-flag summary to {flag_path}, token lookup to {token_path}, "
            "and token stats to {stats_path}"
        ),
        flag_path=str(per_flag_output),
        token_path=str(token_lookup_output),
        stats_path=str(token_stats_output),
    )


if __name__ == "__main__":
    app()
