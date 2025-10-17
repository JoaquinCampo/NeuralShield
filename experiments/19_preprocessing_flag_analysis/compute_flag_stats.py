"""
Compute preprocessing flag statistics for normal vs anomalous HTTP requests.

This script samples requests from the SR_BH_2020 dataset, runs them through the
configured preprocessing pipeline, and records aggregate flag statistics. Results
are saved as JSON so they can be inspected or plotted later.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import types
from collections import Counter
from pathlib import Path


def ensure_optional_dependencies() -> None:
    """
    Provide lightweight fallbacks when optional dependencies are missing.

    When running inside constrained sandboxes we might not have access to the
    full dependency set (e.g. `loguru`, `idna`). The preprocessors only need
    minimal functionality, so we register tiny stand-ins that satisfy imports.
    """

    if "loguru" not in sys.modules:
        loguru_module = types.ModuleType("loguru")

        class _Logger:
            def debug(self, *args, **kwargs) -> None:
                return None

            info = debug
            warning = debug
            error = debug

        loguru_module.logger = _Logger()
        sys.modules["loguru"] = loguru_module

    if "idna" not in sys.modules:
        idna_module = types.ModuleType("idna")

        def _encode(text: str) -> bytes:
            try:
                return text.encode("idna")
            except Exception:
                return text.encode()

        def _decode(data: bytes) -> str:
            try:
                return data.decode("idna")
            except Exception:
                return data.decode()

        idna_module.encode = _encode
        idna_module.decode = _decode
        sys.modules["idna"] = idna_module


def reservoir_sample(
    path: Path, *, label: str, limit: int, rng: random.Random
) -> tuple[list[str], int]:
    """
    Reservoir sample `limit` requests matching `label` from a JSONL dataset.

    Returns the sampled list and the total number of matching records observed.
    """
    sample: list[str] = []
    seen = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            if obj.get("label") != label:
                continue

            seen += 1
            request = obj["request"]

            if len(sample) < limit:
                sample.append(request)
            else:
                idx = rng.randrange(seen)
                if idx < limit:
                    sample[idx] = request

    return sample, seen


class FlagStatsAccumulator:
    """
    Incrementally aggregate preprocessing flag statistics.
    """

    def __init__(self, preprocess):
        self._preprocess = preprocess
        self.flag_counts: list[int] = []
        self.unique_counts: list[int] = []
        self.freq: Counter[str] = Counter()
        self.presence: Counter[str] = Counter()
        self.zero_flags = 0
        self.total = 0

    def add(self, request: str) -> None:
        processed = self._preprocess(request)
        flags = extract_flags(processed)

        if not flags:
            self.zero_flags += 1

        self.flag_counts.append(len(flags))
        unique_flags = set(flags)
        self.unique_counts.append(len(unique_flags))

        self.freq.update(flags)
        self.presence.update(unique_flags)
        self.total += 1

    def finalize(self) -> dict[str, object]:
        total = self.total if self.total else 1

        summaries = {
            "flag_count_summary": summarize(self.flag_counts),
            "unique_flag_summary": summarize(self.unique_counts),
            "zero_flag_fraction": self.zero_flags / self.total if self.total else 0.0,
            "top_flags": [],
        }

        top_flags: list[dict[str, object]] = []
        for flag, total_hits in self.freq.most_common(50):
            per_request = total_hits / total
            presence_rate = self.presence[flag] / total
            top_flags.append(
                {
                    "flag": flag,
                    "total": int(total_hits),
                    "per_request": per_request,
                    "presence_rate": presence_rate,
                }
            )

        summaries["top_flags"] = top_flags
        return summaries


FLAG_SET = {
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
    "QNUL",
    "QNONASCII",
    "QLONG",
    "QSQLI_QUOTE_SEMI",
    "PCTSPACE_PAIR",
    "HOME",
    "MULTIPLESLASH",
    "MULTIPLESLASH_HEAVY",
    "DOTCUR",
    "DOTDOT",
    "XSS_TAG",
    "FLAG_RISK_HIGH",
    "FLAG_OVERFLOW",
    "PIPE_REPEAT",
    "BRACE_REPEAT",
    "STRUCT_GAP",
    "FLAG_RISK_HIGH_SUPERFLAGTOKEN_EXPERIMENTAL",
    "QUOTE_SUPERFLAGTOKEN_EXPERIMENTAL",
    "SEMICOLON_SUPERFLAGTOKEN_EXPERIMENTAL",
    "QSQLI_QUOTE_SEMI_SUPERFLAGTOKEN_EXPERIMENTAL",
    "QRAWSEMI_SUPERFLAGTOKEN_EXPERIMENTAL",
    "ANGLE_SUPERFLAGTOKEN_EXPERIMENTAL",
    "XSS_TAG_SUPERFLAGTOKEN_EXPERIMENTAL",
    "PIPE_SUPERFLAGTOKEN_EXPERIMENTAL",
    "PCTSPACE_SUPERFLAGTOKEN_EXPERIMENTAL",
    "QNUL_SUPERFLAGTOKEN_EXPERIMENTAL",
}

PREFIX_FLAGS = ("QARRAY:", "QREPEAT:", "STRUCT_GAP:")


def is_flag(token: str) -> bool:
    return token in FLAG_SET or any(token.startswith(prefix) for prefix in PREFIX_FLAGS)


def extract_flags(processed: str) -> list[str]:
    """
    Extract flag tokens from the processed request.

    Flags are appended at the end of lines (either space or comma separated),
    so we scan tokens in reverse to collect trailing flag groups.
    """
    flags: list[str] = []

    for raw_line in processed.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        tokens = stripped.split()
        if not tokens:
            continue

        start = 1 if tokens[0].startswith("[") else 0
        body = tokens[start:]

        collected: list[str] = []
        for token in reversed(body):
            candidates = [part for part in token.split(",") if part]
            if candidates and all(is_flag(part) for part in candidates):
                for part in reversed(candidates):
                    collected.insert(0, part)
            else:
                break

        flags.extend(collected)

    return flags


def summarize(values: list[int]) -> dict[str, float]:
    """
    Compute basic summary statistics for a list of integers.
    """
    n = len(values)
    if n == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "min": 0.0,
        }

    sorted_vals = sorted(values)
    mean_val = sum(sorted_vals) / n
    if n % 2:
        median_val = float(sorted_vals[n // 2])
    else:
        median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2

    def percentile(p: float) -> float:
        if n == 0:
            return 0.0
        rank = max(1, math.ceil((p / 100) * n))
        return float(sorted_vals[rank - 1])

    return {
        "mean": mean_val,
        "median": median_val,
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": float(sorted_vals[-1]),
        "min": float(sorted_vals[0]),
    }


def collect_flag_stats(requests: list[str], *, preprocess) -> dict[str, object]:
    """
    Run the preprocessing pipeline over each request and aggregate flag statistics.
    """
    accumulator = FlagStatsAccumulator(preprocess)
    for request in requests:
        accumulator.add(request)
    return accumulator.finalize()


def process_all_requests(
    path: Path, *, label: str, accumulator: FlagStatsAccumulator
) -> int:
    """
    Process every request for the given label found in `path`.

    Returns the number of processed requests.
    """
    count = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            if obj.get("label") != label:
                continue

            accumulator.add(obj["request"])
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute preprocessing flag statistics."
    )
    parser.add_argument(
        "--valid-path",
        type=Path,
        default=Path("src/neuralshield/data/SR_BH_2020/train.jsonl"),
        help="JSONL file containing normal requests.",
    )
    parser.add_argument(
        "--attack-path",
        type=Path,
        default=Path("src/neuralshield/data/SR_BH_2020/test.jsonl"),
        help="JSONL file containing attack requests.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50_000,
        help="Number of requests to sample per label.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reservoir sampling.",
    )
    parser.add_argument(
        "--pipeline",
        choices=("balanced", "csic-overfit", "srbh-overfit"),
        default="balanced",
        help="Which preprocessing pipeline to execute.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("flag_stats_summary.json"),
        help="Destination JSON file for summary statistics.",
    )
    args = parser.parse_args()

    ensure_optional_dependencies()

    # Import after we have stubs in place.
    if args.pipeline == "balanced":
        from neuralshield.preprocessing.pipeline import preprocess
    elif args.pipeline == "csic-overfit":
        from neuralshield.preprocessing.pipeline_csic_overfit import (
            preprocess_csic_overfit as preprocess,
        )
    elif args.pipeline == "srbh-overfit":
        from neuralshield.preprocessing.pipeline_srbh_overfit import (
            preprocess_srbh_overfit as preprocess,
        )
    else:
        raise ValueError(f"Unsupported pipeline '{args.pipeline}'")

    rng = random.Random(args.seed)

    report: dict[str, object] = {
        "sample_sizes": {},
        "source_totals": {},
        "label_stats": {},
    }

    if args.sample_size > 0:
        valid_sample, valid_total = reservoir_sample(
            args.valid_path, label="valid", limit=args.sample_size, rng=rng
        )
        attack_sample, attack_total = reservoir_sample(
            args.attack_path, label="attack", limit=args.sample_size, rng=rng
        )

        valid_acc = FlagStatsAccumulator(preprocess)
        for request in valid_sample:
            valid_acc.add(request)

        attack_acc = FlagStatsAccumulator(preprocess)
        for request in attack_sample:
            attack_acc.add(request)

        report["sample_sizes"] = {
            "valid": valid_acc.total,
            "attack": attack_acc.total,
        }
        report["source_totals"] = {
            "valid_train_total": valid_total,
            "attack_pool_total": attack_total,
        }
        report["label_stats"] = {
            "valid": valid_acc.finalize(),
            "attack": attack_acc.finalize(),
        }
    else:
        valid_acc = FlagStatsAccumulator(preprocess)
        valid_total = process_all_requests(
            args.valid_path, label="valid", accumulator=valid_acc
        )

        attack_acc = FlagStatsAccumulator(preprocess)
        attack_total = process_all_requests(
            args.attack_path, label="attack", accumulator=attack_acc
        )

        report["sample_sizes"] = {
            "valid": valid_acc.total,
            "attack": attack_acc.total,
        }
        report["source_totals"] = {
            "valid_train_total": valid_total,
            "attack_pool_total": attack_total,
        }
        report["label_stats"] = {
            "valid": valid_acc.finalize(),
            "attack": attack_acc.finalize(),
        }

    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Saved flag statistics to {args.output}")
    valid_mean = report["label_stats"]["valid"]["flag_count_summary"]["mean"]  # type: ignore[index]
    attack_mean = report["label_stats"]["attack"]["flag_count_summary"]["mean"]  # type: ignore[index]
    print(f"Valid mean flags/request: {valid_mean:.2f}")
    print(f"Attack mean flags/request: {attack_mean:.2f}")


if __name__ == "__main__":
    main()
