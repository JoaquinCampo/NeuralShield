"""Flag analysis utilities for extracting and analyzing preprocessing flags."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from neuralshield.preprocessing.pipeline import preprocess


# Flag set from preprocessing pipeline
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
    "QNONASCII",
    "QLONG",
    "HOME",
    "MULTIPLESLASH",
    "DOTCUR",
    "DOTDOT",
}

PREFIX_FLAGS = ("QARRAY:", "QREPEAT:")


def is_flag(token: str) -> bool:
    """Check if a token is a flag."""
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


class FlagAnalyzer:
    """Analyze flags across requests and compute statistics."""

    def __init__(self):
        self.attack_flag_counts: Counter[str] = Counter()
        self.benign_flag_counts: Counter[str] = Counter()
        self.attack_flag_presence: Counter[str] = Counter()
        self.benign_flag_presence: Counter[str] = Counter()
        self.attack_request_count = 0
        self.benign_request_count = 0
        self.flag_cooccurrence: dict[tuple[str, str], int] = {}
        self.attack_flag_counts_per_request: list[int] = []
        self.benign_flag_counts_per_request: list[int] = []

    def add_request(self, request: str, label: str) -> list[str]:
        """Process a request and add its flags to statistics."""
        processed = preprocess(request)
        flags = extract_flags(processed)
        unique_flags = set(flags)

        if label == "attack":
            self.attack_flag_counts.update(flags)
            self.attack_flag_presence.update(unique_flags)
            self.attack_request_count += 1
            self.attack_flag_counts_per_request.append(len(flags))
        else:
            self.benign_flag_counts.update(flags)
            self.benign_flag_presence.update(unique_flags)
            self.benign_request_count += 1
            self.benign_flag_counts_per_request.append(len(flags))

        # Track co-occurrence (pairs)
        flag_list = sorted(unique_flags)
        for i, flag1 in enumerate(flag_list):
            for flag2 in flag_list[i + 1 :]:
                pair = (flag1, flag2)
                self.flag_cooccurrence[pair] = self.flag_cooccurrence.get(pair, 0) + 1

        return flags

    def compute_statistics(self) -> dict[str, Any]:
        """Compute comprehensive flag statistics."""
        all_flags = set(self.attack_flag_presence.keys()) | set(
            self.benign_flag_presence.keys()
        )

        flag_stats = {}
        for flag in all_flags:
            attack_count = self.attack_flag_counts[flag]
            benign_count = self.benign_flag_counts[flag]
            attack_presence = self.attack_flag_presence[flag]
            benign_presence = self.benign_flag_presence[flag]

            attack_rate = (
                attack_presence / self.attack_request_count
                if self.attack_request_count > 0
                else 0.0
            )
            benign_rate = (
                benign_presence / self.benign_request_count
                if self.benign_request_count > 0
                else 0.0
            )
            signal_strength = attack_rate - benign_rate

            flag_stats[flag] = {
                "attack_count": attack_count,
                "benign_count": benign_count,
                "attack_presence_rate": attack_rate,
                "benign_presence_rate": benign_rate,
                "signal_strength": signal_strength,
                "attack_per_request": (
                    attack_count / self.attack_request_count
                    if self.attack_request_count > 0
                    else 0.0
                ),
                "benign_per_request": (
                    benign_count / self.benign_request_count
                    if self.benign_request_count > 0
                else 0.0
                ),
            }

        return {
            "flag_statistics": flag_stats,
            "summary": {
                "total_flags": len(all_flags),
                "attack_requests": self.attack_request_count,
                "benign_requests": self.benign_request_count,
                "attack_avg_flags_per_request": (
                    sum(self.attack_flag_counts_per_request)
                    / len(self.attack_flag_counts_per_request)
                    if self.attack_flag_counts_per_request
                    else 0.0
                ),
                "benign_avg_flags_per_request": (
                    sum(self.benign_flag_counts_per_request)
                    / len(self.benign_flag_counts_per_request)
                    if self.benign_flag_counts_per_request
                    else 0.0
                ),
            },
            "cooccurrence": self.flag_cooccurrence,
        }

