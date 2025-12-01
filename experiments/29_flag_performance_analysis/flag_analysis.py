"""Flag analysis utilities for extracting and analyzing preprocessing flags."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_selection import mutual_info_classif

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
        
        # Extended tracking for advanced analysis
        self.flag_sequences: list[tuple[list[str], str]] = []  # (flag_sequence, label)
        self.flag_presence_matrix: list[dict[str, bool]] = []  # Per-request flag presence
        self.labels: list[str] = []  # Labels for each request
        self.flag_pair_labels: dict[tuple[str, str], list[str]] = {}  # Labels for each pair
        self.flag_frequencies: Counter[str] = Counter()  # Overall frequency

    def add_request(self, request: str, label: str) -> list[str]:
        """Process a request and add its flags to statistics."""
        processed = preprocess(request)
        flags = extract_flags(processed)
        unique_flags = set(flags)
        
        # Track sequence (order matters)
        unique_flag_list = sorted(unique_flags)  # Sorted for consistency
        self.flag_sequences.append((unique_flag_list, label))
        
        # Track per-request presence for MI/correlation
        presence_dict = {flag: True for flag in unique_flags}
        self.flag_presence_matrix.append(presence_dict)
        self.labels.append(label)
        
        # Track frequencies
        self.flag_frequencies.update(flags)

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

        # Track co-occurrence (pairs) with labels
        flag_list = sorted(unique_flags)
        for i, flag1 in enumerate(flag_list):
            for flag2 in flag_list[i + 1 :]:
                pair = (flag1, flag2)
                self.flag_cooccurrence[pair] = self.flag_cooccurrence.get(pair, 0) + 1
                if pair not in self.flag_pair_labels:
                    self.flag_pair_labels[pair] = []
                self.flag_pair_labels[pair].append(label)

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

    def compute_mutual_information(self) -> dict[str, float]:
        """Compute mutual information for each flag."""
        if not self.flag_presence_matrix or not self.labels:
            return {}

        # Build flag presence matrix
        all_flags = set()
        for presence_dict in self.flag_presence_matrix:
            all_flags.update(presence_dict.keys())

        flag_list = sorted(all_flags)
        flag_matrix = np.array(
            [
                [1 if flag in presence_dict else 0 for flag in flag_list]
                for presence_dict in self.flag_presence_matrix
            ]
        )

        # Convert labels to binary
        y = np.array([1 if label == "attack" else 0 for label in self.labels])

        # Compute MI
        mi_scores = mutual_info_classif(flag_matrix, y, random_state=42)

        return dict(zip(flag_list, mi_scores.tolist()))

    def compute_correlation_matrix(self) -> tuple[list[str], np.ndarray]:
        """Compute correlation matrix between flags."""
        if not self.flag_presence_matrix:
            return [], np.array([])

        # Build flag presence matrix
        all_flags = set()
        for presence_dict in self.flag_presence_matrix:
            all_flags.update(presence_dict.keys())

        flag_list = sorted(all_flags)
        flag_matrix = np.array(
            [
                [1 if flag in presence_dict else 0 for flag in flag_list]
                for presence_dict in self.flag_presence_matrix
            ]
        )

        # Compute correlation
        corr_matrix = np.corrcoef(flag_matrix.T)

        return flag_list, corr_matrix

    def compute_interaction_effects(self) -> dict[tuple[str, str], dict[str, float]]:
        """Compute interaction effects (attack rates) for flag pairs."""
        interaction_effects = {}

        for pair, labels_list in self.flag_pair_labels.items():
            attack_count = sum(1 for label in labels_list if label == "attack")
            benign_count = sum(1 for label in labels_list if label != "attack")
            total = len(labels_list)

            if total > 0:
                attack_rate = attack_count / total
                benign_rate = benign_count / total
                signal_strength = attack_rate - benign_rate

                interaction_effects[pair] = {
                    "attack_rate": attack_rate,
                    "benign_rate": benign_rate,
                    "signal_strength": signal_strength,
                    "attack_count": attack_count,
                    "benign_count": benign_count,
                    "total_count": total,
                }

        return interaction_effects

    def compute_rarity_stats(self) -> dict[str, dict[str, float]]:
        """Compute rarity statistics for each flag."""
        total_requests = self.attack_request_count + self.benign_request_count
        if total_requests == 0:
            return {}

        rarity_stats = {}
        for flag, count in self.flag_frequencies.items():
            frequency = count / total_requests
            rarity_stats[flag] = {
                "frequency": frequency,
                "rarity": 1.0 - frequency,  # Inverse of frequency
                "total_count": count,
                "requests_with_flag": self.attack_flag_presence.get(flag, 0)
                + self.benign_flag_presence.get(flag, 0),
            }

        return rarity_stats

    def compute_family_stats(self) -> dict[str, dict[str, Any]]:
        """Compute statistics grouped by flag families."""
        # Define flag families
        flag_families = {
            "encoding": [
                "DOUBLEPCT",
                "PCTSLASH",
                "PCTBACKSLASH",
                "PCTSPACE",
                "PCTCONTROL",
                "PCTNULL",
                "PCTSUSPICIOUS",
            ],
            "query": [
                "QBARE",
                "QEMPTYVAL",
                "QNONASCII",
                "QLONG",
                "QRAWSEMI",
                "QSEMISEP",
                "QNUL",
            ],
            "dangerous_chars": [
                "QUOTE",
                "ANGLE",
                "SEMICOLON",
                "PIPE",
                "BACKSLASH",
                "BRACE",
                "PAREN",
            ],
            "path": ["DOTDOT", "DOTCUR", "MULTIPLESLASH", "HOME"],
            "header": [
                "BADHDRCONT",
                "OBSFOLD",
                "BADCRLF",
                "DUPHDR",
                "HOPBYHOP",
                "BADHDRNAME",
            ],
            "unicode": [
                "FULLWIDTH",
                "CONTROL",
                "MIXEDSCRIPT",
                "INVALID_UNICODE",
                "UNICODE_FORMAT",
                "MATH_UNICODE",
            ],
            "other": [
                "UNUSUAL_METHOD",
                "WSPAD",
                "SPACE",
                "NUL",
                "HOSTMISMATCH",
                "IDNA",
                "BADHOST",
                "HTMLENT",
            ],
        }

        family_stats = {}
        for family_name, family_flags in flag_families.items():
            family_attack_count = sum(
                self.attack_flag_counts.get(flag, 0) for flag in family_flags
            )
            family_benign_count = sum(
                self.benign_flag_counts.get(flag, 0) for flag in family_flags
            )
            family_attack_presence = sum(
                self.attack_flag_presence.get(flag, 0) for flag in family_flags
            )
            family_benign_presence = sum(
                self.benign_flag_presence.get(flag, 0) for flag in family_flags
            )

            attack_rate = (
                family_attack_presence / self.attack_request_count
                if self.attack_request_count > 0
                else 0.0
            )
            benign_rate = (
                family_benign_presence / self.benign_request_count
                if self.benign_request_count > 0
                else 0.0
            )

            family_stats[family_name] = {
                "attack_count": family_attack_count,
                "benign_count": family_benign_count,
                "attack_presence_rate": attack_rate,
                "benign_presence_rate": benign_rate,
                "signal_strength": attack_rate - benign_rate,
                "flags_in_family": len(family_flags),
            }

        return family_stats

    def compute_frequency_distributions(
        self,
    ) -> dict[str, dict[str, float | int]]:
        """Compute detailed frequency distribution statistics for each flag."""
        distributions = {}

        for flag in set(self.attack_flag_presence.keys()) | set(
            self.benign_flag_presence.keys()
        ):
            # Count occurrences per request
            flag_counts = []
            for presence_dict in self.flag_presence_matrix:
                if flag in presence_dict:
                    # Count how many times flag appears (approximate from total count)
                    flag_counts.append(1)  # Presence = 1, could be extended

            if flag_counts:
                counts_array = np.array(flag_counts)
                distributions[flag] = {
                    "mean": float(np.mean(counts_array)),
                    "median": float(np.median(counts_array)),
                    "p25": float(np.percentile(counts_array, 25)),
                    "p75": float(np.percentile(counts_array, 75)),
                    "p95": float(np.percentile(counts_array, 95)),
                    "p99": float(np.percentile(counts_array, 99)),
                    "std": float(np.std(counts_array)),
                    "min": int(np.min(counts_array)),
                    "max": int(np.max(counts_array)),
                    "total_occurrences": self.flag_frequencies.get(flag, 0),
                }

        return distributions

    def compute_sequence_stats(self, top_n: int = 20) -> list[tuple[tuple[str, ...], int, float]]:
        """Compute statistics for flag sequences."""
        sequence_counts: Counter[tuple[str, ...]] = Counter()
        sequence_labels: dict[tuple[str, ...], list[str]] = {}

        for sequence, label in self.flag_sequences:
            seq_tuple = tuple(sequence)
            sequence_counts[seq_tuple] += 1
            if seq_tuple not in sequence_labels:
                sequence_labels[seq_tuple] = []
            sequence_labels[seq_tuple].append(label)

        # Compute attack rates for sequences
        sequence_stats = []
        for sequence, count in sequence_counts.most_common(top_n):
            labels_list = sequence_labels[sequence]
            attack_count = sum(1 for label in labels_list if label == "attack")
            attack_rate = attack_count / len(labels_list) if labels_list else 0.0
            sequence_stats.append((sequence, count, attack_rate))

        return sequence_stats

