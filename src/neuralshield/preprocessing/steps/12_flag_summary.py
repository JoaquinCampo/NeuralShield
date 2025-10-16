"""Emit aggregated flag summaries and overflow markers."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Dict, Iterable, List

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


class FlagSummaryEmitter(HttpPreprocessor):
    """Summarise per-request flag families and emit overflow indicators."""

    CONFIG_PATH = Path("src/neuralshield/preprocessing/config.toml")
    DEFAULT_OVERFLOW_THRESHOLD = 12

    ORDERED_FAMILIES = [
        "danger",
        "encoding",
        "unicode",
        "query",
        "header",
        "traversal",
        "network",
        "structure",
    ]

    FAMILY_MAP: Dict[str, set[str]] = {
        "danger": {
            "ANGLE",
            "QUOTE",
            "SEMICOLON",
            "BRACE",
            "PIPE",
            "BACKSLASH",
            "SPACE",
            "NUL",
            "MIXEDSCRIPT",
        },
        "encoding": {
            "DOUBLEPCT",
            "PCTSLASH",
            "PCTBACKSLASH",
            "PCTSPACE",
            "PCTCONTROL",
            "PCTNULL",
            "PCTSUSPICIOUS",
            "HTMLENT",
        },
        "unicode": {
            "FULLWIDTH",
            "CONTROL",
            "UNICODE_FORMAT",
            "MATH_UNICODE",
            "INVALID_UNICODE",
        },
        "query": {
            "QSEMISEP",
            "QRAWSEMI",
            "QBARE",
            "QEMPTYVAL",
            "QNUL",
            "QNONASCII",
            "QLONG",
            "QARRAY",
            "QREPEAT",
            "QKEY_SYMBOL",
            "QKEY_EMPTY",
        },
        "header": {
            "BADHDRCONT",
            "OBSFOLD",
            "BADCRLF",
            "BADHDRNAME",
            "DUPHDR",
            "HDRMERGE",
            "WSPAD",
        },
        "traversal": {
            "DOTCUR",
            "DOTDOT",
        },
        "network": {
            "HOSTMISMATCH",
            "IDNA",
            "BADHOST",
            "UNUSUAL_METHOD",
        },
        "structure": {
            "HDRNORM",
            "HOPBYHOP",
            "PAREN",
            "MULTIPLESLASH",
            "HOME",
        },
    }

    FLAG_EXCLUSIONS = {"FLAG_OVERFLOW"}
    KNOWN_FLAGS = set().union(*FAMILY_MAP.values()).union({"FLAG_OVERFLOW"})

    def __init__(self) -> None:
        self.overflow_threshold = self._load_overflow_threshold()

    def process(self, request: str) -> str:
        lines = [line for line in request.split("\n") if not line.startswith("[FLAG_SUMMARY]")]

        family_counts = {family: 0 for family in self.ORDERED_FAMILIES}
        other_count = 0
        risk_total = 0

        flags_line_index = None
        qmeta_index = None

        for idx, line in enumerate(lines):
            if line.startswith("[FLAGS] "):
                flags_line_index = idx
            if line.startswith("[QMETA] "):
                qmeta_index = idx

            for token in self._extract_flag_tokens(line):
                base = self._normalize_flag(token)
                if base in self.FLAG_EXCLUSIONS:
                    continue
                family = self._flag_family(base)
                if family:
                    family_counts[family] += 1
                    if family != "structure":
                        risk_total += 1
                else:
                    other_count += 1
                    risk_total += 1

        if risk_total > self.overflow_threshold:
            lines, flags_line_index, qmeta_index = self._ensure_flag(
                lines,
                flags_line_index,
                qmeta_index,
                "FLAG_OVERFLOW",
            )

        summary_parts: List[str] = []
        for family in self.ORDERED_FAMILIES:
            summary_parts.append(f"{family}={family_counts.get(family, 0)}")
        summary_parts.append(f"other={other_count}")
        summary_parts.append(f"total={risk_total}")

        summary_line = f"[FLAG_SUMMARY] {' '.join(summary_parts)}"

        insert_index = qmeta_index if qmeta_index is not None else len(lines)
        lines.insert(insert_index, summary_line)

        return "\n".join(lines)

    def _load_overflow_threshold(self) -> int:
        try:
            with self.CONFIG_PATH.open("rb") as cfg_file:
                data = tomllib.load(cfg_file)
            return int(
                data["tool"]["neuralshield"]["flag_summary"].get(
                    "overflow_threshold", self.DEFAULT_OVERFLOW_THRESHOLD
                )
            )
        except Exception:
            return self.DEFAULT_OVERFLOW_THRESHOLD

    def _extract_flag_tokens(self, line: str) -> Iterable[str]:
        if not line:
            return []
        if (
            line.startswith("[METHOD]")
            or line.startswith("[HAGG]")
            or line.startswith("[QMETA]")
        ):
            return []

        tokens = line.split()
        if not tokens:
            return []

        start_idx = 1 if tokens[0].startswith("[") else 0
        flags: List[str] = []
        for token in tokens[start_idx:]:
            candidate = token.strip()
            if not candidate:
                continue
            base = self._normalize_flag(candidate)
            if base in self.KNOWN_FLAGS or base.isupper():
                flags.append(candidate)

        return flags

    def _normalize_flag(self, token: str) -> str:
        return token.split(":", 1)[0]

    def _flag_family(self, flag: str) -> str | None:
        for family, members in self.FAMILY_MAP.items():
            if flag in members:
                return family
        return None

    def _ensure_flag(
        self,
        lines: list[str],
        flags_index: int | None,
        qmeta_index: int | None,
        flag: str,
    ) -> tuple[list[str], int | None, int | None]:
        if flags_index is not None:
            existing = lines[flags_index][8:].split()
            if flag not in existing:
                updated = list(dict.fromkeys(existing + [flag]))
                lines[flags_index] = f"[FLAGS] {' '.join(updated)}"
            return lines, flags_index, qmeta_index

        insert_at = qmeta_index if qmeta_index is not None else len(lines)
        lines.insert(insert_at, f"[FLAGS] {flag}")

        if qmeta_index is not None:
            qmeta_index += 1

        return lines, insert_at, qmeta_index
