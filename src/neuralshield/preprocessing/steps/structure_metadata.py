"""
Helpers for aggregating structural metadata flags.

The preprocessing pipeline occasionally observes structural signals that are
useful for diagnostics but noisy for anomaly detection (e.g., ubiquitous casing
normalisation). Rather than emitting these as inline flags, we aggregate them
under a single `[STRUCT]` metadata line so downstream consumers can choose how
to use them without inflating per-request flag counts.
"""

from __future__ import annotations

from typing import Iterable

STRUCT_PREFIX = "[STRUCT]"


def merge_structure_flags(lines: list[str], flags: Iterable[str]) -> None:
    """
    Merge structural flags into the `[STRUCT]` metadata line.

    Args:
        lines: Mutable list of request lines.
        flags: Iterable of structural flag strings to merge.
    """
    new_flags = {flag for flag in flags if flag}
    if not new_flags:
        return

    struct_index = None
    existing_flags: set[str] = set()

    for idx, line in enumerate(lines):
        if line.startswith(f"{STRUCT_PREFIX} "):
            struct_index = idx
            existing_flags.update(
                part for part in line[len(STRUCT_PREFIX) + 1 :].split() if part
            )
            break

    combined = sorted(existing_flags.union(new_flags))
    if not combined:
        return

    struct_line = f"{STRUCT_PREFIX} {' '.join(combined)}"

    if struct_index is not None:
        lines[struct_index] = struct_line
    else:
        lines.append(struct_line)
