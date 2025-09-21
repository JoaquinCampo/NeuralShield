"""Header obs-fold unfolding processor.

Implements the rules from specs/steps/lineas-plegadas-obs-fold.md to
normalize obsolete header folding (obs-fold) by joining continuation lines,
emitting flags for folded headers, and flagging malformed continuations.
"""

from __future__ import annotations

from typing import List, Set

from loguru import logger

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor

HEADER_PREFIX = "[HEADER] "
CONTINUATION_PREFIXES = (" ", "\t")
OBSFOLD_FLAG = "OBSFOLD"
BADHDRCONT_FLAG = "BADHDRCONT"
BADCRLF_FLAG = "BADCRLF"


class ObsFoldUnfolder(HttpPreprocessor):
    """Unfold obsolete header continuations (obs-fold)."""

    def __init__(self) -> None:
        # Stateless processor
        pass

    @staticmethod
    def _sanitize_segment(segment: str) -> tuple[str, bool]:
        """Replace embedded CR/LF characters with spaces.

        Returns sanitized segment and whether CR/LF were removed.
        """

        if "\r" not in segment and "\n" not in segment:
            return segment, False

        logger.debug("Removing embedded CR/LF from header segment")
        sanitized = segment.replace("\r", " ").replace("\n", " ")
        return sanitized, True

    @staticmethod
    def _append_segment(current: str, segment: str) -> str:
        """Join continuation segment ensuring a single space separator."""

        base = current.rstrip(" \t")
        if not segment:
            return base
        if not base:
            return segment
        return f"{base} {segment}"

    def process(self, request: str) -> str:
        """Process structured request and unfold header continuations."""

        lines = request.strip().split("\n")
        processed_lines: List[str] = []

        current_header: str | None = None
        current_flags: Set[str] = set()
        badhdrcont_emitted = False

        def flush_current() -> None:
            nonlocal current_header, current_flags
            if current_header is None:
                return
            final_content = current_header.rstrip(" \t")
            processed_lines.append(f"{HEADER_PREFIX}{final_content}")
            for flag in sorted(current_flags):
                processed_lines.append(flag)
            current_header = None
            current_flags = set()

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if not line.startswith(HEADER_PREFIX):
                flush_current()
                processed_lines.append(line)
                continue

            header_content = line[len(HEADER_PREFIX) :]

            if header_content.startswith(CONTINUATION_PREFIXES):
                if current_header is None:
                    if not badhdrcont_emitted:
                        processed_lines.append(BADHDRCONT_FLAG)
                        badhdrcont_emitted = True
                    logger.debug("Observed header continuation without base header")
                    continue

                continuation = header_content.lstrip(" \t")
                continuation, had_crlf = self._sanitize_segment(continuation)
                if had_crlf:
                    current_flags.add(BADCRLF_FLAG)
                current_flags.add(OBSFOLD_FLAG)
                current_header = self._append_segment(current_header, continuation)
                continue

            flush_current()
            base_content, had_crlf = self._sanitize_segment(header_content)
            current_header = base_content
            current_flags = set()
            if had_crlf:
                current_flags.add(BADCRLF_FLAG)

        flush_current()

        return "\n".join(processed_lines)
