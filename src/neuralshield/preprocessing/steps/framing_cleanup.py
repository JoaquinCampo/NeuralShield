"""Framing cleanup preprocessing step."""

from __future__ import annotations

import unicodedata
from typing import List, Tuple

from loguru import logger

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


class FramingCleanup(HttpPreprocessor):
    """Remove BOM and edge control characters, emitting flags."""

    _BOM = "\ufeff"

    def process(self, request: str) -> str:
        """Strip framing artifacts from *request* and append flags if needed."""
        flags: List[str] = []
        working, bom_removed = self._strip_bom(request)
        if bom_removed:
            flags.append("BOMREMOVED")

        working, leading_removed, trailing_removed = self._strip_edge_controls(working)
        if leading_removed or trailing_removed:
            flags.append("EDGECTRLREMOVED")

        if not flags:
            return working

        logger.debug(
            "Framing cleanup removed {} (leading_controls={}, trailing_controls={})",
            " ".join(flags),
            leading_removed,
            trailing_removed,
        )

        if not working.endswith(("\n", "\r")):
            working += "\n"

        return f"{working}{' '.join(flags)}"

    def _strip_bom(self, request: str) -> Tuple[str, bool]:
        if request.startswith(self._BOM):
            return request[len(self._BOM) :], True
        return request, False

    def _strip_edge_controls(self, request: str) -> Tuple[str, int, int]:
        leading = 0
        trailing = 0
        start = 0
        end = len(request)

        while start < end and self._is_edge_control(request[start]):
            start += 1
            leading += 1

        while end > start and self._is_edge_control(request[end - 1]):
            end -= 1
            trailing += 1

        if start == 0 and end == len(request):
            return request, leading, trailing

        return request[start:end], leading, trailing

    @staticmethod
    def _is_edge_control(ch: str) -> bool:
        return unicodedata.category(ch) == "Cc" and ch not in "\t\r\n"
