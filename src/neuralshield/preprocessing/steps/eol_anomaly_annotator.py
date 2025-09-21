"""EOL anomaly annotation preprocessing step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


@dataclass(frozen=True)
class _Line:
    text: str
    newline: str
    flags: List[tuple[str, str]]


class EolAnomalyAnnotator(HttpPreprocessor):
    """Detect and flag unusual end-of-line patterns without altering bytes."""

    _FLAG_PREFIX = "EOL_"

    def process(self, request: str) -> str:
        lines = self._collect_lines(request)
        if not lines:
            return request

        rendered: List[str] = []
        seen_types: set[str] = set()

        for index, line in enumerate(lines):
            rendered.append(line.text)
            rendered.append(line.newline)

            if self._is_flag_line(line.text):
                for flag, flag_newline in line.flags:
                    rendered.append(flag)
                    rendered.append(flag_newline)
                continue

            existing = {flag for flag, _ in line.flags}
            new_flags: List[str] = []

            eol_type = self._classify_eol(line.newline)
            if eol_type:
                if seen_types and eol_type not in seen_types and "EOLMIX" not in existing:
                    new_flags.append("EOLMIX")
                seen_types.add(eol_type)

            if eol_type == "LF" and "EOL_BARELF" not in existing:
                new_flags.append("EOL_BARELF")
            if eol_type == "CR" and "EOL_BARECR" not in existing:
                new_flags.append("EOL_BARECR")

            if (
                index == len(lines) - 1
                and line.newline == ""
                and "EOL_EOF_NOCRLF" not in existing
            ):
                new_flags.append("EOL_EOF_NOCRLF")

            needs_separator = line.newline == ""

            for flag, flag_newline in line.flags:
                if needs_separator:
                    rendered.append("\n")
                    needs_separator = False
                rendered.append(flag)
                rendered.append(flag_newline)

            for flag in new_flags:
                if needs_separator:
                    rendered.append("\n")
                    needs_separator = False
                rendered.append(flag)
                rendered.append("\n")

        result = "".join(rendered)
        if result and not result.endswith(("\n", "\r")):
            result += "\n"
        return result

    def _collect_lines(self, request: str) -> List[_Line]:
        pairs = list(self._iter_lines(request))
        lines: List[_Line] = []
        pending_orphans: List[tuple[str, str]] = []

        i = 0
        while i < len(pairs):
            text, newline = pairs[i]
            if self._is_flag_line(text):
                pending_orphans.append((text, newline))
                i += 1
                continue

            flags: List[tuple[str, str]] = []
            j = i + 1
            while j < len(pairs) and self._is_flag_line(pairs[j][0]):
                flags.append(pairs[j])
                j += 1

            lines.append(_Line(text=text, newline=newline, flags=flags))
            i = j

        if pending_orphans:
            lines.extend(
                _Line(text=flag, newline=flag_newline, flags=[])
                for flag, flag_newline in pending_orphans
            )

        return lines

    def _iter_lines(self, request: str) -> Iterable[tuple[str, str]]:
        length = len(request)
        if length == 0:
            return

        i = 0
        start = 0
        while i < length:
            ch = request[i]
            if ch == "\r":
                if i + 1 < length and request[i + 1] == "\n":
                    yield request[start:i], "\r\n"
                    i += 2
                    start = i
                    continue
                yield request[start:i], "\r"
                i += 1
                start = i
                continue
            if ch == "\n":
                yield request[start:i], "\n"
                i += 1
                start = i
                continue
            i += 1

        if start < length:
            yield request[start:length], ""

    def _classify_eol(self, newline: str) -> str | None:
        if newline == "\r\n":
            return "CRLF"
        if newline == "\n":
            return "LF"
        if newline == "\r":
            return "CR"
        return None

    def _is_flag_line(self, line: str) -> bool:
        if line.startswith(self._FLAG_PREFIX) or line == "EOLMIX":
            return True
        return "REMOVED" in line
