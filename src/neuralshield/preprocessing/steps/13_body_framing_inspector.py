from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


@dataclass
class HeaderEntry:
    index: int
    raw_name: str
    name: str
    value: str
    existing_flags: set[str]


class BodyFramingInspector(HttpPreprocessor):
    """
    Enforce RFC 9112 body framing rules across Content-Length and Transfer-Encoding.

    Rules implemented:
    - §6: Content-Length value validation and agreement (BADCL, CLMISMATCH)
    - §6: Transfer-Encoding vs Content-Length conflicts (TECLCONFLICT)
    - §6: HTTP/1.0 request MUST NOT send Transfer-Encoding (TEHTTP10)
    - §6: Transfer-Encoding must end with chunked (TEBADEND)
    """

    KNOWN_HEADER_FLAGS = {
        "DUPHDR",
        "HDRMERGE",
        "HOPBYHOP",
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
        "IDNA",
        "BADHOST",
        "HOSTNOTEMPTY",
        "EMPTYHOST",
        "BADHDRNAME",
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
        # Flags introduced by this step
        "BADCL",
        "CLMISMATCH",
        "TEBADEND",
    }

    def process(self, request: str) -> str:
        lines = request.split("\n")
        header_entries = self._rule_collect_headers(lines)
        header_map = {entry.index: entry for entry in header_entries}

        version = self._rule_extract_version(lines)
        inline_flags: dict[int, set[str]] = defaultdict(set)
        global_flags: set[str] = set()

        content_length_entries = [
            entry for entry in header_entries if entry.name == "content-length"
        ]
        transfer_encoding_entries = [
            entry for entry in header_entries if entry.name == "transfer-encoding"
        ]

        valid_lengths = []
        for entry in content_length_entries:
            is_valid, value = self._rule_validate_content_length(entry.value)
            if not is_valid:
                inline_flags[entry.index].add("BADCL")
            else:
                valid_lengths.append(value)

        if len(content_length_entries) > 1 and (
            len(valid_lengths) != len(content_length_entries)
            or len(set(valid_lengths)) > 1
        ):
            for entry in content_length_entries:
                inline_flags[entry.index].add("CLMISMATCH")

        if content_length_entries and transfer_encoding_entries:
            global_flags.add("TECLCONFLICT")

        if version.upper() == "HTTP/1.0" and transfer_encoding_entries:
            global_flags.add("TEHTTP10")

        for entry in transfer_encoding_entries:
            if not self._rule_te_ends_with_chunked(entry.value):
                inline_flags[entry.index].add("TEBADEND")

        rendered_lines = []
        for idx, line in enumerate(lines):
            entry = header_map.get(idx)
            if entry is None:
                rendered_lines.append(line)
                continue

            combined_flags = entry.existing_flags.union(inline_flags.get(idx, set()))
            rendered_lines.append(self._rule_render_header(entry, combined_flags))

        if global_flags:
            rendered_lines.append(self._rule_emit_global_flags(global_flags))

        return "\n".join(rendered_lines)

    def _rule_collect_headers(self, lines: Iterable[str]) -> list[HeaderEntry]:
        """Collect header metadata while separating known inline flags."""

        entries: list[HeaderEntry] = []
        for index, line in enumerate(lines):
            if not line.startswith("[HEADER] "):
                continue
            header_body = line[9:]
            if ":" not in header_body:
                continue
            raw_name, value_part = header_body.split(":", 1)
            raw_name = raw_name.strip()
            value_part = value_part.strip()
            value, existing_flags = self._rule_split_value_and_flags(value_part)
            entries.append(
                HeaderEntry(
                    index=index,
                    raw_name=raw_name,
                    name=raw_name.lower(),
                    value=value,
                    existing_flags=existing_flags,
                )
            )
        return entries

    def _rule_split_value_and_flags(self, value_part: str) -> tuple[str, set[str]]:
        """Separate known inline flags from the header value."""

        tokens = value_part.split()
        extracted_flags: list[str] = []
        while tokens:
            candidate = tokens[-1]
            upper_candidate = candidate.upper()
            if upper_candidate == candidate and upper_candidate in self.KNOWN_HEADER_FLAGS:
                extracted_flags.append(upper_candidate)
                tokens.pop()
            else:
                break
        value = " ".join(tokens).strip()
        return value, set(extracted_flags)

    def _rule_extract_version(self, lines: Iterable[str]) -> str:
        """Return the HTTP-version emitted by RequestStructurer."""

        for line in lines:
            if line.startswith("[VERSION] "):
                return line[10:].strip()
        return "HTTP/1.1"

    def _rule_validate_content_length(self, value: str) -> tuple[bool, int | None]:
        """Return whether the Content-Length is a valid non-negative integer."""

        stripped = value.strip()
        if not stripped.isdigit():
            return False, None
        try:
            parsed = int(stripped, 10)
        except ValueError:
            return False, None
        return True, parsed

    def _rule_te_ends_with_chunked(self, value: str) -> bool:
        """Ensure the transfer-encoding chain ends with chunked."""

        segments = [segment.strip() for segment in value.split(",") if segment.strip()]
        if not segments:
            return False
        final_segment = segments[-1]
        coding = final_segment.split(";", 1)[0].strip().lower()
        return coding == "chunked"

    def _rule_render_header(
        self, entry: HeaderEntry, flags: set[str]
    ) -> str:
        """Render a header line with the combined flag set."""

        rendered = f"[HEADER] {entry.raw_name}: {entry.value}"
        if flags:
            rendered += " " + " ".join(sorted(flags))
        return rendered

    def _rule_emit_global_flags(self, flags: set[str]) -> str:
        """Emit a consolidated body framing report line."""

        return "[BFR] " + " ".join(sorted(flags))
