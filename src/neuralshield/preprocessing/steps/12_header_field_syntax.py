"""
Step 12 – Header Field Syntax

Validate raw header lines against RFC 9112 field-line grammar before any
normalisation occurs. Flags unsafe constructions without mutating the value.
"""

from __future__ import annotations

import re

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


class HeaderFieldSyntax(HttpPreprocessor):
    """Enforce field-name token rules and forbid whitespace before the colon."""
    # RFC 9112 §5 – field-line grammar (token ":" OWS value).

    TOKEN_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")

    def process(self, request: str) -> str:
        lines = request.split("\n")
        processed: list[str] = []

        for line in lines:
            if not line.startswith("[HEADER] "):
                processed.append(line)
                continue

            updated_line = self._rule_inspect_header_line(line)
            processed.append(updated_line)

        return "\n".join(processed)

    def _rule_inspect_header_line(self, line: str) -> str:
        header_body = line[9:]
        flags: list[str] = []

        if ":" not in header_body:
            flags.append("BADFIELD")
            return self._rule_append_flags(line, flags)

        name_part, _ = header_body.split(":", 1)

        if re.search(r"[ \t]", name_part):
            flags.append("PRECOLONWS")

        stripped_name = name_part.strip()
        if not self.TOKEN_PATTERN.fullmatch(stripped_name):
            flags.append("BADFIELDNAME")

        return self._rule_append_flags(line, flags)

    def _rule_append_flags(self, line: str, flags: list[str]) -> str:
        if not flags:
            return line

        for flag in flags:
            token = f" {flag}"
            if token not in line:
                line = f"{line} {flag}"
        return line
