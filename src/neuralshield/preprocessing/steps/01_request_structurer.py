import re

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor
from neuralshield.preprocessing.steps.exceptions import MalformedHttpRequestError


class RequestStructurer(HttpPreprocessor):
    """Structure the HTTP request into a canonical form and flag start-line anomalies."""

    VALID_METHODS = {
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "PATCH",
        "OPTIONS",
        "HEAD",
        "TRACE",
        "CONNECT",
    }

    _LENIENT_SEPARATOR_PATTERN = re.compile(r"[ \t\x0b\x0c\r]+")
    _BARE_CR_PATTERN = re.compile(r"\r(?!\n)")
    _HTTP_VERSION_PATTERN = re.compile(r"^HTTP/\d+\.\d+$")

    def process(self, request: str) -> str:
        """
        Structure the HTTP request into a canonical form.
        """
        if not request or not request.strip():
            raise MalformedHttpRequestError("Empty request")

        # RFC 9112 §2.2 – tolerate bare CR/LF but emit evidence.
        lines = self._split_request_lines(request)
        framing_flags = self._rule_2_2_detect_bare_cr(request)

        # RFC 9112 §3.1 – parse method/target/version and capture separators.
        request_line = lines[0] if lines else ""
        method, url, http_version, separator_flags = self._rule_3_1_parse_request_line(
            request_line
        )

        # RFC 9112 §3.2 – detect whitespace within the request-target.
        target_flags = self._rule_3_2_detect_target_whitespace(url)

        # RFC 9112 §2.3 – ensure HTTP-version grammar.
        version_flags = self._rule_2_3_validate_http_version(http_version)

        # Split URL and query
        path, query_string = self._split_url_query(url)

        # Parse headers (everything after request line until empty line)
        headers = self._parse_headers(lines[1:])

        # Build canonical output
        output_lines = []

        # Add method
        output_lines.append(f"[METHOD] {method}")

        # Add HTTP version for downstream inspectors
        output_lines.append(f"[VERSION] {http_version}")

        # Add URL (path)
        output_lines.append(f"[URL] {path}")

        # Add query parameters
        if query_string:
            query_params = self._split_query_with_entity_protection(query_string)
            for param in query_params:
                output_lines.append(f"[QUERY] {param}")

        # Add headers
        for header in headers:
            output_lines.append(f"[HEADER] {header}")

        combined_flags = sorted(
            separator_flags | target_flags | version_flags | framing_flags
        )
        if combined_flags:
            output_lines.append(f"[FLAGS] {' '.join(combined_flags)}")

        return "\n".join(output_lines)

    def _rule_3_1_parse_request_line( self, request_line: str) -> tuple[str, str, str, set[str]]:
        """RFC 9112 §3.1 – parse method, target, version and flag lenient separators."""
        flags: set[str] = set()

        line = request_line.strip()
        if not line:
            raise MalformedHttpRequestError("Request line is empty")

        parts = [part for part in self._LENIENT_SEPARATOR_PATTERN.split(line) if part]
        if len(parts) != 3:
            raise MalformedHttpRequestError(
                "Invalid request line format: expected method, target, and version"
            )

        separators = list(self._LENIENT_SEPARATOR_PATTERN.finditer(line))
        if len(separators) < 2:
            raise MalformedHttpRequestError(
                "Invalid request line format: expected method, target, and version"
            )

        first_sep = separators[0]
        last_sep = separators[-1]

        method = line[: first_sep.start()]
        target = line[first_sep.end() : last_sep.start()]
        http_version = line[last_sep.end() :]

        if not method or not target or not http_version:
            raise MalformedHttpRequestError(
                "Invalid request line format: missing required components"
            )

        if self._is_lenient_separator(first_sep) or self._is_lenient_separator(last_sep):
            flags.add("LENIENTSEP")

        # Validate method
        if method not in self.VALID_METHODS:
            raise MalformedHttpRequestError(f"Invalid HTTP method: {method}")

        return method, target, http_version, flags

    def _rule_3_2_detect_target_whitespace(self, target: str) -> set[str]:
        """RFC 9112 §3.2 – flag whitespace within the request-target."""
        flags: set[str] = set()

        if self._contains_internal_whitespace(target):
            flags.add("TARGETSPACE")

        return flags

    def _rule_2_3_validate_http_version(self, http_version: str) -> set[str]:
        """RFC 9112 §2.3 – ensure the HTTP-version matches DIGIT.DIGIT grammar."""
        flags: set[str] = set()

        if not self._HTTP_VERSION_PATTERN.match(http_version):
            flags.add("BADVERSION")

        return flags

    def _split_url_query(self, url: str) -> tuple[str, str]:
        """Split URL on first '?' into path and query string."""
        if "?" in url:
            path, query_string = url.split("?", 1)
        else:
            path, query_string = url, ""

        return path, query_string

    def _split_query_with_entity_protection(self, query_string: str) -> list[str]:
        """Split query string on '&' while protecting HTML entities."""
        if not query_string:
            return []

        parts = []
        current_part = ""
        i = 0

        while i < len(query_string):
            if query_string[i] == "&":
                if self._is_part_of_html_entity(query_string, i):
                    current_part += query_string[i]
                else:
                    if current_part:
                        parts.append(current_part)
                    current_part = ""
                i += 1
            else:
                current_part += query_string[i]
                i += 1

        if current_part:
            parts.append(current_part)

        return parts

    def _is_part_of_html_entity(self, text: str, ampersand_pos: int) -> bool:
        """Check if the '&' at the given position is part of an HTML entity."""
        remaining = text[ampersand_pos:]

        if remaining.startswith("&#"):
            end_pos = remaining.find(";", 2)
            if end_pos != -1:
                entity_content = remaining[2:end_pos]
                if entity_content.startswith("x") and len(entity_content) > 1:
                    return all(
                        c in "0123456789abcdefABCDEF" for c in entity_content[1:]
                    )
                return entity_content.isdigit()

        semicolon_pos = remaining.find(";")
        if semicolon_pos != -1 and semicolon_pos > 1:
            entity_name = remaining[1:semicolon_pos]
            return entity_name.isalpha()

        return False

    def _parse_headers(self, header_lines: list[str]) -> list[str]:
        """Parse headers until the first empty line."""
        headers = []

        for line in header_lines:
            if self._rule_3_1_skip_pre_header_whitespace(line, headers):
                continue
            if line == "":
                break
            headers.append(line)

        return headers

    def _rule_3_1_skip_pre_header_whitespace(self, line: str, headers: list[str]) -> bool:
        """RFC 9112 §3.1 – ignore blank lines until the first header is seen."""

        return line == "" and not headers

    def _rule_2_2_detect_bare_cr(self, request: str) -> set[str]:
        """RFC 9112 §2.2 – identify bare carriage returns outside content."""
        flags: set[str] = set()
        if self._BARE_CR_PATTERN.search(request):
            flags.add("BARECR")
        return flags

    def _split_request_lines(self, request: str) -> list[str]:
        """Split a raw HTTP message into lines tolerating bare CR separators."""
        return re.split(r"\r\n|\n|\r", request)

    def _is_lenient_separator(self, match: re.Match[str]) -> bool:
        """Check if a separator contains characters beyond a single SP."""
        separator = match.group(0)
        return separator != " " or len(separator) != 1

    def _contains_internal_whitespace(self, value: str) -> bool:
        """Return True if the provided string contains linear whitespace."""
        return bool(self._LENIENT_SEPARATOR_PATTERN.search(value))
