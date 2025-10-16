from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


class WhitespaceCollapse(HttpPreprocessor):
    """
    Normalize whitespace within header values to eliminate formatting variations.

    Step 04: Whitespace Collapse
    - Collapses sequences of spaces and tabs to single space
    - Trims leading and trailing whitespace
    - Preserves semantic spacing between tokens
    - Flags when modifications are made for security evidence
    - Skips processing for redacted values (<SECRET:...>)
    """
    # Spec references:
    # - Internal style guidelines for header value normalisation.
    # - Records `WSPAD` evidence when whitespace adjustments are made.

    def process(self, request: str) -> str:
        """
        Process the HTTP request to normalize whitespace in header values.

        Args:
            request: The HTTP request as a string with structured lines

        Returns:
            The processed request with normalized whitespace and WSPAD flags
        """
        lines = request.split("\n")
        processed_lines = []

        for line in lines:
            if line.startswith("[HEADER] "):
                processed_lines.append(self._rule_whitespace_collapse(line))
            else:
                processed_lines.append(line)

        return "\n".join(processed_lines)

    def _rule_whitespace_collapse(self, line: str) -> str:
        """Normalize header value whitespace and emit WSPAD evidence."""
        # Extract the header content after "[HEADER] "
        header_content = line[9:]

        # Skip processing for redacted values
        if self._rule_whitespace_is_redacted_value(header_content):
            return line

        # Parse header name and value
        colon_index = header_content.find(":")
        if colon_index == -1:
            return line  # Malformed header, pass through unchanged

        name = header_content[:colon_index].strip()

        # Extract the part after the colon
        after_colon = header_content[colon_index + 1 :]

        # Check if there's exactly one space after the colon followed by the value
        if after_colon.startswith(" "):
            value_part = after_colon[1:]  # Remove the normal space
        else:
            value_part = after_colon  # No space after colon

        # Normalize whitespace in the value part
        normalized_value = self._rule_whitespace_normalize_value(value_part)

        # Check for whitespace anomalies beyond normal formatting
        had_whitespace_issues = self._rule_whitespace_detect_anomaly(
            after_colon, value_part
        )

        if had_whitespace_issues:
            # Normalize to standard format: name: value (single space after colon)
            return f"[HEADER] {name}: {normalized_value} WSPAD"
        else:
            # Already in correct format, return as-is
            return line

    def _rule_whitespace_is_redacted_value(self, header_content: str) -> bool:
        """
        Check if the header contains a redacted value that should be preserved.

        Args:
            header_content: The header content (name: value)

        Returns:
            True if the value is redacted and should be preserved unchanged
        """
        # Look for <SECRET:...> pattern in the value part
        if ":" not in header_content:
            return False

        value_part = header_content.split(":", 1)[1].strip()
        return value_part.startswith("<SECRET:") and value_part.endswith(">")

    def _parse_header_line(self, header_content: str) -> tuple[str | None, str]:
        """
        Parse a header line into name and value.

        Args:
            header_content: Raw header content like "Host: example.com"

        Returns:
            Tuple of (name, value) or (None, "") for malformed headers
        """
        if not header_content or ":" not in header_content:
            return None, ""

        name, value = header_content.split(":", 1)
        return name.strip(), value.strip()

    def _rule_whitespace_normalize_value(self, value: str) -> str:
        """
        Normalize whitespace within a header value.

        Args:
            value: The original header value

        Returns:
            The normalized value with collapsed whitespace
        """
        import re

        # Collapse sequences of tabs and spaces to single space
        normalized = re.sub(r"[\t ]+", " ", value)

        # Trim leading and trailing whitespace
        normalized = normalized.strip()

        return normalized

    def _rule_whitespace_detect_anomaly(
        self, after_colon: str, value_part: str
    ) -> bool:
        """
        Determine if whitespace formatting deviates from the canonical form.
        """
        return (
            after_colon.startswith("   ")
            or "  " in value_part
            or "\t" in after_colon
            or "\t" in value_part
        )
