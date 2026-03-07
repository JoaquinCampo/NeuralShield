from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor, split_line_content


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
                processed_line = self._process_header_line(line)
                processed_lines.append(processed_line)
            else:
                processed_lines.append(line)

        return "\n".join(processed_lines)

    def _process_header_line(self, line: str) -> str:
        """
        Process a single header line for whitespace normalization.

        Args:
            line: The header line in format "[HEADER] name: value"

        Returns:
            The processed header line with normalized whitespace and flags
        """
        # Use split_line_content to separate flags, but keep the raw
        # header text for whitespace comparison (split_line_content joins
        # with single spaces, which would mask the very issue we detect).
        _, existing_flags = split_line_content(line, "[HEADER] ")

        # Reconstruct raw header content by stripping trailing flags from line
        raw_header = line[9:]  # Remove "[HEADER] "
        # Strip known trailing flags
        for flag in sorted(existing_flags, key=lambda f: len(f), reverse=True):
            if raw_header.endswith(f" {flag}"):
                raw_header = raw_header[: -(len(flag) + 1)]

        # Skip processing for redacted values
        if self._is_redacted_value(raw_header):
            return line

        colon_index = raw_header.find(":")
        if colon_index == -1:
            return line  # Malformed header, pass through unchanged

        name = raw_header[:colon_index].strip()
        raw_value = raw_header[colon_index + 1 :]
        normalized_value = self._normalize_whitespace(raw_value)
        normalized_header = f"{name}: {normalized_value}"

        # Emit WSPAD if we changed the header formatting.
        all_flags = set(existing_flags)
        if raw_header != normalized_header:
            all_flags.add("WSPAD")

        rebuilt = f"[HEADER] {normalized_header}"
        if all_flags:
            rebuilt += f" {' '.join(sorted(all_flags))}"
        return rebuilt

    def _is_redacted_value(self, header_content: str) -> bool:
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

    def _normalize_whitespace(self, value: str) -> str:
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
