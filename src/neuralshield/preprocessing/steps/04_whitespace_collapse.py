from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor
from neuralshield.preprocessing.http_preprocessor import split_line_content


class WhitespaceCollapse(HttpPreprocessor):
    """
    Detect whitespace anomalies within header values without rewriting content.

    Step 04: Whitespace Collapse
    - Detects padding/tab anomalies in header values
    - Emits WSPAD when anomalous whitespace is observed
    - Does not modify header content (evidence preservation)
    - Skips processing for redacted values (<SECRET:...>)
    """

    def process(self, request: str) -> str:
        """
        Process the HTTP request to detect whitespace anomalies in header values.

        Args:
            request: The HTTP request as a string with structured lines

        Returns:
            The processed request with WSPAD flags where applicable
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
        Process a single header line for whitespace anomaly detection.

        Args:
            line: The header line in format "[HEADER] name: value"

        Returns:
            The processed header line with added flags (content preserved)
        """
        header_content, existing_flags = split_line_content(line, "[HEADER] ")

        # Skip processing for redacted values
        if self._is_redacted_value(header_content):
            return line

        colon_index = header_content.find(":")
        if colon_index == -1:
            return line  # Malformed header, pass through unchanged

        after_colon = header_content[colon_index + 1 :]

        has_tabs = "\t" in after_colon
        has_multi_spaces = "  " in after_colon
        has_trailing_ws = after_colon.endswith((" ", "\t"))
        has_leading_tab = after_colon.startswith("\t")
        has_leading_multi_space = after_colon.startswith("  ")

        anomalous = (
            has_tabs
            or has_multi_spaces
            or has_trailing_ws
            or has_leading_tab
            or has_leading_multi_space
        )

        all_flags = set(existing_flags)
        if anomalous:
            all_flags.add("WSPAD")

        rebuilt = f"[HEADER] {header_content}"
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

    # NOTE: This step is detection-only. We intentionally do not normalize
    # whitespace here; normalization without explicit evidence would violate the
    # "no-destructiveness" property described in the thesis.
