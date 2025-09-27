import html

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


class HtmlEntityDecodeOnce(HttpPreprocessor):
    """
    Detect HTML entities in URL and query components without decoding them.

    Step 05: HTML Entity Decode Once
    - Detects HTML entities (&lt;, &#64;, &#x40;, etc.) in [URL] and [QUERY] lines
    - Preserves entities unchanged for evidence preservation
    - Emits HTMLENT flag when entities are detected
    - Security-focused: identifies entity-based evasion attempts
    """

    def process(self, request: str) -> str:
        """
        Process structured HTTP request lines, detecting HTML entities in URL and query components.

        Args:
            request: Structured HTTP request from percent decode processing

        Returns:
            Processed request with HTMLENT flags where entities are detected
        """
        lines = request.split("\n")
        processed_lines = []

        for line in lines:
            if line.strip() == "":
                processed_lines.append(line)
                continue

            # Only process URL and QUERY lines
            if line.startswith("[URL] ") or line.startswith("[QUERY] "):
                processed_line, flags = self._process_component_line(line)
                processed_lines.append(processed_line)
                # Flags are already attached to the processed_line
            else:
                # Pass through METHOD and HEADER lines unchanged
                processed_lines.append(line)

        return "\n".join(processed_lines)

    def _process_component_line(self, line: str) -> tuple[str, list[str]]:
        """
        Process a single URL or QUERY line for HTML entity detection.

        Args:
            line: Line in format "[TYPE] content [existing_flags...]"

        Returns:
            tuple: (processed_line_with_flags, empty_list)
        """
        # Parse line to separate content from existing flags
        if line.startswith("[URL] "):
            prefix = "[URL]"
            line_content = line[6:]  # Remove '[URL] '
        elif line.startswith("[QUERY] "):
            prefix = "[QUERY]"
            line_content = line[8:]  # Remove '[QUERY] '
        else:
            return line, []

        # Split content from existing flags
        parts = line_content.split()
        content = parts[0] if parts else ""
        existing_flags = set(parts[1:]) if len(parts) > 1 else set()

        # Detect HTML entities
        new_flags = []
        if self._has_html_entities(content):
            new_flags.append("HTMLENT")

        # Combine existing and new flags
        all_flags = existing_flags | set(new_flags)
        flags_str = " ".join(sorted(all_flags)) if all_flags else ""

        # Reconstruct line with all flags
        processed_line = f"{prefix} {content}"
        if flags_str:
            processed_line += f" {flags_str}"

        return processed_line, []  # Return empty list since flags are attached

    def _has_html_entities(self, text: str) -> bool:
        """
        Detect if text contains valid HTML entities using html.unescape().

        Args:
            text: The text to check for HTML entities

        Returns:
            bool: True if valid HTML entities are found
        """
        # Use html.unescape() to detect valid HTML entities
        # If unescaping changes the string, valid entities were present
        return html.unescape(text) != text
