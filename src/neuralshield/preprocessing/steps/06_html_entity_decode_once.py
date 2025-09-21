"""
06 HTML Entity Decode Once - Decode HTML entities exactly once.
"""

import html
import re
from typing import List

from loguru import logger

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor

# Constants
HTMLENT_FLAG = "HTMLENT"

# Compiled regex patterns
HTML_ENTITY_PATTERN = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]*|#(?:\d+|x[0-9a-fA-F]+));")


class HTMLEntityDecodeOnce(HttpPreprocessor):
    """
    Decode HTML entities exactly once.
    
    Processes [URL] and [QUERY] lines to:
    - Decode HTML entities like &#x2f;, &lt;, etc.
    - Emit HTMLENT flag if entities were found and decoded
    - Respect delimiter preservation policy
    """

    def _html_entity_decode_once(self, text: str) -> tuple[str, List[str]]:
        """
        Detect and decode HTML entities once.

        Args:
            text: Text to decode

        Returns:
            Tuple of (decoded_text, flags)
        """
        flags: List[str] = []

        if HTML_ENTITY_PATTERN.search(text):
            decoded = html.unescape(text)
            if decoded != text:
                flags.append(HTMLENT_FLAG)
                logger.debug("HTML entities detected and decoded")
                return decoded, flags

        return text, flags

    def _process_line_content(self, content: str) -> tuple[str, List[str]]:
        """
        Process line content for HTML entity decoding.

        Args:
            content: Content to process

        Returns:
            Tuple of (processed_content, flags)
        """
        return self._html_entity_decode_once(content)

    def process(self, request: str) -> str:
        """
        Process structured HTTP request applying HTML entity decoding.

        Args:
            request: Structured HTTP request with [BRACKET] format

        Returns:
            Processed request with HTML entity decoding applied
        """
        lines = request.strip().split("\n")
        processed_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip existing flag lines to avoid duplication
            if not line.startswith("[") and any(
                flag == line.strip()
                or f" {flag} " in f" {line.strip()} "
                or line.strip().startswith(f"{flag} ")
                or line.strip().endswith(f" {flag}")
                for flag in [HTMLENT_FLAG]
            ):
                processed_lines.append(line)
                continue

            # Process URL and QUERY lines only
            if line.startswith("[URL] "):
                content = line[6:]  # Remove '[URL] ' prefix
                processed_content, flags = self._process_line_content(content)
                processed_lines.append(f"[URL] {processed_content}")
                if flags:
                    processed_lines.append(" ".join(flags))

            elif line.startswith("[QUERY] "):
                content = line[8:]  # Remove '[QUERY] ' prefix
                processed_content, flags = self._process_line_content(content)
                processed_lines.append(f"[QUERY] {processed_content}")
                if flags:
                    processed_lines.append(" ".join(flags))

            else:
                # Pass through METHOD, HEADER, and other lines unchanged
                processed_lines.append(line)

        return "\n".join(processed_lines)
