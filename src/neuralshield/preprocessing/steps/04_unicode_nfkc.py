"""
04 Unicode NFKC and Control Detection - Normalize Unicode and detect anomalies.
"""

import unicodedata
from typing import List, Set

from loguru import logger

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor

# Constants
FULLWIDTH_FLAG = "FULLWIDTH"
CONTROL_FLAG = "CONTROL"

# Fullwidth character range (U+FF00–U+FFEF)
FULLWIDTH_RANGE = range(0xFF00, 0xFFF0)


class UnicodeNFKC(HttpPreprocessor):
    """
    Apply Unicode NFKC normalization and detect control characters.
    
    Processes [URL] and [QUERY] lines to:
    - Apply NFKC normalization
    - Detect fullwidth characters
    - Detect control characters
    - Emit appropriate flags
    """

    def _is_fullwidth_char(self, char: str) -> bool:
        """Check if character is in fullwidth range."""
        return ord(char) in FULLWIDTH_RANGE

    def _apply_nfkc_and_detect_fullwidth(self, text: str) -> tuple[str, bool]:
        """
        Apply NFKC normalization and detect fullwidth characters.

        Args:
            text: Text to normalize

        Returns:
            Tuple of (normalized_text, has_fullwidth_or_changed)
        """
        # Check for fullwidth characters before normalization
        has_fullwidth = any(self._is_fullwidth_char(char) for char in text)

        # Apply NFKC normalization
        normalized = unicodedata.normalize("NFKC", text)

        # Flag if we had fullwidth chars or if normalization changed the text
        changed = normalized != text or has_fullwidth

        if changed:
            logger.debug("FULLWIDTH characters detected and normalized")

        return normalized, changed

    def _detect_control_chars(self, text: str) -> bool:
        """
        Detect control characters in text.

        Args:
            text: Text to analyze

        Returns:
            True if control characters detected
        """
        # Check for %00 sequences (null bytes) without decoding them
        if "%00" in text.upper():
            logger.debug("CONTROL character detected: %00 sequence")
            return True

        # Check for actual control characters in the text
        for char in text:
            if unicodedata.category(char) == "Cc":
                logger.debug(f"CONTROL character detected: {repr(char)}")
                return True

        return False

    def _process_line_content(self, content: str) -> tuple[str, List[str]]:
        """
        Process line content for Unicode normalization and control detection.

        Args:
            content: Content to process

        Returns:
            Tuple of (processed_content, flags)
        """
        flags: Set[str] = set()

        # Apply NFKC normalization and detect fullwidth
        normalized_content, has_fullwidth = self._apply_nfkc_and_detect_fullwidth(content)
        if has_fullwidth:
            flags.add(FULLWIDTH_FLAG)

        # Detect control characters
        if self._detect_control_chars(normalized_content):
            flags.add(CONTROL_FLAG)

        # Return content and sorted flags
        sorted_flags = sorted(flags)
        return normalized_content, sorted_flags

    def process(self, request: str) -> str:
        """
        Process structured HTTP request applying Unicode normalization and control detection.

        Args:
            request: Structured HTTP request with [BRACKET] format

        Returns:
            Processed request with Unicode normalization and flags
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
                for flag in [FULLWIDTH_FLAG, CONTROL_FLAG]
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
