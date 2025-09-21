"""
05 Percent Decode Once - Apply percent decoding exactly once.
"""

import re
from typing import List, Set

from loguru import logger

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor

# Constants
DOUBLEPCT_FLAG = "DOUBLEPCT"
PCTSLASH_FLAG = "PCTSLASH"
PCTBACKSLASH_FLAG = "PCTBACKSLASH"

# Compiled regex patterns
PERCENT_PATTERN = re.compile(r"%[0-9A-Fa-f]{2}")


class PercentDecodeOnce(HttpPreprocessor):
    """
    Apply percent-decode exactly once per component.
    
    Processes [URL] and [QUERY] lines to:
    - Apply percent-decode exactly once
    - Preserve %00 (null bytes) 
    - Detect double encoding (DOUBLEPCT)
    - Detect preserved delimiters (PCTSLASH, PCTBACKSLASH)
    """

    def _decode_hex_pair_safely(self, match: re.Match[str], preserve_null: bool = True, preserve_control: bool = True, preserve_space: bool = True) -> str:
        """Decode a single hex pair, with various preservation options."""
        hex_value = match.group(0).upper()
        
        # Always preserve %00 (null bytes)
        if preserve_null and hex_value == "%00":
            return match.group(0)
        
        # Preserve spaces in URLs (%20)
        if preserve_space and hex_value == "%20":
            return match.group(0)
            
        # Preserve control characters if requested
        if preserve_control:
            try:
                char_code = int(hex_value[1:], 16)
                # Preserve control characters (0x01-0x1F except tab which is 0x09)
                if 0x01 <= char_code <= 0x1F:
                    return match.group(0)
            except ValueError:
                pass
        
        try:
            return chr(int(hex_value[1:], 16))
        except ValueError:
            return match.group(0)  # Keep invalid sequences

    def _has_valid_hex_pairs(self, text: str) -> bool:
        """Check if text contains valid percent-encoded hex pairs."""
        return bool(PERCENT_PATTERN.search(text))

    def _has_decodable_hex_pairs(self, text: str) -> bool:
        """Check if text contains percent-encoded hex pairs that we would actually decode."""
        # Find all hex pairs but exclude %00 which we never decode
        pairs = PERCENT_PATTERN.findall(text)
        return any(pair.upper() != "%00" for pair in pairs)

    def _percent_decode_once(self, text: str, is_url: bool = False) -> tuple[str, List[str]]:
        """
        Apply percent-decode exactly once and detect anomalies.

        Args:
            text: Text to decode

        Returns:
            Tuple of (decoded_text, flags)
        """
        flags: Set[str] = set()

        def decode_once(s: str, is_url: bool = False) -> str:
            """Decode percent sequences once, with context-aware preservation."""
            if is_url:
                # In URLs, preserve more characters to avoid breaking the format
                return PERCENT_PATTERN.sub(
                    lambda m: self._decode_hex_pair_safely(m, preserve_null=True, preserve_control=True, preserve_space=True), s
                )
            else:
                # In queries, be more permissive but still preserve dangerous chars
                return PERCENT_PATTERN.sub(
                    lambda m: self._decode_hex_pair_safely(m, preserve_null=True, preserve_control=False, preserve_space=False), s
                )

        # First pass: try one decode
        decoded_once = decode_once(text, is_url)

        # If nothing changed, no encoding was present
        if decoded_once == text:
            return text, []

        # Check for multi-level encoding (excluding %00 which we never decode)
        if self._has_decodable_hex_pairs(decoded_once):
            flags.add(DOUBLEPCT_FLAG)
            logger.debug("Double percent encoding detected")

            # For multi-level encoding, try one more decode to handle triple+
            # encoding but stop if no more decodable hex pairs would remain
            decoded_twice = decode_once(decoded_once, is_url)

            # If a second decode still leaves decodable hex pairs, use it
            if decoded_twice != decoded_once and self._has_decodable_hex_pairs(decoded_twice):
                result = decoded_twice
            else:
                result = decoded_once
        else:
            # Single encoding detected - use the decoded result
            result = decoded_once

        # Check for preserved delimiters in path components
        if "%2F" in result.upper():
            flags.add(PCTSLASH_FLAG)
            logger.debug("PCTSLASH detected: %2F preserved")
        if "%5C" in result.upper():
            flags.add(PCTBACKSLASH_FLAG)
            logger.debug("PCTBACKSLASH detected: %5C preserved")

        return result, sorted(flags)

    def _process_line_content(self, content: str, is_url: bool = False) -> tuple[str, List[str]]:
        """
        Process line content for percent decoding.

        Args:
            content: Content to process
            is_url: Whether this is URL content (more conservative decoding)

        Returns:
            Tuple of (processed_content, flags)
        """
        return self._percent_decode_once(content, is_url)

    def process(self, request: str) -> str:
        """
        Process structured HTTP request applying percent decoding.

        Args:
            request: Structured HTTP request with [BRACKET] format

        Returns:
            Processed request with percent decoding applied
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
                for flag in [DOUBLEPCT_FLAG, PCTSLASH_FLAG, PCTBACKSLASH_FLAG]
            ):
                processed_lines.append(line)
                continue

            # Process URL and QUERY lines only
            if line.startswith("[URL] "):
                content = line[6:]  # Remove '[URL] ' prefix
                processed_content, flags = self._process_line_content(content, is_url=True)
                processed_lines.append(f"[URL] {processed_content}")
                if flags:
                    processed_lines.append(" ".join(flags))

            elif line.startswith("[QUERY] "):
                content = line[8:]  # Remove '[QUERY] ' prefix
                processed_content, flags = self._process_line_content(content, is_url=False)
                processed_lines.append(f"[QUERY] {processed_content}")
                if flags:
                    processed_lines.append(" ".join(flags))

            else:
                # Pass through METHOD, HEADER, and other lines unchanged
                processed_lines.append(line)

        return "\n".join(processed_lines)
