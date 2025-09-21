"""
Dangerous Character + Script Mixing Detector

Implements caracteres-peligrosos-script-mixing.md specification.
Detects dangerous characters useful for XSS/SQLi/RCE/traversal attacks
and script mixing for homograph attacks.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Set, Tuple

from loguru import logger

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor

# Dangerous character flags
ANGLE_FLAG = "ANGLE"
QUOTE_FLAG = "QUOTE"
SEMICOLON_FLAG = "SEMICOLON"
PAREN_FLAG = "PAREN"
BRACE_FLAG = "BRACE"
PIPE_FLAG = "PIPE"
BACKSLASH_FLAG = "BACKSLASH"
SPACE_FLAG = "SPACE"
NUL_FLAG = "NUL"
QNUL_FLAG = "QNUL"

# Script mixing flag
MIXEDSCRIPT_FLAG = "MIXEDSCRIPT"

# Character detection patterns (both literal and percent-encoded)
DANGEROUS_CHARS = {
    ANGLE_FLAG: re.compile(r"([<>]|%3[CE])", re.IGNORECASE),  # < > and %3C %3E (angle brackets for HTML/XML tags)
    QUOTE_FLAG: re.compile(r'([\'"]|%2[27])', re.IGNORECASE),  # ' " and %22 %27 (quotes for string escaping/injection)
    SEMICOLON_FLAG: re.compile(r"(;|%3B)", re.IGNORECASE),  # ; and %3B (semicolons for command separation/SQL injection)
    PAREN_FLAG: re.compile(r"([()]|%2[89])", re.IGNORECASE),  # ( ) and %28 %29 (parentheses for function calls/SQL injection)
    BRACE_FLAG: re.compile(r"([{}]|%7[BD])", re.IGNORECASE),  # { } and %7B %7D (braces for template injection/code blocks)
    PIPE_FLAG: re.compile(r"(\||%7C)", re.IGNORECASE),  # | and %7C (pipes for command chaining/shell injection)
    BACKSLASH_FLAG: re.compile(r"(\\|%5C)", re.IGNORECASE),  # \ and %5C (backslashes for path traversal/escaping)
    SPACE_FLAG: re.compile(r"( |%20)", re.IGNORECASE),  # space and %20 (spaces suspicious in URL paths)
    NUL_FLAG: re.compile(r"(\x00|%00)", re.IGNORECASE),  # null byte and %00 (null byte injection/string termination)
}


@dataclass
class ComponentType:
    """Component type for context-aware processing."""

    URL = "URL"
    QUERY = "QUERY"
    HEADER = "HEADER"


class DangerousCharacterDetector(HttpPreprocessor):
    """
    Detects dangerous characters and script mixing in HTTP requests.

    Implements caracteres-peligrosos-script-mixing.md specification for:
    - Dangerous character detection (XSS/SQLi/RCE/traversal)
    - Script mixing detection (homograph attacks)
    - Context-aware flagging by component (URL/QUERY/HEADER)
    """

    def __init__(self) -> None:
        # Stateless design following Zen of Python
        pass

    def _detect_dangerous_chars(self, text: str, component: str) -> Set[str]:
        """
        Detect dangerous characters in text based on component context.

        Args:
            text: Text to analyze
            component: Component type (URL/QUERY/HEADER)

        Returns:
            Set of detected flag names
        """
        flags: Set[str] = set()

        # Skip analysis for redacted secrets
        if text.startswith("<SECRET:") and text.endswith(">"):
            return flags

        # Check each dangerous character pattern
        for flag, pattern in DANGEROUS_CHARS.items():
            if pattern.search(text):
                # Special handling for component-specific rules
                if flag == SPACE_FLAG and component != ComponentType.URL:
                    # SPACE only flagged in URL paths (suspicious in paths)
                    continue

                elif flag == SEMICOLON_FLAG and component == ComponentType.HEADER:
                    # SEMICOLON is legitimate in headers (Accept-Language, Cookie, etc.)
                    continue
                elif flag == NUL_FLAG and component == ComponentType.QUERY:
                    # For query values, emit both NUL and QNUL
                    flags.add(NUL_FLAG)
                    flags.add(QNUL_FLAG)
                    continue

                flags.add(flag)
                logger.debug(f"Detected {flag} in {component}: {text[:50]}...")

        return flags

    def _get_unicode_script(self, char: str) -> str:
        """
        Get Unicode script for a character.

        Args:
            char: Single Unicode character

        Returns:
            Script name (e.g., 'Latin', 'Cyrillic', 'Greek', 'Common')
        """
        try:
            return (
                unicodedata.name(char, "").split()[0]
                if unicodedata.name(char, "")
                else "Unknown"
            )
        except (ValueError, IndexError):
            # Fallback to script property detection
            try:
                # Use unicodedata category as proxy for script detection
                category = unicodedata.category(char)
                if category.startswith("L"):  # Letter categories
                    # Simple heuristic based on code point ranges
                    code_point = ord(char)
                    if 0x0000 <= code_point <= 0x007F:
                        return "Latin"
                    elif 0x0400 <= code_point <= 0x04FF:
                        return "Cyrillic"
                    elif 0x0370 <= code_point <= 0x03FF:
                        return "Greek"
                    else:
                        return "Other"
                else:
                    return "Common"  # Punctuation, numbers, symbols
            except (ValueError, TypeError):
                return "Unknown"

    def _detect_script_mixing(self, text: str) -> bool:
        """
        Detect script mixing in text (homograph attacks).

        Args:
            text: Text to analyze for script mixing

        Returns:
            True if script mixing detected, False otherwise
        """
        if not text or len(text) < 2:
            return False

        # Skip analysis for redacted secrets
        if text.startswith("<SECRET:") and text.endswith(">"):
            return False

        # Decode percent-encoded characters for script analysis
        import urllib.parse

        try:
            decoded_text = urllib.parse.unquote(text)
        except Exception:
            decoded_text = text  # Fallback to original if decode fails

        # Collect scripts for all alphabetic characters
        scripts: Set[str] = set()

        for char in decoded_text:
            if char.isalpha():  # Only consider alphabetic characters
                script = self._get_unicode_script(char)

                # Map script names to our target scripts
                if (
                    script in ["LATIN", "Latin"]
                    or (0x0041 <= ord(char) <= 0x005A)
                    or (0x0061 <= ord(char) <= 0x007A)
                ):
                    scripts.add("Latin")
                elif script in ["CYRILLIC", "Cyrillic"] or (
                    0x0400 <= ord(char) <= 0x04FF
                ):
                    scripts.add("Cyrillic")
                elif script in ["GREEK", "Greek"] or (0x0370 <= ord(char) <= 0x03FF):
                    scripts.add("Greek")
                # Ignore Common/Inherited scripts (punctuation, numbers)

        # Detect mixing: ≥2 different scripts (excluding Common/Inherited)
        mixed = len(scripts) >= 2

        if mixed:
            logger.debug(
                f"Script mixing detected in '{decoded_text[:30]}...': {scripts}"
            )

        return mixed

    def _process_line_content(
        self, content: str, component: str
    ) -> Tuple[str, List[str]]:
        """
        Process line content for dangerous characters and script mixing.

        Args:
            content: Content to analyze
            component: Component type (URL/QUERY/HEADER)

        Returns:
            Tuple of (processed_content, detected_flags)
        """
        all_flags: Set[str] = set()

        # Detect dangerous characters
        char_flags = self._detect_dangerous_chars(content, component)
        all_flags.update(char_flags)

        # Detect script mixing for relevant tokens
        if component in [ComponentType.URL, ComponentType.QUERY]:
            # Check for script mixing in the content
            if self._detect_script_mixing(content):
                all_flags.add(MIXEDSCRIPT_FLAG)
        elif component == ComponentType.HEADER:
            # For headers, check script mixing only in the header VALUE (after colon)
            # to avoid false positives from Latin header names with non-Latin values
            if ":" in content:
                header_value = content.split(":", 1)[1].strip()
                if self._detect_script_mixing(header_value):
                    all_flags.add(MIXEDSCRIPT_FLAG)
            else:
                # Fallback: if no colon, check entire content
                if self._detect_script_mixing(content):
                    all_flags.add(MIXEDSCRIPT_FLAG)

        # Return content unchanged and sorted flags
        sorted_flags = sorted(all_flags)
        return content, sorted_flags

    def process(self, request: str) -> str:
        """
        Process HTTP request to detect dangerous characters and script mixing.

        Args:
            request: Structured HTTP request with [BRACKET] format

        Returns:
            Processed request with dangerous character flags
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
                for flag in [
                    ANGLE_FLAG,
                    QUOTE_FLAG,
                    SEMICOLON_FLAG,
                    PAREN_FLAG,
                    BRACE_FLAG,
                    PIPE_FLAG,
                    BACKSLASH_FLAG,
                    SPACE_FLAG,
                    NUL_FLAG,
                    QNUL_FLAG,
                    MIXEDSCRIPT_FLAG,
                ]
            ):
                processed_lines.append(line)
                continue

            # Process specific component types
            if line.startswith("[URL] "):
                content = line[6:]  # Remove '[URL] ' prefix
                processed_content, flags = self._process_line_content(
                    content, ComponentType.URL
                )
                processed_lines.append(f"[URL] {processed_content}")
                if flags:
                    processed_lines.append(" ".join(flags))

            elif line.startswith("[QUERY] "):
                content = line[8:]  # Remove '[QUERY] ' prefix
                processed_content, flags = self._process_line_content(
                    content, ComponentType.QUERY
                )
                processed_lines.append(f"[QUERY] {processed_content}")
                if flags:
                    processed_lines.append(" ".join(flags))

            elif line.startswith("[HEADER] "):
                content = line[9:]  # Remove '[HEADER] ' prefix
                processed_content, flags = self._process_line_content(
                    content, ComponentType.HEADER
                )
                processed_lines.append(f"[HEADER] {processed_content}")
                if flags:
                    processed_lines.append(" ".join(flags))

            else:
                # Pass through METHOD and other lines unchanged
                processed_lines.append(line)

        return "\n".join(processed_lines)
