import re
from typing import Set

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor
from neuralshield.preprocessing.steps.structure_metadata import (
    merge_structure_flags,
)


class DangerousCharactersScriptMixing(HttpPreprocessor):
    """
    Detect dangerous characters and script mixing for attack detection.

    Step 05: Dangerous Characters and Script Mixing
    - Detects 11 types of dangerous characters in URLs, queries, and headers
    - Handles both literal and percent-encoded forms
    - Applies context-specific detection rules
    - Identifies homograph attacks through script mixing analysis
    - Emits security flags for attack detection and evidence preservation
    """

    # Step 05 specific flags (used for idempotency)
    STEP05_FLAGS = {
        "ANGLE",
        "QUOTE",
        "SEMICOLON",
        "PAREN",
        "BRACE",
        "PIPE",
        "BACKSLASH",
        "SPACE",
        "NUL",
        "QNUL",
        "MIXEDSCRIPT",
    }

    # Dangerous character detection patterns (literal and percent-encoded)
    DANGEROUS_PATTERNS = {
        "ANGLE": [r"[<>]", r"%3[Cc]", r"%3[Ee]"],
        "QUOTE": [r'[\'"]', r"%27", r"%22"],
        "SEMICOLON": [r";", r"%3[Bb]"],
        "PAREN": [r"[()]", r"%28", r"%29"],
        "BRACE": [r"[{}]", r"%7[Bb]", r"%7[Dd]"],
        "PIPE": [r"\|", r"%7[Cc]"],
        "BACKSLASH": [r"\\", r"%5[Cc]"],
        "SPACE": [r" ", r"%20"],
        "NUL": [r"\x00", r"%00"],
    }

    # Unicode script ranges for script mixing detection
    SCRIPT_RANGES = {
        "LATIN": (0x0041, 0x007A),  # A-Z, a-z
        "CYRILLIC": (0x0400, 0x04FF),  # А-я
        "GREEK": (0x0370, 0x03FF),  # Α-ω
    }

    def __init__(self):
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for flag_name, patterns in self.DANGEROUS_PATTERNS.items():
            self.compiled_patterns[flag_name] = [re.compile(p) for p in patterns]

    def process(self, request: str) -> str:
        """
        Process the HTTP request to detect dangerous characters and script mixing.

        Args:
            request: The HTTP request as a string with structured lines

        Returns:
            The processed request with dangerous character and script mixing flags
        """
        lines = request.split("\n")
        processed_lines = []
        structure_flags: set[str] = set()

        for line in lines:
            if line.startswith("[URL] "):
                processed_line, removed = self._process_url_line(line)
            elif line.startswith("[QUERY] "):
                processed_line, removed = self._process_query_line(line)
            elif line.startswith("[HEADER] "):
                processed_line, removed = self._process_header_line(line)
            else:
                processed_line = line
                removed = set()

            structure_flags.update(removed)
            processed_lines.append(processed_line)

        merge_structure_flags(processed_lines, structure_flags)

        return "\n".join(processed_lines)

    def _process_url_line(self, line: str) -> tuple[str, set[str]]:
        """Process a URL line for dangerous characters and script mixing."""
        # Split line into content and existing flags
        parts = line.split()
        content = parts[1] if len(parts) > 1 else ""  # Content after "[URL]"
        existing_flags = set()

        # Extract any existing flags (uppercase words)
        for part in parts[2:]:  # Skip "[URL]" and content
            if part.isupper() and part in self.STEP05_FLAGS:
                existing_flags.add(part)

        flags = set()

        # Detect dangerous characters (all flagged in URL context)
        char_flags = self._detect_dangerous_characters(content, context="URL")
        flags.update(char_flags)

        # Detect script mixing
        if self._detect_script_mixing(content):
            flags.add("MIXEDSCRIPT")

        # Combine with existing flags
        all_flags = existing_flags.union(flags)
        structure_flags: set[str] = set()

        if "PAREN" in all_flags:
            all_flags.discard("PAREN")
            structure_flags.add("PAREN")

        # Emit flags if any detected
        if all_flags:
            sorted_flags = sorted(all_flags)
            return f"[URL] {content} {' '.join(sorted_flags)}", structure_flags
        else:
            return f"[URL] {content}", structure_flags

    def _process_query_line(self, line: str) -> tuple[str, set[str]]:
        """Process a query line for dangerous characters."""
        # Split line into content and existing flags
        parts = line.split()
        content = parts[1] if len(parts) > 1 else ""  # Content after "[QUERY]"
        existing_flags = set()

        # Extract any existing flags (uppercase words)
        for part in parts[2:]:  # Skip "[QUERY]" and content
            if part.isupper() and part in self.STEP05_FLAGS:
                existing_flags.add(part)

        flags = set()

        # Detect dangerous characters
        char_flags = self._detect_dangerous_characters(content, context="QUERY")
        flags.update(char_flags)

        # Special handling for null bytes in query context
        if "NUL" in flags:
            flags.add("QNUL")

        # Combine with existing flags
        all_flags = existing_flags.union(flags)
        structure_flags: set[str] = set()

        if "PAREN" in all_flags:
            all_flags.discard("PAREN")
            structure_flags.add("PAREN")

        # Emit flags if any detected
        if all_flags:
            sorted_flags = sorted(all_flags)
            return f"[QUERY] {content} {' '.join(sorted_flags)}", structure_flags
        else:
            return f"[QUERY] {content}", structure_flags

    def _process_header_line(self, line: str) -> tuple[str, set[str]]:
        """Process a header line for dangerous characters and script mixing."""
        # Split line into parts: "[HEADER]" "name:" "value" [flags...]
        parts = line.split()
        content = (
            " ".join(parts[1:]) if len(parts) > 1 else ""
        )  # Everything after "[HEADER]"
        existing_flags = set()

        # Extract any existing flags (uppercase words at the end)
        filtered_parts = []
        for part in parts[1:]:  # Skip "[HEADER]"
            if part.isupper() and part in self.STEP05_FLAGS:
                existing_flags.add(part)
            else:
                filtered_parts.append(part)

        # Reconstruct content without existing flags
        content = " ".join(filtered_parts)

        flags = set()

        # Detect dangerous characters (semicolon NOT flagged in headers)
        char_flags = self._detect_dangerous_characters(content, context="HEADER")
        flags.update(char_flags)

        # Detect script mixing in header values only
        if self._detect_script_mixing(content):
            flags.add("MIXEDSCRIPT")

        # Combine with existing flags
        all_flags = existing_flags.union(flags)
        structure_flags: set[str] = set()

        if "PAREN" in all_flags:
            all_flags.discard("PAREN")
            structure_flags.add("PAREN")

        # Emit flags if any detected
        if all_flags:
            sorted_flags = sorted(all_flags)
            return f"[HEADER] {content} {' '.join(sorted_flags)}", structure_flags
        else:
            return f"[HEADER] {content}", structure_flags

    def _detect_dangerous_characters(self, content: str, context: str) -> Set[str]:
        """
        Detect dangerous characters in the given content.

        Args:
            content: The text content to analyze
            context: 'URL', 'QUERY', or 'HEADER'

        Returns:
            Set of detected flag names
        """
        flags = set()

        # Check each dangerous character pattern
        for flag_name, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    # Apply context-specific rules
                    if self._should_flag_character(flag_name, context):
                        flags.add(flag_name)
                    break  # Found one pattern match, no need to check others

        return flags

    def _should_flag_character(self, flag_name: str, context: str) -> bool:
        """
        Determine if a character flag should be emitted based on context.

        Args:
            flag_name: The flag name (e.g., 'SEMICOLON')
            context: 'URL', 'QUERY', or 'HEADER'

        Returns:
            True if the flag should be emitted
        """
        # Context-specific rules
        if context == "HEADER":
            if flag_name == "SEMICOLON":
                return False  # Semicolons are legitimate in headers (cookies, etc.)
            elif flag_name == "SPACE":
                return False  # Spaces are not suspicious in header values

        # URL-specific rules
        if context == "URL":
            # All dangerous characters are flagged in URLs
            # SPACE is particularly suspicious in URL paths
            return True

        # QUERY context - all dangerous characters flagged
        return True

    def _detect_script_mixing(self, content: str) -> bool:
        """
        Detect if the content contains mixed scripts (homograph attack indicator).

        Args:
            content: The text content to analyze

        Returns:
            True if mixed scripts are detected
        """
        # Find all alphabetic characters
        alphabetic_chars = []
        for char in content:
            if char.isalpha():
                alphabetic_chars.append(char)

        if len(alphabetic_chars) < 2:
            return False  # Need at least 2 alphabetic chars to detect mixing

        # Identify scripts for each character
        scripts_found = set()
        for char in alphabetic_chars:
            char_script = self._identify_script(char)
            if char_script:
                scripts_found.add(char_script)

        # Flag if 2 or more different scripts found
        return len(scripts_found) >= 2

    def _identify_script(self, char: str) -> str:
        """
        Identify the script of a Unicode character.

        Args:
            char: Single character to analyze

        Returns:
            Script name ('LATIN', 'CYRILLIC', 'GREEK') or empty string
        """
        codepoint = ord(char)

        for script_name, (start, end) in self.SCRIPT_RANGES.items():
            if start <= codepoint <= end:
                return script_name

        return ""  # Unknown or unsupported script
