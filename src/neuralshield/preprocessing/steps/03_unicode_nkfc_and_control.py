import unicodedata

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


class UnicodeNFKCAndControl(HttpPreprocessor):
    """
    Apply Unicode NFKC normalization to URL and QUERY content and detect anomalies.
    """

    def process(self, request: str) -> str:
        """
        Process structured HTTP request lines, applying NFKC normalization to URL and QUERY content
        and detecting fullwidth/control character anomalies.

        Args:
            request: Structured HTTP request from RequestStructurer

        Returns:
            Processed request with normalization and flags
        """
        lines = request.split("\n")
        processed_lines = []

        for line in lines:
            if line.strip() == "":
                processed_lines.append(line)
                continue

            # Only process URL and QUERY lines
            if line.startswith("[URL] ") or line.startswith("[QUERY] "):
                processed_line, flags = self._process_content_line(line)
                processed_lines.append(processed_line)
                # Flags are already attached to the processed_line
            else:
                # Pass through METHOD and HEADER lines unchanged
                processed_lines.append(line)

        return "\n".join(processed_lines)

    def _process_content_line(self, line: str) -> tuple[str, list[str]]:
        """
        Process a single URL or QUERY line: normalize content and detect anomalies.

        Args:
            line: Line in format "[TYPE] content"

        Returns:
            tuple: (processed_line, list_of_flags)
        """
        # Split prefix from content
        if line.startswith("[URL] "):
            prefix = "[URL]"
            content = line[6:]  # Remove '[URL] '
        elif line.startswith("[QUERY] "):
            prefix = "[QUERY]"
            content = line[8:]  # Remove '[QUERY] '
        else:
            return line, []

        # Detect all Unicode security issues before normalization
        has_fullwidth = self._has_fullwidth_characters(content)
        has_unicode_format = self._has_unicode_formatting_chars(content)
        has_math_unicode = self._has_mathematical_unicode(content)
        has_invalid_unicode = self._has_invalid_unicode(content)

        # Apply NFKC normalization
        normalized_content = unicodedata.normalize("NFKC", content)

        # Detect control characters in normalized content
        has_control = self._has_control_characters(normalized_content)

        # Determine flags to emit
        flags = []
        if has_fullwidth or normalized_content != content:
            flags.append("FULLWIDTH")
        if has_control:
            flags.append("CONTROL")
        if has_unicode_format:
            flags.append("UNICODE_FORMAT")
        if has_math_unicode:
            flags.append("MATH_UNICODE")
        if has_invalid_unicode:
            flags.append("INVALID_UNICODE")

        # Sort flags alphabetically as per spec
        flags.sort()

        # Reconstruct the line with normalized content and flags
        processed_line = f"{prefix} {normalized_content}"
        if flags:
            processed_line += f" {' '.join(flags)}"

        return (
            processed_line,
            [],
        )  # Return empty flags since they're attached to the line

    def _has_fullwidth_characters(self, text: str) -> bool:
        """
        Detect fullwidth characters in the text (U+FF00-U+FFEF range).

        Fullwidth characters are often used for filter evasion.
        """
        for char in text:
            code = ord(char)
            if 0xFF00 <= code <= 0xFFEF:
                return True
        return False

    def _has_control_characters(self, text: str) -> bool:
        """
        Detect control characters in the text (Unicode category Cc).

        Control characters indicate potential injection attempts or binary data.
        Also detects %00 sequences (null bytes) without decoding them.
        """
        # Check for literal %00 sequences (null bytes in percent encoding)
        if "%00" in text:
            return True

        # Check for Unicode control characters (category Cc)
        for char in text:
            category = unicodedata.category(char)
            if category == "Cc":
                return True

        return False

    def _has_unicode_formatting_chars(self, text: str) -> bool:
        """
        Detect zero-width and bidirectional formatting characters.

        These characters can hide content or manipulate text rendering.
        """
        for char in text:
            code = ord(char)
            # Zero-width characters
            if code in (0x200B, 0x200C, 0x200D, 0xFEFF):
                return True
            # Bidirectional text controls
            if 0x202A <= code <= 0x202E:
                return True
            # Other formatting characters (category Cf)
            if unicodedata.category(char) == "Cf":
                return True

        return False

    def _has_mathematical_unicode(self, text: str) -> bool:
        """
        Detect mathematical alphanumeric symbols.

        These symbols visually resemble letters and can create homoglyph attacks.
        """
        for char in text:
            code = ord(char)
            # Mathematical alphanumeric symbols
            if 0x1D400 <= code <= 0x1D7FF:
                return True

        return False

    def _has_invalid_unicode(self, text: str) -> bool:
        """
        Detect private use and invalid Unicode characters.

        Private use characters may contain hidden malicious data.
        Non-characters are invalid Unicode code points.
        """
        for char in text:
            code = ord(char)
            # Private use characters
            if 0xE000 <= code <= 0xF8FF:
                return True
            # Non-characters
            if code in (0xFFFE, 0xFFFF):
                return True
            # Other non-characters in planes
            if (code & 0xFFFF) in (0xFFFE, 0xFFFF):
                return True

        return False
