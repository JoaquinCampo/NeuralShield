import unicodedata

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor, split_line_content


class UnicodeNFKCAndControl(HttpPreprocessor):
    """
    Detect Unicode anomalies in URL and QUERY content using NFKC as a diagnostic reference.

    NFKC normalization is used only for comparison (flag detection); the original
    content is preserved in the output so that downstream steps and encoders see
    the actual bytes the client sent.
    """

    _PREFIXES = {"[URL] ": "[URL]", "[QUERY] ": "[QUERY]"}

    def process(self, request: str) -> str:
        lines = request.split("\n")
        processed_lines = []

        for line in lines:
            if line.strip() == "":
                processed_lines.append(line)
                continue

            matched = False
            for tag_prefix, tag in self._PREFIXES.items():
                if line.startswith(tag_prefix):
                    content, existing_flags = split_line_content(line, tag_prefix)
                    new_flags = self._detect_unicode_issues(content)
                    all_flags = sorted(existing_flags | new_flags)
                    rebuilt = f"{tag} {content}"
                    if all_flags:
                        rebuilt += f" {' '.join(all_flags)}"
                    processed_lines.append(rebuilt)
                    matched = True
                    break

            if not matched:
                processed_lines.append(line)

        return "\n".join(processed_lines)

    def _detect_unicode_issues(self, content: str) -> set[str]:
        """Detect Unicode anomalies using NFKC as a diagnostic reference."""
        flags: set[str] = set()

        has_fullwidth = self._has_fullwidth_characters(content)
        normalized = unicodedata.normalize("NFKC", content)

        if has_fullwidth or normalized != content:
            flags.add("FULLWIDTH")
        if self._has_control_characters(content):
            flags.add("CONTROL")
        if self._has_unicode_formatting_chars(content):
            flags.add("UNICODE_FORMAT")
        if self._has_mathematical_unicode(content):
            flags.add("MATH_UNICODE")
        if self._has_invalid_unicode(content):
            flags.add("INVALID_UNICODE")

        return flags

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
