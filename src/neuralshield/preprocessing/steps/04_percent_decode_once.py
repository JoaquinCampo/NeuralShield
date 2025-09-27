import re

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


class PercentDecodeOnce(HttpPreprocessor):
    """
    Apply percent-decode exactly once per component with context-aware preservation.

    Step 04: Percent Decode Once
    - Decodes %hh sequences exactly once per URL/QUERY component
    - Preserves dangerous encodings (%00, %20, controls, delimiters) as evidence
    - Detects double/multiple encoding attempts (DOUBLEPCT flag)
    - Context-aware: URLs more conservative than queries
    - Flags: DOUBLEPCT, PCTSLASH, PCTBACKSLASH, PCTSPACE, PCTCONTROL, PCTNULL, PCTSUSPICIOUS
    """

    def process(self, request: str) -> str:
        """
        Process structured HTTP request lines, applying percent-decoding exactly once
        with intelligent preservation of dangerous encodings.

        Args:
            request: Structured HTTP request from Unicode NFKC processing

        Returns:
            Processed request with selective decoding and security flags
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
        Process a single URL or QUERY line with percent decoding.

        Args:
            line: Line in format "[TYPE] content"

        Returns:
            tuple: (processed_line, list_of_flags)
        """
        # Split prefix from content
        if line.startswith("[URL] "):
            prefix = "[URL]"
            content = line[6:]  # Remove '[URL] '
            context = "URL"
        elif line.startswith("[QUERY] "):
            prefix = "[QUERY]"
            content = line[8:]  # Remove '[QUERY] '
            context = "QUERY"
        else:
            return line, []

        # Apply percent decoding with context-aware preservation
        decoded_content, flags = self._percent_decode_once(content, context)

        # Reconstruct the line with decoded content and flags
        processed_line = f"{prefix} {decoded_content}"
        if flags:
            processed_line += f" {' '.join(flags)}"

        return (
            processed_line,
            [],
        )  # Return empty flags since they're attached to the line

    def _percent_decode_once(self, text: str, context: str) -> tuple[str, list[str]]:
        """
        Apply percent-decoding exactly once with context-aware preservation.

        Args:
            text: The text to decode
            context: 'URL' or 'QUERY' for different preservation rules

        Returns:
            tuple: (decoded_text, list_of_flags)
        """
        flags = []

        # Apply selective decoding: decode safe encodings, preserve dangerous ones
        decoded = self._selective_percent_decode(text, context)

        # Check for double encoding: look for remaining patterns that are not dangerous
        remaining_patterns_in_decoded = re.findall(r"%[0-9A-Fa-f]{2}", decoded)
        dangerous_encodings = self._get_dangerous_encodings(context)

        # Filter out patterns that are intentionally preserved (dangerous)
        non_dangerous_patterns = [
            p
            for p in remaining_patterns_in_decoded
            if p.upper() not in dangerous_encodings
        ]

        # If there are remaining patterns that could be decoded (not dangerous), it's double encoding
        if non_dangerous_patterns:
            flags.append("DOUBLEPCT")

        # Apply context-aware flagging for preserved dangerous encodings
        if context == "URL":
            flags.extend(self._check_url_preservation(decoded))
        else:  # QUERY
            flags.extend(self._check_query_preservation(decoded))

        # Sort flags alphabetically as per spec
        flags = sorted(set(flags))  # Remove duplicates and sort

        return decoded, flags

    def _selective_percent_decode(self, text: str, context: str) -> str:
        """
        Selectively decode percent encodings based on context.

        Only decode "safe" encodings, preserve dangerous ones as evidence.
        """

        def should_decode_percent(match):
            encoded = match.group(0).upper()
            hex_value = encoded[1:]  # Remove %

            # Always preserve dangerous encodings
            dangerous_encodings = self._get_dangerous_encodings(context)
            if encoded in dangerous_encodings:
                return encoded  # Preserve as-is

            # For other encodings, decode them (they're considered safe)
            try:
                # Convert hex to character
                char_code = int(hex_value, 16)
                return chr(char_code)
            except (ValueError, OverflowError):
                return encoded  # Invalid encoding, preserve

        # Replace all %HH patterns using the selective function
        result = re.sub(r"%[0-9A-Fa-f]{2}", should_decode_percent, text)
        return result

    def _get_dangerous_encodings(self, context: str) -> set[str]:
        """
        Get the set of dangerous percent encodings that should be preserved.
        """
        dangerous = {
            "%00",  # Null byte
        }

        if context == "URL":
            dangerous.update(
                {
                    "%20",  # Space
                    "%01",
                    "%02",
                    "%03",
                    "%04",
                    "%05",
                    "%06",
                    "%07",
                    "%08",
                    "%09",
                    "%0A",
                    "%0B",
                    "%0C",
                    "%0D",
                    "%0E",
                    "%0F",
                    "%10",
                    "%11",
                    "%12",
                    "%13",
                    "%14",
                    "%15",
                    "%16",
                    "%17",
                    "%18",
                    "%19",
                    "%1A",
                    "%1B",
                    "%1C",
                    "%1D",
                    "%1E",
                    "%1F",  # Control characters
                    "%2F",  # Forward slash
                    "%5C",  # Backslash
                }
            )
        else:  # QUERY
            dangerous.update(
                {
                    "%20",  # Space
                    "%01",
                    "%02",
                    "%03",
                    "%04",
                    "%05",
                    "%06",
                    "%07",
                    "%08",
                    "%09",
                    "%0A",
                    "%0B",
                    "%0C",
                    "%0D",
                    "%0E",
                    "%0F",
                    "%10",
                    "%11",
                    "%12",
                    "%13",
                    "%14",
                    "%15",
                    "%16",
                    "%17",
                    "%18",
                    "%19",
                    "%1A",
                    "%1B",
                    "%1C",
                    "%1D",
                    "%1E",
                    "%1F",  # Control characters
                }
            )

        return dangerous

    def _check_url_preservation(self, text: str) -> list[str]:
        """
        Check for preserved encodings in URL context (conservative approach).

        URLs preserve almost everything suspicious.
        """
        flags = []

        # Check for null bytes
        if "%00" in text:
            flags.append("PCTNULL")

        # Check for spaces
        if "%20" in text:
            flags.append("PCTSPACE")

        # Check for control characters (%01-%1F)
        control_pattern = r"%0[1-9A-F]|%1[0-9A-F]"
        if re.search(control_pattern, text, re.IGNORECASE):
            flags.append("PCTCONTROL")

        # Check for path delimiters
        if "%2F" in text.upper():  # Case-insensitive
            flags.append("PCTSLASH")
        if "%5C" in text.upper():
            flags.append("PCTBACKSLASH")

        # Check for other suspicious patterns
        suspicious_patterns = [
            "%3C",
            "%3E",  # < >
            "%22",
            "%27",  # " '
            "%3B",
            "%28",
            "%29",  # ; ( )
            "%7B",
            "%7D",  # { }
            "%7C",
            "%60",  # | `
            "%24",
            "%40",  # $ @
        ]

        for pattern in suspicious_patterns:
            if pattern.upper() in text.upper():
                flags.append("PCTSUSPICIOUS")
                break  # Only flag once for suspicious encodings

        return flags

    def _check_query_preservation(self, text: str) -> list[str]:
        """
        Check for preserved encodings in query context (moderate preservation).
        """
        flags = []

        # Check for null bytes
        if "%00" in text:
            flags.append("PCTNULL")

        # Check for spaces (preserved in queries too per spec)
        if "%20" in text:
            flags.append("PCTSPACE")

        # Check for control characters (%01-%1F)
        control_pattern = r"%0[1-9A-F]|%1[0-9A-F]"
        if re.search(control_pattern, text, re.IGNORECASE):
            flags.append("PCTCONTROL")

        # Check for other suspicious patterns in queries
        suspicious_patterns = [
            "%3C",
            "%3E",  # < >
            "%22",
            "%27",  # " '
            "%3B",
            "%28",
            "%29",  # ; ( )
            "%7B",
            "%7D",  # { }
            "%7C",
            "%60",  # | `
            "%24",
            "%40",  # $ @
            "%2F",
            "%5C",  # / \ (path delimiters suspicious in queries)
        ]

        for pattern in suspicious_patterns:
            if pattern.upper() in text.upper():
                flags.append("PCTSUSPICIOUS")
                break

        return flags
