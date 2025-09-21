"""
02 EOL Anomaly Annotator - Detect and flag line ending anomalies.
"""

from loguru import logger

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor
from neuralshield.preprocessing.steps.exceptions import MalformedHttpRequestError


class EOLAnnotator(HttpPreprocessor):
    """
    Detect and flag End-Of-Line anomalies in HTTP requests.

    This processor identifies non-standard line endings and adds flag markers
    to help detect potential security issues or malformed requests.

    EOL Token Types:
    - CRLF: Standard "\\r\\n" sequence
    - Bare LF: "\\n" not preceded by "\\r"
    - Bare CR: "\\r" not followed by "\\n"

    Flags Added:
    - EOL_BARELF: Line uses only bare LF endings
    - EOL_BARECR: Line uses only bare CR endings
    - EOLMIX: Line mixes different EOL types
    - EOL_EOF_NOCRLF: File ends without proper line ending
    """

    # EOL token constants
    CRLF_TOKEN = "CRLF"
    LF_TOKEN = "LF"
    CR_TOKEN = "CR"

    # Flag constants
    BARE_LF_FLAG = "EOL_BARELF"
    BARE_CR_FLAG = "EOL_BARECR"
    MIXED_EOL_FLAG = "EOLMIX"
    EOF_NO_CRLF_FLAG = "EOL_EOF_NOCRLF"

    KNOWN_FLAGS = {BARE_LF_FLAG, BARE_CR_FLAG, MIXED_EOL_FLAG, EOF_NO_CRLF_FLAG}

    def _extract_line_content(self, text: str, start_pos: int) -> tuple[str, int]:
        """Extract line content up to the first EOL character."""
        text_length = len(text)
        current_pos = start_pos

        while current_pos < text_length and text[current_pos] not in "\r\n":
            current_pos += 1

        line_content = text[start_pos:current_pos]
        return line_content, current_pos

    def _scan_eol_sequence(self, text: str, start_pos: int) -> tuple[set[str], int]:
        """Scan and categorize EOL tokens in sequence."""
        text_length = len(text)
        current_pos = start_pos
        eol_tokens = set()

        while current_pos < text_length and text[current_pos] in "\r\n":
            char = text[current_pos]

            if char == "\r":
                # Check for CRLF sequence
                if current_pos + 1 < text_length and text[current_pos + 1] == "\n":
                    eol_tokens.add(self.CRLF_TOKEN)
                    current_pos += 2
                else:
                    eol_tokens.add(self.CR_TOKEN)
                    current_pos += 1
            else:  # char == '\n'
                eol_tokens.add(self.LF_TOKEN)
                current_pos += 1

        return eol_tokens, current_pos

    def _get_flag_for_eol_pattern(
        self, eol_tokens: set[str], is_eof: bool
    ) -> str | None:
        """Determine the appropriate flag based on EOL pattern."""
        if is_eof:
            return self.EOF_NO_CRLF_FLAG

        if eol_tokens == {self.LF_TOKEN}:
            return self.BARE_LF_FLAG

        if eol_tokens == {self.CR_TOKEN}:
            return self.BARE_CR_FLAG

        if len(eol_tokens) >= 2:
            return self.MIXED_EOL_FLAG

        # Standard CRLF or empty set
        return None

    def _extract_next_line_content(self, text: str, start_pos: int) -> str:
        """Extract the content of the next line without EOL characters."""
        text_length = len(text)
        end_pos = start_pos

        while end_pos < text_length and text[end_pos] not in "\r\n":
            end_pos += 1

        return text[start_pos:end_pos]

    def _should_skip_flag(self, flag: str, text: str, next_line_pos: int) -> bool:
        """Check if flag should be skipped to maintain idempotency."""
        if next_line_pos >= len(text):
            return True

        next_line_content = self._extract_next_line_content(text, next_line_pos)
        return next_line_content in self.KNOWN_FLAGS

    def _append_flag_line(
        self, output_parts: list[str], flag: str, needs_newline: bool
    ) -> None:
        """Append a flag line to the output with proper formatting."""
        if needs_newline:
            output_parts.append("\n")
        output_parts.append(flag)
        output_parts.append("\n")

    def _log_flag_addition(
        self, flag: str, eol_tokens: set[str], start_pos: int, end_pos: int
    ) -> None:
        """Log the addition of a flag for debugging."""
        token_names = ",".join(sorted(eol_tokens)) if eol_tokens else "<none>"
        logger.debug(
            f"EOLAnnotator: adding flag '{flag}' for tokens [{token_names}] "
            f"at position [{start_pos}:{end_pos}]"
        )

    def _process_line(self, text: str, position: int) -> tuple[list[str], int]:
        """Process a single line and return output parts and next position."""
        text_length = len(text)
        output_parts = []

        # Extract line content
        line_content, eol_start_pos = self._extract_line_content(text, position)
        is_existing_flag = line_content in self.KNOWN_FLAGS

        # Scan EOL sequence
        eol_tokens, next_position = self._scan_eol_sequence(text, eol_start_pos)

        # Preserve original content and EOL sequence
        output_parts.append(line_content)
        if next_position > eol_start_pos:
            output_parts.append(text[eol_start_pos:next_position])

        # Determine flag if needed
        if not is_existing_flag:
            is_at_eof = eol_start_pos == text_length
            flag = self._get_flag_for_eol_pattern(eol_tokens, is_at_eof)

            if flag is not None and not self._should_skip_flag(
                flag, text, next_position
            ):
                self._log_flag_addition(flag, eol_tokens, eol_start_pos, next_position)
                needs_newline = is_at_eof
                self._append_flag_line(output_parts, flag, needs_newline)

        # Ensure we advance position
        final_position = (
            max(next_position, position + 1)
            if next_position == position
            else next_position
        )
        return output_parts, final_position

    def process(self, request: str) -> str:
        """Process HTTP request and add EOL anomaly flags."""
        if not request:
            raise MalformedHttpRequestError("Empty HTTP request")

        output_parts = []
        position = 0
        text_length = len(request)

        while position < text_length:
            line_parts, position = self._process_line(request, position)
            output_parts.extend(line_parts)

        return "".join(output_parts)
