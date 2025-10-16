from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


class HeaderUnfoldObsFold(HttpPreprocessor):
    """
    Unfold obs-fold header continuations.

    Step 02: Header Unfold (Obs-fold)
    - Detects and unfolds obsolete header folding (obs-fold) to produce
      single-line headers
    - Unfolds headers that continue on next line with SP/HTAB
    - Normalizes continuation separators to single space
    - Detects and flags embedded CR/LF characters
    - Handles malformed continuation lines
    """

    def process(self, request: str) -> str:
        """
        Process the request to unfold obs-fold header continuations.

        Args:
            request: The structured HTTP request with [HEADER] lines

        Returns:
            Processed request with unfolded headers and security flags
        """
        lines = request.split("\n")
        processed_lines: list[str] = []

        current_header: str | None = None
        current_header_index: int | None = None
        seen_first_header = False

        for line in lines:
            if not line.startswith("[HEADER] "):
                if self._rule_skip_pre_header_whitespace(line, seen_first_header):
                    continue
                processed_lines.append(line)
                current_header = None
                current_header_index = None
                continue

            header_content = line[9:]

            if self._rule_5_2_is_continuation_line(header_content):
                current_header, current_header_index = self._rule_5_2_handle_continuation(
                    header_content,
                    processed_lines,
                    current_header,
                    current_header_index,
                )
                continue

            seen_first_header = True
            processed_line, current_header = self._rule_5_2_handle_new_header(
                line, header_content
            )
            processed_lines.append(processed_line)
            current_header_index = len(processed_lines) - 1

        return "\n".join(processed_lines)

    def _rule_5_2_is_valid_header_line(self, header_content: str) -> bool:
        """RFC 9112 §5.2 – header field-lines must contain a colon."""
        return ":" in header_content

    def _rule_5_2_is_continuation_line(self, header_content: str) -> bool:
        """RFC 9112 §5.2 – obs-fold continuation starts with SP / HTAB."""
        if not header_content:
            return False

        first_char = header_content[0]
        return first_char in (" ", "\t")  # SP (0x20) or HTAB (0x09)

    def _rule_5_2_unfold_header(self, current_header: str, continuation: str) -> str:
        """RFC 9112 §5.2 – unfold continuation with single SP separator."""
        # Trim leading whitespace from continuation
        trimmed_continuation = continuation.lstrip(" \t")

        # Join with single space
        if trimmed_continuation:
            return f"{current_header} {trimmed_continuation}"
        else:
            # Empty continuation after trimming - still join with space
            return f"{current_header} "

    def _rule_detect_embedded_crlf(self, content: str) -> bool:
        """Detect embedded CR/LF evidence inside a header segment."""
        return "\r" in content or "\n" in content

    def _rule_5_2_handle_continuation(
        self,
        continuation_content: str,
        processed_lines: list[str],
        current_header: str | None,
        current_header_index: int | None,
    ) -> tuple[str | None, int | None]:
        """Handle obs-fold continuation lines per RFC 9112 §5.2."""
        if current_header is None or current_header_index is None:
            processed_lines.append("[HEADER] BADHDRCONT")
            return current_header, current_header_index

        unfolded_header = self._rule_5_2_unfold_header(
            current_header, continuation_content
        )
        updated_line = f"[HEADER] {unfolded_header} OBSFOLD"
        if self._rule_detect_embedded_crlf(continuation_content):
            updated_line += " BADCRLF"

        processed_lines[current_header_index] = updated_line
        return unfolded_header, current_header_index

    def _rule_5_2_handle_new_header(
        self,
        original_line: str,
        header_content: str,
    ) -> tuple[str, str | None]:
        """Handle a fresh [HEADER] field-line, attaching evidence as needed."""
        if not self._rule_5_2_is_valid_header_line(header_content):
            return f"{original_line} BADCRLF", header_content

        line_to_add = original_line
        if self._rule_detect_embedded_crlf(header_content):
            line_to_add += " BADCRLF"

        return line_to_add, header_content

    def _rule_skip_pre_header_whitespace(self, line: str, seen_first_header: bool) -> bool:
        """RFC 9112 §2.2 – ignore blank lines before the first header field."""

        return line.strip() == "" and not seen_first_header
