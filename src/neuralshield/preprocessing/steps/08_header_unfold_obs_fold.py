from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


class HeaderUnfoldObsFold(HttpPreprocessor):
    """
    Unfold obs-fold header continuations.

    Step 08: Header Unfold (Obs-fold)
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
        processed_lines = []

        # State tracking for header context
        current_header = None
        current_header_index = None

        for line in lines:
            if line.strip() == "":
                processed_lines.append(line)
                continue

            if line.startswith("[HEADER] "):
                header_content = line[9:]  # Remove "[HEADER] "

                if self._is_continuation_line(header_content):
                    # This is a continuation line
                    if current_header is None:
                        # Orphaned continuation - no preceding header
                        processed_lines.append("[HEADER] BADHDRCONT")
                        continue  # Skip orphaned continuation
                    else:
                        # Valid continuation - join to current header
                        unfolded_header = self._unfold_header(
                            current_header, header_content
                        )
                        # Rebuild the original header at its source index
                        # Attach OBSFOLD to the updated line
                        updated_line = f"[HEADER] {unfolded_header} OBSFOLD"
                        if self._has_embedded_crlf(header_content):
                            updated_line += " BADCRLF"
                        # Replace original header line (not the last processed line)
                        if current_header_index is not None:
                            processed_lines[current_header_index] = updated_line
                        else:
                            # Fallback: if index is missing, modify the most recent line
                            processed_lines[-1] = updated_line

                        # Update current_header to allow multiple continuations
                        current_header = unfolded_header

                        # Move to next input line after handling this continuation
                        continue
                elif not self._is_valid_header_line(header_content):
                    # Malformed header (no colon) - indicates embedded CRLF
                    current_header = header_content
                    # Attach BADCRLF flag inline
                    processed_lines.append(f"{line} BADCRLF")
                    current_header_index = len(processed_lines) - 1
                else:
                    # This is a new valid header line
                    current_header = header_content
                    line_to_add = line
                    # Check for embedded CRLF in header name/value - attach inline
                    if self._has_embedded_crlf(header_content):
                        line_to_add += " BADCRLF"
                    processed_lines.append(line_to_add)
                    current_header_index = len(processed_lines) - 1
            else:
                # Non-header line
                processed_lines.append(line)

        return "\n".join(processed_lines)

    def _is_valid_header_line(self, header_content: str) -> bool:
        """
        Check if a header line is valid (contains a colon for name:value format).

        Args:
            header_content: The header content to check

        Returns:
            True if the header line contains a colon (valid format)
        """
        return ":" in header_content

    def _is_continuation_line(self, header_content: str) -> bool:
        """
        Check if a header line is a continuation (starts with SP or HTAB).

        Args:
            header_content: The header content to check

        Returns:
            True if this is a continuation line
        """
        if not header_content:
            return False

        first_char = header_content[0]
        return first_char in (" ", "\t")  # SP (0x20) or HTAB (0x09)

    def _unfold_header(self, current_header: str, continuation: str) -> str:
        """
        Unfold a header by joining it with its continuation.

        Args:
            current_header: The current header content
            continuation: The continuation line content

        Returns:
            The unfolded header with single space separator
        """
        # Trim leading whitespace from continuation
        trimmed_continuation = continuation.lstrip(" \t")

        # Join with single space
        if trimmed_continuation:
            return f"{current_header} {trimmed_continuation}"
        else:
            # Empty continuation after trimming - still join with space
            return f"{current_header} "

    def _has_embedded_crlf(self, content: str) -> bool:
        """
        Check if content contains embedded CR/LF characters.

        Args:
            content: The content to check

        Returns:
            True if CR or LF characters are found
        """
        return "\r" in content or "\n" in content
