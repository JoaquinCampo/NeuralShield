"""
00 Framing Cleanup - Remove BOM and edge control characters.
"""

import unicodedata

from loguru import logger

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


class FramingCleanup(HttpPreprocessor):
    """
    Remove framing artifacts from the absolute borders of HTTP request strings.

    This processor removes:
    - BOM (Byte Order Mark) at the beginning
    - Non-printable control characters at the beginning and end

    It preserves all content within the HTTP message structure and
    only modifies the absolute edges to ensure robust parsing downstream.
    """

    def process(self, request: str) -> str:
        """
        Remove framing artifacts from HTTP request.

        Args:
            request: Raw HTTP request string

        Returns:
            HTTP request string with edge artifacts removed
        """

        original_length = len(request)
        processed = request

        # Remove BOM at the beginning
        bom_removed = False
        if processed.startswith("\ufeff"):
            processed = processed[1:]
            bom_removed = True
            logger.debug("Removed BOM from beginning of HTTP request")

        # Remove control characters from the beginning
        leading_controls = 0
        while (
            processed
            and unicodedata.category(processed[0]) == "Cc"
            and processed[0] not in "\t\r\n"
        ):
            processed = processed[1:]
            leading_controls += 1

        # Remove control characters from the end
        trailing_controls = 0
        while (
            processed
            and unicodedata.category(processed[-1]) == "Cc"
            and processed[-1] not in "\t\r\n"
        ):
            processed = processed[:-1]
            trailing_controls += 1

        # Log what was removed
        if bom_removed or leading_controls > 0 or trailing_controls > 0:
            total_removed = original_length - len(processed)
            logger.debug(
                "Framing cleanup: removed {total} chars "
                "(BOM: {bom}, leading controls: {lead}, trailing controls: {trail})",
                total=total_removed,
                bom=bom_removed,
                lead=leading_controls,
                trail=trailing_controls,
            )

        return processed
