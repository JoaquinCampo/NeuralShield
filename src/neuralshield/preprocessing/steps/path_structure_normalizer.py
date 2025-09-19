"""
PathStructureNormalizer: Implements path structure normalization per colapsar-slash-dot-no-resolver-dotdot.md

Handles:
- Collapse multiple slashes (//) to single slash (/)
- Remove current directory segments (/.)
- Preserve parent directory segments (..) without resolution
- Detect and flag structural anomalies

Flags emitted:
- MULTIPLESLASH: When // sequences are collapsed
- DOTDOT: When .. segments are present
- HOME: When path is exactly /
"""

import re
from typing import List

from loguru import logger

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor

# Constants for flags
MULTIPLESLASH_FLAG = "MULTIPLESLASH"
DOTDOT_FLAG = "DOTDOT"
HOME_FLAG = "HOME"


class PathStructureNormalizer(HttpPreprocessor):
    """
    Path structure normalizer implementing
    colapsar-slash-dot-no-resolver-dotdot.md specification.

    Processes [URL] lines to:
    - Collapse multiple slashes (//) to single slash
    - Remove current directory segments (.)
    - Preserve parent directory segments (..) for traversal detection
    - Flag structural anomalies

    Only processes [URL] lines, passes other lines unchanged.
    Emits flags immediately after lines where anomalies are detected.

    Example:
    Input:
        [URL] /foo//bar/.//baz

    Output:
        [URL] /foo/bar/baz
        MULTIPLESLASH
    """

    def __init__(self) -> None:
        # No mutable state - all processing is stateless
        pass

    def _normalize_path_structure(self, path: str) -> tuple[str, List[str]]:
        """
        Normalize path structure and detect anomalies.

        Returns normalized path and list of flags.
        """
        flags: List[str] = []

        # Handle empty path
        if not path:
            return "/", [HOME_FLAG]

        # Split path into segments by '/' without decoding %2F
        # Keep track of original structure
        segments = path.split("/")

        # Check for multiple slashes by looking for empty segments between slashes
        # (but not leading empty segment which indicates absolute path)
        has_multiple_slashes = any(
            segment == ""
            for i, segment in enumerate(segments)
            if i > 0 and i < len(segments) - 1
        )

        # Remove empty segments (from multiple slashes) but preserve leading slash
        normalized_segments = []

        # Handle absolute path (starts with /)
        if segments and segments[0] == "":
            normalized_segments.append("")  # Preserve leading slash marker
            segments = segments[1:]  # Remove empty first element

        # Process remaining segments
        for segment in segments:
            if segment == "":
                # Empty segment from multiple slashes - skip it
                continue
            elif segment == ".":
                # Current directory - skip it (collapses /.)
                continue
            elif segment == "..":
                # Parent directory - preserve it (don't resolve)
                normalized_segments.append(segment)
                if DOTDOT_FLAG not in flags:
                    flags.append(DOTDOT_FLAG)
                    logger.debug("DOTDOT detected: .. segment found")
            else:
                # Regular segment - keep it
                normalized_segments.append(segment)

        # Reconstruct path
        if not normalized_segments or (
            len(normalized_segments) == 1 and normalized_segments[0] == ""
        ):
            # Empty path or only root slash
            normalized_path = "/"
            if HOME_FLAG not in flags:
                flags.append(HOME_FLAG)
                logger.debug("HOME detected: path is root /")
        else:
            # Join segments with single slashes
            if normalized_segments[0] == "":
                # Absolute path
                normalized_path = "/" + "/".join(normalized_segments[1:])
            else:
                # Relative path (shouldn't happen in HTTP but handle gracefully)
                normalized_path = "/".join(normalized_segments)

        # Flag multiple slashes if detected
        if has_multiple_slashes:
            flags.append(MULTIPLESLASH_FLAG)
            logger.debug("MULTIPLESLASH detected: // sequences found")

        # Sort flags alphabetically for consistent output
        flags.sort()
        return normalized_path, flags

    def process(self, request: str) -> str:
        """
        Process structured HTTP request applying path structure normalization.

        Expects input in pipeline format with [METHOD], [URL], [QUERY], [HEADER] prefixes.
        Only processes [URL] lines for path structure normalization.
        Other lines are passed through unchanged.

        Implements path normalization from colapsar-slash-dot-no-resolver-dotdot.md:
        1. Segment by / without decoding %2F
        2. Collapse multiple slashes (//) to single slash
        3. Remove current directory segments (.)
        4. Preserve parent directory segments (..) without resolution
        5. Emit flags immediately after processed lines
        """
        lines = request.strip().split("\n")
        processed_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip existing flag lines to avoid duplication
            if not line.startswith("[") and any(
                flag in line
                for flag in [
                    MULTIPLESLASH_FLAG,
                    DOTDOT_FLAG,
                    HOME_FLAG,
                ]
            ):
                continue

            # Process URL lines only
            if line.startswith("[URL] "):
                path = line[6:]  # Remove '[URL] ' prefix
                normalized_path, flags = self._normalize_path_structure(path)
                processed_lines.append(f"[URL] {normalized_path}")
                if flags:
                    processed_lines.append(" ".join(flags))
            else:
                # Pass through METHOD, QUERY, HEADER, and other lines unchanged
                processed_lines.append(line)

        return "\n".join(processed_lines)
