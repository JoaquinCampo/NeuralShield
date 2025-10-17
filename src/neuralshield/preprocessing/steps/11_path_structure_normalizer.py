import re
from typing import List, Set, Tuple

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor
from neuralshield.preprocessing.steps.structure_metadata import (
    merge_structure_flags,
)


class PathStructureNormalizer(HttpPreprocessor):
    """
    Normalize URL path structure while preserving security-relevant traversal indicators.

    Step 11: Path Structure Normalizer
    - Collapses multiple slashes to single slashes
    - Removes current directory segments (.)
    - Preserves parent directory segments (..) for traversal detection
    - Emits flags for all structural anomalies
    - Never resolves traversal to maintain attack signatures
    - Security-focused canonicalization
    """

    def process(self, request: str) -> str:
        """
        Process structured HTTP request lines, normalizing URL path structure.

        Args:
            request: Structured HTTP request from query parsing

        Returns:
            Processed request with normalized URL paths and security flags
        """
        lines = request.split("\n")
        processed_lines = []
        structure_flags: set[str] = set()

        for line in lines:
            if line.strip() == "":
                processed_lines.append(line)
                continue

            if line.startswith("[URL] "):
                # Extract URL content and normalize path structure
                url_content = line[6:].strip()  # Remove '[URL] ' and trim whitespace

                # Normalize path and get flags
                normalized_path, flags = self._normalize_path_structure(url_content)

                # Reconstruct line with normalized path and flags
                processed_line = f"[URL] {normalized_path}"

                if "MULTIPLESLASH" in flags:
                    flags.remove("MULTIPLESLASH")
                    structure_flags.add("MULTIPLESLASH")

                if flags:
                    processed_line += f" {' '.join(sorted(flags))}"

                processed_lines.append(processed_line)
            else:
                # Pass through other lines unchanged
                processed_lines.append(line)

        merge_structure_flags(processed_lines, structure_flags)

        return "\n".join(processed_lines)

    def _normalize_path_structure(self, url_path: str) -> Tuple[str, Set[str]]:
        """
        Normalize URL path structure according to security-first rules.

        Args:
            url_path: Raw URL path from [URL] line

        Returns:
            tuple: (normalized_path, flags_set)
        """
        flags = set()

        # Handle empty or root paths
        if not url_path or url_path == "/":
            flags.add("HOME")
            return "/", flags

        # Track if path is absolute
        is_absolute = url_path.startswith("/")

        # Segment the path by literal slashes (not %2F)
        segments = self._segment_path(url_path)

        # Track anomalies during processing
        has_multiple_slashes = self._detect_multiple_slashes(segments)
        has_current_dir = self._detect_current_directory(segments)
        has_parent_dir = self._detect_parent_directory(segments)

        # Apply normalization rules
        normalized_segments = self._normalize_segments(segments)

        # Set flags based on detected anomalies
        if has_multiple_slashes:
            flags.add("MULTIPLESLASH")
        if has_current_dir:
            flags.add("DOTCUR")
        if has_parent_dir:
            flags.add("DOTDOT")

        # Reconstruct path
        normalized_path = self._reconstruct_path(normalized_segments, is_absolute)

        # Handle root path canonicalization
        if normalized_path == "/" or normalized_path == "":
            flags.add("HOME")
            normalized_path = "/"

        return normalized_path, flags

    def _segment_path(self, path: str) -> List[str]:
        """
        Split path into segments by literal slashes, preserving %2F encoding.

        Args:
            path: URL path string

        Returns:
            List of path segments
        """
        # Split by literal '/' but preserve leading slash info
        if path.startswith("/"):
            segments = path[1:].split("/")  # Remove leading slash, split
            segments.insert(0, "")  # Re-insert empty segment for leading slash
        else:
            segments = path.split("/")

        return segments

    def _detect_multiple_slashes(self, segments: List[str]) -> bool:
        """
        Detect if path contains multiple consecutive slashes.

        Args:
            segments: Path segments

        Returns:
            True if multiple slashes detected
        """
        return "" in segments  # Empty segments indicate multiple slashes

    def _detect_current_directory(self, segments: List[str]) -> bool:
        """
        Detect if path contains current directory segments (.).

        Args:
            segments: Path segments

        Returns:
            True if current directory segments detected
        """
        return "." in segments

    def _detect_parent_directory(self, segments: List[str]) -> bool:
        """
        Detect if path contains parent directory segments (..).

        Args:
            segments: Path segments

        Returns:
            True if parent directory segments detected
        """
        return ".." in segments

    def _normalize_segments(self, segments: List[str]) -> List[str]:
        """
        Apply normalization rules to path segments.

        Args:
            segments: Raw path segments

        Returns:
            Normalized path segments
        """
        normalized = []

        for segment in segments:
            if segment == "":
                # Skip empty segments (from multiple slashes)
                continue
            elif segment == ".":
                # Skip current directory segments
                continue
            else:
                # Keep regular segments and parent directory segments
                normalized.append(segment)

        return normalized

    def _reconstruct_path(self, segments: List[str], is_absolute: bool) -> str:
        """
        Reconstruct path from normalized segments.

        Args:
            segments: Normalized path segments
            is_absolute: Whether the original path was absolute

        Returns:
            Reconstructed path string
        """
        if not segments:
            return "/"

        # Join segments
        path = "/".join(segments)

        # Add leading slash for absolute paths
        if is_absolute:
            path = "/" + path

        # Remove trailing slash unless it's just root
        if path.endswith("/") and len(path) > 1:
            path = path.rstrip("/")

        return path


class PathStructureNormalizerSrbh(PathStructureNormalizer):
    """SR_BH-tuned variant that highlights heavy slash abuse and missing benign anchors."""

    @staticmethod
    def _count_empty_groups(segments: List[str], is_absolute: bool) -> int:
        start_idx = 1 if is_absolute and segments else 0
        count = 0
        in_group = False
        for segment in segments[start_idx:]:
            if segment == "":
                if not in_group:
                    count += 1
                    in_group = True
            else:
                in_group = False
        return count

    def _normalize_path_structure(self, url_path: str) -> Tuple[str, Set[str]]:
        normalized_path, flags = super()._normalize_path_structure(url_path)

        segments = self._segment_path(url_path)
        is_absolute = url_path.startswith("/")
        if self._count_empty_groups(segments, is_absolute) > 1:
            flags.add("MULTIPLESLASH_HEAVY")

        return normalized_path, flags

    def process(self, request: str) -> str:
        processed = super().process(request)
        lines = processed.split("\n")

        observed: Set[str] = set()
        for line in lines:
            tokens = line.split()
            if not tokens:
                continue
            start_idx = 1 if tokens[0].startswith("[") else 0
            for token in tokens[start_idx:]:
                if token in {"HOPBYHOP", "HOME"}:
                    observed.add(token)

        missing = [
            f"STRUCT_GAP:{flag}"
            for flag in ("HOPBYHOP", "HOME")
            if flag not in observed
        ]

        if not missing:
            return processed

        gap_index = None
        for idx, line in enumerate(lines):
            if line.startswith("[STRUCT_GAP] "):
                gap_index = idx
                existing = {
                    part for part in line[len("[STRUCT_GAP] ") :].split() if part
                }
                updated = sorted(existing.union(missing))
                lines[idx] = "[STRUCT_GAP] " + " ".join(updated)
                break

        if gap_index is None:
            lines.append("[STRUCT_GAP] " + " ".join(sorted(missing)))

        return "\n".join(lines)
