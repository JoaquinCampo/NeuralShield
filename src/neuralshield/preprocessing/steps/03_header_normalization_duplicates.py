from typing import Any

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor
from neuralshield.preprocessing.steps.structure_metadata import (
    merge_structure_flags,
)


class HeaderNormalizationDuplicates(HttpPreprocessor):
    """
    Normalize header names and handle duplicates.

    Step 03: Header Normalization and Duplicates
    - Normalizes header names to lowercase per RFC 9110
    - Validates header name characters
    - Detects and handles duplicate headers
    - Flags hop-by-hop headers in requests
    - Emits security flags for anomalies
    """

    # Headers that can be safely merged with comma separation
    MERGEABLE_HEADERS = {
        "accept",
        "accept-encoding",
        "accept-language",
        "cache-control",
        "pragma",
        "link",
        "www-authenticate",
    }

    # Hop-by-hop headers that should not appear in requests
    HOP_BY_HOP_HEADERS = {
        "connection",
        "te",
        "upgrade",
        "trailer",
    }

    # Valid characters for header names per RFC 9110 token definition
    VALID_HEADER_NAME_CHARS = set(
        "!#$%&'*+-.^_`|~"  # Special characters
        + "0123456789"  # Digits
        + "abcdefghijklmnopqrstuvwxyz"  # Lowercase
        + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # Uppercase
    )

    def process(self, request: str) -> str:
        """
        Process the request to normalize header names and detect duplicates.

        Args:
            request: The structured HTTP request with [HEADER] lines

        Returns:
            Processed request with normalized headers, inline flags, and aggregates
        """
        lines = request.split("\n")
        processed_lines = []
        structure_flags: set[str] = set()

        # Two-pass processing: collect then emit
        headers_map: dict[str, list[str]] = {}
        header_flags: dict[str, set[str]] = {}
        global_flags: set[str] = {}
        aggregates = {}

        (
            headers_map,
            header_flags,
            global_flags,
            aggregates,
            struct_updates,
        ) = self._collect_and_normalize_headers(lines)

        structure_flags.update(struct_updates)

        hop_results = self._check_hop_by_hop_headers(headers_map, header_flags)
        header_flags.update(hop_results["header_flags"])
        if hop_results["has_hopbyhop"]:
            aggregates["hopbyhop"] = 1
            structure_flags.add("HOPBYHOP")

        for flags in header_flags.values():
            flags.discard("HOPBYHOP")

        # Emit all non-header, non-flag lines (method, url, query, etc.)
        # Skip existing flags to avoid duplication on reprocessing
        for line in lines:
            if not line.startswith("[HEADER]") and not self._is_security_flag(line):
                processed_lines.append(line)

        # Emit normalized headers in canonical order with inline flags
        processed_lines.extend(self._emit_normalized_headers(headers_map, header_flags))

        # Emit aggregates
        processed_lines.append(self._emit_aggregates(aggregates))

        # Emit global flags if any
        if global_flags:
            processed_lines.append(self._emit_global_flags(global_flags))

        merge_structure_flags(processed_lines, structure_flags)

        return "\n".join(processed_lines)

    def _is_security_flag(self, line: str) -> bool:
        """
        Check if a line is a security flag from this step.

        Args:
            line: The line to check

        Returns:
            True if this is a security flag that should be filtered out
        """
        return (
            line.startswith("BADHDRNAME:")
            or line.startswith("DUPHDR:")
            or line == "HOPBYHOP"
            or line.startswith("HOPBYHOP:")
            or line == "HDRNORM"
            or line == "HDRMERGE"
            or line.startswith("[F:")
            or line.startswith("[HAGG]")
            or line.startswith("[HGF]")
        )

    def _collect_and_normalize_headers(
        self, lines: list[str]
    ) -> tuple[
        dict[str, list[str]],
        dict[str, set[str]],
        set[str],
        dict[str, int],
        set[str],
    ]:
        """
        First pass: collect all headers, normalize names,
        detect duplicates and anomalies.

        Returns:
            Tuple of (headers_map, header_flags, global_flags, aggregates, structure_flags) where:
            - headers_map: dict[normalized_name, list[values]]
            - header_flags: dict[normalized_name, set[flag_strings]]
            - global_flags: set of global flag strings
            - aggregates: dict of statistical aggregates
            - structure_flags: set of structural evidence flags
        """
        headers_map: dict[str, list[str]] = {}
        header_flags: dict[str, set[str]] = {}
        global_flags: set[str] = set()
        structure_flags: set[str] = set()
        aggregates = {
            "h_count": 0,
            "dup_names": 0,
            "hopbyhop": 0,
            "bad_names": 0,
            "total_bytes": 0,
        }

        # Track normalization changes for HDRNORM
        has_normalization_changes = False

        for line in lines:
            if not line.startswith("[HEADER] "):
                continue

            header_content = line[9:]  # Remove "[HEADER] "

            # Parse header name and value
            name, value = self._parse_header_line(header_content)
            if name is None:
                continue  # Skip malformed headers

            # Validate and normalize name
            normalized_name, name_flags = self._normalize_header_name(name)

            # Check if normalization changed the name (original had different casing)
            if name != normalized_name:
                has_normalization_changes = True

            # Initialize header flags for this name if not exists
            if normalized_name not in header_flags:
                header_flags[normalized_name] = set()

            # Add name validation flags
            for flag in name_flags:
                if flag == "BADHDRNAME":
                    header_flags[normalized_name].add("BADHDRNAME")
                    aggregates["bad_names"] += 1
                # Other name flags are handled per header

            # Store header value
            if normalized_name not in headers_map:
                headers_map[normalized_name] = []
            headers_map[normalized_name].append(value)

        # Process duplicates and generate header flags
        duplicate_results = self._process_duplicates(headers_map, header_flags)
        header_flags.update(duplicate_results["header_flags"])
        aggregates["dup_names"] = duplicate_results["dup_count"]

        # Check for hop-by-hop headers
        hop_results = self._check_hop_by_hop_headers(headers_map, header_flags)
        header_flags.update(hop_results["header_flags"])
        if hop_results["has_hopbyhop"]:
            aggregates["hopbyhop"] = 1

        # Set HDRNORM if any normalization occurred
        if has_normalization_changes:
            structure_flags.add("HDRNORM")

        # Calculate aggregates
        aggregates["h_count"] = sum(len(values) for values in headers_map.values())
        aggregates["total_bytes"] = sum(
            len(",".join(values)) for values in headers_map.values()
        )

        return headers_map, header_flags, global_flags, aggregates, structure_flags

    def _parse_header_line(self, header_content: str) -> tuple[str | None, str]:
        """
        Parse a header line into name and value.

        Args:
            header_content: Raw header content like "Host: example.com"

        Returns:
            Tuple of (name, value) or (None, "") for malformed headers
        """
        if not header_content or not header_content.strip():
            return None, ""

        # Find the first colon to separate name from value
        colon_index = header_content.find(":")
        if colon_index == -1:
            # Malformed header: no colon found
            return None, ""

        name = header_content[:colon_index]
        if colon_index + 1 < len(header_content):
            value = header_content[colon_index + 1 :]
        else:
            value = ""

        # Trim whitespace from name only (values handled by Step 10)
        name = name.strip()
        # value = value.strip()  # Removed: Step 10 handles value whitespace

        # Validate that we have a non-empty name
        if not name:
            return None, ""

        return name, value

    def _normalize_header_name(self, name: str) -> tuple[str, set[str]]:
        """
        Normalize header name and detect anomalies.

        Args:
            name: Original header name

        Returns:
            Tuple of (normalized_name, flags)
        """
        flags = set()
        normalized_name = name.lower()

        # NOTE: HDRNORM is now handled globally in _collect_and_normalize_headers

        # Validate characters per RFC 9110 token rules
        invalid_chars = set()
        for char in normalized_name:
            if char not in self.VALID_HEADER_NAME_CHARS:
                invalid_chars.add(char)

        if invalid_chars:
            # Emit BADHDRNAME flag (without header name, since it's inline)
            flags.add("BADHDRNAME")

        return normalized_name, flags

    def _process_duplicates(
        self, headers_map: dict[str, list[str]], header_flags: dict[str, set[str]]
    ) -> dict[str, Any]:
        """
        Process duplicate headers and generate appropriate flags.

        Args:
            headers_map: Map of normalized names to list of values
                (modified in place for merging)
            header_flags: Dict to store per-header flags

        Returns:
            Dict with header_flags updates and duplicate count
        """
        dup_count = 0

        # Process each header for duplicates
        for name, values in list(headers_map.items()):
            if len(values) > 1:
                dup_count += 1

                # Initialize flags for this header if not exists
                if name not in header_flags:
                    header_flags[name] = set()

                # This header has duplicates
                header_flags[name].add("DUPHDR")

                # Merge values for mergeable headers (except set-cookie)
                if name in self.MERGEABLE_HEADERS:
                    merged_value = ", ".join(values)
                    headers_map[name] = [merged_value]
                    header_flags[name].add("HDRMERGE")
                # set-cookie gets flagged but values remain separate (handled in output)

        return {"header_flags": header_flags, "dup_count": dup_count}

    def _check_hop_by_hop_headers(
        self, headers_map: dict[str, list[str]], header_flags: dict[str, set[str]]
    ) -> dict[str, Any]:
        """
        Check for hop-by-hop headers in request context.

        Args:
            headers_map: Map of normalized names to list of values
            header_flags: Dict to store per-header flags

        Returns:
            Dict with header_flags updates and hop-by-hop indicator
        """
        has_hopbyhop = False

        for name in headers_map.keys():
            if name in self.HOP_BY_HOP_HEADERS:
                has_hopbyhop = True

                # Initialize flags for this header if not exists
                if name not in header_flags:
                    header_flags[name] = set()

                header_flags[name].add("HOPBYHOP")

        return {"header_flags": header_flags, "has_hopbyhop": has_hopbyhop}

    def _emit_normalized_headers(
        self, headers_map: dict[str, list[str]], header_flags: dict[str, set[str]]
    ) -> list[str]:
        """
        Emit headers in canonical order with inline flags.

        Args:
            headers_map: Map of normalized names to list of values
            header_flags: Map of header names to their flag sets

        Returns:
            List of formatted header lines with inline flags
        """
        header_lines = []

        # Separate set-cookie from other headers
        set_cookie_headers = []
        other_headers = []

        for name, values in headers_map.items():
            if name == "set-cookie":
                # Set-cookie values remain separate (one header line per value)
                set_cookie_headers.extend(values)
            else:
                # For other headers, emit each value as a separate header line
                # This handles both merged headers (1 value) and non-mergeable duplicates (multiple values)
                for value in values:
                    other_headers.append((name, value))

        # Sort other headers alphabetically by name, then by value for stability
        other_headers.sort(key=lambda x: (x[0], x[1]))

        # Emit other headers first with inline flags
        for name, value in other_headers:
            flags = header_flags.get(name, set())
            if flags:
                # Sort flags by severity: HOPBYHOP > BADHDRNAME > DUPHDR > HDRMERGE
                severity_order = {
                    "HOPBYHOP": 0,
                    "BADHDRNAME": 1,
                    "DUPHDR": 2,
                    "HDRMERGE": 3,
                }
                sorted_flags = sorted(flags, key=lambda f: severity_order.get(f, 99))
                flag_str = ",".join(sorted_flags)
                header_lines.append(f"[HEADER] {name}: {value} {flag_str}")
            else:
                header_lines.append(f"[HEADER] {name}: {value}")

        # Emit set-cookie headers last (in original order) with inline flags
        for value in set_cookie_headers:
            flags = header_flags.get("set-cookie", set())
            if flags:
                severity_order = {
                    "HOPBYHOP": 0,
                    "BADHDRNAME": 1,
                    "DUPHDR": 2,
                    "HDRMERGE": 3,
                }
                sorted_flags = sorted(flags, key=lambda f: severity_order.get(f, 99))
                flag_str = ",".join(sorted_flags)
                header_lines.append(f"[HEADER] set-cookie: {value} {flag_str}")
            else:
                header_lines.append(f"[HEADER] set-cookie: {value}")

        return header_lines

    def _emit_aggregates(self, aggregates: dict[str, int]) -> str:
        """
        Emit aggregates in [HAGG] format.

        Args:
            aggregates: Dict of aggregate statistics

        Returns:
            Formatted aggregates string
        """
        return (
            f"[HAGG] h_count={aggregates['h_count']} "
            f"dup_names={aggregates['dup_names']} "
            f"hopbyhop={aggregates['hopbyhop']} "
            f"bad_names={aggregates['bad_names']} "
            f"total_bytes={aggregates['total_bytes']}"
        )

    def _emit_global_flags(self, global_flags: set[str]) -> str:
        """
        Emit global flags in [HGF] format.

        Args:
            global_flags: Set of global flag strings

        Returns:
            Formatted global flags string
        """
        if not global_flags:
            return ""

        # Sort flags alphabetically for determinism
        sorted_flags = sorted(global_flags)
        flag_str = ",".join(sorted_flags)
        return f"[HGF] {flag_str}"
