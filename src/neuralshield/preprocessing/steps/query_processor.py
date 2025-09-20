"""
Query parameter processing implementing query-decodificar-una-vez.md specification.

This module provides comprehensive query string parsing with:
- Robust separator detection (& vs ; vs mixed)
- Percent-decode exactly once per token
- Multiplicidad preservation and anomaly detection
- Various Q* flags for security and compliance
"""

import re
import urllib.parse
from dataclasses import dataclass
from typing import List, Set, Tuple

from loguru import logger

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor

# Query-specific flags
QSEMISEP_FLAG = "QSEMISEP"  # Semicolon separator detected as dominant
QRAWSEMI_FLAG = "QRAWSEMI"  # Semicolon present but not dominant pattern
DOUBLEPCT_FLAG = "DOUBLEPCT"  # Double percent encoding detected
QBARE_FLAG = "QBARE"  # Parameter without = (bare key)
QEMPTYVAL_FLAG = "QEMPTYVAL"  # Parameter with = but empty value
QNUL_FLAG = "QNUL"  # NUL byte in value after decode
QNONASCII_FLAG = "QNONASCII"  # Non-ASCII characters in key or value
QARRAY_FLAG = "QARRAY"  # Array notation [] suffix in key
QLONG_FLAG = "QLONG"  # Value exceeds length threshold
QREPEAT_PREFIX = "QREPEAT:"  # Key repetition flag


@dataclass
class QueryParameter:
    """Represents a parsed query parameter with metadata."""

    key: str
    value: str
    is_bare: bool = False
    is_empty_value: bool = False
    has_double_encoding: bool = False
    has_nul: bool = False
    has_nonascii: bool = False
    is_array: bool = False
    is_long: bool = False


@dataclass
class QueryParseResult:
    """Result of query string parsing."""

    parameters: List[QueryParameter]
    separator_flags: Set[str]
    parameter_count: int
    key_list: List[str]  # Preserves order and repetition
    repeated_keys: Set[str]


class QueryProcessor(HttpPreprocessor):
    """
    Comprehensive query parameter processor implementing query-decodificar-una-vez.md.

    Processes structured HTTP requests to parse query parameters with:
    - Intelligent separator detection (& vs ; heuristics)
    - Exact-once percent decoding with double-encoding detection
    - Multiplicidad preservation and anomaly flagging
    - Security-focused parameter analysis
    """

    def __init__(self, long_value_threshold: int = 1024) -> None:
        """
        Initialize query processor.

        Args:
            long_value_threshold: Length threshold for QLONG flag (default: 1024 bytes)
        """
        self.long_value_threshold = long_value_threshold

    def _detect_separator(self, query_string: str) -> Tuple[str, Set[str]]:
        """
        Detect optimal separator using heuristics from specification.

        Implements logic:
        - Default to &
        - If ; present and pattern is k=v(;k=v)+ with few &, use mixed ;&
        - If ; present but mixed arbitrarily, use & and flag QRAWSEMI

        Args:
            query_string: Raw query string

        Returns:
            Tuple of (separator, flags_set)
        """
        flags: Set[str] = set()

        if ";" not in query_string:
            return "&", flags

        # Count separators
        semicolon_count = query_string.count(";")
        ampersand_count = query_string.count("&")

        # Check if semicolon is dominant pattern k=v(;k=v)+
        semicolon_pattern = re.compile(r"^\w+=[^;&]*(?:;\w+=[^;&]*)*$")
        is_semicolon_dominant = (
            semicolon_count > 0
            and ampersand_count <= semicolon_count // 2  # Few & vs ;
            and semicolon_pattern.match(query_string)  # Clean k=v;k=v pattern
        )

        if is_semicolon_dominant:
            flags.add(QSEMISEP_FLAG)
            return ";&", flags  # Mixed separator for robust parsing
        else:
            # Semicolon present but not dominant - flag as raw semicolon
            if semicolon_count > 0:
                flags.add(QRAWSEMI_FLAG)
            return "&", flags

    def _percent_decode_once(self, text: str) -> Tuple[str, bool]:
        """
        Apply percent-decode exactly once, detecting double encoding.

        Args:
            text: Text to decode

        Returns:
            Tuple of (decoded_text, has_double_encoding)
        """
        try:
            # First decode pass
            decoded = urllib.parse.unquote(text)

            # Check if valid %hh patterns remain after decode
            percent_pattern = re.compile(r"%[0-9a-fA-F]{2}")
            has_double_encoding = bool(percent_pattern.search(decoded))

            return decoded, has_double_encoding

        except Exception as e:
            logger.warning(f"Percent decode failed for '{text}': {e}")
            return text, False

    def _analyze_parameter(self, key: str, value: str, token: str) -> QueryParameter:
        """
        Analyze a single query parameter for anomalies and metadata.

        Args:
            key: Parameter key (after decode)
            value: Parameter value (after decode)
            token: Original token before parsing

        Returns:
            QueryParameter with analysis results
        """
        # Detect basic structure
        is_bare = "=" not in token
        is_empty_value = "=" in token and value == ""

        # Decode and analyze
        decoded_key, key_double_encoding = self._percent_decode_once(key)
        decoded_value, value_double_encoding = self._percent_decode_once(value)
        has_double_encoding = key_double_encoding or value_double_encoding

        # Security analysis
        has_nul = "\x00" in decoded_value
        has_nonascii = any(ord(c) > 127 for c in decoded_key + decoded_value)
        is_array = decoded_key.endswith("[]")
        is_long = len(decoded_value) > self.long_value_threshold

        return QueryParameter(
            key=decoded_key,
            value=decoded_value,
            is_bare=is_bare,
            is_empty_value=is_empty_value,
            has_double_encoding=has_double_encoding,
            has_nul=has_nul,
            has_nonascii=has_nonascii,
            is_array=is_array,
            is_long=is_long,
        )

    def _parse_query_string(self, query_string: str) -> QueryParseResult:
        """
        Parse complete query string implementing full specification.

        Args:
            query_string: Raw query string (without leading ?)

        Returns:
            QueryParseResult with parsed parameters and metadata
        """
        if not query_string:
            return QueryParseResult(
                parameters=[],
                separator_flags=set(),
                parameter_count=0,
                key_list=[],
                repeated_keys=set(),
            )

        # 1. Detect separator and get flags
        separator, separator_flags = self._detect_separator(query_string)

        # 2. Split into tokens using HTML-entity aware method
        tokens = self._split_query_with_entity_protection(query_string, separator)

        # 3. Parse each token
        parameters = []
        key_list = []
        key_counts: dict[str, int] = {}

        for token in tokens:
            if not token:  # Skip empty tokens
                continue

            # Split on first = if present
            if "=" in token:
                key, value = token.split("=", 1)
            else:
                key, value = token, ""

            # Analyze parameter
            param = self._analyze_parameter(key, value, token)
            parameters.append(param)

            # Track keys for multiplicidad
            key_list.append(param.key)
            key_counts[param.key] = key_counts.get(param.key, 0) + 1

        # Find repeated keys
        repeated_keys = {key for key, count in key_counts.items() if count > 1}

        return QueryParseResult(
            parameters=parameters,
            separator_flags=separator_flags,
            parameter_count=len(parameters),
            key_list=key_list,
            repeated_keys=repeated_keys,
        )

    def _generate_flags(self, result: QueryParseResult) -> List[str]:
        """
        Generate all applicable flags based on parse result.

        Args:
            result: Query parse result

        Returns:
            List of flag strings
        """
        flags: List[str] = []

        # Separator flags
        flags.extend(result.separator_flags)

        # Parameter-specific flags
        for param in result.parameters:
            if param.is_bare:
                flags.append(QBARE_FLAG)
            if param.is_empty_value:
                flags.append(QEMPTYVAL_FLAG)
            if param.has_double_encoding:
                flags.append(DOUBLEPCT_FLAG)
            if param.has_nul:
                flags.append(QNUL_FLAG)
            if param.has_nonascii:
                flags.append(QNONASCII_FLAG)
            if param.is_array:
                flags.append(f"{QARRAY_FLAG}:{param.key}")
            if param.is_long:
                flags.append(QLONG_FLAG)

        # Repetition flags
        for key in result.repeated_keys:
            flags.append(f"{QREPEAT_PREFIX}{key}")

        # Remove duplicates while preserving order
        seen = set()
        unique_flags = []
        for flag in flags:
            if flag not in seen:
                seen.add(flag)
                unique_flags.append(flag)

        return unique_flags

    def _split_query_with_entity_protection(
        self, query_string: str, separator: str
    ) -> List[str]:
        """
        Split query string while protecting HTML entities from being broken.

        Args:
            query_string: Query string to split
            separator: Detected separator ("&", ";", or ";&")

        Returns:
            List of parameter tokens with HTML entities preserved
        """
        # HTML entity pattern - same as RequestStructurer
        html_entity_pattern = re.compile(
            r"&(?:[a-zA-Z][a-zA-Z0-9]*|#(?:\d+|x[0-9a-fA-F]+));"
        )

        # Find and protect HTML entities
        entities: List[str] = []

        def replace_entity(match):
            placeholder = f"__HTMLENT_{len(entities)}__"
            entities.append(match.group(0))
            return placeholder

        protected_query = html_entity_pattern.sub(replace_entity, query_string)

        # Now split safely
        if separator == ";&":
            # Mixed separator - split on both & and ;
            tokens = re.split(r"[&;]", protected_query)
        else:
            tokens = protected_query.split(separator)

        # Restore HTML entities in each token
        restored_tokens = []
        for token in tokens:
            restored_token = token
            for i, entity in enumerate(entities):
                restored_token = restored_token.replace(f"__HTMLENT_{i}__", entity)
            restored_tokens.append(restored_token)

        return restored_tokens

    def _join_query_parts_safely(self, query_parts: List[str]) -> str:
        """
        Safely join query parts without breaking HTML entities.

        The RequestStructurer correctly parses HTML entities, but when we reconstruct
        the query string by joining with '&', we can break entities like &#x3c;
        This method preserves the original structure.

        Args:
            query_parts: List of query parameter strings from [QUERY] lines

        Returns:
            Properly reconstructed query string
        """
        if not query_parts:
            return ""

        # Simply join with & - the RequestStructurer has already done the hard work
        # of correctly parsing HTML entities, so we just need to reconstruct
        return "&".join(query_parts)

    def _format_query_output(
        self, result: QueryParseResult, flags: List[str]
    ) -> List[str]:
        """
        Format query processing output using tokenization-optimized individual parameter lines.

        New format:
        [QPARAM] {key} [flags...]
        [QSEP] [separator_flags...]
        [QMETA] count=N [global_flags...]

        Args:
            result: Query parse result
            flags: Generated flags

        Returns:
            List of output lines
        """
        lines = []

        if result.parameter_count > 0:
            # Track key repetitions for QREPEAT flags
            key_counts: dict[str, int] = {}
            for param in result.parameters:
                key = param.key if param.key else "<empty>"
                key_counts[key] = key_counts.get(key, 0) + 1

            # Output individual parameters with their specific flags
            for param in result.parameters:
                param_flags = []

                # Use <empty> for empty keys for clarity
                display_key = param.key if param.key else "<empty>"

                # Add parameter-specific flags
                if param.is_bare:
                    param_flags.append(QBARE_FLAG)
                if param.is_empty_value:
                    param_flags.append(QEMPTYVAL_FLAG)
                if param.has_double_encoding:
                    param_flags.append(DOUBLEPCT_FLAG)
                if param.has_nul:
                    param_flags.append(QNUL_FLAG)
                if param.has_nonascii:
                    param_flags.append(QNONASCII_FLAG)
                if param.is_array:
                    param_flags.append(f"{QARRAY_FLAG}:{display_key}")
                if param.is_long:
                    param_flags.append(QLONG_FLAG)

                # Add repetition flag if this key appears multiple times
                if key_counts.get(param.key if param.key else "<empty>", 0) > 1:
                    param_flags.append(f"{QREPEAT_PREFIX}{display_key}")

                # Format parameter line
                if param_flags:
                    lines.append(f"[QPARAM] {display_key} {' '.join(param_flags)}")
                else:
                    lines.append(f"[QPARAM] {display_key}")

            # Add separator information if present
            separator_flags = [
                flag for flag in flags if flag in [QSEMISEP_FLAG, QRAWSEMI_FLAG]
            ]
            if separator_flags:
                lines.append(f"[QSEP] {' '.join(separator_flags)}")

            # Add global metadata
            global_flags = [
                flag
                for flag in flags
                if flag not in separator_flags and not flag.startswith(QREPEAT_PREFIX)
            ]
            if global_flags:
                lines.append(
                    f"[QMETA] count={result.parameter_count} {' '.join(global_flags)}"
                )
            else:
                lines.append(f"[QMETA] count={result.parameter_count}")

        return lines

    def process(self, request: str) -> str:
        """
        Process structured HTTP request to parse and analyze query parameters.

        Expects input with [METHOD], [URL], [QUERY], [HEADER] prefixes.
        Processes [QUERY] lines and replaces them with parsed parameter analysis.

        Args:
            request: Structured HTTP request string

        Returns:
            Processed request with query analysis
        """
        lines = request.strip().split("\n")
        processed_lines = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            # Skip existing flag lines from other processors
            if not line.startswith("[") and any(
                flag == line.strip()
                or f" {flag} " in f" {line.strip()} "
                or line.strip().startswith(f"{flag} ")
                or line.strip().endswith(f" {flag}")
                for flag in [
                    "Q:",
                    "KEYS:",
                    QSEMISEP_FLAG,
                    QRAWSEMI_FLAG,
                    DOUBLEPCT_FLAG,
                    QBARE_FLAG,
                    QEMPTYVAL_FLAG,
                    QNUL_FLAG,
                    QNONASCII_FLAG,
                    QARRAY_FLAG,
                    QLONG_FLAG,
                    QREPEAT_PREFIX,
                ]
            ):
                i += 1
                continue

            if line.startswith("[QUERY] "):
                # Extract query string from current QUERY line format
                query_param = line[8:]  # Remove '[QUERY] ' prefix

                # Collect all QUERY lines (may have flags interspersed due to pipeline order)
                query_parts = [query_param]
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if next_line.startswith("[QUERY] "):
                        query_parts.append(next_line[8:])
                        j += 1
                    elif not next_line.startswith("[") and any(
                        flag == next_line.strip()
                        or f" {flag} " in f" {next_line.strip()} "
                        or next_line.strip().startswith(f"{flag} ")
                        or next_line.strip().endswith(f" {flag}")
                        for flag in [
                            DOUBLEPCT_FLAG,
                            QBARE_FLAG,
                            QEMPTYVAL_FLAG,
                            QNUL_FLAG,
                            QNONASCII_FLAG,
                            QARRAY_FLAG,
                            QLONG_FLAG,
                            QREPEAT_PREFIX,
                            # Include flags from other processors that might appear
                            "FULLWIDTH",
                            "CONTROL",
                            "HTMLENT",
                        ]
                    ):
                        # Skip flag lines between QUERY lines
                        j += 1
                    else:
                        # Hit a different type of line, stop collecting
                        break

                # Reconstruct query string from parts
                # Use HTML-entity aware joining to prevent breaking entities like &#x3c;
                query_string = (
                    self._join_query_parts_safely(query_parts)
                    if query_parts != [""]
                    else ""
                )

                # Process the complete query string
                result = self._parse_query_string(query_string)
                flags = self._generate_flags(result)
                output_lines = self._format_query_output(result, flags)

                # Add output lines
                processed_lines.extend(output_lines)

                # Skip processed QUERY lines
                i = j
            else:
                # Pass through other lines unchanged
                processed_lines.append(line)
                i += 1

        return "\n".join(processed_lines)
