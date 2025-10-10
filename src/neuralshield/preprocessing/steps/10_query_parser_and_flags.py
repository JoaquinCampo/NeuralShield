import re
from typing import Dict, List, Set, Tuple
from urllib.parse import unquote

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


class QueryParserAndFlags(HttpPreprocessor):
    """
    Comprehensive query parameter parsing with anomaly detection.

    Step 10: Query Parser and Flags
    - Parses query parameters with intelligent separator detection (& vs ;)
    - Performs per-token percent decoding for analysis
    - Detects 9 types of parameter anomalies
    - Emits structured output with flags and metadata
    - Supports value shape detection and redaction
    - Security-focused with evidence preservation
    """

    # Configuration
    LONG_VALUE_THRESHOLD = 1024  # bytes

    # Value shape patterns
    SHAPE_PATTERNS = {
        "jwt": re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*$"),
        "uuid": re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
        ),
        "ipv4": re.compile(
            r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
        ),
        "ipv6": re.compile(r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$"),
        "b64url": re.compile(r"^[A-Za-z0-9_-]*$"),
        "b64": re.compile(r"^[A-Za-z0-9+/]*={0,2}$"),
        "hex": re.compile(r"^[0-9a-fA-F]+$"),
        "email": re.compile(r"^[^@]+@[^@]+\.[^@]+$"),
        "uaxurl": re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.I),
        "num": re.compile(r"^-?\d+(\.\d+)?$"),
        "lower": re.compile(r"^[a-z]+$"),
        "upper": re.compile(r"^[A-Z]+$"),
        "alpha": re.compile(r"^[a-zA-Z]+$"),
        "alnum": re.compile(r"^[a-zA-Z0-9]+$"),
        "lowernum": re.compile(r"^[a-z0-9]+$"),
        "uppernum": re.compile(r"^[A-Z0-9]+$"),
    }

    def process(self, request: str) -> str:
        """
        Process structured HTTP request lines, parsing query parameters with anomaly detection.

        Args:
            request: Structured HTTP request from HTML entity detection

        Returns:
            Processed request with parsed query parameters and comprehensive flags
        """
        lines = request.split("\n")
        processed_lines = []
        query_lines = []
        separator_flags = set()
        all_global_flags = set()
        total_param_count = 0

        # First pass: collect all query data
        for line in lines:
            if line.strip() == "":
                processed_lines.append(line)
                continue

            if line.startswith("[QUERY] "):
                query_lines.append(line)
            else:
                # Process any accumulated query data before non-query lines
                if query_lines:
                    query_output, sep_flags, global_flags, param_count = (
                        self._process_all_queries(query_lines)
                    )
                    processed_lines.extend(query_output)
                    separator_flags.update(sep_flags)
                    all_global_flags.update(global_flags)
                    total_param_count += param_count
                    query_lines = []

                processed_lines.append(line)

        # Process any remaining query data
        if query_lines:
            query_output, sep_flags, global_flags, param_count = (
                self._process_all_queries(query_lines)
            )
            processed_lines.extend(query_output)
            separator_flags.update(sep_flags)
            all_global_flags.update(global_flags)
            total_param_count += param_count

        # Add separator metadata if any
        if separator_flags:
            processed_lines.append(f"[QSEP] {' '.join(sorted(separator_flags))}")

        # Add summary metadata
        global_flags_str = (
            " ".join(sorted(all_global_flags)) if all_global_flags else ""
        )
        if global_flags_str:
            processed_lines.append(
                f"[QMETA] count={total_param_count} {global_flags_str}"
            )
        else:
            processed_lines.append(f"[QMETA] count={total_param_count}")

        return "\n".join(processed_lines)

    def _process_all_queries(
        self, query_lines: List[str]
    ) -> Tuple[List[str], Set[str], Set[str], int]:
        """
        Process all consecutive [QUERY] lines as a single query unit.

        Returns:
            tuple: (output_lines, separator_flags, global_flags, param_count)
        """
        all_params = []
        separator_flags = set()

        # Process each query line, checking for separators
        for line in query_lines:
            # Parse [QUERY] line: extract key=value and existing flags
            line_content = line[8:]  # Remove '[QUERY] '

            # Split by spaces to separate key=value from existing flags
            parts = line_content.split()
            if parts:
                param = parts[0]
                existing_flags = set(parts[1:]) if len(parts) > 1 else set()
            else:
                # Empty or malformed [QUERY] line; preserve as bare parameter
                param = ""
                existing_flags = set()

            # Check if this parameter contains separators that need splitting
            sep_type, sep_flags = self._detect_separator_type(param)
            separator_flags.update(sep_flags)

            if sep_type == ";":
                # Split by semicolons for semicolon-dominant patterns
                sub_params = [p.strip() for p in param.split(";") if p.strip()]
                # For each sub-param, preserve the existing flags
                for sub_param in sub_params:
                    all_params.append((sub_param, existing_flags.copy()))
            else:
                # Single parameter with existing flags
                all_params.append((param, existing_flags))
        # Now process all parameters
        parsed_params = []
        key_counts: Dict[str, int] = {}
        global_flags = set()

        for param, existing_flags in all_params:
            parsed_param, new_flags = self._parse_single_parameter(param)
            # Combine existing flags with new flags
            combined_flags = existing_flags | new_flags
            parsed_params.append((parsed_param, combined_flags))

            # Track key repetitions
            if "=" in param:
                key = param.split("=", 1)[0]
                key_counts[key] = key_counts.get(key, 0) + 1

            # Collect global flags (both existing and new)
            global_flags.update(combined_flags)

        # Apply repetition flags
        for i, (parsed_param, param_flags) in enumerate(parsed_params):
            param_str, _ = all_params[i]
            if "=" in param_str:
                key = param_str.split("=", 1)[0]
                if key_counts[key] > 1:
                    param_flags.add(f"QREPEAT:{key}")
                    global_flags.add(f"QREPEAT:{key}")

            parsed_params[i] = (parsed_param, param_flags)

        # Generate output lines (without QMETA, that's handled by caller)
        output_lines = []
        for parsed_param, param_flags in parsed_params:
            flags_str = " ".join(sorted(param_flags)) if param_flags else ""
            if flags_str:
                output_lines.append(f"{parsed_param} {flags_str}")
            else:
                output_lines.append(parsed_param)

        return output_lines, separator_flags, global_flags, len(parsed_params)

    def _parse_query_parameters(self, query_string: str) -> List[str]:
        """
        Parse query string with comprehensive anomaly detection.

        Args:
            query_string: Raw query string from [QUERY] line

        Returns:
            List of output lines: [QUERY] lines + [QSEP] + [QMETA]
        """
        # Step 1: Detect separator type
        separator_type, separator_flags = self._detect_separator_type(query_string)

        # Step 2: Split query into parameters
        parameters = self._split_query_parameters(query_string, separator_type)

        # Step 3: Parse each parameter
        parsed_params = []
        key_counts: Dict[str, int] = {}
        global_flags = set()

        for param in parameters:
            parsed_param, param_flags = self._parse_single_parameter(param)
            parsed_params.append((parsed_param, param_flags))

            # Track key repetitions
            if "=" in param:
                key = param.split("=", 1)[0]
                key_counts[key] = key_counts.get(key, 0) + 1

            # Collect global flags
            global_flags.update(param_flags)

        # Step 4: Apply repetition flags
        for i, (parsed_param, param_flags) in enumerate(parsed_params):
            if "=" in parameters[i]:
                key = parameters[i].split("=", 1)[0]
                if key_counts[key] > 1:
                    param_flags.add(f"QREPEAT:{key}")
                    global_flags.add(f"QREPEAT:{key}")

            parsed_params[i] = (parsed_param, param_flags)

        # Step 5: Generate output lines
        output_lines = []

        # Parameter lines
        for parsed_param, param_flags in parsed_params:
            flags_str = " ".join(sorted(param_flags)) if param_flags else ""
            if flags_str:
                output_lines.append(f"{parsed_param} {flags_str}")
            else:
                output_lines.append(parsed_param)

        # Separator metadata
        if separator_flags:
            output_lines.append(f"[QSEP] {' '.join(sorted(separator_flags))}")

        # Summary metadata
        count = len(parsed_params)
        global_flags_str = " ".join(sorted(global_flags)) if global_flags else ""
        if global_flags_str:
            output_lines.append(f"[QMETA] count={count} {global_flags_str}")
        else:
            output_lines.append(f"[QMETA] count={count}")

        return output_lines

    def _detect_separator_type(self, query_string: str) -> Tuple[str, Set[str]]:
        """
        Detect separator type using intelligent heuristics.

        Returns:
            tuple: (separator_type, flags_set)
        """
        # Default to ampersand
        separator_type = "&"
        flags = set()

        # Count separators
        ampersand_count = query_string.count("&")
        semicolon_count = query_string.count(";")

        if semicolon_count > 0:
            # Check if semicolon is dominant
            # Pattern: k=v(;k=v)+ with few &
            if ampersand_count <= 1 and self._is_semicolon_dominant(query_string):
                separator_type = ";"
                flags.add("QSEMISEP")
            else:
                # Raw semicolon present but not dominant
                flags.add("QRAWSEMI")

        return separator_type, flags

    def _is_semicolon_dominant(self, query_string: str) -> bool:
        """
        Check if semicolon-separated pattern is dominant.
        Pattern: k=v(;k=v)+ with optional trailing components
        """
        # Simple heuristic: at least 2 semicolon-separated k=v pairs
        parts = query_string.split(";")
        kv_pairs = 0

        for part in parts:
            part = part.strip()
            if "=" in part and len(part.split("=", 1)) == 2:
                kv_pairs += 1

        return kv_pairs >= 2

    def _split_query_parameters(self, query_string: str, separator: str) -> List[str]:
        """
        Split query string by detected separator.
        """
        return [
            param.strip() for param in query_string.split(separator) if param.strip()
        ]

    def _parse_single_parameter(self, param: str) -> Tuple[str, Set[str]]:
        """
        Parse single parameter and detect anomalies.

        Args:
            param: Raw parameter string (e.g., "key=value")

        Returns:
            tuple: (formatted_output, flags_set)
        """
        flags = set()

        # Check for bare key (no =)
        if "=" not in param:
            flags.add("QBARE")
            return f"[QUERY] {param}", flags

        # Split key and value
        key, value = param.split("=", 1)

        # Check for empty value
        if not value:
            flags.add("QEMPTYVAL")

        # Apply per-token percent decoding for analysis
        try:
            decoded_value = unquote(value)
        except Exception:
            decoded_value = value  # Fallback to original

        # Check for null bytes
        if "\x00" in decoded_value:
            flags.add("QNUL")

        # Check for non-ASCII
        if not all(ord(c) < 128 for c in key + decoded_value):
            flags.add("QNONASCII")

        # Check for array notation
        if key.endswith("[]"):
            flags.add(f"QARRAY:{key}")

        # Check for long values
        if len(decoded_value.encode("utf-8")) > self.LONG_VALUE_THRESHOLD:
            flags.add("QLONG")

        # Apply redaction if needed (use decoded value for display)
        display_value = decoded_value
        if self._should_redact_value(key, decoded_value):
            shape = self._detect_value_shape(decoded_value)
            length = len(decoded_value.encode("utf-8"))
            display_value = f"<SECRET:{shape}:{length}>"
        elif self._should_shape_value(decoded_value):
            shape = self._detect_value_shape(decoded_value)
            length = len(decoded_value.encode("utf-8"))
            display_value = f"<{shape}:{length}>"

        # Format output
        output = f"[QUERY] {key}={display_value}"

        return output, flags

    def _has_double_percent_encoding(self, value: str) -> bool:
        """
        Check if value contains evidence of double percent encoding.
        """
        # Look for remaining %XX patterns after single decode
        remaining_patterns = re.findall(r"%[0-9A-Fa-f]{2}", value)
        return len(remaining_patterns) > 0

    def _detect_value_shape(self, value: str) -> str:
        """
        Detect the shape/category of a value.

        Returns:
            shape_name: One of the predefined shapes or 'mixed'
        """
        # Test against all patterns
        for shape_name, pattern in self.SHAPE_PATTERNS.items():
            if pattern.match(value):
                return shape_name

        # Default to mixed
        return "mixed"

    def _should_redact_value(self, key: str, value: str) -> bool:
        """
        Determine if a value should be redacted based on key/value patterns.
        """
        # Redact sensitive keys
        sensitive_keys = {
            "password",
            "token",
            "secret",
            "key",
            "auth",
            "api_key",
            "apikey",
            "bearer",
        }
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            return True

        # Redact JWTs, long hex strings, etc.
        shape = self._detect_value_shape(value)
        if shape in {"jwt", "hex"} and len(value) > 20:
            return True

        return False

    def _should_shape_value(self, value: str) -> bool:
        """
        Determine if a value should be shaped (replaced with <shape:len>) instead of shown raw.
        """
        # Shape values that are clearly structured (JWTs, UUIDs, IPs, etc.)
        shape = self._detect_value_shape(value)
        structured_shapes = {"jwt", "uuid", "ipv4", "ipv6", "email", "uaxurl"}

        return shape in structured_shapes
