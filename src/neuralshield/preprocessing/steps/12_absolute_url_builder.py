"""
Absolute URL Builder - Step 12

This step constructs canonical absolute URLs from HTTP request components
to enable security analysis and SSRF detection. It handles multiple request
forms (origin-form, absolute-form, authority-form, asterisk-form) and validates
host header consistency.

Key Features:
- Builds canonical absolute URLs: scheme://host[:port]/path[?query]
- Validates host header consistency with HOSTMISMATCH flag
- Applies IDNA encoding for internationalized domains
- Handles IPv4, IPv6, and domain name formats
- Emits hybrid flags: HOSTMISMATCH (global), IDNA/BADHOST (inline)

Security Benefits:
- SSRF detection through URL/host header validation
- Host header injection prevention
- International domain spoofing detection
- Canonical URL representation for security analysis
"""

import ipaddress
import re
from typing import List, Optional, Tuple

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


class AbsoluteUrlBuilder(HttpPreprocessor):
    """
    Absolute URL Builder - Step 12

    Constructs canonical absolute URLs from HTTP request components while
    validating host header consistency and applying security checks.

    Flag Strategy:
    - HOSTMISMATCH: Global flag for request-level security issues
    - IDNA: Inline flag with affected host header (Unicode processing evidence)
    - BADHOST: Inline flag with affected host header (validation failure)
    """

    # Constants
    DEFAULT_SCHEME = "http"
    DEFAULT_PORTS = {"http": 80, "https": 443}

    # Flag definitions
    STEP12_GLOBAL_FLAGS = {"HOSTMISMATCH"}
    STEP12_INLINE_FLAGS = {"IDNA", "BADHOST"}

    # Host validation patterns
    HOST_PATTERNS = {
        "ipv4": re.compile(r"^(\d{1,3}\.){3}\d{1,3}$"),
        "ipv6": re.compile(r"^\[([0-9a-fA-F:]+)\]$"),
        "ipv6_with_port": re.compile(r"^\[([0-9a-fA-F:]+)\](?::(\d+))?$"),
        "domain": re.compile(
            r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$"
        ),
    }

    def __init__(self):
        """Initialize the Absolute URL Builder."""
        super().__init__()
        # Pre-compile regex patterns for performance
        self._compiled_patterns = {
            "ipv4": re.compile(r"^(\d{1,3}\.){3}\d{1,3}$"),
            "ipv6": re.compile(r"^\[([0-9a-fA-F:]+)\]$"),
            "ipv6_with_port": re.compile(r"^\[([0-9a-fA-F:]+)\]:(\d+)$"),
            "domain": re.compile(
                r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$"
            ),
            "port": re.compile(r"^:(\d+)$"),
        }

    def process(self, request: str) -> str:
        """
        Process HTTP request to build absolute URLs and validate host consistency.

        Args:
            request: HTTP request as string with structured lines

        Returns:
            Enhanced request with [URL_ABS] line and security flags
        """
        lines = request.split("\n")
        processed_lines = []
        global_flags = set()

        # Extract key components
        method = None
        url = None
        headers = []
        host_header = None

        # Parse the request
        for line in lines:
            if line.startswith("[METHOD] "):
                method = line[9:].strip()
                processed_lines.append(line)
            elif line.startswith("[URL] "):
                url = line[6:].strip()
                processed_lines.append(line)
            elif line.startswith("[HEADER] "):
                header_content = line[9:].strip()
                headers.append(header_content)

                # Extract host header
                if header_content.lower().startswith("host:"):
                    host_header = header_content

                processed_lines.append(line)
            else:
                processed_lines.append(line)

        # Process URL construction and validation
        if method and url is not None:
            url_abs_line, inline_flags, request_global_flags = self._process_request(
                method, url, host_header
            )

            # Add URL_ABS line after URL line
            if url_abs_line:
                # Find URL line and insert URL_ABS after it
                for i, line in enumerate(processed_lines):
                    if line.startswith("[URL] "):
                        processed_lines.insert(i + 1, url_abs_line)
                        break

            # Apply inline flags to headers
            if inline_flags:
                processed_lines = self._apply_inline_flags(
                    processed_lines, inline_flags
                )

            # Collect global flags
            global_flags.update(request_global_flags)

        # Emit global flags at end
        if global_flags:
            processed_lines.append(self._emit_global_flags(global_flags))

        return "\n".join(processed_lines)

    def _process_request(
        self, method: str, url: str, host_header: Optional[str]
    ) -> Tuple[Optional[str], dict, set]:
        """
        Process a single request to build absolute URL and validate components.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request target URL
            host_header: Host header content (if present)

        Returns:
            Tuple of (url_abs_line, inline_flags_dict, global_flags_set)
        """
        global_flags = set()
        inline_flags = {}

        # Parse host header
        host_info = self._parse_host_header(host_header)
        host, port, host_valid, host_error = host_info

        # Detect request form
        form_type = self._detect_request_form(url)

        # Process based on form type
        url_abs = None

        if form_type == "origin-form":
            url_abs, form_flags = self._handle_origin_form(url, host_info)
            global_flags.update(form_flags.get("global", set()))
            inline_flags.update(form_flags.get("inline", {}))

        elif form_type == "absolute-form":
            url_abs, form_flags = self._handle_absolute_form(url, host_info)
            global_flags.update(form_flags.get("global", set()))
            inline_flags.update(form_flags.get("inline", {}))

        elif form_type == "authority-form":
            url_abs, form_flags = self._handle_authority_form(method, url, host_info)
            global_flags.update(form_flags.get("global", set()))
            inline_flags.update(form_flags.get("inline", {}))

        elif form_type == "asterisk-form":
            url_abs, form_flags = self._handle_asterisk_form(method, host_info)
            global_flags.update(form_flags.get("global", set()))
            inline_flags.update(form_flags.get("inline", {}))

        # Create URL_ABS line if we have a result
        url_abs_line = f"[URL_ABS] {url_abs}" if url_abs else None

        return url_abs_line, inline_flags, global_flags

    def _parse_host_header(
        self, host_header: Optional[str]
    ) -> Tuple[Optional[str], Optional[int], bool, Optional[str]]:
        """
        Parse Host header into components.

        Args:
            host_header: Raw host header content (may include existing flags)

        Returns:
            Tuple of (host, port, is_valid, error_type)
        """
        if not host_header or not host_header.strip():
            return None, None, False, "missing"

        # Extract host part (remove "host:" prefix if present)
        host_content = host_header.strip()
        if host_content.lower().startswith("host:"):
            host_content = host_content[5:].strip()

        if not host_content:
            return None, None, False, "missing"

        # Strip any existing flags from previous steps (e.g., MIXEDSCRIPT, BADHOST)
        # Split by spaces and take only the first part (the actual host:port)
        host_parts = host_content.split()
        if len(host_parts) > 1:
            # First part should be the host:port, rest are flags
            host_content = host_parts[0]

        host = None
        port = None

        # Handle IPv6 with port: [::1]:8080
        ipv6_with_port_match = self._compiled_patterns["ipv6_with_port"].match(
            host_content
        )
        if ipv6_with_port_match:
            host = f"[{ipv6_with_port_match.group(1)}]"
            port_str = ipv6_with_port_match.group(2)
            port = int(port_str) if port_str else None
        else:
            # Handle IPv6 without port: [::1]
            ipv6_match = self._compiled_patterns["ipv6"].match(host_content)
            if ipv6_match:
                host = host_content
                port = None
            else:
                # Handle IPv4/domain with optional port: example.com:8080 or 192.168.1.1:8080
                if ":" in host_content:
                    parts = host_content.rsplit(":", 1)
                    if len(parts) == 2:
                        potential_host, potential_port = parts

                        # Validate port
                        try:
                            port = int(potential_port)
                            if not (1 <= port <= 65535):
                                return None, None, False, "invalid_port"
                            host = potential_host
                        except ValueError:
                            return None, None, False, "invalid_port"
                    else:
                        host = host_content
                else:
                    # No port specified
                    host = host_content

        # Validate host format
        if not host:
            return None, None, False, "empty_host"

        # Validate host format
        validation_result = self._validate_host_format(host)
        if not validation_result[0]:
            return None, port, False, validation_result[1]

        return host, port, True, None

    def _validate_host_format(self, host: str) -> Tuple[bool, Optional[str]]:
        """
        Validate host format and return detailed error if invalid.

        Args:
            host: Host string to validate

        Returns:
            Tuple of (is_valid, error_type)
        """
        if not host or not host.strip():
            return False, "empty"

        # First priority: Check for IPv6 format [address] or [address]:port
        if host.startswith("["):
            if "]:" in host:
                # IPv6 with port: [::1]:8080
                ipv6_end = host.find("]:")
                ipv6_addr = host[1:ipv6_end]
                port_part = host[ipv6_end + 1 :]
                # Remove leading colon if present
                if port_part.startswith(":"):
                    port_part = port_part[1:]
                try:
                    port_num = int(port_part)
                    if not (1 <= port_num <= 65535):
                        return False, "invalid_port"
                except ValueError:
                    return False, "invalid_port"
            elif host.endswith("]"):
                # IPv6 without port: [::1]
                ipv6_addr = host[1:-1]
            else:
                return False, "invalid_ipv6"

            # Validate IPv6 address using ipaddress module
            try:
                ipaddress.IPv6Address(ipv6_addr)
                return True, None
            except ipaddress.AddressValueError:
                return False, "invalid_ipv6"

        # Second priority: Check for IPv4 format using ipaddress module
        # First check if it looks like an IPv4 address (has 3 dots)
        if host.count(".") == 3:
            try:
                ipaddress.IPv4Address(host)
                return True, None
            except ipaddress.AddressValueError:
                # Looks like IPv4 but invalid - reject completely
                return False, "invalid_ipv4"
        # Not an IPv4 format, continue to domain validation

        # Check for domain format (allow Unicode for IDNA processing)
        # First do basic validation before regex matching
        if ".." in host:
            return False, "consecutive_dots"
        if host.startswith("-") or host.endswith("-"):
            return False, "leading_trailing_dash"
        if len(host.encode("utf-8")) > 253:  # RFC 1035 limit (bytes)
            return False, "too_long"

        # Use more permissive regex that allows long domains and Unicode
        # Allow domains with at least one dot (basic FQDN check)
        if "." not in host:
            return False, "no_tld"

        unicode_domain_pattern = re.compile(
            r"^[a-zA-Z0-9\u0080-\uFFFF]([a-zA-Z0-9\u0080-\uFFFF\-]*[a-zA-Z0-9\u0080-\uFFFF])?(\.[a-zA-Z0-9\u0080-\uFFFF]([a-zA-Z0-9\u0080-\uFFFF\-]*[a-zA-Z0-9\u0080-\uFFFF])?)*$"
        )

        if unicode_domain_pattern.match(host):
            return True, None

        return False, "invalid_format"

    def _detect_request_form(self, url: str) -> str:
        """
        Detect the HTTP request target form.

        Args:
            url: Request target string

        Returns:
            Form type: "origin-form", "absolute-form", "authority-form", "asterisk-form"
        """
        if url == "*":
            return "asterisk-form"

        if "://" in url:
            return "absolute-form"

        if ":" in url and "/" not in url:
            # Authority-form: host:port (no scheme, no path)
            return "authority-form"

        # Default to origin-form: /path?query
        return "origin-form"

    def _handle_origin_form(
        self, url: str, host_info: Tuple
    ) -> Tuple[Optional[str], dict]:
        """
        Handle origin-form requests: /path?query

        Args:
            url: Origin-form URL (e.g., "/api/users")
            host_info: Parsed host header info (host, port, is_valid, error)

        Returns:
            Tuple of (absolute_url, flags_dict)
        """
        host, port, host_valid, host_error = host_info
        flags = {"global": set(), "inline": {}}

        if not host_valid:
            # Invalid or missing host header
            flags["inline"]["host"] = "BADHOST"
            return None, flags

        # Apply IDNA processing if needed
        processed_host, idna_flag = self._process_idna_encoding(host)
        if idna_flag:
            flags["inline"]["host"] = "IDNA"

        # Build absolute URL
        scheme = self.DEFAULT_SCHEME
        port_str = f":{port}" if port and port != self.DEFAULT_PORTS.get(scheme) else ""

        absolute_url = f"{scheme}://{processed_host}{port_str}{url}"

        return absolute_url, flags

    def _handle_absolute_form(
        self, url: str, host_info: Tuple
    ) -> Tuple[Optional[str], dict]:
        """
        Handle absolute-form requests: http://host:port/path?query

        Args:
            url: Absolute-form URL
            host_info: Parsed host header info

        Returns:
            Tuple of (absolute_url, flags_dict)
        """
        host, port, host_valid, host_error = host_info
        flags = {"global": set(), "inline": {}}

        # For absolute-form, we use the URL as-is, but validate against Host header
        if host_valid:
            # Extract host from URL for comparison
            try:
                from urllib.parse import urlparse

                parsed = urlparse(url)
                url_host = parsed.hostname
                url_port = parsed.port

                # Apply IDNA processing to URL host if needed
                processed_url_host, idna_flag = self._process_idna_encoding(
                    url_host or ""
                )
                if idna_flag:
                    flags["inline"]["host"] = "IDNA"

                # Compare hosts (normalize for comparison)
                if url_host and host:
                    url_host_normalized = processed_url_host.lower()
                    host_normalized = (
                        self._process_idna_encoding(host)[0].lower() if host else ""
                    )

                    # Remove brackets from IPv6 for comparison
                    if url_host_normalized.startswith(
                        "["
                    ) and url_host_normalized.endswith("]"):
                        url_host_normalized = url_host_normalized[1:-1]
                    if host_normalized.startswith("[") and host_normalized.endswith(
                        "]"
                    ):
                        host_normalized = host_normalized[1:-1]

                    if url_host_normalized != host_normalized:
                        flags["global"].add("HOSTMISMATCH")

            except Exception:
                # If URL parsing fails, treat as valid absolute URL
                pass

        return url, flags

    def _handle_authority_form(
        self, method: str, url: str, host_info: Tuple
    ) -> Tuple[Optional[str], dict]:
        """
        Handle authority-form requests: host:port (CONNECT method)

        Args:
            method: HTTP method
            url: Authority-form target
            host_info: Parsed host header info

        Returns:
            Tuple of (absolute_url, flags_dict)
        """
        flags = {"global": set(), "inline": {}}

        # Authority-form is only valid for CONNECT method
        if method.upper() != "CONNECT":
            flags["inline"]["host"] = "BADHOST"
            return None, flags

        # Apply IDNA processing if needed
        processed_url, idna_flag = self._process_idna_encoding(url)
        if idna_flag:
            flags["inline"]["host"] = "IDNA"

        # For CONNECT, return the authority as-is (no scheme)
        return processed_url, flags

    def _handle_asterisk_form(
        self, method: str, host_info: Tuple
    ) -> Tuple[Optional[str], dict]:
        """
        Handle asterisk-form requests: * (OPTIONS method)

        Args:
            method: HTTP method
            host_info: Parsed host header info

        Returns:
            Tuple of (absolute_url, flags_dict)
        """
        host, port, host_valid, host_error = host_info
        flags = {"global": set(), "inline": {}}

        if not host_valid:
            flags["inline"]["host"] = "BADHOST"
            return None, flags

        # Apply IDNA processing if needed
        processed_host, idna_flag = self._process_idna_encoding(host)
        if idna_flag:
            flags["inline"]["host"] = "IDNA"

        # Build absolute URL for OPTIONS *
        scheme = self.DEFAULT_SCHEME
        port_str = f":{port}" if port and port != self.DEFAULT_PORTS.get(scheme) else ""

        absolute_url = f"{scheme}://{processed_host}{port_str}/*"

        return absolute_url, flags

    def _process_idna_encoding(self, host: str) -> Tuple[str, bool]:
        """
        Apply IDNA (Internationalized Domain Names in Applications) encoding.

        Args:
            host: Host string that may contain Unicode characters

        Returns:
            Tuple of (processed_host, idna_flag_emitted)
        """
        if not host:
            return host, False

        # Check if host contains non-ASCII characters
        if not self._contains_unicode(host):
            return host, False

        # For IPv6 addresses, don't apply IDNA to the address part
        if host.startswith("[") and host.endswith("]"):
            # IPv6 address - don't encode
            return host, False
        elif "]:" in host:
            # IPv6 with port: [::1]:8080 - encode only the host part
            ipv6_end = host.find("]:")
            ipv6_part = host[: ipv6_end + 1]  # Include the closing bracket
            port_part = host[ipv6_end + 1 :]  # The port part
            encoded_host = ipv6_part  # IPv6 part stays as-is
            if port_part:
                encoded_host += port_part
            return encoded_host, False

        # Try to apply IDNA encoding
        try:
            import idna

            encoded_host = idna.encode(host).decode("ascii")
            return encoded_host, True
        except ImportError:
            # IDNA module not available - for now, just flag Unicode presence
            # In production, this should be installed
            return host, True  # Still flag that Unicode was detected
        except Exception:
            # If IDNA encoding fails, return original and don't flag
            # This prevents breaking valid but non-IDNA domains
            return host, False

    def _contains_unicode(self, text: str) -> bool:
        """
        Check if text contains non-ASCII Unicode characters.

        Args:
            text: Text to check

        Returns:
            True if contains Unicode characters
        """
        return any(ord(char) > 127 for char in text)

    def _apply_inline_flags(self, lines: List[str], inline_flags: dict) -> List[str]:
        """
        Apply inline flags to header lines.

        Args:
            lines: List of processed lines
            inline_flags: Dict mapping header types to flags

        Returns:
            Modified lines with inline flags applied
        """
        modified_lines = []

        for line in lines:
            if line.startswith("[HEADER] "):
                header_content = line[9:].strip()  # Remove "[HEADER] " prefix

                # Check if this header needs flags
                for flag_target, flag_value in inline_flags.items():
                    if flag_target == "host" and header_content.lower().startswith(
                        "host:"
                    ):
                        # Add flag to host header
                        modified_line = f"[HEADER] {header_content} {flag_value}"
                        modified_lines.append(modified_line)
                        break
                else:
                    # No flag needed for this header
                    modified_lines.append(line)
            else:
                modified_lines.append(line)

        return modified_lines

    def _emit_global_flags(self, global_flags: set) -> str:
        """
        Emit global flags in NeuralShield format.

        Args:
            global_flags: Set of global flag strings

        Returns:
            Formatted global flags line
        """
        if not global_flags:
            return ""

        # Sort flags alphabetically for determinism
        sorted_flags = sorted(global_flags)
        return " ".join(sorted_flags)
