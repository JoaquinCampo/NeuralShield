"""
Absolute URL Builder - Step 06

Build a canonical absolute URL for each request and enforce RFC 9112 host
requirements across the different request target forms.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


@dataclass
class HostInfo:
    """Structured representation of the Host header."""

    value: str | None
    host: str | None
    port: int | None
    was_present: bool
    is_empty: bool
    is_valid: bool
    error: str | None


class AbsoluteUrlBuilder(HttpPreprocessor):
    """
    Construct canonical absolute URLs while validating Host header consistency.

    RFC 9112 references covered here:
    - §3.2: host header presence, authority matching, and special-form rules.
    """

    DEFAULT_SCHEME = "http"
    DEFAULT_PORTS = {"http": 80, "https": 443}

    def __init__(self) -> None:
        super().__init__()
        self._compiled_patterns = {
            "ipv4": re.compile(r"^(\d{1,3}\.){3}\d{1,3}$"),
            "ipv6": re.compile(r"^\[([0-9a-fA-F:]+)\]$"),
            "ipv6_with_port": re.compile(r"^\[([0-9a-fA-F:]+)\]:(\d+)$"),
        }

    def process(self, request: str) -> str:
        lines = request.split("\n")
        processed_lines: list[str] = []
        global_flags: set[str] = set()

        method: str | None = None
        url: str | None = None
        host_header: str | None = None

        for line in lines:
            if line.startswith("[METHOD] "):
                method = line[9:].strip()
                processed_lines.append(line)
            elif line.startswith("[URL] "):
                url = line[6:].strip()
                processed_lines.append(line)
            elif line.startswith("[HEADER] "):
                header_content = line[9:].strip()
                if header_content.lower().startswith("host:"):
                    host_header = header_content
                processed_lines.append(line)
            else:
                processed_lines.append(line)

        if method and url is not None:
            # RFC 9112 §3.2 – evaluate host requirements for the detected request form.
            url_abs_line, inline_flags, request_global_flags = self._rule_process_request(
                method, url, host_header
            )

            if url_abs_line:
                for index, line in enumerate(processed_lines):
                    if line.startswith("[URL] "):
                        processed_lines.insert(index + 1, url_abs_line)
                        break

            if inline_flags:
                processed_lines = self._rule_apply_inline_flags(processed_lines, inline_flags)

            global_flags.update(request_global_flags)

        if global_flags:
            processed_lines.append(self._rule_emit_global_flags(global_flags))

        return "\n".join(processed_lines)

    # --------------------------------------------------------------------- #
    # Core processing
    # --------------------------------------------------------------------- #

    def _rule_process_request(
        self,
        method: str,
        url: str,
        host_header: Optional[str],
    ) -> tuple[str | None, dict[str, set[str]], set[str]]:
        inline_flags: dict[str, set[str]] = {}
        global_flags: set[str] = set()

        host_info = self._rule_parse_host_header(host_header)
        form_type = self._rule_detect_request_form(url)

        url_abs: str | None = None

        if form_type == "origin-form":
            url_abs, inline, global_ = self._rule_handle_origin_form(url, host_info)
        elif form_type == "absolute-form":
            url_abs, inline, global_ = self._rule_handle_absolute_form(url, host_info)
        elif form_type == "authority-form":
            url_abs, inline, global_ = self._rule_handle_authority_form(method, url, host_info)
        else:  # asterisk-form
            url_abs, inline, global_ = self._rule_handle_asterisk_form(method, host_info)

        self._rule_merge_inline_flags(inline_flags, inline)
        global_flags.update(global_)

        url_abs_line = f"[URL_ABS] {url_abs}" if url_abs else None
        return url_abs_line, inline_flags, global_flags

    # --------------------------------------------------------------------- #
    # Host parsing & validation helpers
    # --------------------------------------------------------------------- #

    def _rule_parse_host_header(self, host_header: Optional[str]) -> HostInfo:
        if host_header is None or not host_header.strip():
            return HostInfo(
                value=None,
                host=None,
                port=None,
                was_present=False,
                is_empty=False,
                is_valid=False,
                error="missing",
            )

        host_content = host_header.strip()
        if host_content.lower().startswith("host:"):
            host_content = host_content[5:].strip()

        if host_content == "":
            return HostInfo(
                value="",
                host=None,
                port=None,
                was_present=True,
                is_empty=True,
                is_valid=False,
                error="empty",
            )

        # Remove inline flags that previous steps may have appended.
        host_value = host_content.split()[0]

        host, port, split_error = self._split_host_port(host_value)
        if split_error:
            return HostInfo(
                value=host_value,
                host=None,
                port=None,
                was_present=True,
                is_empty=False,
                is_valid=False,
                error=split_error,
            )

        is_valid, validation_error = self._rule_validate_host_format(host)

        return HostInfo(
            value=host_value,
            host=host,
            port=port,
            was_present=True,
            is_empty=False,
            is_valid=is_valid,
            error=validation_error,
        )

    def _split_host_port(
        self,
        host_value: str,
    ) -> tuple[str | None, int | None, str | None]:
        match = self._compiled_patterns["ipv6_with_port"].match(host_value)
        if match:
            address = f"[{match.group(1)}]"
            port = int(match.group(2))
            if not (1 <= port <= 65535):
                return None, None, "invalid_port"
            return address, port, None

        if self._compiled_patterns["ipv6"].match(host_value):
            return host_value, None, None

        if ":" in host_value:
            potential_host, potential_port = host_value.rsplit(":", 1)
            try:
                port = int(potential_port)
            except ValueError:
                return None, None, "invalid_port"
            if not (1 <= port <= 65535):
                return None, None, "invalid_port"
            return potential_host, port, None

        return host_value, None, None

    def _rule_validate_host_format(self, host: str | None) -> tuple[bool, str | None]:
        if not host or not host.strip():
            return False, "empty"

        if host.startswith("["):
            if not host.endswith("]"):
                return False, "invalid_ipv6"
            try:
                ipaddress.IPv6Address(host[1:-1])
                return True, None
            except ipaddress.AddressValueError:
                return False, "invalid_ipv6"

        if host.count(".") == 3:
            try:
                ipaddress.IPv4Address(host)
                return True, None
            except ipaddress.AddressValueError:
                return False, "invalid_ipv4"

        if ".." in host:
            return False, "consecutive_dots"
        if host.startswith("-") or host.endswith("-"):
            return False, "leading_trailing_dash"
        if len(host.encode("utf-8")) > 253:
            return False, "too_long"
        if "." not in host:
            return False, "no_tld"

        unicode_domain_pattern = re.compile(
            r"^[a-zA-Z0-9\u0080-\uFFFF]([a-zA-Z0-9\u0080-\uFFFF\-]*[a-zA-Z0-9\u0080-\uFFFF])?"
            r"(\.[a-zA-Z0-9\u0080-\uFFFF]([a-zA-Z0-9\u0080-\uFFFF\-]*[a-zA-Z0-9\u0080-\uFFFF])?)*$"
        )
        if unicode_domain_pattern.match(host):
            return True, None

        return False, "invalid_format"

    # --------------------------------------------------------------------- #
    # Request form handlers
    # --------------------------------------------------------------------- #

    def _rule_handle_origin_form(
        self,
        url: str,
        host_info: HostInfo,
    ) -> tuple[str | None, dict[str, set[str]], set[str]]:
        inline: dict[str, set[str]] = {}
        global_flags: set[str] = set()

        if not host_info.was_present:
            global_flags.add("HOSTMISSING")
            return None, inline, global_flags

        if host_info.is_empty:
            self._rule_add_inline_flag(inline, "host", "EMPTYHOST")
            return None, inline, global_flags

        if not host_info.is_valid or not host_info.host:
            self._rule_add_inline_flag(inline, "host", "BADHOST")
            return None, inline, global_flags

        processed_host, idna_flag = self._rule_process_idna(host_info.host)
        if idna_flag:
            self._rule_add_inline_flag(inline, "host", "IDNA")

        scheme = self.DEFAULT_SCHEME
        default_port = self.DEFAULT_PORTS.get(scheme)
        port = host_info.port
        port_str = f":{port}" if port and port != default_port else ""

        absolute_url = f"{scheme}://{processed_host}{port_str}{url}"
        return absolute_url, inline, global_flags

    def _rule_handle_absolute_form(
        self,
        url: str,
        host_info: HostInfo,
    ) -> tuple[str | None, dict[str, set[str]], set[str]]:
        inline: dict[str, set[str]] = {}
        global_flags: set[str] = set()

        parsed = urlparse(url)
        url_host = parsed.hostname or ""
        url_port = parsed.port
        scheme = parsed.scheme or self.DEFAULT_SCHEME
        default_port = self.DEFAULT_PORTS.get(scheme)

        if not host_info.was_present:
            global_flags.add("HOSTMISSING")
        elif host_info.is_empty:
            self._rule_add_inline_flag(inline, "host", "EMPTYHOST")
        elif not host_info.is_valid or not host_info.host:
            self._rule_add_inline_flag(inline, "host", "BADHOST")
        else:
            header_host_norm, header_idna = self._rule_process_idna(host_info.host)
            if header_idna:
                self._rule_add_inline_flag(inline, "host", "IDNA")

            url_host_norm, _ = self._rule_process_idna(url_host)

            header_port = host_info.port or default_port
            target_port = url_port or default_port

            if self._rule_hosts_mismatch(
                header_host_norm,
                header_port,
                url_host_norm,
                target_port,
            ):
                global_flags.add("HOSTMISMATCH")

        return url, inline, global_flags

    def _rule_handle_authority_form(
        self,
        method: str,
        url: str,
        host_info: HostInfo,
    ) -> tuple[str | None, dict[str, set[str]], set[str]]:
        inline: dict[str, set[str]] = {}
        global_flags: set[str] = set()

        if method.upper() != "CONNECT":
            self._rule_add_inline_flag(inline, "host", "BADHOST")
            return None, inline, global_flags

        target_host, target_port = self._rule_split_authority_target(url)
        if target_host is None or target_port is None:
            global_flags.add("BADAUTHORITY")
            return None, inline, global_flags

        processed_target, target_idna = self._rule_process_idna(target_host)
        if target_idna:
            # No header line to annotate; the canonical target is already encoded.
            pass

        if not host_info.was_present:
            global_flags.add("HOSTMISSING")
        elif host_info.is_empty:
            self._rule_add_inline_flag(inline, "host", "EMPTYHOST")
        elif not host_info.is_valid or not host_info.host:
            self._rule_add_inline_flag(inline, "host", "BADHOST")
        else:
            header_host_norm, header_idna = self._rule_process_idna(host_info.host)
            if header_idna:
                self._rule_add_inline_flag(inline, "host", "IDNA")

            if self._rule_hosts_mismatch(
                header_host_norm,
                host_info.port,
                processed_target,
                target_port,
            ):
                global_flags.add("HOSTMISMATCH")

        absolute_authority = f"{processed_target}:{target_port}"
        return absolute_authority, inline, global_flags

    def _rule_handle_asterisk_form(
        self,
        method: str,
        host_info: HostInfo,
    ) -> tuple[str | None, dict[str, set[str]], set[str]]:
        inline: dict[str, set[str]] = {}
        global_flags: set[str] = set()

        if not host_info.was_present:
            global_flags.add("HOSTMISSING")
            return None, inline, global_flags

        if not host_info.is_empty:
            if not host_info.is_valid:
                self._rule_add_inline_flag(inline, "host", "BADHOST")
            else:
                self._rule_add_inline_flag(inline, "host", "HOSTNOTEMPTY")

        return None, inline, global_flags

    # --------------------------------------------------------------------- #
    # Utility helpers
    # --------------------------------------------------------------------- #

    def _rule_detect_request_form(self, url: str) -> str:
        if url == "*":
            return "asterisk-form"
        if "://" in url:
            return "absolute-form"
        if ":" in url and "/" not in url:
            return "authority-form"
        return "origin-form"

    def _rule_split_authority_target(self, value: str) -> tuple[str | None, int | None]:
        if ":" not in value:
            return None, None
        host, port_str = value.rsplit(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            return None, None
        if not (1 <= port <= 65535):
            return None, None
        return host, port

    def _rule_process_idna(self, host: str | None) -> tuple[str, bool]:
        if not host:
            return "", False

        if host.startswith("[") and host.endswith("]"):
            return host, False

        if "]:" in host:
            closing = host.find("]:")
            ipv6_part = host[: closing + 1]
            port_part = host[closing + 1 :]
            return ipv6_part + port_part, False

        if not any(ord(char) > 127 for char in host):
            return host, False

        try:
            import idna

            encoded = idna.encode(host).decode("ascii")
            return encoded, True
        except Exception:
            return host, True

    def _rule_hosts_mismatch(
        self,
        header_host: str,
        header_port: int | None,
        target_host: str,
        target_port: int | None,
    ) -> bool:
        def normalize(value: str) -> str:
            if value.startswith("[") and value.endswith("]"):
                return value[1:-1].lower()
            return value.lower()

        if normalize(header_host) != normalize(target_host):
            return True

        if header_port is not None and target_port is not None:
            return header_port != target_port

        return False

    def _rule_add_inline_flag(
        self,
        inline_flags: dict[str, set[str]],
        key: str,
        value: str,
    ) -> None:
        inline_flags.setdefault(key, set()).add(value)

    def _rule_merge_inline_flags(
        self,
        destination: dict[str, set[str]],
        source: dict[str, set[str]],
    ) -> None:
        for key, values in source.items():
            destination.setdefault(key, set()).update(values)

    def _rule_apply_inline_flags(
        self,
        lines: list[str],
        inline_flags: dict[str, set[str]],
    ) -> list[str]:
        if not inline_flags:
            return lines

        output: list[str] = []
        for line in lines:
            if line.startswith("[HEADER] "):
                header_content = line[9:].strip()
                if (
                    "host" in inline_flags
                    and header_content.lower().startswith("host:")
                    and inline_flags["host"]
                ):
                    flags = " ".join(sorted(inline_flags["host"]))
                    output.append(f"[HEADER] {header_content} {flags}")
                else:
                    output.append(line)
            else:
                output.append(line)
        return output

    def _rule_emit_global_flags(self, global_flags: set[str]) -> str:
        return " ".join(sorted(global_flags))
