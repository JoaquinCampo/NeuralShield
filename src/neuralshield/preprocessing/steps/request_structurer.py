"""Request structuring preprocessing step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor
from neuralshield.preprocessing.steps.exceptions import MalformedHttpRequestError


@dataclass(frozen=True)
class _RequestParts:
    method: str
    target: str
    version: str


class RequestStructurer(HttpPreprocessor):
    """Parse an HTTP request into canonical structured lines."""

    _ALLOWED_METHODS = {
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "PATCH",
        "OPTIONS",
        "HEAD",
        "TRACE",
        "CONNECT",
    }

    def process(self, request: str) -> str:
        if not request:
            raise MalformedHttpRequestError("Empty HTTP request")

        normalized_lines = self._normalize_lines(request)

        if not normalized_lines:
            raise MalformedHttpRequestError("Missing request line")
        index = self._skip_eol_flags(normalized_lines, 0)
        if index >= len(normalized_lines):
            raise MalformedHttpRequestError("Missing request line")

        request_line = normalized_lines[index]
        index += 1
        request_flags, index = self._consume_eol_flags(normalized_lines, index)

        parts = self._parse_request_line(request_line)

        path, query = self._split_target(parts.target)
        query_tokens = self._split_query(query)

        structured_lines: List[str] = [f"[METHOD] {parts.method}"]
        structured_lines.append(f"[URL] {path}")

        for token in query_tokens:
            if token:
                structured_lines.append(f"[QUERY] {token}")

        structured_lines.extend(request_flags)

        headers_with_flags: List[Tuple[str, List[str]]] = []
        while index < len(normalized_lines):
            line = normalized_lines[index]
            if line == "":
                index += 1
                break
            if ":" not in line:
                break
            header_line = line
            index += 1
            header_flags, index = self._consume_eol_flags(normalized_lines, index)
            headers_with_flags.append((header_line, header_flags))

        for header_line, header_flags in headers_with_flags:
            structured_lines.append(f"[HEADER] {header_line}")
            structured_lines.extend(header_flags)

        trailing_lines = normalized_lines[index:]
        if trailing_lines:
            structured_lines.append("")
            structured_lines.extend(trailing_lines)

        result = "\n".join(structured_lines)
        if structured_lines:
            result += "\n"
        return result

    def _skip_eol_flags(self, lines: List[str], start: int) -> int:
        while start < len(lines) and self._is_eol_flag(lines[start]):
            start += 1
        return start

    def _consume_eol_flags(self, lines: List[str], start: int) -> Tuple[List[str], int]:
        flags: List[str] = []
        while start < len(lines) and self._is_eol_flag(lines[start]):
            flags.append(lines[start])
            start += 1
        return flags, start

    @staticmethod
    def _is_eol_flag(line: str) -> bool:
        return line.startswith("EOL_") or line == "EOLMIX"

    def _normalize_lines(self, request: str) -> List[str]:
        request = request.replace("\r\n", "\n")
        request = request.replace("\\r\\n", "\n")
        request = request.replace("\r", "\n")
        lines = request.split("\n")

        while lines and lines[-1] == "":
            lines.pop()

        return lines

    def _parse_request_line(self, line: str) -> _RequestParts:
        parts = line.split()
        if len(parts) != 3:
            raise MalformedHttpRequestError("Malformed request line")

        method, target, version = parts

        if method not in self._ALLOWED_METHODS:
            raise MalformedHttpRequestError(f"Unsupported method: {method}")

        if not version.upper().startswith("HTTP/"):
            raise MalformedHttpRequestError(f"Invalid HTTP version: {version}")

        return _RequestParts(method=method, target=target, version=version)

    def _split_target(self, target: str) -> Tuple[str, str]:
        if "?" not in target:
            return target, ""

        path, query = target.split("?", 1)
        return path, query

    def _split_query(self, query: str) -> List[str]:
        if not query:
            return []

        return list(self._iter_query_tokens(query))

    def _iter_query_tokens(self, query: str) -> Iterable[str]:
        token: List[str] = []
        i = 0
        length = len(query)
        while i < length:
            ch = query[i]
            if ch == "&":
                if self._is_entity(query, i):
                    token.append(ch)
                    i += 1
                    continue
                yield "".join(token)
                token = []
                i += 1
                continue

            token.append(ch)
            i += 1

        yield "".join(token)

    def _is_entity(self, query: str, index: int) -> bool:
        semicolon = query.find(";", index + 1)
        if semicolon == -1:
            return False

        next_amp = query.find("&", index + 1)
        if 0 <= next_amp < semicolon:
            return False

        candidate = query[index:semicolon + 1]
        return "=" not in candidate
