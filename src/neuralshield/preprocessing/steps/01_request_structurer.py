"""
01 Request Structurer - Parse raw HTTP into structured format.
"""

import re

from loguru import logger

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor
from neuralshield.preprocessing.steps.exceptions import MalformedHttpRequestError

VALID_METHODS = (
    "GET",
    "POST", 
    "PUT",
    "DELETE",
    "PATCH",
    "OPTIONS",
    "HEAD",
    "TRACE",
    "CONNECT",
)


class RequestStructurer(HttpPreprocessor):
    """
    Transform HTTP request into structured [METHOD]/[URL]/[QUERY]/[HEADER] format.
    
    Parses the request line, splits URL from query, tokenizes query parameters,
    and formats headers. Uses HTML-entity aware splitting to avoid breaking
    entities during query parameter parsing.
    """

    def _parse_request_line(self, request_line: str) -> tuple[str, str, str]:
        """
        Parse the HTTP request line to extract method, path, and query string.

        Args:
            request_line: First line of HTTP request (e.g., "GET /path?query HTTP/1.1")

        Returns:
            Tuple of (method, path, query)

        Raises:
            MalformedHttpRequestError: If request line is malformed
        """
        logger.debug("Parsing HTTP request line")

        parts = request_line.strip().split()
        if len(parts) != 3:
            raise MalformedHttpRequestError(
                f"Invalid request line format: {request_line}"
            )

        method, url, http_version = parts

        # Validate HTTP version format
        if not http_version.startswith("HTTP/"):
            raise MalformedHttpRequestError(f"Invalid HTTP version: {http_version}")

        # Validate method
        if method.upper() not in VALID_METHODS:
            raise MalformedHttpRequestError(f"Invalid method: {method}")

        # Extract path and query string from URL
        if "?" in url:
            path, query_string = url.split("?", 1)
        else:
            path, query_string = url, ""

        logger.debug(
            "Parsed request line: method={}, path={}, query={}",
            method,
            path,
            query_string,
        )

        return method, path, query_string

    def _parse_query_parameters(self, query_string: str) -> list[str]:
        """
        Parse query string into individual parameter strings with HTML entity protection.

        Args:
            query_string: Query string portion of URL (after ?)

        Returns:
            List of query parameter strings
        """
        logger.debug("Parsing query parameters from: {}", query_string)

        if not query_string:
            return []

        # HTML entity pattern to protect & characters within entities
        html_entity_pattern = re.compile(
            r"&(?:[a-zA-Z][a-zA-Z0-9]*|#(?:\d+|x[0-9a-fA-F]+));"
        )

        # Find all HTML entities and replace with temporary placeholders
        entities: list[str] = []

        def replace_entity(match):
            placeholder = f"__HTMLENT_{len(entities)}__"
            entities.append(match.group(0))
            return placeholder

        protected_query = html_entity_pattern.sub(replace_entity, query_string)

        # Now split on & safely (HTML entity & characters are protected)
        query_params = []
        for param in protected_query.split("&"):
            # Restore HTML entities in this parameter
            restored_param = param
            for i, entity in enumerate(entities):
                restored_param = restored_param.replace(f"__HTMLENT_{i}__", entity)
            query_params.append(restored_param)

        logger.debug("Parsed {} query parameters", len(query_params))
        return query_params

    def _parse_headers(self, headers_section: str) -> list[str]:
        """
        Parse headers preserving original lines.

        Args:
            headers_section: Raw headers section of HTTP request

        Returns:
            List of header lines
        """
        logger.debug("Parsing HTTP headers")

        if headers_section == "":
            return []

        headers = headers_section.split("\n")

        logger.debug("Parsed {} headers", len(headers))
        return headers

    def process(self, request: str) -> str:
        """
        Transform HTTP request into structured format.

        Args:
            request: Cleaned HTTP request string

        Returns:
            Structured request string with [METHOD], [URL], [QUERY], [HEADER] prefixes

        Raises:
            MalformedHttpRequestError: If request cannot be parsed
        """
        logger.debug("Processing HTTP request for structuring")

        if not request.strip():
            raise MalformedHttpRequestError("Empty HTTP request")

        # Split request into lines
        lines = request.split("\n")

        if not lines:
            raise MalformedHttpRequestError("No lines in HTTP request")

        # Parse request line
        request_line = lines[0]
        method, path, query_string = self._parse_request_line(request_line)

        # Find the end of headers (first empty line)
        headers_end = 1
        for i in range(1, len(lines)):
            if lines[i] == "" or lines[i] == "\r":
                headers_end = i
                break
        else:
            # No empty line found, headers go to end of request
            headers_end = len(lines)

        # Extract headers section
        headers_section = "\n".join(lines[1:headers_end])

        # Parse components
        query_params = self._parse_query_parameters(query_string)
        headers = self._parse_headers(headers_section)

        # Format output
        result_lines = []
        result_lines.append(f"[METHOD] {method}")
        result_lines.append(f"[URL] {path}")

        # Add query parameters
        for param in query_params:
            result_lines.append(f"[QUERY] {param}")

        # Add headers
        for header in headers:
            result_lines.append(f"[HEADER] {header}")

        result = "\n".join(result_lines)
        logger.debug("Successfully structured HTTP request: {} {}", method, path)
        return result
