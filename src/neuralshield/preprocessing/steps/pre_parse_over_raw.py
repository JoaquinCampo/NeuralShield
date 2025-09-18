import unicodedata

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
    "PRI",
)
METHOD = str
PATH = str
QUERY = str
HEADER = str
QUERIES = list[QUERY]
HEADERS = list[HEADER]


class RemoveFramingArtifacts(HttpPreprocessor):
    """
    Remove framing artifacts from the HTTP request.
    """

    def _remove_framing_artifacts(self, http_request: str) -> str:
        """
        Remove BOM and control characters from the edges of HTTP request strings.

        This function cleans framing artifacts from the absolute borders
        of the HTTP request string. It removes:
        - BOM (Byte Order Mark) at the beginning
        - Non-printable control characters at the beginning and end

        It preserves all content within the HTTP message structure and
        only modifies the absolute edges to ensure robust parsing downstream.

        Args:
            http_request: Raw HTTP request string that may contain framing artifacts

        Returns:
            HTTP request string with edge artifacts removed
        """

        logger.debug("Removing framing artifacts from HTTP request")

        original_length = len(http_request)
        processed = http_request

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
                "Pre-parse cleanup: removed {total} chars "
                "(BOM: {bom}, leading controls: {lead}, trailing controls: {trail})",
                total=total_removed,
                bom=bom_removed,
                lead=leading_controls,
                trail=trailing_controls,
            )

        return processed

    def process(self, request: str) -> str:
        return self._remove_framing_artifacts(http_request=request)


class RequestStructurer(HttpPreprocessor):
    """
    Transform HTTP request into structured [METHOD]/[URL]/[QUERY]/[HEADER] format.
    """

    def _parse_request_line(
        self,
        request_line: str,
    ) -> tuple[METHOD, PATH, QUERY]:
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

        if method.upper() not in VALID_METHODS:
            raise MalformedHttpRequestError(f"Invalid method: {method}")

        return method, path, query_string

    def _parse_query_parameters(
        self,
        query_string: str,
    ) -> QUERIES:
        """
        Parse query string into individual parameter strings.

        Args:
            query_string: Query string portion of URL (after ?)

        Returns:
            List of formatted query parameters like ["a=1", "b=2", "a=3"]
            (preserving original encoding)
        """
        logger.debug("Parsing query parameters from: {}", query_string)

        if not query_string:
            return []

        query_params = []
        for param in query_string.split("&"):
            query_params.append(param)

        logger.debug("Parsed {} query parameters", len(query_params))
        return query_params

    def _parse_headers(
        self,
        headers_section: str,
    ) -> HEADERS:
        """
        Parse headers preserving original lines.

        Args:
            headers_section: Raw headers section of HTTP request

        Returns:
            List of formatted headers like:
            ["host:example.com","user-agent:Mozilla/5.0"]
        """
        logger.debug("Parsing HTTP headers")

        if headers_section == "":
            return []

        headers = headers_section.split("\n")

        logger.debug("Parsed {} headers", len(headers))
        return headers

    def _format_output(
        self,
        method: METHOD,
        path: PATH,
        query_params: QUERIES,
        headers: HEADERS,
    ) -> str:
        """
        Format the parsed components into the target output format.

        Args:
            method: HTTP method
            path: URL path
            query_params: List of query parameters
            headers: List of headers

        Returns:
            Formatted output string
        """
        logger.debug(
            "Formatting output with {} query params and {} headers",
            len(query_params),
            len(headers),
        )

        lines = []

        # Add method and URL
        lines.append(f"[METHOD] {method}")
        lines.append(f"[URL] {path}")

        # Add query parameters
        for param in query_params:
            lines.append(f"[QUERY] {param}")

        # Add headers
        for header in headers:
            lines.append(f"[HEADER] {header}")

        return "\n".join(lines)

    def process(self, request: str) -> str:
        """
        Transform HTTP request into structured format.

        Args:
            request: Raw HTTP request string

        Returns:
            Formatted request string

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
        result = self._format_output(method, path, query_params, headers)

        logger.debug(f"Successfully structured HTTP request: {method} {path} \n\n")
        return result
