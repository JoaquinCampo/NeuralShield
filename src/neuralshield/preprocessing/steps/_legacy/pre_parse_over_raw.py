import re
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

        HTML-entity aware: protects & characters that are part of HTML entities
        like &#x3c; from being treated as parameter separators.

        Args:
            query_string: Query string portion of URL (after ?)

        Returns:
            List of formatted query parameters like ["a=1", "b=2", "a=3"]
            (preserving original encoding and HTML entities)
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
            f"Formatting output with {len(query_params)}"
            f" query params and {len(headers)} headers",
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


# NOTE: This step is not enabled in the config for now.
class LineJumpCatcher(HttpPreprocessor):
    """
    Detect and flag End-Of-Line anomalies in HTTP requests.

    This processor identifies non-standard line endings and adds flag markers
    to help detect potential security issues or malformed requests.

    EOL Token Types:
    - CRLF: Standard "\r\n" sequence
    - Bare LF: "\n" not preceded by "\r"
    - Bare CR: "\r" not followed by "\n"

    Flags Added:
    - EOL_BARELF: Line uses only bare LF endings
    - EOL_BARECR: Line uses only bare CR endings
    - EOLMIX: Line mixes different EOL types
    - EOL_EOF_NOCRLF: File ends without proper line ending
    """

    # EOL token constants
    CRLF_TOKEN = "CRLF"
    LF_TOKEN = "LF"
    CR_TOKEN = "CR"

    # Flag constants
    BARE_LF_FLAG = "EOL_BARELF"
    BARE_CR_FLAG = "EOL_BARECR"
    MIXED_EOL_FLAG = "EOLMIX"
    EOF_NO_CRLF_FLAG = "EOL_EOF_NOCRLF"

    KNOWN_FLAGS = {BARE_LF_FLAG, BARE_CR_FLAG, MIXED_EOL_FLAG, EOF_NO_CRLF_FLAG}

    def _extract_line_content(self, text: str, start_pos: int) -> tuple[str, int]:
        """Extract line content up to the first EOL character."""
        text_length = len(text)
        current_pos = start_pos

        while current_pos < text_length and text[current_pos] not in "\r\n":
            current_pos += 1

        line_content = text[start_pos:current_pos]
        return line_content, current_pos

    def _scan_eol_sequence(self, text: str, start_pos: int) -> tuple[set[str], int]:
        """Scan and categorize EOL tokens in sequence."""
        text_length = len(text)
        current_pos = start_pos
        eol_tokens = set()

        while current_pos < text_length and text[current_pos] in "\r\n":
            char = text[current_pos]

            if char == "\r":
                # Check for CRLF sequence
                if current_pos + 1 < text_length and text[current_pos + 1] == "\n":
                    eol_tokens.add(self.CRLF_TOKEN)
                    current_pos += 2
                else:
                    eol_tokens.add(self.CR_TOKEN)
                    current_pos += 1
            else:  # char == '\n'
                eol_tokens.add(self.LF_TOKEN)
                current_pos += 1

        return eol_tokens, current_pos

    def _get_flag_for_eol_pattern(
        self, eol_tokens: set[str], is_eof: bool
    ) -> str | None:
        """Determine the appropriate flag based on EOL pattern."""
        if is_eof:
            return self.EOF_NO_CRLF_FLAG

        if eol_tokens == {self.LF_TOKEN}:
            return self.BARE_LF_FLAG

        if eol_tokens == {self.CR_TOKEN}:
            return self.BARE_CR_FLAG

        if len(eol_tokens) >= 2:
            return self.MIXED_EOL_FLAG

        # Standard CRLF or empty set
        return None

    def _extract_next_line_content(self, text: str, start_pos: int) -> str:
        """Extract the content of the next line without EOL characters."""
        text_length = len(text)
        end_pos = start_pos

        while end_pos < text_length and text[end_pos] not in "\r\n":
            end_pos += 1

        return text[start_pos:end_pos]

    def _should_skip_flag(self, flag: str, text: str, next_line_pos: int) -> bool:
        """Check if flag should be skipped to maintain idempotency."""
        if next_line_pos >= len(text):
            return True

        next_line_content = self._extract_next_line_content(text, next_line_pos)
        return next_line_content in self.KNOWN_FLAGS

    def _append_flag_line(
        self, output_parts: list[str], flag: str, needs_newline: bool
    ) -> None:
        """Append a flag line to the output with proper formatting."""
        if needs_newline:
            output_parts.append("\n")
        output_parts.append(flag)
        output_parts.append("\n")

    def _log_flag_addition(
        self, flag: str, eol_tokens: set[str], start_pos: int, end_pos: int
    ) -> None:
        """Log the addition of a flag for debugging."""
        token_names = ",".join(sorted(eol_tokens)) if eol_tokens else "<none>"
        logger.debug(
            f"LineJumpCatcher: adding flag '{flag}' for tokens [{token_names}] "
            f"at position [{start_pos}:{end_pos}]"
        )

    def _process_line(self, text: str, position: int) -> tuple[list[str], int]:
        """Process a single line and return output parts and next position."""
        text_length = len(text)
        output_parts = []

        # Extract line content
        line_content, eol_start_pos = self._extract_line_content(text, position)
        is_existing_flag = line_content in self.KNOWN_FLAGS

        # Scan EOL sequence
        eol_tokens, next_position = self._scan_eol_sequence(text, eol_start_pos)

        # Preserve original content and EOL sequence
        output_parts.append(line_content)
        if next_position > eol_start_pos:
            output_parts.append(text[eol_start_pos:next_position])

        # Determine flag if needed
        if not is_existing_flag:
            is_at_eof = eol_start_pos == text_length
            flag = self._get_flag_for_eol_pattern(eol_tokens, is_at_eof)

            if flag is not None and not self._should_skip_flag(
                flag, text, next_position
            ):
                self._log_flag_addition(flag, eol_tokens, eol_start_pos, next_position)
                needs_newline = is_at_eof
                self._append_flag_line(output_parts, flag, needs_newline)

        # Ensure we advance position
        final_position = (
            max(next_position, position + 1)
            if next_position == position
            else next_position
        )
        return output_parts, final_position

    def process(self, request: str) -> str:
        """Process HTTP request and add EOL anomaly flags."""
        if not request:
            raise MalformedHttpRequestError("Empty HTTP request")

        output_parts = []
        position = 0
        text_length = len(request)

        while position < text_length:
            line_parts, position = self._process_line(request, position)
            output_parts.extend(line_parts)

        return "".join(output_parts)
