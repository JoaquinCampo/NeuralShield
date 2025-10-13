from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor
from neuralshield.preprocessing.steps.exceptions import MalformedHttpRequestError


class RequestStructurer(HttpPreprocessor):
    """
    Structure the HTTP request into a canonical form.
    """

    VALID_METHODS = {
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
        """
        Structure the HTTP request into a canonical form.
        """
        if not request or not request.strip():
            raise MalformedHttpRequestError("Empty request")

        lines = request.split("\n")

        # Parse request line
        request_line = lines[0] if lines else ""
        method, url, http_version, flags = self._parse_request_line(request_line)

        # Split URL and query
        path, query_string = self._split_url_query(url)

        # Parse headers (everything after request line until empty line)
        headers = self._parse_headers(lines[1:])

        # Build canonical output
        output_lines = []

        # Add method
        output_lines.append(f"[METHOD] {method}")

        # Add URL (path)
        output_lines.append(f"[URL] {path}")

        # Add query parameters
        if query_string:
            query_params = self._split_query_with_entity_protection(query_string)
            for param in query_params:
                output_lines.append(f"[QUERY] {param}")

        # Add headers
        for header in headers:
            output_lines.append(f"[HEADER] {header}")

        # Add flags if any
        if flags:
            output_lines.append(f"[FLAGS] {' '.join(flags)}")

        return "\n".join(output_lines)

    def _parse_request_line(self, request_line: str) -> tuple[str, str, str, list[str]]:
        """Parse the HTTP request line into method, URL, HTTP version, and flags."""
        parts = request_line.split(" ")
        flags = []

        if len(parts) != 3:
            raise MalformedHttpRequestError(
                f"Invalid request line format: expected 3 parts, got {len(parts)}"
            )

        method, url, http_version = parts

        # Flag unusual methods instead of rejecting them
        if method not in self.VALID_METHODS:
            flags.append("UNUSUAL_METHOD")

        # Validate HTTP version
        if not http_version.startswith("HTTP/"):
            raise MalformedHttpRequestError(
                f"Invalid HTTP version format: {http_version}"
            )

        return method, url, http_version, flags

    def _split_url_query(self, url: str) -> tuple[str, str]:
        """Split URL on first '?' into path and query string."""
        if "?" in url:
            path, query_string = url.split("?", 1)
        else:
            path, query_string = url, ""

        return path, query_string

    def _split_query_with_entity_protection(self, query_string: str) -> list[str]:
        """Split query string on '&' while protecting HTML entities."""
        if not query_string:
            return []

        # Use regex to split on '&' that are not part of HTML entities
        # HTML entities like &#x26; or &#38; should not be split
        parts = []
        current_part = ""
        i = 0

        while i < len(query_string):
            if query_string[i] == "&":
                # Check if this '&' is part of an HTML entity
                if self._is_part_of_html_entity(query_string, i):
                    current_part += query_string[i]
                else:
                    # This is a parameter separator
                    if current_part:
                        parts.append(current_part)
                    current_part = ""
                i += 1
            else:
                current_part += query_string[i]
                i += 1

        # Add the last part
        if current_part:
            parts.append(current_part)

        return parts

    def _is_part_of_html_entity(self, text: str, ampersand_pos: int) -> bool:
        """Check if the '&' at the given position is part of an HTML entity."""
        # Look ahead to see if this looks like an HTML entity
        # Patterns: &#digits; &#xhex; &name;
        remaining = text[ampersand_pos:]

        # Check for numeric entities: &#123; or &#x1A;
        if remaining.startswith("&#"):
            end_pos = remaining.find(";", 2)
            if end_pos != -1:
                entity_content = remaining[2:end_pos]
                # Check if it's a valid numeric or hex entity
                if entity_content.startswith("x") and len(entity_content) > 1:
                    # Hex entity: check if the rest are hex digits
                    return all(
                        c in "0123456789abcdefABCDEF" for c in entity_content[1:]
                    )
                else:
                    # Decimal entity: check if all are digits
                    return entity_content.isdigit()

        # Check for named entities: &amp; &lt; etc.
        semicolon_pos = remaining.find(";")
        if semicolon_pos != -1 and semicolon_pos > 1:
            entity_name = remaining[1:semicolon_pos]
            # Simple check for valid entity name (letters only)
            return entity_name.isalpha()

        return False

    def _parse_headers(self, header_lines: list[str]) -> list[str]:
        """Parse headers until the first empty line."""
        headers = []

        for line in header_lines:
            if line == "":
                # Empty line marks end of headers
                break
            headers.append(line)

        return headers
