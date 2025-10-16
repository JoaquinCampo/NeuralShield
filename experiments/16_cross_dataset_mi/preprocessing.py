#!/usr/bin/env python3
"""
Paper's preprocessing pipeline (Section 3.2).

Simpler than NeuralShield's 13 steps - just the 5 steps from the paper.
"""

from urllib.parse import unquote


def strip_headers(request: str) -> str:
    """
    Extract only request line and body, removing HTTP headers.

    This helps focus on attack payloads rather than header artifacts
    when datasets have synthetic/uniform headers.

    Args:
        request: Full HTTP request with headers

    Returns:
        Request line + body (if present)
    """
    lines = request.split("\n")

    if not lines:
        return request

    # First line is always the request line (METHOD URL HTTP/x.x)
    request_line = lines[0]

    # Find body (after blank line, if POST/PUT)
    body = ""
    found_blank = False
    for line in lines[1:]:
        if not line.strip():
            found_blank = True
            continue
        if found_blank:
            # After blank line = body
            body += " " + line.strip()

    result = request_line
    if body:
        result += " " + body.strip()

    return result


def paper_preprocess(request: str, strip_http_headers: bool = False) -> str:
    """
    Apply paper's 5-step preprocessing.

    Steps from paper (Section 3.2):
    1. Header filters (simplified: we already have filtered requests)
    2. URL decode
    3. UTF-8 decode (implicit in Python strings)
    4. URL decode (again - handles double encoding)
    5. Lowercase

    Args:
        request: Raw HTTP request string
        strip_http_headers: If True, remove HTTP headers before preprocessing

    Returns:
        Preprocessed request string
    """
    # Step 0 (optional): Strip headers to focus on payloads
    if strip_http_headers:
        request = strip_headers(request)

    # Step 1: Header filters - already done in CSIC/SRBH datasets
    # (they only contain relevant parts)

    # Step 2: URL decode
    decoded = unquote(request)

    # Step 3: UTF-8 decode - Python strings are already Unicode
    # (no action needed)

    # Step 4: URL decode again (double encoding attack protection)
    decoded = unquote(decoded)

    # Step 5: Lowercase
    processed = decoded.lower()

    return processed
