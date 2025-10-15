#!/usr/bin/env python3
"""
Paper's preprocessing pipeline (Section 3.2).

Simpler than NeuralShield's 13 steps - just the 5 steps from the paper.
"""

from urllib.parse import unquote


def paper_preprocess(request: str) -> str:
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

    Returns:
        Preprocessed request string
    """
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
