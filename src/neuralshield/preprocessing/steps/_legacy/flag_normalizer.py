import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from loguru import logger

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor

# Types
ComponentType = Literal["URL", "QUERY"]


@dataclass
class LineContext:
    """Context for processing a single line."""

    component: ComponentType
    nfkc_changed: bool = False


@dataclass
class TransformResult:
    """Result from a transformation step."""

    content: str
    flags: set[str]
    changed: bool


# Constants
FULLWIDTH_FLAG = "FULLWIDTH"
CONTROL_FLAG = "CONTROL"
HTMLENT_FLAG = "HTMLENT"
DOUBLEPCT_FLAG = "DOUBLEPCT"

# Fullwidth character range (U+FF00–U+FFEF)
FULLWIDTH_RANGE = range(0xFF00, 0xFFF0)

# Compiled regex patterns
HTML_ENTITY_PATTERN = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]*|#(?:\d+|x[0-9a-fA-F]+));")
PERCENT_PATTERN = re.compile(r"%[0-9A-Fa-f]{2}")


# Utility functions
def is_fullwidth_char(char: str) -> bool:
    """Check if character is in fullwidth range."""
    return ord(char) in FULLWIDTH_RANGE


def has_valid_hex_pairs(text: str) -> bool:
    """Check if text contains valid percent-encoded hex pairs."""
    return bool(PERCENT_PATTERN.search(text))


def has_decodable_hex_pairs(text: str) -> bool:
    """Check if text contains percent-encoded hex pairs that we would actually decode."""
    # Find all hex pairs but exclude %00 which we never decode
    pairs = PERCENT_PATTERN.findall(text)
    return any(pair.upper() != "%00" for pair in pairs)


def decode_hex_pair_safely(match: re.Match[str], preserve_null: bool = True) -> str:
    """Decode a single hex pair, optionally preserving %00."""
    if preserve_null and match.group(0).upper() == "%00":
        return match.group(0)
    try:
        return chr(int(match.group(0)[1:], 16))
    except ValueError:
        return match.group(0)  # Keep invalid sequences


# Pure transformer functions
def unicode_normalizer(text: str, ctx: LineContext) -> TransformResult:
    """
    Apply NFKC normalization and detect fullwidth characters.

    Returns normalized text and FULLWIDTH flag if changes detected.
    """
    # Check for fullwidth characters before normalization
    has_fullwidth = any(is_fullwidth_char(char) for char in text)

    # Apply NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)

    flags: set[str] = set()
    changed = normalized != text or has_fullwidth

    if changed:
        flags.add(FULLWIDTH_FLAG)
        ctx.nfkc_changed = True
        logger.debug("FULLWIDTH characters detected and normalized")

    return TransformResult(content=normalized, flags=flags, changed=changed)


def control_char_detector(text: str, _ctx: LineContext) -> TransformResult:
    """
    Detect control characters and %00 sequences without decoding.

    Preserves %00 sequences while flagging them as control characters.
    """
    flags: set[str] = set()

    # Check for %00 sequences (null bytes) without decoding them
    if "%00" in text.upper():
        flags.add(CONTROL_FLAG)
        logger.debug("CONTROL character detected: %00 sequence")

    # Check for actual control characters in the text
    for char in text:
        if unicodedata.category(char) == "Cc":
            flags.add(CONTROL_FLAG)
            logger.debug(f"CONTROL character detected: {repr(char)}")
            break

    return TransformResult(content=text, flags=flags, changed=False)


def html_entity_decoder(text: str, _ctx: LineContext) -> TransformResult:
    """
    Detect and decode HTML entities once.

    Returns decoded text and HTMLENT flag if entities found.
    """
    flags: set[str] = set()

    if HTML_ENTITY_PATTERN.search(text):
        decoded = html.unescape(text)
        if decoded != text:
            flags.add(HTMLENT_FLAG)
            logger.debug("HTML entities detected and decoded")
            return TransformResult(content=decoded, flags=flags, changed=True)

    return TransformResult(content=text, flags=flags, changed=False)


def percent_encoding_analyzer(text: str, ctx: LineContext) -> TransformResult:
    """
    Analyze and normalize percent encoding.

    - Detects multi-level encoding and decodes to single level
    - Preserves single-level encoding by default
    - Decodes single-level alphanumeric codes only if NFKC created them
    - Never decodes %00
    """
    flags: set[str] = set()

    def decode_once(s: str) -> str:
        """Decode percent sequences once, preserving %00."""
        return PERCENT_PATTERN.sub(
            lambda m: decode_hex_pair_safely(m, preserve_null=True), s
        )

    # First pass: try one decode
    decoded_once = decode_once(text)

    # If nothing changed, no encoding was present
    if decoded_once == text:
        return TransformResult(content=text, flags=flags, changed=False)

    # Check for multi-level encoding (excluding %00 which we never decode)
    if has_decodable_hex_pairs(decoded_once):
        flags.add(DOUBLEPCT_FLAG)
        logger.debug("Double percent encoding detected")

        # For multi-level encoding, try one more decode to handle triple+
        # encoding but stop if no more decodable hex pairs would remain
        # (to preserve single-level)
        decoded_twice = decode_once(decoded_once)

        # If a second decode still leaves decodable hex pairs, use it
        # (handles %252520 → %20). Otherwise stick with one decode
        # (handles %252E → %2E).
        if decoded_twice != decoded_once and has_decodable_hex_pairs(decoded_twice):
            return TransformResult(content=decoded_twice, flags=flags, changed=True)
        else:
            return TransformResult(content=decoded_once, flags=flags, changed=True)

    # Single encoding detected
    # If NFKC normalization changed this line, decode alphanumeric codes only
    if ctx.nfkc_changed:

        def decode_alnum_only(match: re.Match[str]) -> str:
            if match.group(0).upper() == "%00":
                return match.group(0)  # Never decode %00
            try:
                ch = chr(int(match.group(0)[1:], 16))
                return ch if ch.isalnum() else match.group(0)
            except ValueError:
                return match.group(0)

        result = PERCENT_PATTERN.sub(decode_alnum_only, text)
        changed = result != text
        if changed:
            logger.debug("Single-level alphanumeric percent codes decoded after NFKC")
        return TransformResult(content=result, flags=flags, changed=changed)

    # Preserve original single encoding
    return TransformResult(content=text, flags=flags, changed=False)


class FlagsNormalizer(HttpPreprocessor):
    """
    Core flags normalizer implementing normalizar-flags.md specification.

    Detects and flags anomalies while producing canonical representation:
    - FULLWIDTH: Fullwidth characters normalized by NFKC
    - CONTROL: Control characters (Unicode category Cc)
    - HTMLENT: HTML entities like &#x2f;, &lt;
    - DOUBLEPCT: Double percent encoding like %252E

    Works on structured pipeline format with [METHOD], [URL], [QUERY], [HEADER] prefixes.
    Only processes [URL] and [QUERY] lines where encoding attacks typically occur.
    Emits flags immediately after processed lines where anomalies are detected.

    Example:
    Input:
        [METHOD] GET
        [URL] /a&#x2f;b%252Ec%00d
        [QUERY] param=％76alue
        [HEADER] Host: example.com

    Output:
        [METHOD] GET
        [URL] /a/b%2Ec%00d
        CONTROL DOUBLEPCT HTMLENT
        [QUERY] param=value
        FULLWIDTH
        [HEADER] Host: example.com
    """

    def __init__(self) -> None:
        # No mutable state - all processing is stateless
        pass

    def _process_line_content(
        self, content: str, component: ComponentType
    ) -> tuple[str, list[str]]:
        """
        Apply all transformations to line content.

        Returns processed content and list of flags.
        """
        ctx = LineContext(component=component)
        all_flags: set[str] = set()

        # Apply transformers in spec order
        result = unicode_normalizer(content, ctx)
        current_content = result.content
        all_flags.update(result.flags)

        result = control_char_detector(current_content, ctx)
        current_content = result.content
        all_flags.update(result.flags)

        result = html_entity_decoder(current_content, ctx)
        current_content = result.content
        all_flags.update(result.flags)

        result = percent_encoding_analyzer(current_content, ctx)
        current_content = result.content
        all_flags.update(result.flags)

        # Return content and sorted flags
        sorted_flags = sorted(all_flags)
        return current_content, sorted_flags

    def process(self, request: str) -> str:
        """
        Process structured HTTP request applying core normalization and flag detection.

        Expects input in pipeline format with
        [METHOD], [URL], [QUERY], [HEADER] prefixes.
        Only processes [URL] and [QUERY] lines where encoding attacks typically occur.
        [METHOD] and [HEADER] lines are passed through unchanged.

        Implements the orden de operaciones from normalizar-flags.md:
        1. Unicode NFKC normalization (FULLWIDTH detection)
        2. Control character detection
        3. HTML entity decoding (HTMLENT detection)
        4. Percent decoding (DOUBLEPCT detection)
        5. Immediate flag emission per line
        """
        lines = request.strip().split("\n")
        processed_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip existing flag lines to avoid duplication
            if not line.startswith("[") and any(
                flag == line.strip()
                or f" {flag} " in f" {line.strip()} "
                or line.strip().startswith(f"{flag} ")
                or line.strip().endswith(f" {flag}")
                for flag in [
                    FULLWIDTH_FLAG,
                    CONTROL_FLAG,
                    HTMLENT_FLAG,
                    DOUBLEPCT_FLAG,
                ]
            ):
                continue

            # Process URL and QUERY lines only
            if line.startswith("[URL] "):
                content = line[6:]  # Remove '[URL] ' prefix
                processed_content, flags = self._process_line_content(content, "URL")
                processed_lines.append(f"[URL] {processed_content}")
                if flags:
                    processed_lines.append(" ".join(flags))

            elif line.startswith("[QUERY] "):
                content = line[8:]  # Remove '[QUERY] ' prefix
                processed_content, flags = self._process_line_content(content, "QUERY")
                processed_lines.append(f"[QUERY] {processed_content}")
                if flags:
                    processed_lines.append(" ".join(flags))

            else:
                # Pass through METHOD, HEADER, and other lines unchanged
                processed_lines.append(line)

        return "\n".join(processed_lines)
