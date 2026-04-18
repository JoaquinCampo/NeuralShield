import re
from abc import ABC, abstractmethod
from typing import Tuple, Set

# Known flags that can appear inline on tagged lines.
# Used by split_line_content to separate content from trailing flags.
_KNOWN_FLAGS: Set[str] = {
    # Phase 1
    "UNUSUAL_METHOD",
    # Phase 2
    "OBSFOLD",
    "BADHDRCONT",
    "BADCRLF",
    "BADHDRNAME",
    "DUPHDR",
    "HDRMERGE",
    "HOPBYHOP",
    "HDRNORM",
    "WSPAD",
    # Phase 3
    "ANGLE",
    "QUOTE",
    "SEMICOLON",
    "PAREN",
    "BRACE",
    "PIPE",
    "BACKSLASH",
    "SPACE",
    "NUL",
    "QNUL",
    "MIXEDSCRIPT",
    "MIXEDSEP",
    "HOSTMISMATCH",
    "IDNA",
    "BADHOST",
    "FULLWIDTH",
    "CONTROL",
    "UNICODE_FORMAT",
    "MATH_UNICODE",
    "INVALID_UNICODE",
    "DOUBLEPCT",
    "PCTSLASH",
    "PCTBACKSLASH",
    "PCTSPACE",
    "PCTCONTROL",
    "PCTNULL",
    "PCTSUSPICIOUS",
    "HTMLENT",
    "QSEMISEP",
    "QRAWSEMI",
    "QBARE",
    "QEMPTYVAL",
    "QNONASCII",
    "QLONG",
    "HOME",
    "MULTIPLESLASH",
    "DOTCUR",
    "DOTDOT",
}

# Regex to match parametric flags like QARRAY:<key> and QREPEAT:<key>
_PARAMETRIC_FLAG_RE = re.compile(r"^Q(?:ARRAY|REPEAT):\S+$")


def split_line_content(line: str, prefix: str) -> Tuple[str, Set[str]]:
    """Split a tagged line into (content, flags).

    Given a line like ``[URL] /path ANGLE QUOTE`` and prefix ``[URL] ``,
    returns ``("/path", {"ANGLE", "QUOTE"})``.

    Flags are identified by matching against the known flag set.  Content
    tokens that happen to look like flags but aren't known are kept as
    content.
    """
    raw = line[len(prefix) :]
    if not raw:
        return "", set()

    # Walk backwards from the end collecting flags without modifying the content.
    # This must not collapse whitespace (tabs, multiple spaces), otherwise evidence
    # such as WSPAD cannot be detected reliably downstream.
    flags: Set[str] = set()
    work = raw

    while True:
        # Remove trailing whitespace only for token detection.
        stripped = work.rstrip()
        if not stripped:
            break

        # Find the last whitespace-separated token.
        m = re.search(r"\s+(\S+)$", stripped)
        if not m:
            break

        token = m.group(1)
        if token in _KNOWN_FLAGS or _PARAMETRIC_FLAG_RE.match(token):
            flags.add(token)
            # Remove the token (and the whitespace before it) from the working string.
            work = stripped[: m.start()]
            continue
        break

    # If flags were found, trim the remaining content's trailing whitespace.
    # Once flags are appended, trailing whitespace becomes ambiguous.
    content = work.rstrip() if flags else work
    return content, flags


class HttpPreprocessor(ABC):
    """
    Abstract base class for HTTP preprocessors.

    All HTTP preprocessors must inherit from this class and implement the
    `process` method.
    """

    @abstractmethod
    def process(self, request: str) -> str: ...
