import re
from abc import ABC, abstractmethod
from typing import Tuple, Set

# Known flags that can appear inline on tagged lines.
# Used by split_line_content to separate content from trailing flags.
_KNOWN_FLAGS: Set[str] = {
    # Phase 1
    "UNUSUAL_METHOD",
    # Phase 2
    "OBSFOLD", "BADHDRCONT", "BADCRLF", "BADHDRNAME", "DUPHDR",
    "HDRMERGE", "HOPBYHOP", "HDRNORM", "WSPAD",
    # Phase 3
    "ANGLE", "QUOTE", "SEMICOLON", "PAREN", "BRACE", "PIPE",
    "BACKSLASH", "SPACE", "NUL", "QNUL", "MIXEDSCRIPT", "MIXEDSEP",
    "HOSTMISMATCH", "IDNA", "BADHOST",
    "FULLWIDTH", "CONTROL", "UNICODE_FORMAT", "MATH_UNICODE", "INVALID_UNICODE",
    "DOUBLEPCT", "PCTSLASH", "PCTBACKSLASH", "PCTSPACE", "PCTCONTROL",
    "PCTNULL", "PCTSUSPICIOUS",
    "HTMLENT",
    "QSEMISEP", "QRAWSEMI", "QBARE", "QEMPTYVAL", "QNONASCII", "QLONG",
    "HOME", "MULTIPLESLASH", "DOTCUR", "DOTDOT",
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
    raw = line[len(prefix):]
    parts = raw.split()

    if not parts:
        return "", set()

    # Walk backwards from the end collecting flags
    flags: Set[str] = set()
    split_idx = len(parts)
    for i in range(len(parts) - 1, -1, -1):
        token = parts[i]
        if token in _KNOWN_FLAGS or _PARAMETRIC_FLAG_RE.match(token):
            flags.add(token)
            split_idx = i
        else:
            break  # stop at first non-flag token from the right

    content = " ".join(parts[:split_idx])
    return content, flags


class HttpPreprocessor(ABC):
    """
    Abstract base class for HTTP preprocessors.

    All HTTP preprocessors must inherit from this class and implement the
    `process` method.
    """

    @abstractmethod
    def process(self, request: str) -> str: ...
