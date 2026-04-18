from urllib.parse import urlsplit, urlunsplit

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor, split_line_content


class UrlAbsFinalizer(HttpPreprocessor):
    """Ensure [URL_ABS] stays consistent with the final normalized [URL].

    The pipeline computes [URL_ABS] early (Step 06) to validate request-target
    form and host consistency. Later steps may normalize the path ([URL]).

    This step updates the path component of [URL_ABS] to match the final [URL]
    when [URL] is an origin-form path (starts with '/'). For non-origin-form
    targets (absolute-form, authority-form), this step is intentionally
    conservative and leaves [URL_ABS] unchanged.
    """

    def process(self, request: str) -> str:
        lines = request.split("\n")

        url_path: str | None = None
        url_abs_idx: int | None = None
        url_abs_value: str | None = None

        for idx, line in enumerate(lines):
            if line.startswith("[URL] ") and url_path is None:
                content, _flags = split_line_content(line, "[URL] ")
                url_path = content
                continue

            if line.startswith("[URL_ABS] ") and url_abs_idx is None:
                url_abs_idx = idx
                url_abs_value = line[10:].strip()

        if not url_path or not url_path.startswith("/"):
            return request
        if url_abs_idx is None or not url_abs_value:
            return request

        try:
            parts = urlsplit(url_abs_value)
        except Exception:
            return request

        # Only update well-formed absolute URLs.
        if not parts.scheme or not parts.netloc:
            return request

        if parts.path == url_path:
            return request

        updated = urlunsplit((parts.scheme, parts.netloc, url_path, parts.query, parts.fragment))
        lines[url_abs_idx] = f"[URL_ABS] {updated}"
        return "\n".join(lines)
