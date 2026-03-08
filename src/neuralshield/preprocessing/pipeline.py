import tomllib
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from typing import Iterable, Sequence

from neuralshield.preprocessing.http_preprocessor import (
    HttpPreprocessor,
    _KNOWN_FLAGS,
    _PARAMETRIC_FLAG_RE,
)


# TODO(audit): Verify Figure 4.2 in PDF — Phase 2 flag list is incomplete
#   (missing BADHDRCONT, BADCRLF, HDRMERGE, HOPBYHOP, HDRNORM). See audit 1.10.
# TODO(audit): Regenerate Listing 4.2 end-to-end example from actual pipeline
#   output after all bug fixes are applied. See audit 1.12.


class PreprocessorPipeline:
    """Callable pipeline with optional batched execution support."""

    def __init__(
        self,
        steps: Sequence[HttpPreprocessor],
        *,
        max_workers: int | None = None,
    ) -> None:
        self._steps = tuple(steps)
        self._max_workers = max_workers

    def __call__(self, request: str) -> str:
        """Process a single request through every configured step."""

        for step in self._steps:
            request = step.process(request)
        return self._finalize_artifact(request)

    @staticmethod
    def _finalize_artifact(request: str) -> str:
        """Finalize the canonical artifact.

        - Enforce a stable line order.
        - Ensure exactly one aggregated [FLAGS] line (when any flags exist).
        - Do not modify line contents, only structure/order/deduplication.
        """
        lines = [ln for ln in request.split("\n") if ln != ""]
        all_flags: set[str] = set()

        def add_token(token: str) -> None:
            if not token:
                return
            if token in _KNOWN_FLAGS or _PARAMETRIC_FLAG_RE.match(token):
                all_flags.add(token)

        def add_token_maybe_csv(token: str) -> None:
            # Some steps may emit comma-separated tokens (legacy).
            if "," in token:
                for sub in token.split(","):
                    add_token(sub.strip())
                return
            add_token(token.strip())

        # Keep categorized lines to re-emit in a stable order.
        method_line: str | None = None
        url_line: str | None = None
        url_abs_line: str | None = None
        query_lines: list[str] = []
        header_lines: list[str] = []
        hagg_line: str | None = None
        hgf_line: str | None = None
        qsep_line: str | None = None
        qmeta_line: str | None = None
        other_lines: list[str] = []

        for line in lines:
            # Collect flags already on any [FLAGS] line (we will remove all and re-add one).
            if line.startswith("[FLAGS] "):
                for token in line[8:].split():
                    add_token_maybe_csv(token)
                continue

            if line.startswith("[METHOD] "):
                method_line = line
                continue
            if line.startswith("[URL] "):
                url_line = line
                for token in line[6:].split():
                    add_token_maybe_csv(token)
                continue
            if line.startswith("[URL_ABS] "):
                url_abs_line = line
                continue
            if line.startswith("[QUERY] "):
                query_lines.append(line)
                for token in line[8:].split():
                    add_token_maybe_csv(token)
                continue
            if line.startswith("[HEADER] "):
                header_lines.append(line)
                for token in line[9:].split():
                    add_token_maybe_csv(token)
                continue
            if line.startswith("[HAGG] "):
                hagg_line = line
                continue
            if line.startswith("[HGF] "):
                hgf_line = line
                for token in line[6:].split():
                    add_token_maybe_csv(token)
                continue
            if line.startswith("[QSEP] "):
                qsep_line = line
                for token in line[6:].split():
                    add_token_maybe_csv(token)
                continue
            if line.startswith("[QMETA] "):
                qmeta_line = line
                parts = line[7:].split()
                for token in parts[1:]:
                    add_token_maybe_csv(token)
                continue

            other_lines.append(line)

        out_lines: list[str] = []
        for ln in (method_line, url_line, url_abs_line):
            if ln is not None:
                out_lines.append(ln)
        out_lines.extend(query_lines)
        out_lines.extend(header_lines)
        for ln in (hagg_line, hgf_line, qsep_line, qmeta_line):
            if ln is not None:
                out_lines.append(ln)
        out_lines.extend(other_lines)

        if all_flags:
            out_lines.append(f"[FLAGS] {' '.join(sorted(all_flags))}")

        return "\n".join(out_lines)

    def batch(self, batch: Sequence[str]) -> list[str]:
        """Process a batch of requests, preserving order."""

        if not batch:
            return []
        if len(batch) == 1:
            return [self(batch[0])]
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            return list(executor.map(self, batch))


def pipeline(
    steps: Iterable[HttpPreprocessor], *, max_workers: int | None = None
) -> PreprocessorPipeline:
    """Create an HTTP request preprocessing pipeline from a sequence of steps."""

    steps_list = list(steps)
    return PreprocessorPipeline(steps_list, max_workers=max_workers)


def resolve(dotted: str) -> HttpPreprocessor:
    """
    Resolve a dotted import path to an HTTP preprocessing class instance.

    Accepts "module.path:ClassName", imports the module, retrieves the class,
    and returns an instantiated `HttpPreprocessor`.
    """
    module_name, class_name = dotted.split(":", 1)
    preprocessor_cls: type[HttpPreprocessor] = getattr(
        import_module(module_name), class_name
    )
    return preprocessor_cls()


def load_order_from_config(config_path: Path) -> list[str]:
    """
    Load the order of steps from a config file.
    """
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    try:
        return cfg["tool"]["neuralshield"]["pipeline_order"]["order"]
    except KeyError:
        raise ValueError("Pipeline order not found in config")


preprocess: PreprocessorPipeline = pipeline(
    resolve(name)
    for name in load_order_from_config(Path(__file__).with_name("config.toml"))
)
