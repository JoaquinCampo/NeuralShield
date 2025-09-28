import tomllib
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from typing import Iterable, Sequence

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


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
        return request

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
    for name in load_order_from_config(
        Path("src/neuralshield/preprocessing/config.toml")
    )
)
