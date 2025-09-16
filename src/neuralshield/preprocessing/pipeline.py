"""
HTTP request preprocessing pipeline for neuralshield.

This module creates and manages preprocessing pipelines for HTTP requests.
Each preprocessing step is an instance of `HttpPreprocessor` implementing a
`process()` method that takes an HTTP request string and returns a normalized
or transformed version suitable for machine learning models.
"""

import tomllib
from importlib import import_module
from pathlib import Path
from typing import Callable, Iterable, List, Type

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor

PreprocessorFunc = Callable[[str], str]


def pipeline(steps: Iterable[HttpPreprocessor]) -> PreprocessorFunc:
    """
    Create an HTTP request preprocessing pipeline from a sequence of steps.

    Args:
        steps: An iterable of `HttpPreprocessor` instances to execute in order.

    Returns:
        A function that applies all provided steps' `process()` in order.
    """

    def run(s: str) -> str:
        for step in steps:
            s = step.process(s)
        return s

    return run


def resolve(dotted: str) -> HttpPreprocessor:
    """
    Resolve a dotted import path to an HTTP preprocessing class instance.

    Accepts "module.path:ClassName", imports the module, retrieves the class,
    and returns an instantiated `HttpPreprocessor`.
    """
    module_name, class_name = dotted.split(":", 1)
    preprocessor_cls: Type[HttpPreprocessor] = getattr(
        import_module(module_name), class_name
    )
    return preprocessor_cls()


def load_order_from_config(config_path: Path) -> List[str]:
    """
    Load the order of steps from a config file.
    """
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    try:
        return cfg["tool"]["neuralshield"]["pipeline_order"]["order"]
    except KeyError:
        raise ValueError("Pipeline order not found in config")


preprocess: PreprocessorFunc = pipeline(
    resolve(name)
    for name in load_order_from_config(
        Path("src/neuralshield/preprocessing/config.toml")
    )
)
