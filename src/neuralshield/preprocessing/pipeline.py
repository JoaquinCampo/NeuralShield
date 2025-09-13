"""
HTTP request preprocessing pipeline for neuralshield.

This module provides functionality to create and manage preprocessing pipelines
for HTTP requests. Each preprocessing step is a function that takes an HTTP
request string and returns a normalized/transformed version suitable for
machine learning models.
"""

import tomllib
from importlib import import_module
from pathlib import Path
from typing import Callable, Iterable, List

HttpPreprocessor = Callable[[str], str]


def pipeline(steps: Iterable[HttpPreprocessor]) -> HttpPreprocessor:
    """
    Create an HTTP request preprocessing pipeline from a sequence of steps.

    Args:
        steps: An iterable of HttpPreprocessor functions to be executed in sequence.

    Returns:
        An HttpPreprocessor function that applies all the provided steps in order.
    """

    def run(s: str) -> str:
        for step in steps:
            s = step(s)
        return s

    return run


def resolve(dotted: str) -> HttpPreprocessor:
    """
    Resolve a dotted import path to an HTTP preprocessing function.

    Takes a string in the format "module.path:function_name" and dynamically
    imports the module and retrieves the specified HTTP preprocessor function.
    """
    module_name, func_name = dotted.split(":", 1)
    return getattr(import_module(module_name), func_name)


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


preprocess: HttpPreprocessor = pipeline(
    resolve(name)
    for name in load_order_from_config(
        Path("src/neuralshield/preprocessing/config.toml")
    )
)
