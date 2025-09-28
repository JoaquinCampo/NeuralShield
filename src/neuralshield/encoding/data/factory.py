from __future__ import annotations

from typing import Callable, Mapping

from loguru import logger

from neuralshield.encoding.data.base import DatasetReader

ReaderFactory = type[DatasetReader]

_reader_registry: dict[str, ReaderFactory] = {}


def register_reader(name: str) -> Callable[[ReaderFactory], ReaderFactory]:
    """Decorator to register a dataset reader class under ``name``."""

    normalized = name.lower()

    def decorator(cls: ReaderFactory) -> ReaderFactory:
        if normalized in _reader_registry:
            raise ValueError(f"Dataset reader '{normalized}' is already registered")
        _reader_registry[normalized] = cls
        logger.debug("Registered dataset reader {name}", name=normalized)
        return cls

    return decorator


def get_reader(name: str) -> ReaderFactory:
    """Return a reader factory by name."""

    normalized = name.lower()
    try:
        return _reader_registry[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset reader '{normalized}'") from exc


def available_readers() -> Mapping[str, ReaderFactory]:
    """Return the registered dataset readers."""

    return dict(_reader_registry)
