from __future__ import annotations

from typing import Callable

from loguru import logger

from neuralshield.encoding.models.base import RequestEncoder

EncoderFactory = type[RequestEncoder]

_encoder_registry: dict[str, EncoderFactory] = {}


def register_encoder(name: str) -> Callable[[EncoderFactory], EncoderFactory]:
    """Decorator to register an encoder class."""

    normalized = name.lower()

    def decorator(cls: EncoderFactory) -> EncoderFactory:
        if not issubclass(cls, RequestEncoder):
            raise TypeError("register_encoder expects a RequestEncoder subclass")
        if normalized in _encoder_registry:
            raise ValueError(f"Encoder '{normalized}' is already registered")
        _encoder_registry[normalized] = cls
        logger.debug("Registered encoder {name}", name=normalized)
        return cls

    return decorator


def get_encoder(name: str) -> EncoderFactory:
    """Return a factory for the requested encoder."""

    normalized = name.lower()
    try:
        return _encoder_registry[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown encoder '{normalized}'") from exc


def available_encoders() -> dict[str, EncoderFactory]:
    """Return a copy of the encoder registry."""

    return dict(_encoder_registry)
