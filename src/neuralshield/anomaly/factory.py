from __future__ import annotations

from typing import Callable

from loguru import logger

from neuralshield.anomaly.base import AnomalyDetector

DetectorFactory = type[AnomalyDetector]

_detector_registry: dict[str, DetectorFactory] = {}


def register_detector(name: str) -> Callable[[DetectorFactory], DetectorFactory]:
    """Decorator to register a detector class."""

    normalized = name.lower()

    def decorator(cls: DetectorFactory) -> DetectorFactory:
        if not issubclass(cls, AnomalyDetector):
            raise TypeError("register_detector expects an AnomalyDetector subclass")
        if normalized in _detector_registry:
            raise ValueError(f"Detector '{normalized}' is already registered")
        _detector_registry[normalized] = cls
        logger.debug("Registered detector {name}", name=normalized)
        return cls

    return decorator


def get_detector(name: str) -> DetectorFactory:
    """Return a factory for the requested detector."""

    normalized = name.lower()
    try:
        return _detector_registry[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown detector '{normalized}'") from exc


def available_detectors() -> dict[str, DetectorFactory]:
    """Return a copy of the detector registry."""

    return dict(_detector_registry)
