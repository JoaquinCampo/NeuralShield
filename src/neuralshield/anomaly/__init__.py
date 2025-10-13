"""Anomaly detection utilities for NeuralShield."""

from neuralshield.anomaly import (
    isolation_forest,  # noqa: F401 - ensure registration
    mahalanobis,  # noqa: F401 - ensure registration
    ocsvm,  # noqa: F401 - ensure registration
)
from neuralshield.anomaly.base import AnomalyDetector
from neuralshield.anomaly.factory import (
    available_detectors,
    get_detector,
    register_detector,
)
from neuralshield.anomaly.isolation_forest import IsolationForestDetector
from neuralshield.anomaly.mahalanobis import MahalanobisDetector
from neuralshield.anomaly.ocsvm import OCSVMDetector

__all__ = [
    "AnomalyDetector",
    "IsolationForestDetector",
    "MahalanobisDetector",
    "OCSVMDetector",
    "available_detectors",
    "get_detector",
    "register_detector",
    "main",
]


def main() -> None:
    """Entry point that defers importing Typer CLI until needed."""

    from neuralshield.anomaly.cli import main as cli_main

    cli_main()
