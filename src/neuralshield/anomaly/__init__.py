"""Anomaly detection utilities for NeuralShield."""

from neuralshield.anomaly.model import (
    IsolationForestDetector,
)

__all__ = [
    "IsolationForestDetector",
    "main",
]


def main() -> None:
    """Entry point that defers importing Typer CLI until needed."""

    from neuralshield.anomaly.cli import main as cli_main

    cli_main()
