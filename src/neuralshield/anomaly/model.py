"""DEPRECATED: Use neuralshield.anomaly.isolation_forest instead.

This module is kept for backward compatibility only.
"""

from __future__ import annotations

import warnings

# Import from new location
from neuralshield.anomaly.isolation_forest import IsolationForestDetector

warnings.warn(
    "neuralshield.anomaly.model is deprecated. "
    "Use neuralshield.anomaly.isolation_forest instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["IsolationForestDetector"]
