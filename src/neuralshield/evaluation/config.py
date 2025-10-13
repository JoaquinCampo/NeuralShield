from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict


class EvaluationConfig(BaseModel):
    """Configuration for anomaly detection evaluation."""

    positive_label: str = "attack"
    negative_label: str = "valid"

    model_config = ConfigDict(validate_assignment=True)


@dataclass
class EvaluationResult:
    """Results from evaluating an anomaly detector."""

    # Core classification metrics
    precision: float
    recall: float
    f1_score: float
    accuracy: float

    # Anomaly detection specific
    fpr: float  # False positive rate
    specificity: float  # True negative rate (1 - FPR)

    # Raw confusion matrix counts
    tp: int
    fp: int
    tn: int
    fn: int

    # Sample counts
    total_samples: int
    positive_samples: int
    negative_samples: int
