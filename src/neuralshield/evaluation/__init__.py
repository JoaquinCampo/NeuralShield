"""Evaluation module for NeuralShield anomaly detection."""

from neuralshield.evaluation.base import Evaluator
from neuralshield.evaluation.classifier import ClassificationEvaluator
from neuralshield.evaluation.config import EvaluationConfig, EvaluationResult
from neuralshield.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
)

__all__ = [
    "Evaluator",
    "ClassificationEvaluator",
    "EvaluationConfig",
    "EvaluationResult",
    "calculate_classification_metrics",
    "calculate_confusion_matrix",
]
