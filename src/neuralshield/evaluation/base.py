from __future__ import annotations

from abc import ABC, abstractmethod

from neuralshield.evaluation.config import EvaluationConfig, EvaluationResult


class Evaluator(ABC):
    """Base class for anomaly detection evaluators."""

    def __init__(self, config: EvaluationConfig = EvaluationConfig()) -> None:
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        predictions: list[bool],
        labels: list[str],
    ) -> EvaluationResult:
        """
        Evaluate predictions against ground truth labels.

        Args:
            predictions: Binary predictions (True = anomaly detected)
            labels: Ground truth labels

        Returns:
            EvaluationResult containing all metrics
        """
        ...
