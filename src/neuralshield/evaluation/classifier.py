from __future__ import annotations

from loguru import logger

from neuralshield.evaluation.base import Evaluator
from neuralshield.evaluation.config import EvaluationResult
from neuralshield.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
)


class ClassificationEvaluator(Evaluator):
    """Evaluate binary classification for anomaly detection."""

    def evaluate(
        self,
        predictions: list[bool],
        labels: list[str],
    ) -> EvaluationResult:
        """
        Calculate all classification metrics for anomaly detection.

        Args:
            predictions: Binary predictions (True = anomaly detected)
            labels: Ground truth labels

        Returns:
            EvaluationResult with precision, recall, F1, FPR, and more

        Raises:
            ValueError: If predictions and labels have different lengths or are empty
        """
        if len(predictions) != len(labels):
            raise ValueError(
                f"Predictions ({len(predictions)}) and labels ({len(labels)}) "
                "must have the same length"
            )

        if not predictions:
            raise ValueError("Cannot evaluate empty predictions")

        logger.debug(
            "Starting evaluation",
            total_samples=len(predictions),
            positive_label=self.config.positive_label,
            negative_label=self.config.negative_label,
        )

        # Calculate confusion matrix
        tp, fp, tn, fn = calculate_confusion_matrix(
            predictions,
            labels,
            self.config.positive_label,
            self.config.negative_label,
        )

        # Calculate all metrics using sklearn
        metrics = calculate_classification_metrics(
            predictions,
            labels,
            self.config.positive_label,
        )

        # Build result
        result = EvaluationResult(
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            accuracy=metrics["accuracy"],
            fpr=metrics["fpr"],
            specificity=metrics["specificity"],
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            total_samples=len(predictions),
            positive_samples=tp + fn,
            negative_samples=tn + fp,
        )

        logger.info(
            "Evaluation complete",
            precision=result.precision,
            recall=result.recall,
            f1_score=result.f1_score,
            fpr=result.fpr,
        )

        return result
