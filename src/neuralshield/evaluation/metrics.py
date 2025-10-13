from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_confusion_matrix(
    predictions: list[bool],
    labels: list[str],
    positive_label: str,
    negative_label: str,  # noqa: ARG001
) -> tuple[int, int, int, int]:
    """
    Calculate confusion matrix counts from predictions and labels.

    Args:
        predictions: Binary predictions (True = anomaly detected)
        labels: Ground truth labels (e.g., "attack", "valid")
        positive_label: Label indicating positive class (attack)
        negative_label: Label indicating negative class (valid)

    Returns:
        Tuple of (tp, fp, tn, fn)
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"Predictions ({len(predictions)}) and labels ({len(labels)}) "
            "must have the same length"
        )

    if not predictions:
        raise ValueError("Cannot calculate confusion matrix from empty predictions")

    # Convert labels to binary (True = positive class)
    y_true = np.array([label == positive_label for label in labels])
    y_pred = np.array(predictions)

    # Use sklearn's confusion_matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()

    return int(tp), int(fp), int(tn), int(fn)


def calculate_classification_metrics(
    predictions: list[bool],
    labels: list[str],
    positive_label: str,
) -> dict[str, float]:
    """
    Calculate all classification metrics using sklearn.

    Args:
        predictions: Binary predictions
        labels: Ground truth labels
        positive_label: Label indicating positive class

    Returns:
        Dictionary containing precision, recall, f1_score, accuracy, fpr, specificity
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length")

    if not predictions:
        raise ValueError("Cannot calculate metrics from empty predictions")

    # Convert to binary
    y_true = np.array([label == positive_label for label in labels])
    y_pred = np.array(predictions)

    # Leverage sklearn for all metrics
    precision = float(precision_score(y_true, y_pred, zero_division=0.0))
    recall = float(recall_score(y_true, y_pred, zero_division=0.0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0.0))
    accuracy = float(accuracy_score(y_true, y_pred))

    # Calculate FPR and specificity from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()

    # FPR = FP / (FP + TN)
    fpr = float(fp / max(fp + tn, 1))

    # Specificity = TN / (TN + FP) = 1 - FPR
    specificity = float(tn / max(tn + fp, 1))

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "fpr": fpr,
        "specificity": specificity,
    }
