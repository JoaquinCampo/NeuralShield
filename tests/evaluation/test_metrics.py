"""Tests for evaluation metrics calculations."""

import pytest

from neuralshield.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
)


def test_perfect_classification():
    """Test metrics with perfect predictions."""
    predictions = [True, True, False, False]
    labels = ["attack", "attack", "valid", "valid"]

    tp, fp, tn, fn = calculate_confusion_matrix(
        predictions, labels, positive_label="attack", negative_label="valid"
    )

    assert tp == 2
    assert fp == 0
    assert tn == 2
    assert fn == 0

    metrics = calculate_classification_metrics(predictions, labels, "attack")
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["specificity"] == 1.0


def test_all_false_positives():
    """Test with all false positives."""
    predictions = [True, True, True, True]
    labels = ["valid", "valid", "valid", "valid"]

    tp, fp, tn, fn = calculate_confusion_matrix(
        predictions, labels, positive_label="attack", negative_label="valid"
    )

    assert tp == 0
    assert fp == 4
    assert tn == 0
    assert fn == 0

    metrics = calculate_classification_metrics(predictions, labels, "attack")
    assert metrics["precision"] == 0.0  # No true positives
    assert metrics["recall"] == 0.0  # No actual attacks
    assert metrics["f1_score"] == 0.0
    assert metrics["accuracy"] == 0.0
    assert metrics["fpr"] == 1.0  # All normals flagged
    assert metrics["specificity"] == 0.0


def test_all_false_negatives():
    """Test with all false negatives."""
    predictions = [False, False, False, False]
    labels = ["attack", "attack", "attack", "attack"]

    tp, fp, tn, fn = calculate_confusion_matrix(
        predictions, labels, positive_label="attack", negative_label="valid"
    )

    assert tp == 0
    assert fp == 0
    assert tn == 0
    assert fn == 4

    metrics = calculate_classification_metrics(predictions, labels, "attack")
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0  # Missed all attacks
    assert metrics["f1_score"] == 0.0
    assert metrics["accuracy"] == 0.0


def test_mixed_predictions():
    """Test with mixed predictions."""
    predictions = [True, False, True, False, True, False]
    labels = ["attack", "attack", "valid", "valid", "attack", "valid"]

    tp, fp, tn, fn = calculate_confusion_matrix(
        predictions, labels, positive_label="attack", negative_label="valid"
    )

    # TP: indices 0, 4
    # FP: index 2
    # TN: indices 3, 5
    # FN: index 1
    assert tp == 2
    assert fp == 1
    assert tn == 2
    assert fn == 1

    metrics = calculate_classification_metrics(predictions, labels, "attack")
    assert metrics["precision"] == pytest.approx(2 / 3)  # 2 / (2 + 1)
    assert metrics["recall"] == pytest.approx(2 / 3)  # 2 / (2 + 1)
    assert metrics["f1_score"] == pytest.approx(2 / 3)
    assert metrics["accuracy"] == pytest.approx(4 / 6)  # (2 + 2) / 6
    assert metrics["fpr"] == pytest.approx(1 / 3)  # 1 / (1 + 2)
    assert metrics["specificity"] == pytest.approx(2 / 3)  # 2 / (2 + 1)


def test_empty_predictions_raises():
    """Test that empty predictions raise ValueError."""
    with pytest.raises(ValueError, match="empty predictions"):
        calculate_confusion_matrix([], [], "attack", "valid")

    with pytest.raises(ValueError, match="empty predictions"):
        calculate_classification_metrics([], [], "attack")


def test_mismatched_lengths_raises():
    """Test that mismatched lengths raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        calculate_confusion_matrix([True, False], ["attack"], "attack", "valid")

    with pytest.raises(ValueError, match="same length"):
        calculate_classification_metrics([True, False], ["attack"], "attack")


def test_imbalanced_dataset():
    """Test with realistic imbalanced dataset (99% normal, 1% attack)."""
    # 99 normal requests, 1 attack
    predictions = [False] * 95 + [True] * 5  # Detector flags 5 as anomalies
    labels = ["valid"] * 99 + ["attack"] * 1

    tp, fp, tn, fn = calculate_confusion_matrix(
        predictions, labels, positive_label="attack", negative_label="valid"
    )

    # True positive: 1 attack caught (assuming it's in the last 5)
    # False positives: 4 normal flagged as attacks
    # True negatives: 95 normal correctly identified
    # False negatives: 0 (attack was caught)
    assert tp == 1
    assert fp == 4
    assert tn == 95
    assert fn == 0

    metrics = calculate_classification_metrics(predictions, labels, "attack")
    assert metrics["precision"] == pytest.approx(1 / 5)  # 1 / (1 + 4)
    assert metrics["recall"] == 1.0  # Caught the only attack
    assert metrics["fpr"] == pytest.approx(4 / 99)  # 4 / (4 + 95)
    assert metrics["specificity"] == pytest.approx(95 / 99)  # 95 / (95 + 4)


def test_custom_labels():
    """Test with custom positive/negative labels."""
    predictions = [True, False, True]
    labels = ["malicious", "benign", "malicious"]

    tp, fp, tn, fn = calculate_confusion_matrix(
        predictions, labels, positive_label="malicious", negative_label="benign"
    )

    assert tp == 2
    assert fp == 0
    assert tn == 1
    assert fn == 0

    metrics = calculate_classification_metrics(predictions, labels, "malicious")
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
