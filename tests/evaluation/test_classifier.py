"""Tests for ClassificationEvaluator."""

import pytest

from neuralshield.evaluation import (
    ClassificationEvaluator,
    EvaluationConfig,
)


def test_evaluator_with_perfect_predictions():
    """Test evaluator with perfect predictions."""
    config = EvaluationConfig(positive_label="attack", negative_label="valid")
    evaluator = ClassificationEvaluator(config)

    predictions = [True, True, False, False]
    labels = ["attack", "attack", "valid", "valid"]

    result = evaluator.evaluate(predictions, labels)

    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1_score == 1.0
    assert result.accuracy == 1.0
    assert result.fpr == 0.0
    assert result.specificity == 1.0
    assert result.tp == 2
    assert result.fp == 0
    assert result.tn == 2
    assert result.fn == 0
    assert result.total_samples == 4
    assert result.positive_samples == 2
    assert result.negative_samples == 2


def test_evaluator_with_mixed_predictions():
    """Test evaluator with mixed predictions."""
    evaluator = ClassificationEvaluator()  # Use default config

    predictions = [True, False, True, False]
    labels = ["attack", "valid", "attack", "attack"]

    result = evaluator.evaluate(predictions, labels)

    assert result.tp == 2  # Indices 0, 2
    assert result.fp == 0
    assert result.tn == 1  # Index 1
    assert result.fn == 1  # Index 3
    assert result.total_samples == 4
    assert result.positive_samples == 3
    assert result.negative_samples == 1
    assert result.recall == pytest.approx(2 / 3)
    assert result.precision == 1.0


def test_evaluator_empty_predictions_raises():
    """Test that empty predictions raise ValueError."""
    evaluator = ClassificationEvaluator()

    with pytest.raises(ValueError, match="empty predictions"):
        evaluator.evaluate([], [])


def test_evaluator_mismatched_lengths_raises():
    """Test that mismatched lengths raise ValueError."""
    evaluator = ClassificationEvaluator()

    with pytest.raises(ValueError, match="same length"):
        evaluator.evaluate([True], ["attack", "valid"])


def test_evaluator_custom_labels():
    """Test evaluator with custom labels."""
    config = EvaluationConfig(positive_label="malicious", negative_label="benign")
    evaluator = ClassificationEvaluator(config)

    predictions = [True, False, True]
    labels = ["malicious", "benign", "malicious"]

    result = evaluator.evaluate(predictions, labels)

    assert result.tp == 2
    assert result.tn == 1
    assert result.fp == 0
    assert result.fn == 0
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_evaluator_high_false_positive_rate():
    """Test evaluator with high false positive rate."""
    evaluator = ClassificationEvaluator()

    # Detector is too sensitive
    predictions = [True] * 10
    labels = ["valid"] * 9 + ["attack"] * 1

    result = evaluator.evaluate(predictions, labels)

    assert result.tp == 1
    assert result.fp == 9
    assert result.tn == 0
    assert result.fn == 0
    assert result.fpr == 1.0  # All normals flagged
    assert result.specificity == 0.0
    assert result.recall == 1.0  # Caught the attack
    assert result.precision == pytest.approx(1 / 10)


def test_evaluator_high_false_negative_rate():
    """Test evaluator with high false negative rate."""
    evaluator = ClassificationEvaluator()

    # Detector is too conservative
    predictions = [False] * 10
    labels = ["attack"] * 9 + ["valid"] * 1

    result = evaluator.evaluate(predictions, labels)

    assert result.tp == 0
    assert result.fp == 0
    assert result.tn == 1
    assert result.fn == 9
    assert result.recall == 0.0  # Missed all attacks
    assert result.fpr == 0.0  # No false alarms
    assert result.specificity == 1.0
