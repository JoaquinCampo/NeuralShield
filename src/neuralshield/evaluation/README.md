# Evaluation Module

Clean, minimal evaluation module for NeuralShield anomaly detection.

## Purpose

Evaluate binary classification performance for anomaly detection with security-aware metrics.

## Features

- **Classification metrics**: Precision, Recall, F1-score, Accuracy
- **Anomaly-specific metrics**: False Positive Rate (FPR), Specificity
- **Confusion matrix**: Raw TP, FP, TN, FN counts
- **Leverages sklearn**: Uses battle-tested implementations
- **Type-safe**: Full type hints and Pydantic validation
- **Well-tested**: Comprehensive test coverage

## Quick Start

```python
from neuralshield.evaluation import ClassificationEvaluator, EvaluationConfig

# Configure
config = EvaluationConfig(
    positive_label="attack",
    negative_label="valid"
)

# Evaluate
evaluator = ClassificationEvaluator(config)
result = evaluator.evaluate(
    predictions=[True, False, True],
    labels=["attack", "valid", "attack"]
)

# Access results
print(f"Precision: {result.precision:.3f}")
print(f"Recall: {result.recall:.3f}")
print(f"F1-Score: {result.f1_score:.3f}")
print(f"FPR: {result.fpr:.3f}")
```

## Metrics

### Classification Metrics

- **Precision**: Of detected anomalies, how many are actual attacks?
- **Recall**: Of actual attacks, how many were detected?
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall correctness (less meaningful with class imbalance)

### Anomaly Detection Metrics

- **FPR** (False Positive Rate): What fraction of normal traffic is flagged?
- **Specificity**: How well we avoid false alarms (1 - FPR)

### Confusion Matrix

- **TP** (True Positives): Attacks correctly detected
- **FP** (False Positives): Normal traffic incorrectly flagged
- **TN** (True Negatives): Normal traffic correctly passed
- **FN** (False Negatives): Attacks missed

## API

### `EvaluationConfig`

Configuration for evaluation.

**Fields**:

- `positive_label: str = "attack"` - Label indicating attacks
- `negative_label: str = "valid"` - Label indicating normal traffic

### `ClassificationEvaluator`

Main evaluator class.

**Methods**:

- `evaluate(predictions: list[bool], labels: list[str]) -> EvaluationResult`

### `EvaluationResult`

Dataclass containing all metrics and counts.

## Design Principles

Following NeuralShield's principles:

- **Simple over complex**: Single evaluator, clear interface
- **Explicit over implicit**: All metrics clearly defined
- **Readability counts**: Self-documenting code and results
- **Practicality beats purity**: Leverages sklearn, not reinventing the wheel

## Future Extensions (Phase 2+)

- Score-based metrics (AUC-ROC, AUC-PR)
- Cross-validation support
- Model comparison utilities
- CLI integration
- W&B logging integration
- Threshold optimization

## Testing

Run tests:

```bash
uv run pytest tests/evaluation/ -v
```

## Example

See `examples/evaluation_example.py` for a complete working example.
