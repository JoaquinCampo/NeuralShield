# Pipeline Evolution Summary

## Key Improvements in New Pipeline

1. **Structural Flag Isolation**: Common structural flags (HDRNORM, HOPBYHOP, PAREN, MULTIPLESLASH, HOME) moved to [STRUCT] line to reduce noise
2. **Flag Family Summaries**: New [FLAG_SUMMARY] line with categorized counts (danger, encoding, unicode, query, header, traversal, network, structure)
3. **Risk-focused Metrics**: Separates structural signals from risk indicators
4. **Overflow Detection**: FLAG_OVERFLOW when risk flags exceed threshold

## Statistical Comparison

### CSIC Dataset (72K valid, 25K attack)
- **Aggregate counts preserved**: Valid 4.00, Attack 4.33 (identical to old pipeline)
- **Better feature representation**: Family summaries provide richer, more stable features
- **Attack signals clearer**: QUOTE (16.4% vs 0.3%), SEMICOLON (6.8%), ANGLE (5.2%)

### SR_BH Dataset (100K valid, 382K attack) 
- **Consistent results**: Valid 4.32, Attack 5.14 (matches old pipeline)
- **Structural cleanup**: Traversal flags now properly isolated under [STRUCT]

## Benefits for Anomaly Detection

1. **Reduced noise**: Structural flags don't inflate per-request counts
2. **Family-level features**: More robust than individual flag presence
3. **Better generalization**: Family summaries travel better across datasets
4. **Risk prioritization**: Clear separation of concerning vs benign signals

## Migration Notes

- Existing models may need retraining due to [FLAG_SUMMARY] features
- Old individual flag features still available but deprecated for new models
- Consider family counts + overflow flags for next-generation detectors
