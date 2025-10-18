# Experiment 21 · Flag Correlation

Goal: quantify how each preprocessing flag correlates with attack traffic in the SR_BH 2020 dataset so we can prioritise “important” tokens for weighted pooling and anomaly detection.

Plan:

1. Run the preprocessing pipeline over both `train.jsonl` (all normal) and `test.jsonl` (mixed) under `src/neuralshield/data/SR_BH_2020/`.
2. For every request, extract the emitted flag tokens (including parameterised ones such as `QARRAY:` and `QREPEAT:`).
3. Aggregate counts, per-label conditional probabilities, and simple lift/odds-ratio metrics to highlight which flags skew strongly toward attacks.
4. Persist the aggregated statistics as a JSON/CSV artifact and a concise Markdown summary for downstream experiments.

`compute_flag_stats.py` will emit per-flag counts, conditional probabilities, and odds ratios keyed by `attack` vs `valid`. Use the results to select high-correlation flags for weighted pooling experiments.

Usage:

```bash
uv run python experiments/21_flag_correlation/compute_flag_stats.py --output experiments/21_flag_correlation/results.json
```

## Cross-dataset Tier 1 Flags

Comparing SR_BH 2020 and CSIC results surfaces a stable intersection of flags that consistently spike in attack traffic:

- `DOUBLEPCT`
- `PCTSUSPICIOUS`
- `PCTCONTROL`
- `QRAWSEMI`
- `QNUL`
- `QARRAY:` (and `QREPEAT:` where present)
- `PCTSPACE`
- `PIPE`
- `QUOTE`

These serve as Tier 1 “high-importance” cues for weighted pooling or feature weighting without overfitting to a single dataset. Flags that fire on every request (e.g., `PAREN`, `HOPBYHOP`, `HDRNORM`) should remain unweighted to avoid drowning out the attack-specific signals.
