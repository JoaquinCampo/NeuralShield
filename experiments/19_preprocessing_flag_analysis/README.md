# Experiment 19 – Preprocessing Flag Analysis

## Purpose

Quantify how NeuralShield’s preprocessing pipeline behaves on normal versus attack traffic. The working hypothesis is that benign requests accumulate few, mostly structural flags while malicious payloads trigger more—and more diverse—flags that downstream anomaly detectors can exploit.

## Data

- **Datasets:**  
  - SR_BH_2020 JSONL export (`train.jsonl`, `test.jsonl`)  
  - CSIC HTTP dataset (combined `csic_dataset.jsonl`)
- **Coverage:**  
  - SR_BH: 100 000 valid (entire train split) + 382 620 attack (entire test split)  
  - CSIC: 72 000 valid + 25 065 attack (entire combined dataset)
- **Why SR_BH:** Large, modern crawl containing both benign browsing sessions and numerous injected payloads (SQLi, traversal, protocol abuse).

## Method

1. Ensure optional dependencies resolve (the script provides tiny stubs for `loguru`/`idna` so it also runs inside restricted sandboxes).
2. Load the preprocessing pipeline from `neuralshield.preprocessing.pipeline`.
3. Reservoir-sample the target count per label (or stream the full corpus when `--sample-size <= 0`).
4. Run each request through all preprocessing steps (now 13 with the summary emitter).
5. Parse trailing flag tokens plus structural metadata (`[STRUCT]`) and the new `[FLAG_SUMMARY]` families.
6. Aggregate per-request flag counts, unique-flag counts, and per-flag frequency/presence statistics.
7. Persist the summary as JSON for downstream analysis.

Re-run with:

```bash
# Balanced baseline (current default)
uv run python experiments/19_preprocessing_flag_analysis/compute_flag_stats.py --sample-size -1

# CSIC overfit heuristics
uv run python experiments/19_preprocessing_flag_analysis/compute_flag_stats.py \
  --pipeline csic-overfit \
  --valid-path src/neuralshield/data/CSIC/csic_dataset.jsonl \
  --attack-path src/neuralshield/data/CSIC/csic_dataset.jsonl \
  --sample-size -1 \
  --output experiments/19_preprocessing_flag_analysis/flag_stats_csic_overfit.json

# SR_BH overfit heuristics
uv run python experiments/19_preprocessing_flag_analysis/compute_flag_stats.py \
  --pipeline srbh-overfit \
  --valid-path src/neuralshield/data/SR_BH_2020/train.jsonl \
  --attack-path src/neuralshield/data/SR_BH_2020/test.jsonl \
  --sample-size -1 \
  --output experiments/19_preprocessing_flag_analysis/flag_stats_srbh_overfit.json
```

The command rewrites `flag_stats_summary.json` unless you pass `--output`. Use `--sample-size`, `--seed`, or custom dataset paths as needed (positive `--sample-size` retains sampling).

### Pipeline Variants

- `neuralshield.preprocessing.pipeline_csic_overfit.preprocess_csic_overfit` – enables the CSIC-specific heuristics (`QSQLI_QUOTE_SEMI`, `XSS_TAG`, `FLAG_RISK_HIGH`).
- `neuralshield.preprocessing.pipeline_srbh_overfit.preprocess_srbh_overfit` – enables the SR_BH-specific heuristics (`PIPE_REPEAT`, `BRACE_REPEAT`, `PCTSPACE_PAIR`, `STRUCT_GAP:*`).

## Key Metrics

| Metric                 | Valid | Attack | Notes                                     |
| ---------------------- | ----- | ------ | ----------------------------------------- |
| Mean flags / request   | 4.32  | 5.14   | +0.82 absolute lift for attacks           |
| Median flags / request | 4     | 5      | Attack distribution stays higher          |
| 95th percentile        | 7     | 8      | Tail compressed by structural filtering   |
| Max flags observed     | 17    | 29     | Worst cases concentrate many anomalies    |
| Mean unique flags      | 4.04  | 4.55   | Attacks touch more distinct steps         |

No sample hit zero flags because Step 03 (`HDRNORM`) always emits at least the normalization evidence flag.

## Flag Differentials

### SR_BH_2020

- **Attack leaning:** `MULTIPLESLASH` (+56 pp presence), `QUOTE` (+35 pp), `PIPE` (+19 pp), `PCTSPACE` (+6.6 pp), `BRACE` (+5.7 pp). These stem from the dangerous-character and traversal steps.
- **Encoding evidence:** Encoded whitespace and slash preservation (`PCTSPACE`, `PCTSLASH`) now surface in the `[FLAG_SUMMARY]` counts without bloating inline flags.
- **Benign leaning:** `HOME` (root paths) and `HOPBYHOP` (hop-by-hop headers) mostly occur on normal traffic; with `[STRUCT]` they no longer affect totals but still provide context.
- **Low-value features:** `HDRNORM`, `PAREN` remain nearly universal and are confined to `[STRUCT]`, keeping per-request flag counts focused on anomalies.

### CSIC

- **Attack leaning:** `QUOTE` (+8 pp), `SEMICOLON` (+3.4 pp), `QRAWSEMI` (+2.6 pp), `ANGLE` (+2.6 pp) still characterise malicious payloads.
- **Noise reduction:** The new `QNONASCII` threshold drops the delta to ~0.03 pp, so occasional locale-specific characters on valid requests no longer trigger the flag.
- **Benign leaning:** Structural evidence (`HDRNORM`, `MULTIPLESLASH`, `HOPBYHOP`, `PAREN`) sits under `[STRUCT]`, emphasising how templated CSIC traffic differs from SR_BH.
- **Zero-flag cases:** none (minimum 4 flags) because CSIC requests always include the structural annotations emitted by earlier steps.

## Outputs

- `compute_flag_stats.py` – script to produce the statistics.
- `flag_stats_summary.json` – serialized results from the full SR_BH_2020 run (100 k valid + 382 620 attack).
- `flag_stats_csic.json` – serialized results from the full CSIC dataset (72 k valid + 25 065 attack).

## Next Steps

1. Turn per-flag presence counts into features and benchmark a simple detector (e.g., logistic regression or Mahalanobis) to quantify gains over raw text.
2. Audit query-specific flags (`QARRAY:*`, `QBARE`, `PCTSUSPICIOUS`) on difficult cases to refine preprocessing heuristics.
3. Extend the analysis to CSIC or mixed datasets to check whether the observed signal generalizes across corpora.
