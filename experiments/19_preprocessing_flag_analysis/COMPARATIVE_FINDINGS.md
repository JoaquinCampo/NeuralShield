# Preprocessing Flag Comparison – SR_BH vs CSIC

## Scope
- **Pipeline:** `neuralshield.preprocessing.pipeline` (13-step HTTP normalisation / flagging stack with structural summaries)
- **Datasets:** SR_BH_2020 (`train.jsonl`, `test.jsonl`) and CSIC (`csic_dataset.jsonl`)
- **Granularity:** Full corpora — 100 000 valid + 382 620 attack requests for SR_BH, 72 000 valid + 25 065 attack requests for CSIC
- **Artifacts:** `flag_stats_summary.json` (SR_BH) and `flag_stats_csic.json` (CSIC)

## Aggregate Flag Volume
| Dataset | Label | Mean flags | Median | p95 | Max | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| SR_BH | valid | 4.32 | 4 | 7 | 17 | structural suppression narrows the tail |
| SR_BH | attack | 5.14 | 5 | 8 | 29 | +0.82 lift versus valids |
| CSIC | valid | 4.00 | 4 | 4 | 7 | minimum 4 because of structural flags |
| CSIC | attack | 4.33 | 4 | 6 | 11 | +0.33 lift versus valids |

**Takeaway:** SR_BH still shows a clear separation, though structural suppression compresses the benign tail. CSIC’s gap shrank, but attacks retain slightly higher counts and rely more on query quirks than raw volume.

## High-Signal Flag Differentials
### SR_BH Attacks vs Valids
- `MULTIPLESLASH` (+56 pp presence) – structural traversal attempts now surfaced under `[STRUCT]`
- `QUOTE` (+35 pp) and `PIPE` (+19 pp) – injection-heavy characters in URL/query/header fields
- `PCTSPACE` (+6.6 pp), `BRACE` (+5.7 pp) – encoded whitespace and delimiter usage remain strong differentiators
- Severity-tagged encodings keep counts low while the `[FLAG_SUMMARY]` line highlights the encoding family impact
- Structural signals (`HOME`, `HOPBYHOP`) remain primarily benign and no longer inflate totals

### CSIC Attacks vs Valids
- `QUOTE` (+8 pp), `SEMICOLON` (+3.4 pp), `QRAWSEMI` (+2.6 pp), `ANGLE` (+2.6 pp) – query manipulation remains the standout signal
- `QNONASCII` now differs by only ~0.03 pp after introducing the rarity threshold
- Structural flags are fully captured under `[STRUCT]`, keeping per-request totals uniform at ≥4
- Double-encoding features appear less frequently but are easier to spot via the encoding family counts

## Cross-Dataset Observations
- **Shared useful features:** `QUOTE`, encoded whitespace (`PCTSPACE`), and traversal hints appear in both datasets, albeit with different magnitudes.
- **Dataset-specific signals:** SR_BH emphasises traversal and encoding abuse; CSIC highlights query-shape anomalies (`QRAWSEMI`, `SEMICOLON`).
- **Structural evidence:** `[STRUCT]` lines now isolate ubiquitous normalisation flags so they no longer distort totals.
- **Family summaries:** The new `[FLAG_SUMMARY]` line provides consistent features (danger/encoding/query/…) that travel well across datasets.

## Recommendations
1. **Feature engineering:** Build feature sets around volume (counts), presence (binary panel), and high-value flag families (encoding, dangerous characters, query anomalies).
2. **Model transferability:** Train detectors on shared flag subsets to test generalisation between SR_BH and CSIC; evaluate dataset-specific augmentations separately.
3. **Monitoring:** Track family counts and overflow events; rising encoding totals or frequent `FLAG_OVERFLOW` markers warrant investigation.
4. **Pipeline tuning:** Leverage `[STRUCT]` and `[FLAG_SUMMARY]` to keep models focused on discriminative signals; adjust thresholds as new datasets arrive.
