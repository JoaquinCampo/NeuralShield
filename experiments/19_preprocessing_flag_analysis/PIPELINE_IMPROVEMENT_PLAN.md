# Preprocessing Pipeline Improvement Plan

## Goal

Increase separation between normal and anomalous HTTP requests by refining the flagging strategy discovered in Experiment 19.

## Proposed Changes

1. **Suppress Low-Value Structural Flags**

   - Stop emitting `HDRNORM`, `PAREN`, `HOPBYHOP`, `MULTIPLESLASH` as per-line flags.
   - If historical tracking is necessary, aggregate them into a single metadata field (e.g., `[STRUCTURE] hdrnorm,paren,…`).
   - Outcomes: reduce flag noise and free downstream models to focus on discriminative signals.

2. **Enrich Percent-Decoding Evidence (Step 08)**

   - Add severity-aware variants: `DOUBLEPCT:M`, `PCTSUSPICIOUS:H`, `PCTCONTROL:H`, etc.
   - Normalise mutually exclusive flags so each request records at most one high-level encoding flag per incident.
   - Outcomes: maintain concise flag counts while emphasising high-risk encodings.

3. **Refine Query Analysis (Step 10)**

   - Require rarity thresholds before emitting `QRAWSEMI`, `SEMICOLON`, or `QNONASCII`; compute token frequencies to avoid “always-on” flags in CSIC.
   - Add new shape flags for suspicious parameter names (e.g., `QKEY_SYMBOL`, `QKEY_EMPTY`).
   - Outcomes: better separation for query manipulation attacks, fewer false positives on templated traffic.

4. **Introduce Family-Level Summaries**
   - Emit a `[FLAG_SUMMARY] danger=3 encoding=2 query=1 traversal=0` line per request.
   - Define families and severity weights centrally so downstream models can consume consistent features.
   - Outcomes: simplifies feature engineering and allows coarse anomaly thresholds without inspecting every flag.

## Implementation Steps

1. Update step modules (`03`, `05`, `08`, `10`, `11`) to apply suppression, severity annotations, and conditional emissions.
2. Extend `config.toml` with new thresholds, family definitions, and overflow settings.
3. Add unit tests covering:
   - Suppressed structural flags on clean inputs.
   - Severity annotations for percent-decoding.
   - Query flag emission conditional on rarity thresholds.
   - `[FLAG_SUMMARY]` generation and overflow detection.
4. Re-run Experiment 19 scripts (`compute_flag_stats.py` with SR_BH and CSIC) to verify improved attack-to-valid separation.
5. Document the behaviour changes in `docs/PREPROCESSING_FLAGS_ANALYSIS.md` and update experiment findings.

## Risks & Mitigations

- **Over-suppression:** Ensure structural flags remain available (e.g., aggregated metadata) for debugging.
- **Threshold tuning:** Begin with conservative rarity/percentile settings; adjust after observing new flag distributions.
- **Backward compatibility:** Version the pipeline (e.g., `pipeline_v2`) or feature-flag changes to avoid breaking existing models.
