# Preprocessing Pipeline Improvement Plan

## Goal

Increase separation between normal and anomalous HTTP requests by refining the flagging strategy discovered in Experiment 19.

## Proposed Changes

### CSIC-Specific Adjustments

1. **Tighten Query Heuristics (Step 10)**
   - Drop `RAW_SEMICOLON_MIN_COUNT` from 2 → 1 so single `;--` payloads emit `QRAWSEMI`.
     Why: `QRAWSEMI` currently separates CSIC attacks from valids by 2.6 pp, yet 34 % of malicious rows rely on a single raw semicolon that the threshold suppresses.
   - Add a compound `QSQLI_QUOTE_SEMI` flag when a parameter carries both `QUOTE` and `SEMICOLON`.
     Why: 8.1 % of CSIC attacks show the `QUOTE`+`SEMICOLON` combo (vs 0.1 % of valids); an explicit flag keeps the joint signal even if future balancing suppresses either input flag.
   - Expand tag detection to flag `<script>` injections via a new `XSS_TAG` entry in Step 05.
     Why: CSIC XSS payloads use literal `<script>` and currently only raise `ANGLE`, leaving downstream models to guess intent despite a 2.6 pp presence delta.

2. **Escalate Risk Summaries**
   - Teach `FlagSummaryEmitter` to append `FLAG_RISK_HIGH` whenever `total ≥ 4`.
     Why: No CSIC valid request exceeds three risk flags, while 7.8 % of attacks do; the derived flag converts that gap into a single binary feature.
   - Ensure new query and XSS flags map into the `danger`/`query` families for downstream scoring.
     Why: Family totals drive the logistic baselines we monitor; missing mappings would bury the lift behind the generic `other` bucket.

3. **Visibility for Diagnostics**
   - Increase the `top_flags` export limit in `compute_flag_stats.py` so CSIC-only flags surface in reports.
     Why: The current limit (25) truncates low-frequency CSIC-specific flags, making repeatability checks blind to changes we intentionally introduce.

### SR_BH-Specific Adjustments

1. **Disambiguate Structural Flow (Step 11)**
   - Emit `MULTIPLESLASH_HEAVY` when more than one run of consecutive slashes is detected.
     Why: Attacks in SR_BH show 99.8 % `MULTIPLESLASH` presence but valids retain 43.3 %; differentiating “single collapse” from “path flooding” retains the advantage without muting root paths.
   - Record `STRUCT_GAP:HOPBYHOP` and `STRUCT_GAP:HOME` when those benign anchors are missing.
     Why: 56.7 % of SR_BH valids emit `HOME` and 70.5 % emit `HOPBYHOP`, whereas attacks plummet to 0.1 % and 29.2 %; encoding the absence as a gap makes the benign template explicit.

2. **Strengthen Dangerous Character Signals (Step 05)**
   - Add regexes for repeated pipes (`||`) and encoded braces (`%7B%7B`) to emit `PIPE_REPEAT` and `BRACE_REPEAT`.
     Why: 38.0 % of SR_BH attacks contain pipes (18.8 % valids); focusing on repeated or encoded variants isolates the Struts-style exploits that dominate the dataset.
   - Emit `PCTSPACE_PAIR` in Step 10 when `PCTSPACE` and literal `SPACE` appear in the same parameter.
     Why: Encoded+literal whitespace co-occurs in 6.6 % of SR_BH attacks but almost never in valids (0.6 %), signalling sloppy encoding that downstream models can overfit to.

3. **Promote Percent-Encoding Combos**
   - Highlight requests containing both encoded and literal whitespace by incrementing a dedicated summary counter.
     Why: Surfacing the combination in `[FLAG_SUMMARY]` reduces reliance on per-flag inspection and keeps the anomaly portable to detectors that only ingest the summary line.

### Cross-Cutting Enhancements

1. **Configurable Modes**
   - Introduce toggles in `config.toml` for `overfit_csic`, `overfit_srbh`, and `balanced` presets controlling new heuristics.
     Why: We need to ship dataset-specific heuristics without surprising deployments that expect general-purpose behaviour.

2. **Extended Monitoring**
   - Forward new flags and `FLAG_RISK_HIGH` metrics to wandb dashboards during training and evaluation.
     Why: Visible time-series makes it obvious when the heuristics activate (or regress) during future dataset ingestion.

3. **Documentation & Tests**
   - Update experiment documentation and add regression tests for every new flag/summary path.
     Why: Each heuristics switch needs executable proof and recorded intent so future contributors know the signal’s origin.

## Implementation Steps

1. Add configuration switches and defaults in `src/neuralshield/preprocessing/config.toml`.
2. Implement the CSIC-focused heuristics in Steps 05, 10, and 12; update `compute_flag_stats.py` export limits.
3. Implement the SR_BH-focused heuristics in Steps 05, 10, and 11 plus the structure gap logic.
4. Expand unit tests to cover all new flags, summaries, and config toggles.
5. Re-run Experiment 19 statistics for CSIC and SR_BH; capture before/after deltas in the experiment README.
6. Publish the findings and recommended deployment posture in `docs/PREPROCESSING_FLAGS_ANALYSIS.md`.

## Risks & Mitigations

- **Dataset Lock-In:** Keep the balanced preset as default and require explicit opt-in for dataset-targeted modes.
- **Signal Inflation:** Monitor flag volume after changes to ensure totals stay within overflow thresholds.
- **Regression Risk:** Add focused unit tests and re-run Experiment 19 after each change to catch behavioural drifts.
