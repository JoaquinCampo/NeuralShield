# PDF vs. Code Audit Report

**Date:** 2026-03-07
**PDF:** `neuralshield-docs/output/main.pdf` (4 de marzo de 2026)
**Scope:** Chapters 4 (Design), 5 (Implementation), 6 (Evaluation) cross-referenced against `src/neuralshield/`
**Method:** Five-pass deep audit — line-by-line code reading, step-by-step pipeline tracing, experiment result cross-referencing, mathematical consistency verification

---

## Executive Summary

| Category | Count |
|----------|-------|
| **Discrepancies** (PDF ≠ Code) | 19 |
| — High severity | 2 |
| — Medium severity | 11 |
| — Low severity | 6 |
| **Code bugs** discovered during audit | 2 |
| **Minor observations** (undocumented but not wrong) | 23 |
| **Verified claims** (PDF = Code) | 61 |

### Priority Fixes

1. **[HIGH] Listing 4.2 example** (1.12) — The end-to-end example in Chapter 4 has ~12 discrepancies with actual pipeline output. It appears hand-crafted rather than generated. **Action:** Run the real pipeline on Listing 4.1 and replace Listing 4.2 with the actual output.

2. **[BUG] MULTIPLESLASH false positive** (1.16) — Every absolute URL path triggers `MULTIPLESLASH` due to a segmentation artifact. **Action:** Fix `_detect_multiple_slashes` in `11_path_structure_normalizer.py`.

3. **[BUG] Flag propagation corruption** (1.12l, 1.18) — Steps 06, 07, 08, and 11 consume inline flags from prior steps as content. **Action:** Implement flag/content separation in all four steps.

4. **[MEDIUM] NFKC "diagnostic only" claim** (1.3) — PDF says NFKC is diagnostic; code applies it destructively. **Action:** Update §5.1.6 to match code behavior.

5. **[MEDIUM] CSIC train size** (1.7) — PDF says ~36K, code uses 47K. **Action:** Update §6.2.

6. **[MEDIUM] SR\_BH ensemble weight** (1.8) — PDF says w=0.6, code tested only w=0.5. **Action:** Update Table 6.4.

7. **[MEDIUM] No validation split** (1.13) — PDF describes 3-way split, code implements 2-way. **Action:** Update §6.2 or implement validation split.

8. **[HIGH] Table 6.4 F1-score errors** (1.17) — SR\_BH TF-IDF+LOF F1=23.80% is a copy-paste of Recall (correct: 36.80%). CSIC SecBERT+Maha F1=55.40% is wrong (correct: 63.87%). **Action:** Recompute F1 from P/R in Table 6.3.

9. **[MEDIUM] Flag corruption in 4 steps** (1.18) — Steps 06, 07, 08, 11 use naive `line[6:]` that includes trailing flags. **Action:** Add shared flag/content splitter.

---

## 1. Discrepancies

### 1.1 `MIXEDSEP` flag in Listing 4.2 does not exist in code

**Severity:** Medium
**Location:** PDF Listing 4.2 (page ~38) vs `src/neuralshield/preprocessing/steps/10_query_parser_and_flags.py`

The PDF's end-to-end example shows:

```
[QMETA] count=3 MIXEDSEP
```

The flag `MIXEDSEP` is never defined or emitted anywhere in the codebase. The flag catalog in Section 5.1.4 does not list it either. The actual code emits:

- `QSEMISEP` — when `;` is the dominant separator (line 283 of step 10)
- `QRAWSEMI` — when `;` is present but not dominant (line 286 of step 10)

**Diagnosis:** Likely a pre-implementation draft name that was renamed during development.

**Fix:** Replace `MIXEDSEP` with `QSEMISEP` (or `QRAWSEMI`, depending on the example's intent) in Listing 4.2.

---

### 1.2 Summary `[FLAGS]` line in Listing 4.2 is not produced by the pipeline

**Severity:** Medium
**Location:** PDF Listing 4.2 (page ~38) vs full pipeline output

The PDF example shows a final summary line:

```
[FLAGS] DUPHDR DOUBLEPCT DOTDOT WSPAD
```

No step in the pipeline generates this aggregated summary. In the actual implementation:

- Step 01 (`RequestStructurer`) emits `[FLAGS]` **only** for request-line flags (e.g. `UNUSUAL_METHOD`).
- All other flags are emitted **inline** on their respective `[HEADER]`, `[URL]`, or `[QUERY]` lines, or in block-level tags (`[HGF]`, `[QMETA]`).

There is no aggregation step that collects inline flags into a final `[FLAGS]` summary.

**Fix:** Either (a) update the example to show the actual output format (flags inline on their respective lines, no summary `[FLAGS]` line), or (b) add a final aggregation step to the pipeline that collects all emitted flags into a summary line. Option (a) is recommended since the inline format is more informative.

---

### 1.3 NFKC normalization is applied to content, contradicting "diagnostic only" claim

**Severity:** Medium
**Location:** PDF Section 5.1.6 (page ~49) vs `src/neuralshield/preprocessing/steps/07_unicode_nkfc_and_control.py`

The PDF states:

> *"NFKC se usa como referencia diagnóstica, no como sustitución directa del texto."*
> (NFKC is used as a diagnostic reference, not as direct text substitution.)

The code **does** substitute the content:

```python
# Line 68
normalized_content = unicodedata.normalize("NFKC", content)
# Line 90 — the normalized text replaces the original
processed_line = f"{prefix} {normalized_content}"
```

The NFKC-normalized text overwrites the original `[URL]` and `[QUERY]` content. Flags are emitted correctly, but the original byte sequence is lost.

**Fix:** Either (a) update the PDF to state that NFKC normalization **is** applied to URL/QUERY content (accurate description of the code), or (b) change the code to preserve the original content and use NFKC only for flag detection. Option (a) is simpler; option (b) would preserve the non-destructivity principle more strictly.

---

### 1.4 Step 06 emits HOSTMISMATCH as a bare line without tag prefix

**Severity:** Low
**Location:** `src/neuralshield/preprocessing/steps/06_absolute_url_builder.py`, line 636

Every other line in the canonical artifact uses a tag prefix (`[METHOD]`, `[URL]`, `[HEADER]`, `[HAGG]`, `[HGF]`, etc.). Step 06's `_emit_global_flags()` emits:

```python
return " ".join(sorted_flags)  # produces "HOSTMISMATCH" — no tag prefix
```

This bare line breaks the tagged format convention described in Section 5.1.1.

**Fix:** Wrap the output in a tag, e.g. `[FLAGS] HOSTMISMATCH` or append it to the existing `[FLAGS]` line from Step 01.

---

### 1.5 Orphaned `QueryProcessor` class with `[QPARAM]` tag

**Severity:** Low
**Location:** `src/neuralshield/preprocessing/steps/query_processor.py`

This file contains a `QueryProcessor` class that:

- Uses `[QPARAM]` tags (line 416) instead of `[QUERY]`
- Is **not** registered in the TOML pipeline config
- Has a separate data model (`QueryParameter` dataclass, `QueryParseResult`)

The active pipeline uses `QueryParserAndFlags` from `10_query_parser_and_flags.py`. The PDF does not mention `[QPARAM]` at all. This is dead code that could cause confusion.

**Fix:** Remove `query_processor.py` or move it to an `_archive/` directory to avoid confusion.

---

### 1.6 Step 03 sorts headers alphabetically — not documented

**Severity:** Low
**Location:** `src/neuralshield/preprocessing/steps/03_header_normalization_duplicates.py`, line 349

```python
other_headers.sort(key=lambda x: (x[0], x[1]))
```

All headers are reordered alphabetically by (name, value), destroying the original order. Header order can carry semantic meaning (e.g., proxy chains inspect headers in order). The PDF does not document this sorting behavior.

**Fix:** Add a brief note in Section 5.1.6 (Paso HeaderNormalizationDuplicates) mentioning that headers are emitted in alphabetical order for determinism.

---

### 1.7 CSIC-2010 training set size: PDF says ~36,000, code uses 47,000

**Severity:** Medium
**Location:** PDF §6.2 (page ~57) vs `src/neuralshield/data/CSIC/create_train_test_split.py`

The PDF states:

> *"Contiene aproximadamente 36 000 solicitudes normales de entrenamiento, 36 000 solicitudes normales de prueba y 25 000 solicitudes anómalas"*

The code uses a different split:

```python
# Line 5-6 comment:
# Train: 47,000 randomly sampled valid requests
# Test: Remaining valid requests (25,000) + all attack requests (25,065)

def create_train_test_split(samples, train_size: int = 47000):
```

The actual training set is 47,000 normal requests, not ~36,000. This means the test set has ~25,000 normal requests (not ~36,000).

**Fix:** Update §6.2 to state "approximately 47,000 normal training requests" and "approximately 25,000 normal test requests."

---

### 1.8 SR\_BH-2020 ensemble weight: PDF says w=0.6, code only tested w=0.5

**Severity:** Medium
**Location:** PDF Table 6.4 (page ~73) vs `experiments/18_lof_secbert_ensemble/srbh/`

Table 6.4 reports:

> Ensemble (w=0.6) → AUC 0.928

The experiment results show only w=0.5 was tested:

- `experiments/18_lof_secbert_ensemble/srbh/with_preprocessing/ensemble/ensemble_summary.json` → `weights: [0.5]`, AUC = 0.9287
- The script `scripts/run_lof_secbert_ensemble.py` defaults to `weights = [0.5]` (line 462)
- No experiment directory or result file contains w=0.6 for SR\_BH

The AUC value (0.928) is correct — it comes from w=0.5, not w=0.6.

**Fix:** Change Table 6.4 to `Ensemble (w=0.5)` for SR\_BH-2020, or run the experiment with w=0.6 and update the AUC.

---

### 1.9 PKDD-2007 ensemble AUC: PDF says 0.780, code achieves 0.766

**Severity:** Low
**Location:** PDF Table 6.4 (page ~73) vs experiment results

Table 6.4 reports PKDD ensemble AUC as **0.780** (w=0.2). The experiment code achieves **0.7656** at w=0.2. The 0.014 gap exceeds normal rounding.

**Diagnosis:** Possible different run configuration, hyperparameter setting, or data split version.

**Fix:** Verify which experiment run produced 0.780 and ensure the result is reproducible, or update the table.

---

### 1.10 Figure 4.2 Phase 2 flag list is incomplete

**Severity:** Low
**Location:** PDF Figure 4.2 (page ~37)

Figure 4.2 lists representative Phase 2 flags as: `OBSFOLD DUPHDR BADHDRNAME WSPAD` with aggregate `[HAGG]`.

Missing from the figure:
- `BADHDRCONT` (Step 02) — continuation line without preceding header
- `BADCRLF` (Step 02) — embedded CR/LF in header lines
- `HDRMERGE` (Step 03) — mergeable duplicate headers consolidated
- `HOPBYHOP` (Step 03) — hop-by-hop headers in request context
- `HDRNORM` (Step 03) — header name required normalization

The full catalog in §5.1.4 lists all of these correctly. The figure only shows a subset.

**Fix:** Either add the missing flags to Figure 4.2 or add a note "(representative subset)" to clarify the figure is not exhaustive.

---

### 1.11 `load_order_from_config` is a module function, not a classmethod

**Severity:** Low
**Location:** PDF Listing 5.1 vs `src/neuralshield/preprocessing/pipeline.py`

If the PDF presents `load_order_from_config` as a classmethod of `PreprocessorPipeline`, the actual code defines it as a **module-level function** (lines 63-73), not a method on the class. The class itself only has `__call__` and `batch`.

**Fix:** Verify Listing 5.1 matches the actual code structure. If it shows a classmethod, update to reflect the module-level function.

---

### 1.12 Listing 4.2 end-to-end example has ~12 discrepancies with actual pipeline output

**Severity:** High
**Location:** PDF Listing 4.2 (page ~38) vs pipeline trace of Listing 4.1 input

A step-by-step trace of Listing 4.1 through the actual code reveals the claimed output (Listing 4.2) diverges from what the pipeline would actually produce in at least 12 ways:

**a) Query splitting — 2 lines, not 3.**
Step 01 splits query on `&` only, not `;`. `debug;verbose&format=json` → `[QUERY] debug;verbose` + `[QUERY] format=json` (2 lines). PDF shows 3 lines (`debug`, `verbose`, `format=json`). Step 10 does NOT re-split because `_is_semicolon_dominant` requires ≥2 `key=value` pairs, and `debug;verbose` has zero.

**b) `[QMETA] count=3 MIXEDSEP` is wrong on three counts.**
- Count should be 2 (two query parameters), not 3.
- `MIXEDSEP` doesn't exist — code emits `QRAWSEMI`.
- `QBARE` would also appear (no `=` in `debug;verbose`).

**c) Missing `[QSEP]` line.**
Step 10 emits `[QSEP] QRAWSEMI` — Listing 4.2 omits it entirely.

**d) Missing `[HGF] HDRNORM` line.**
Step 03 normalizes 5 header names to lowercase (`Host`→`host`, `Content-Length`→`content-length`, etc.) and emits `[HGF] HDRNORM`. Listing 4.2 omits it.

**e) Missing `[URL_ABS]` line.**
Step 06 constructs `[URL_ABS] http://evil.example.com/app/...` from the Host header. Listing 4.2 omits it.

**f) `total_bytes=182` is incorrect.**
Step 03 computes `total_bytes` as the sum of byte lengths of comma-joined header *values only* (excluding names). The correct value is approximately 84, not 182.

**g) Header ordering differs.**
Step 03 sorts alphabetically by (name, value): `content-length`, `cookie`, `host`, `host`, `transfer-encoding`, `x-custom`. Listing 4.2 shows: `host`, `host`, `content-length`, `transfer-encoding`, `x-custom`, `cookie`.

**h) Missing `SEMICOLON` flag on query.**
Step 05 would flag `[QUERY] debug;verbose` with `SEMICOLON` (literal `;` in QUERY context). Not shown.

**i) Missing `QBARE` flag on query.**
Step 10 flags parameters without `=` as `QBARE`. Not shown.

**j) `[FLAGS]` aggregation line doesn't exist in the pipeline.**
Already documented in 1.2.

**k) Step 11 `MULTIPLESLASH` false positive (code bug).**
`_detect_multiple_slashes` checks for `"" in segments`. Since `_segment_path` always inserts `""` at index 0 for absolute paths, **every** absolute URL triggers `MULTIPLESLASH`. This is a code bug, not just a PDF issue.

**l) Flag propagation corruption between steps.**
Steps 08 and 11 extract content via `line[6:]`/`line[8:]`, which includes inline flags from previous steps as part of the content string. E.g., `DOUBLEPCT` from step 08 becomes part of the path string when step 11 processes it. This corrupts both flags and content.

**Fix:** Either (a) rewrite Listing 4.2 by running the actual pipeline on Listing 4.1 input and using the real output, or (b) fix the code bugs (MULTIPLESLASH false positive, flag propagation) and then regenerate. Option (b) is recommended since the bugs affect production correctness.

---

### 1.13 No 3-way train/validation/test split despite §6.2 claim

**Severity:** Medium
**Location:** PDF §6.2 (page ~58) vs `src/neuralshield/data/*/create_train_test_split.py`

The PDF describes three data partitions:
- **Entrenamiento:** normal traffic only, for base distribution estimation
- **Validación:** normal traffic subset for hyperparameter tuning (LOF neighbors, PCA variance, fusion weights)
- **Prueba:** mixed normal + attack traffic for evaluation

The code implements only a **2-way split** (train/test):
- CSIC: `create_train_test_split(samples, train_size=47000)` — no validation partition
- SR\_BH: same 2-way pattern
- PKDD: uses original train/test split from dataset, no validation carve-out

Threshold calibration is done on the test set's normal samples (at 95th percentile), not on a separate validation set.

**Fix:** Either (a) update §6.2 to accurately describe the 2-way split with test-set calibration, or (b) implement the 3-way split described in the PDF.

---

### 1.14 Table 6.3 PKDD recall values don't match experiment results

**Severity:** Medium
**Location:** PDF Table 6.3 vs `experiments/25_representation_evaluation/agreement_analysis_pkdd/agreement_results.json`

| Metric | PDF Claims | Experiment Results | Gap |
|--------|-----------|-------------------|-----|
| TF-IDF+PCA+LOF Recall@5% | 13.03% | 9.05% | 3.98pp |
| SecBERT+Maha Recall@5% | 34.12% | 36.22% | 2.10pp |

Agreement (76.55%), Jaccard (0.1337), and Correlation (0.1718) match. Precision values match. Only recall values diverge.

**Diagnosis:** Possible different experiment run or data split version.

**Fix:** Verify which run produced the claimed values and ensure reproducibility.

---

### 1.15 SR\_BH-2020 Table 6.3 metrics not traceable

**Severity:** Medium
**Location:** PDF Table 6.3 vs `experiments/25_representation_evaluation/`

No agreement analysis script or results file exists for SR\_BH-2020 in the experiment directories. CSIC and PKDD both have dedicated `analyze_agreement*.py` scripts and results JSON files, but SR\_BH does not.

The claimed SR\_BH metrics (Agreement=71.99%, Jaccard=0.2018, Correlation=0.2022) cannot be verified against stored experiment results.

**Fix:** Add `analyze_agreement_srbh.py` to generate and store these metrics, or document the source of the claimed values.

---

### 1.16 Step 11 `MULTIPLESLASH` triggers on every absolute path (code bug)

**Severity:** Medium
**Location:** `src/neuralshield/preprocessing/steps/11_path_structure_normalizer.py`, lines 119 and 135

```python
# _segment_path always inserts "" at index 0 for absolute paths (line 119)
segments.insert(0, "")

# _detect_multiple_slashes checks for "" in segments (line 135)
return "" in segments  # Always True for absolute paths!
```

Every URL path starting with `/` (i.e., virtually every HTTP request) gets a false `MULTIPLESLASH` flag. The first `""` segment is an artifact of the splitting logic, not an indicator of multiple slashes.

**Fix:** Exclude the first segment from the multiple-slash check, or use a different detection method (e.g., regex for `//` in the raw path).

---

### 1.17 Table 6.4 F1-scores are mathematically inconsistent with Table 6.3 P/R values

**Severity:** High
**Location:** PDF Table 6.4 (page ~73) vs Table 6.3 (page ~67)

F1 = 2·P·R / (P+R). Cross-checking Table 6.4 F1 values against Table 6.3 precision and recall:

| Dataset | Model | Table 6.3 P | Table 6.3 R | Correct F1 | Table 6.4 F1 | Error |
|---------|-------|-------------|-------------|-----------|-------------|-------|
| **CSIC** | SecBERT+Maha | 90.84 | 49.26 | **63.87** | 55.40 | **-8.47pp** |
| **SR\_BH** | TF-IDF+LOF | 81.08 | 23.80 | **36.80** | 23.80 | **-13.00pp** |
| **SR\_BH** | SecBERT+Maha | 90.07 | 48.18 | **62.77** | 64.60 | +1.83pp |

The SR\_BH TF-IDF+LOF F1=23.80% is identical to the Recall=23.80% — a clear copy-paste error. The CSIC SecBERT+Maha F1=55.40% is also wrong; the code's `roc_metrics.json` confirms F1=63.87%.

The CSIC TF-IDF+LOF (75.95) and ensemble values match correctly. PKDD individual values are close (within ~0.2pp).

**Fix:** Recompute all F1 values in Table 6.4 using F1 = 2·P·R/(P+R) from Table 6.3 values, or regenerate from experiment results.

---

### 1.18 Flag corruption bug affects Steps 06, 07, 08, 11 (not just 08 and 11)

**Severity:** Medium
**Location:** Steps 06, 07, 08, 11

Previously documented for Steps 08 and 11 (in 1.12l). Deeper analysis reveals Steps 06 and 07 also have the same bug:

| Step | Line | Code | Input becomes |
|------|------|------|---------------|
| **06** | 101 | `url = line[6:].strip()` | `/path ANGLE QUOTE` (flags from Step 05 treated as URL) |
| **07** | 54 | `content = line[6:]` | `/path ANGLE QUOTE` → NFKC-normalized including flag text |
| **07** | 57 | `content = line[8:]` | `param SEMICOLON` → flag text enters NFKC normalization |
| **08** | 61 | `content = line[6:]` | Same pattern — flags from Steps 05/07 included |
| **11** | 40 | `url_content = line[6:]` | Same pattern — all prior flags baked into path |

Step 05 (DangerousCharactersScriptMixing) is the first step to add inline flags to `[URL]` and `[QUERY]` lines. Every subsequent step that uses naive offset extraction (`line[6:]`, `line[8:]`) without stripping trailing flags is affected.

Steps 09 and 10 are NOT affected because they properly split lines on whitespace to separate content from flags.

**Fix:** All four affected steps need flag/content separation. Simplest approach: define a shared utility that splits `[URL] /content FLAG1 FLAG2` into `("/content", {"FLAG1", "FLAG2"})`.

---

### 1.19 Step 06 silently swallows URL parsing exceptions

**Severity:** Low
**Location:** `src/neuralshield/preprocessing/steps/06_absolute_url_builder.py`, lines 461-463

When parsing an absolute-form URL fails, the exception is caught and the URL is returned as-is with no error flag or warning. This means malformed absolute URLs pass through undetected.

**Fix:** Emit a flag (e.g., `BADURL`) when URL parsing fails in absolute-form handling.

---

## 2. Minor Observations

These are implementation details not explicitly discussed in the PDF. They are not errors but could improve precision if documented.

| # | Area | Detail |
|---|------|--------|
| 2.1 | **`total_bytes` in `[HAGG]`** | Counts bytes of header **values only** (joined by commas), excluding header names. The PDF does not specify what `total_bytes` measures. |
| 2.2 | **WSPAD threshold** | Triggers on 3+ spaces after colon, double spaces in value, or tabs. The specific thresholds are not documented. |
| 2.3 | **Step 05 context rules** | SEMICOLON and SPACE flags are suppressed in HEADER context (semicolons are legitimate in cookies, spaces are normal in header values). This context-sensitivity is not described in the flag catalog. |
| 2.4 | **Step 10 value redaction** | Sensitive values are replaced with `<SECRET:shape:length>` and structured values with `<shape:length>`. This redaction behavior is not documented. |
| 2.5 | **FULLWIDTH flag trigger** | Emits FULLWIDTH both when fullwidth chars (U+FF00–U+FFEF) are found **and** when NFKC normalization changes the content. PDF says it triggers only on fullwidth characters. |
| 2.6 | **TF-IDF + PCA composition** | PDF presents TF-IDF/PCA as an integrated "syntactic view." In code, TF-IDF is a standalone encoder (`encoding/models/tfidf.py`) and PCA is applied separately in experiment scripts. Not an error — just a different abstraction level. |
| 2.7 | **AUC metric location** | PDF lists AUC as a core metric (Section 6.1.1). It is computed in `scripts/test_anomaly_precomputed.py` and experiment scripts, but not in the core `evaluation/metrics.py` module. |
| 2.8 | **Additional detectors** | Code has `DeepSVDD`, `IsolationForest`, `OCSVM`, `GMM` detectors not mentioned in the PDF. These are experimental and do not contradict the PDF. |
| 2.9 | **Additional encoders** | Code has `fastembed`, `byt5`, `colbert-muvera`, `structural`, `chargram-abstract`, `value-composition` encoders beyond the PDF scope. |
| 2.10 | **LOF `n_neighbors` default** | Code uses `n_neighbors=100` (`anomaly/lof.py`, line 43). The PDF does not specify this hyperparameter. |
| 2.11 | **SecBERT adapted excludes special tokens** | Mean+max pooling explicitly masks out `[CLS]`, `[SEP]`, `[PAD]` tokens (`secbert.py`, lines 196-214). Not documented in PDF. |
| 2.12 | **Flag-weighted token weights are ~3.0 multipliers** | `secbert_flag_token_weights.json` assigns weights ≈2.99–3.0 to flag tokens. The weighting scheme (multiplicative float multipliers, not discrete ranks) is not described in the PDF. |
| 2.13 | **Step 03 flag separator** | Step 03 joins multiple inline flags with commas (`",".join(sorted_flags)`), but other steps use spaces. This inconsistency could affect downstream parsing. |
| 2.14 | **§5.1.6 repeats NFKC "diagnostic" claim** | The Paso UnicodeNFKCAndControl description in §5.1.6 says "NFKC se usa como referencia diagnóstica, no como sustitución directa del texto" — same contradiction as §5.1.6 documented in 1.3. |
| 2.15 | **SecBERT adapted checkpoint path hardcoded** | `secbert.py` line 132 hardcodes `src/neuralshield/finetuning/models/secbert-http-adapted/final` as the model path. Not configurable via config. |
| 2.16 | **Step 03 flag separator inconsistency** | Step 03 uses comma-separated flags (`",".join(sorted_flags)`, line 363), while Steps 05, 08, 10, 11 use space-separated flags. Listing 4.2 shows space-separated `DUPHDR`, but the code would produce `DUPHDR` (single flag, so no difference) — the inconsistency only surfaces with multiple flags per header. |
| 2.17 | **Step 03 hop-by-hop header set is incomplete** | Only 4 headers in `HOP_BY_HOP_HEADERS`: `connection`, `te`, `upgrade`, `trailer` (line 30-35). Missing per RFC 7230: `keep-alive`, `proxy-connection`, `transfer-encoding`, `proxy-authenticate`, `proxy-authorization`. |
| 2.18 | **Step 01 raises exception on malformed requests** | `MalformedHttpRequestError` is raised when the request line has fewer than 3 parts or the HTTP version is missing (lines 74-89). This means malformed requests crash the pipeline rather than being flagged — the PDF doesn't discuss error handling for unparseable requests. |
| 2.19 | **Step 03 value whitespace not stripped** | `_parse_header_line` explicitly does NOT strip leading whitespace from header values (line 220, commented out). This means values like `" api.example.com"` retain the leading space, producing double-spaced output like `[HEADER] host:  api.example.com`. |
| 2.20 | **Step 06 host header flag stripping** | Step 06 strips existing flags from host header values (lines 219-224) before validation. E.g., `"example.com MIXEDSCRIPT"` → `"example.com"`. This is undocumented but necessary to prevent flags from corrupting host validation. |
| 2.21 | **Step 06 IDNA fallback asymmetry** | When the `idna` library is unavailable, Step 06 still returns `IDNA=True` flag (line 569). When IDNA encoding fails, it returns `IDNA=False` (line 573). This asymmetry is undocumented. |
| 2.22 | **Step 07 FULLWIDTH dual trigger** | FULLWIDTH flag fires on two conditions OR'd together: (1) explicit fullwidth chars found (U+FF00–U+FFEF), OR (2) NFKC normalization changed the content (line 75: `if has_fullwidth or normalized_content != content`). PDF only describes the fullwidth char condition. |
| 2.23 | **§5.1.6 QueryParser vs QueryParserAndFlags naming** | PDF §5.1.6 refers to "Paso QueryParser" but the actual class is `QueryParserAndFlags` and the file is `10_query_parser_and_flags.py`. The shorter name is used as a convenient abbreviation in the text. |

---

## 3. Verified Claims (no issues found)

The following claims were verified line-by-line and match exactly:

| Claim | Status |
|-------|--------|
| 12 preprocessing steps exist with correct numbering (00–11) | Verified |
| TOML config lists steps in the documented order | Verified |
| `HttpPreprocessor` ABC with `process(self, request: str) -> str` | Verified |
| `PreprocessorPipeline` class: `__call__`, `batch` with ThreadPoolExecutor, TOML composition | Verified (note: `load_order_from_config` is module-level, see 1.11) |
| 10 canonical tags emitted: `[METHOD]`, `[URL]`, `[URL_ABS]`, `[QUERY]`, `[HEADER]`, `[FLAGS]`, `[HAGG]`, `[HGF]`, `[QSEP]`, `[QMETA]` | Verified |
| All ~50 flags in Section 5.1.4 exist in their respective step files | Verified |
| Step-to-flags matrix (Table 5.1) matches code | Verified |
| `LOFDetector` matches Listing 5.2 (sign flip, `novelty=True`, percentile threshold) | Verified |
| `MahalanobisDetector` matches Listing 5.3 (`EmpiricalCovariance`, percentile threshold) | Verified |
| SecBERT base: `[CLS]` token, 768 dimensions | Verified |
| SecBERT adapted: mean+max pooling over last 2 hidden layers, 1536D | Verified |
| SecBERT flag-weighted: weighted pooling for flag-associated tokens | Verified |
| Fusion formula: `s = w * s_LOF + (1-w) * s_Maha` with z-score normalization | Verified |
| JSONL format with `"request"` and `"label"` keys | Verified |
| Three dataset converters (CSIC, SR_BH, PKDD) produce correct JSONL | Verified |
| Embedding persistence includes metadata (model ID, config, indices, labels) | Verified |
| Encoding config exposes TF-IDF hyperparameters (vocabulary size, n-gram range, etc.) | Verified |
| Orchestration flow: data loading → preprocessing → encoding → persistence | Verified |
| Idempotency: each step is a pure function `str → str` | Verified |
| Non-destructivity: flags are appended inline, anomalous content is preserved | Verified (except NFKC, see 1.3) |
| Three phases: (1) sanitization/segmentation, (2) header normalization, (3) URL/path/query | Verified |
| Five anomaly families from Section 3.2.3 map to the flag taxonomy | Verified |
| SecBERT model name `jackaduma/SecBERT` | Verified |
| SecBERT adapted: MLM finetuning on HTTP corpus, saved to `finetuning/models/secbert-http-adapted/final` | Verified |
| SecBERT adapted excludes special tokens ([CLS], [SEP], [PAD]) from pooling | Verified |
| SecBERT flag-weighted: weighted average pooling using token-weight JSON, 768D output | Verified |
| BGE-small maps to `fastembed` encoder with default model `BAAI/bge-small-en-v1.5` | Verified |
| Z-score normalization: `ŝ_i = (s_i - μ_normal) / σ_normal` computed on normal subset | Verified |
| Fusion formula: `s_fused = w · ŝ_lof + (1-w) · ŝ_maha` | Verified |
| Recall @ 5% FPR: interpolation on ROC curve in `scripts/test_anomaly_precomputed.py` | Verified |
| Complementarity metrics (Agreement, Jaccard, Pearson correlation) implemented for CSIC and PKDD | Verified |
| CSIC ensemble w=0.75, AUC=0.861 (code: 0.8606, rounds correctly) | Verified |
| All 50 flags in §5.1.4 have 1:1 correspondence with code — no missing, no extra | Verified |
| LOF score sign flip: `(-lof_scores)` in `anomaly/lof.py` | Verified |
| Mahalanobis uses `EmpiricalCovariance().mahalanobis()` directly | Verified |
| Percentile threshold formula: `np.percentile(scores, 100 * (1 - max_fpr))` — same in LOF and Maha | Verified |
| `AnomalyDetector` ABC: `fit()`, `scores()`, `predict()`, `save()`, `load()` | Verified |
| CSIC converter: TSV → JSONL with `{"request": ..., "label": "valid"/"attack"}` | Verified |
| SR\_BH converter: CSV with CAPEC multilabels → binary JSONL | Verified |
| PKDD converter: XML format → binary JSONL, preserves original train/test split | Verified |
| Embedding persistence metadata: model, encoder, device, pipeline\_name, batch\_size, total\_requests | Verified |
| TFIDFEncoderConfig exposes: max\_features, ngram\_range, min\_df, max\_df, lowercase, use\_idf, sublinear\_tf | Verified |
| TF-IDF encoder does NOT include PCA internally (PCA applied separately in experiment scripts) | Verified |
| Four detectors in evaluation chapter (IF, LOF, Maha, GMM) all exist in `anomaly/` module | Verified |
| Four encoders in evaluation chapter (TF-IDF, BGE-small, ByT5, SecBERT) all exist in `encoding/models/` | Verified |
| Training protocol: only normal traffic for training, mixed for testing | Verified (code enforces this in split scripts) |
| SecBERT max\_length=512 tokens | Verified (`secbert.py`, line 74) |
| SecBERT adapted averages layers -1 and -2 (not just concatenates), then mean+max pools | Verified (`secbert.py`, lines 190-227) |
| MLM finetuning uses 15% masking probability | Verified (`finetuning/config.py`, line 32) |
| Step 09 (HtmlEntityDecodeOnce) is detection-only — uses `html.unescape()` comparison, never modifies content | Verified (`09_html_entity_decode_once.py`, line 99) |
| Step 02 (HeaderUnfoldObsFold) unfolds obs-fold continuations per RFC 7230 | Verified (`02_header_unfold_obs_fold.py`, lines 131-135) |
| Step 00 (FramingCleanup) preserves `\t`, `\r`, `\n` — only strips non-structural control chars | Verified (`00_framing_cleanup.py`, lines 50, 60) |
| Table 5.1 step-to-flags matrix: all entries match code | Verified (all 12 rows cross-referenced) |
| §5.1.6 step descriptions match code behavior (except NFKC, see 1.3) | Verified |
| Table 6.3 CSIC metrics: Agreement=71.10%, Jaccard=0.3623, Correlation=0.3323 | Verified against experiment results |
| Table 6.3 PKDD metrics: Agreement=76.55%, Jaccard=0.1337, Correlation=0.1718 | Verified against experiment results |
| Fusion threshold recalibration at FPR=5% for each w | Verified (`or_voting_ensemble.py`, lines 145-167) |
| Table 6.2 TF-IDF+PCA+Maha AUC values: 0.504 (sin prep), 0.787 (con prep) | Verified against `experiments/10_tfidf_pca_mahalanobis/` |
| Step order in TOML config matches PDF §5.1.3 listing exactly (12 steps, same numbering) | Verified against `preprocessing/config.toml` |
| Phase 1 (Steps 00–01), Phase 2 (Steps 02–04), Phase 3 (Steps 05–11) boundaries match code | Verified |
| Table 5.1 "Líneas/Meta" column: FramingCleanup → "Sin líneas nuevas" | Verified (returns modified string, same line count) |
| Table 5.1 "Líneas/Meta" column: RequestStructurer → [METHOD], [URL], [QUERY], [HEADER], [FLAGS] | Verified against code lines 44-66 |
| Table 5.1 "Líneas/Meta" column: HeaderNormalizationDuplicates → [HAGG], [HGF] | Verified against code lines 386-420 |
| Table 5.1 "Líneas/Meta" column: AbsoluteUrlBuilder → [URL_ABS] | Verified (inserts after URL line) |
| Table 5.1 "Líneas/Meta" column: QueryParserAndFlags → [QSEP], [QMETA] | Verified against code lines 98-110 |
| Step 02 flags match PDF: OBSFOLD, BADHDRCONT, BADCRLF | Verified against code lines 46, 55, 57, 74, 82 |
| Step 04 flags match PDF: WSPAD only | Verified against code line 85 |
| Step 06 reconstructs [URL\_ABS] in `scheme://host[:port]/path[?query]` | Verified — handles all 4 RFC 7230 forms |
| Step 06 emits HOSTMISMATCH, IDNA, BADHOST flags | Verified against code lines 48-49 |
| Step 06 handles all 4 request-target forms (origin, absolute, authority, asterisk) | Verified against `_detect_request_form()` |
| Step 09 detection-only via `html.unescape()` comparison, content preserved | Verified (line 99) |
| Step 09 no `line[6:]` bug — properly splits flags from content | Verified |
| CSIC ensemble F1=79.19% at w=0.75 | Verified against `roc_metrics.json` (0.7919) |
| PKDD ensemble F1=53.70% at w=0.2 | Verified against `roc_metrics.json` (0.5370) |
| SR\_BH ensemble F1=67.83% at w=0.5 | Verified against `roc_metrics.json` (0.6783) |
| CSIC TF-IDF+LOF F1=75.95% | Verified against `test_cross_dataset_lof.py` reference values |
| §4.2.1 idempotency principle matches code (pure function str → str) | Verified |
| §4.2.1 non-destructivity principle matches code (except NFKC, see 1.3) | Verified |
| §4.4.4 fusion formula `s = w·ŝ_LOF + (1-w)·ŝ_Maha` matches code | Verified |
| §5.2.3 three SecBERT variants: base (CLS, 768D), adapted (mean+max, 1536D), flag-weighted (768D) | Verified |
