# PDF-vs-Code Audit Fix Summary

## Context

A systematic 5-pass audit compared the thesis PDF (`neuralshield-docs`, March 4, 2026) against the actual codebase across Chapters 4 (Design), 5 (Implementation), and 6 (Evaluation).

**Results:** 19 discrepancies identified, 23 minor observations, 61 verified claims.

The guiding principle was **"fix the code to match the PDF"** wherever possible, to minimize document changes. This commit addresses the 9 highest-priority fixes.

Full audit findings: [`docs/PDF_CODE_AUDIT.md`](./PDF_CODE_AUDIT.md)

---

## Fixes Applied

### 1.1 — MIXEDSEP flag missing (Step 10)

**Problem:** PDF Table 4.3 lists `MIXEDSEP` as a flag emitted when both `&` and `;` separators are present in a query string. The code never emitted it.

**Fix:** Added `MIXEDSEP` emission in `_detect_separator_type()` when `ampersand_count > 0 and semicolon_count > 0`.

**File:** `steps/10_query_parser_and_flags.py`

---

### 1.2 — [FLAGS] aggregation missing (Pipeline)

**Problem:** PDF describes a consolidated `[FLAGS]` summary line at the end of the artifact, but the pipeline never aggregated inline flags from tagged lines.

**Fix:** Added `_aggregate_flags()` static method to `PreprocessorPipeline`. Called at the end of `__call__()`, it scans all tagged lines (`[URL]`, `[QUERY]`, `[HEADER]`, `[HGF]`) for inline flags and consolidates them into a single `[FLAGS]` summary line.

**File:** `preprocessing/pipeline.py`

---

### 1.3b — NFKC was destructive, should be diagnostic (Step 07)

**Problem:** PDF Section 4.3.2 describes NFKC normalization as a detection mechanism (flags only). The code was destructively replacing content with the NFKC-normalized form, losing the original attack payload.

**Fix:** Rewrote Step 07 so NFKC is used only for comparison — `unicodedata.normalize("NFKC", content)` is compared against the original to detect anomalies and emit flags (`FULLWIDTH`, `CONTROL`, etc.), but the original content is preserved in the output.

**File:** `steps/07_unicode_nkfc_and_control.py`

---

### 1.4 — HOSTMISMATCH emitted without [FLAGS] prefix (Step 06)

**Problem:** `_emit_global_flags()` returned bare flag text (e.g., `HOSTMISMATCH`) instead of a properly tagged line.

**Fix:** Changed return to `f"[FLAGS] {' '.join(sorted_flags)}"`.

**File:** `steps/06_absolute_url_builder.py`

---

### 1.5 — Orphaned query_processor.py (Dead Code)

**Problem:** `query_processor.py` used `[QPARAM]` tags not referenced anywhere in the pipeline. Not listed in TOML config. Dead code that could cause confusion.

**Fix:** Deleted the file.

---

### 1.10 — Figure 4.2 Phase 2 flag list incomplete (TODO)

**Problem:** Figure 4.2 in the PDF omits several flags that the code actually emits: `BADHDRCONT`, `BADCRLF`, `HDRMERGE`, `HOPBYHOP`, `HDRNORM`.

**Fix:** Added TODO comment in `pipeline.py` to verify and update the figure.

---

### 1.12 — Listing 4.2 end-to-end example stale (TODO)

**Problem:** The walkthrough example in Listing 4.2 may not reflect actual pipeline output after these bug fixes.

**Fix:** Added TODO comment in `pipeline.py` to regenerate the example from actual pipeline output.

---

### 1.16 — MULTIPLESLASH false positive on all absolute paths (Step 11)

**Problem:** `_segment_path()` always inserts `""` at index 0 for absolute paths (to represent the leading `/`). `_detect_multiple_slashes()` checked `"" in segments`, which was **always True** for any absolute path — causing every URL to be falsely flagged as `MULTIPLESLASH`.

**Fix:** Changed to `"" in segments[1:]`, skipping the leading empty segment.

**File:** `steps/11_path_structure_normalizer.py`

---

### 1.18 — Flag corruption across 4 steps (Steps 06, 07, 08, 11)

**Problem:** Four steps used naive `line[6:]` or `line[8:]` slicing to extract content from tagged lines. When prior steps appended inline flags (e.g., `[URL] /path ANGLE QUOTE`), downstream steps would treat `ANGLE QUOTE` as part of the URL content, corrupting processing.

**Fix:**
1. Added `split_line_content(line, prefix)` utility to `http_preprocessor.py`. It walks backwards from the end of the line, collecting tokens that match known flags, and returns a clean `(content, flags)` tuple.
2. Added `_KNOWN_FLAGS` set and `_PARAMETRIC_FLAG_RE` pattern to identify all ~50 flags.
3. Updated all 4 affected steps to use `split_line_content()` and merge flags back after processing.

**Files:**
- `preprocessing/http_preprocessor.py` (new utility)
- `steps/06_absolute_url_builder.py`
- `steps/07_unicode_nkfc_and_control.py`
- `steps/08_percent_decode_once.py`
- `steps/11_path_structure_normalizer.py`

---

## Remaining Items

| ID | Description | Status |
|----|-------------|--------|
| 1.10 | Figure 4.2 flag list update | TODO in code |
| 1.12 | Listing 4.2 regeneration | TODO in code |
| 1.13 | 3-way vs 2-way data split | Pending decision (recommend PDF update) |
| 1.17 | Table 6.4 F1 math errors | Requires PDF correction |
