# Preprocessing - Implementation Plan (pending/partial items)

Date: 2026-03-07

Inputs:
- repo/docs/AUDIT_FIX_SUMMARY.md (PDF-vs-code audit summary)
- NeuralShield-docs/docs/audits/preprocessing-findings.md (repo-only findings + target contract)

Goal:
1) Close all items that are currently Pending or Partial in preprocessing-findings.md, or explicitly resolve them via a documented decision when they imply a contract change.
2) Keep the thesis and the repo aligned; prefer fixing code to match the PDF when feasible, but avoid silent contract drift.
3) Improve reproducibility: tests + a repeatable way to regenerate PDF listings/figures from real pipeline output.

Non-goals:
- Redesign the whole preprocessing DSL unless we explicitly choose that contract.

---

## Contract decision (blocks multiple items)

D0 - Flag encoding in artifact lines

Option A (recommended for compatibility with current PDF):
- Keep current artifact format:
  - Inline flags as space-separated tokens at end of [URL]/[QUERY]/[HEADER] lines.
  - One final consolidated [FLAGS] line that aggregates all emitted flags.

Option B (preprocessing-findings F01 target contract):
- Migrate to explicit suffix:
  - `[TAG] <content> [FLAGS:flag1,flag2,...]`
- Requires broad repo changes + PDF updates.

This plan assumes Option A unless stated otherwise.

---

## Work packages

### WP1 - Robust flag parsing + consistent header-flag emission (F01 partial, F12 partial)

Problem:
- split_line_content() exists and fixes some corruptions, but multiple steps still parse with naive split/join.
- Step 03 currently emits header flags as CSV (e.g. `BADHDRNAME,DUPHDR`), while the pipeline aggregator collects only space-separated tokens. This can hide header-level flags from the final [FLAGS] line.

Changes:
1) Standardize header flag emission to be space-separated tokens (no CSV) OR teach the aggregator to parse CSV tokens.
2) Centralize "content vs flags" parsing so no step treats flags as content.

Acceptance:
- Final [FLAGS] includes header-level flags when present (HOPBYHOP, BADHDRNAME, DUPHDR, HDRMERGE, HDRNORM, etc.).
- No step reintroduces the "flags as content" corruption class.

Notes:
- If we keep CSV for some reason, define it as a supported legacy format and handle it everywhere consistently.

---

### WP2 - QueryParser does not introduce literal spaces or NUL into output (F06, F07)

Problem:
- Step 10 uses unquote(value) and then emits decoded_value as display_value.
- This can introduce literal spaces (from %20) and literal NUL (from %00) into the artifact.

Changes:
- Separate analysis_value vs display_value:
  - analysis_value: decoded for detection only
  - display_value: stable text representation for emission (do not introduce literal spaces/NUL)
- Keep shaping/redaction, but ensure any substitution is explicit (e.g., <SECRET:...>, <shape:len>).

Acceptance:
- `x=%2520` does not become `x=<space>` in emitted artifact.
- `nul=%00` does not include a real NUL byte in emitted artifact.

---

### WP3 - DOUBLEPCT detection covers "dangerous" encodings (F08)

Problem:
- Step 08 marks DOUBLEPCT only if the decoded output still contains non-dangerous %HH patterns.
- This misses `%2520 -> %20` and similar cases, even though they are classic double-encoding evidence.

Changes:
- Mark DOUBLEPCT if there is evidence of `%25` decoding that produces a remaining `%HH` (dangerous or not), or if input contains `%25[0-9A-Fa-f]{2}`.

Acceptance:
- `%2520` yields DOUBLEPCT deterministically.

---

### WP4 - CRLF EOL is not flagged as BADCRLF (F10)

Problem:
- Step 01 splits on "\n" and can leave "\r" embedded in line content when input uses CRLF.
- Step 02 flags BADCRLF when it sees "\r", producing false positives for valid wire-format CRLF.

Changes:
1) In Step 01, normalize line endings before parsing (splitlines() or replace("\r\n", "\n")).
2) In Step 01, detect end-of-headers using strip == "" (not strict equality).
3) In Step 02, ensure BADCRLF only flags embedded CR/LF, not terminators.

Acceptance:
- A normal CRLF-terminated request does not trigger BADCRLF.

---

### WP5 - BADHDRCONT must persist into the final artifact (F09)

Problem:
- Step 02 emits `[HEADER] BADHDRCONT` for orphan continuations.
- Step 03 drops malformed headers (no colon), so BADHDRCONT disappears.

Changes (choose one):
- Option A: accumulate BADHDRCONT as a global flag and ensure it appears in final [FLAGS].
- Option B: emit a dedicated diagnostic line (e.g., `[HERR] orphan-continuation BADHDRCONT`) and ensure it survives and is aggregated.

Acceptance:
- Final artifact contains BADHDRCONT evidence whenever the orphan continuation scenario occurs.

---

### WP6 - Step 05 is detection-only (does not rewrite content) (F11)

Problem:
- DangerousCharactersScriptMixing currently reconstructs header content with split/join.
- This can collapse whitespace and alter evidence without emitting an explicit signal.

Changes:
- Rework Step 05 processing to:
  - parse content vs existing flags
  - detect conditions
  - re-attach flags
  - do not modify the content emitted by previous steps

Acceptance:
- Step 05 is idempotent and does not change payload text (except adding flags).

---

### WP7 - URL_ABS stays consistent with final normalized URL/path (F04)

Problem:
- Step 06 runs before Step 11, so [URL_ABS] can be computed from a pre-normalized [URL].

Changes (choose one):
- Option A: move Step 06 after Step 11 in config.toml.
- Option B: add a finalizer step that recomputes [URL_ABS] at the end from the final [URL] + Host.

Acceptance:
- If [URL] changes due to normalization, [URL_ABS] reflects the final normalized path.

Impact:
- May require regenerating thesis Listing 4.2.

---

### WP8 - Absolute-form request-target handling (introduce [TARGET] or equivalent) (F05)

Problem:
- When the request-target is absolute-form (e.g., `GET http://a.example/x HTTP/1.1`), path normalizers can corrupt it.

Changes:
- Emit `[TARGET] <raw request-target>` in Step 01.
- Emit `[URL]` as path only (origin-form path, or extracted path from absolute-form).
- Ensure path normalization applies to [URL] (path), not [TARGET].
- Update Step 06 to use [TARGET] when present for host consistency logic.

Acceptance:
- absolute-form is preserved (no `http:/...` corruption).
- HOSTMISMATCH / IDNA / BADHOST remain consistent.

Impact:
- This is a contract change that must be reflected in the thesis.

---

### WP9 - One coherent model for global flags (F12 partial)

Problem:
- Multiple "global" outputs exist: [FLAGS] (Step 01 + aggregator), [HGF] (Step 03).
- Aggregator currently may miss some due to formatting differences.

Changes:
- Minimum: ensure aggregator consumes [HGF] correctly.
- Preferred: converge on one canonical global line (keep [FLAGS] only) and deprecate [HGF] if it adds no unique value.

Acceptance:
- No global signal is missed because it was emitted in a different format.

---

### WP10 - Pipeline portability: do not depend on CWD for config.toml (F13)

Problem:
- preprocessing/pipeline.py loads config.toml using a path relative to the current working directory at import time.

Changes:
- Load config.toml relative to the installed package (importlib.resources).
- Prefer lazy initialization to avoid import-time side effects.

Acceptance:
- `from neuralshield.preprocessing.pipeline import preprocess` works from any CWD.

---

### WP11 - Length bucketing step: implement or remove spec (F14)

Problem:
- A length-bucketing spec exists, but no step is implemented or wired into the pipeline.

Decision:
- Option A: implement and add it (post-normalization).
- Option B: mark spec as non-current (or remove) to avoid promising non-existent features.

Acceptance:
- Option A: artifact includes [PLEN]/[PMAX]/[HCNT]/[HLEN] in the documented format.
- Option B: repo no longer implies those lines exist.

---

## Thesis synchronization (PDF-vs-code remaining)

These are directly called out in repo/docs/AUDIT_FIX_SUMMARY.md as remaining items:
- Update Figure 4.2 flag list (Phase 2)
- Regenerate Listing 4.2 end-to-end example from actual pipeline output
- Decide 3-way vs 2-way split (code vs PDF)
- Fix Table 6.4 F1 math errors (PDF)

Recommendation:
- After WP2-WP6 (correctness) and WP7/WP8 decisions (contract-sensitive), regenerate Listing 4.2 and update the thesis.

---

## Execution order (recommended)

To minimize risk and avoid cascaded diffs:
1) WP2, WP3, WP4, WP5 (correctness: query output stability, double encoding, CRLF, persistence)
2) WP6 (detection-only behavior)
3) WP1 + WP9 (format consistency + aggregator coverage)
4) WP10 (portability)
5) WP7 (URL_ABS consistency)
6) WP8 (TARGET / absolute-form) - requires thesis update
7) WP11 (bucketing) - decision needed
8) Thesis regen (Figure 4.2 / Listing 4.2 / tables)

---

## Validation checklist

Repo:
- `uv run ruff check .`
- `uv run ruff format .`
- `uv run mypy src`
- `uv run pytest`

Preprocessing smoke tests:
- Run preprocess on `src/neuralshield/preprocessing/test/in/comprehensive_test.in` and confirm expected invariants.
- Add/extend golden tests for the exact reproductions used in findings (CRLF, %2520, %00, orphan obs-fold).

Thesis:
- Rebuild the PDF and verify listings/figures match actual pipeline output.
