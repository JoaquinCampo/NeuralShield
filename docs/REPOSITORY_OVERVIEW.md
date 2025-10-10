## Repository Overview

NeuralShield is a machine learning-powered Web Application Firewall focused on transforming raw HTTP requests into a stable, idempotent representation and surfacing anomalies via flags. This repository currently implements an initial preprocessing pipeline with a test harness and contains design specifications to guide future steps.

## High-level purpose

- Normalize raw HTTP requests into a deterministic text form suitable for ML/NLP models
- Preserve suspicious signals (e.g., mixed EOLs, obs-fold, double percent encoding) as explicit flags rather than silently fixing them
- Maintain idempotence and determinism across passes and versions

## Project layout

- `pyproject.toml`: Python 3.12 package config (Hatch/hatchling), runtime dependency `loguru`, dev dependency `ipykernel`, Ruff lint/format, mypy config
- `uv.lock`: lockfile for `uv` usage (optional)
- `README.md`: brief project description
- `docs/`: documentation and reports (LaTeX files present); this file lives here
- `src/neuralshield/`
  - `__init__.py`: minimal entrypoint placeholder
  - `preprocessing/`
    - `http_preprocessor.py`: abstract base class for steps (`HttpPreprocessor.process(str) -> str`)
    - `pipeline.py`: constructs the pipeline from TOML config and exposes `preprocess`
    - `config.toml`: ordered list of step classes to run
    - `steps/`
      - `pre_parse_over_raw.py`: concrete steps (see Implemented steps)
      - `exceptions.py`: `MalformedHttpRequestError`
    - `specs/`: authoritative design documents (Spanish) describing planned behavior in detail (see Design specifications)
    - `test/`
      - `in/`: sample raw HTTP requests (`*.in`, no body)
      - `out/`: expected normalized outputs (`*.out`) for the current pipeline
      - `result/`: actual outputs written by the harness (`*.actual`)
      - `diff/`: unified diffs vs expected on failures (`*.diff`)
      - `test_pipeline.py`: test harness runner (see Test harness)

## Runtime and dependencies

- Python: >= 3.12
- Runtime deps: `loguru`
- Packaging: hatch/hatchling; package path rooted at `src/`
- Dev tooling: Ruff (lint/format), mypy, optional `uv`

## Preprocessing pipeline

- Entry point: `neuralshield.preprocessing.pipeline.preprocess`
- Construction: `pipeline()` builds a function by composing step instances configured in `src/neuralshield/preprocessing/config.toml`
- Configured order (see `config.toml`):
  1. `neuralshield.preprocessing.steps.00_framing_cleanup:FramingCleanup`
  2. `neuralshield.preprocessing.steps.01_request_structurer:RequestStructurer`
  3. `neuralshield.preprocessing.steps.02_header_unfold_obs_fold:HeaderUnfoldObsFold`
  4. `neuralshield.preprocessing.steps.03_header_normalization_duplicates:HeaderNormalizationDuplicates`
  5. `neuralshield.preprocessing.steps.04_whitespace_collapse:WhitespaceCollapse`
  6. `neuralshield.preprocessing.steps.05_dangerous_characters_script_mixing:DangerousCharactersScriptMixing`
  7. `neuralshield.preprocessing.steps.06_absolute_url_builder:AbsoluteUrlBuilder`
  8. `neuralshield.preprocessing.steps.07_unicode_nkfc_and_control:UnicodeNFKCAndControl`
  9. `neuralshield.preprocessing.steps.08_percent_decode_once:PercentDecodeOnce`
  10. `neuralshield.preprocessing.steps.09_html_entity_decode_once:HtmlEntityDecodeOnce`
  11. `neuralshield.preprocessing.steps.10_query_parser_and_flags:QueryParserAndFlags`
  12. `neuralshield.preprocessing.steps.11_path_structure_normalizer:PathStructureNormalizer`
- The order is authoritative and can be changed by editing `config.toml` without code changes
- Implementation detail: `pipeline()` materializes the steps into a list to avoid generator exhaustion across calls

## Implemented steps

- **00 Framing Cleanup** trims BOM markers and leading/trailing control characters on the raw request.
- **01 Request Structurer** parses the request line, splits query tokens, and emits `[METHOD]`, `[URL]`, `[QUERY]`, and `[HEADER]` records.
- **02 Header Unfold (Obs-fold)** merges legacy folded header continuations, tagging `OBSFOLD`/`BADCRLF`.
- **03 Header Normalization and Duplicates** lowercases header names, merges allowed duplicates, and emits `[HAGG]`/`[HGF]` metrics.
- **04 Whitespace Collapse** normalizes header value spacing and flags `WSPAD` when adjustments are made.
- **05 Dangerous Characters and Script Mixing** adds inline flags for suspicious characters and mixed-script content across URL, query, and header fields.
- **06 Absolute URL Builder** produces `[URL_ABS]` entries and validates host header consistency (`HOSTMISMATCH`, `IDNA`, `BADHOST`).
- **07 Unicode NFKC and Control** normalizes URL/query text with NFKC and surfaces Unicode anomaly flags.
- **08 Percent Decode Once** applies a single, context-aware percent-decode pass while preserving dangerous encodings.
- **09 HTML Entity Decode Once** detects entity-encoded tokens and adds the `HTMLENT` flag without mutating the source.
- **10 Query Parser and Flags** expands query parameters, redacts sensitive shapes, and aggregates separator/metadata flags.
- **11 Path Structure Normalizer** collapses redundant path segments while preserving traversal evidence (`DOTDOT`, `DOTCUR`, `MULTIPLESLASH`).

- LineJumpCatcher (available, not enabled in config by default)
  - Scans raw text and appends a flag line on the next line for EOL anomalies per original line:
    - `EOL_BARELF` if the EOL run contains only LF(s)
    - `EOL_BARECR` if the EOL run contains only CR(s)
    - `EOLMIX` if the EOL run mixes CRLF/LF/CR
    - `EOL_EOF_NOCRLF` if the last line has no trailing EOL
  - Idempotent: does not duplicate appended flags and does not flag the flag lines themselves

## Output format (current)

- Stable, line-based textual form:
  - `[METHOD] <method>`
  - `[URL] <path>`
  - `[QUERY] <raw-&-split-token>` repeated
  - `[HEADER] <raw-header-line>` repeated
- No body is parsed (tests assume no body)

## Test harness

- Location: `src/neuralshield/preprocessing/test/test_pipeline.py`
- Logging: configured to `INFO` to reduce noise
- Behavior:
  - Discovers `test/in/*.in`
  - For each test:
    - Reads raw input
    - Calls `preprocess(raw)`
    - Writes `test/result/<name>.actual`
    - Compares to `test/out/<name>.out`
    - Writes unified diff to `test/diff/<name>.diff` if different; logs pass/fail
  - Summarizes total and failing counts
- Example invocation (with `uv`):
  - `uv run src/neuralshield/preprocessing/test/test_pipeline.py`

## Design specifications (planned behavior)

The `specs/` directory (Spanish) is the source of truth for future steps. Key topics:

- EOL anomalies and normalization (f0)
  - Detect EOL token usage (CRLF/LF/CR) and emit flags/stats; then canonicalize to CRLF (idempotent)
- Obs-fold (folded headers) detection and unfolding per RFC 9110; emit `OBSFOLD` and ensure single-line headers
- Robust query parsing with `&`/`;` heuristics; percent-decode exactly once per token; multiplicity preservation; Q\* flags (`QREPEAT`, `QEMPTYVAL`, `QBARE`, `QSEMISEP`, `QRAWSEMI`, `QNUL`, `QNONASCII`, `QARRAY`, `QLONG`)
- Path normalization: collapse `//` and `/.` but preserve `..`; metrics `PLEN`, `PMAX`; flags like `MULTIPLESLASH`, `DOTDOT`
- Dangerous characters and Unicode/script-mixing detection; `FULLWIDTH`, `CONTROL`, `MIXEDSCRIPT`, etc.
- Header normalization: lowercase names, whitespace normalization, duplicate merge where safe; flags `DUPHDR:<name>`, `BADHDRNAME:<name>`, `HOPBYHOP:<name>`; metrics `HCNT`, `HLEN`
- Absolute URL construction from origin-form; `HOSTMISMATCH`, `IDNA`
- Shapes and redaction for sensitive values (`<SECRET:shape:len>`), authorization/cookie handling
- Global `FLAGS:[...]` roll-up: deduped, sorted, union of all component flags

## Extensibility

- Add a step by creating a class implementing `HttpPreprocessor.process(self, request: str) -> str`
- Register it in `src/neuralshield/preprocessing/config.toml` with `module:ClassName`
- Keep steps idempotent, deterministic, and high-clarity; prefer early returns and explicit names

## Current limitations (intentional)

- Request bodies are ignored; the pipeline currently canonicalizes only the request line, headers, and query string.
- EOL normalization remains opt-in via `LineJumpCatcher`; canonical CRLF conversion is still a future enhancement.
- Global flag roll-ups beyond the existing `[HAGG]`, `[HGF]`, `[QSEP]`, and `[QMETA]` records remain on the roadmap.

## How to run and interpret tests

1. Place raw requests in `src/neuralshield/preprocessing/test/in/*.in` (no body)
2. Place expected outputs in `.../out/*.out` matching current pipeline behavior
3. Run the harness; check:
   - `.../result/*.actual` for produced output
   - `.../diff/*.diff` for failures; update expected or pipeline config as needed

## Key API and mental model

- Call `neuralshield.preprocessing.pipeline.preprocess(raw_http_request: str) -> str`
- The pipeline is a sequence of idempotent, ordered steps configured by TOML
- The aim is a canonical textual representation plus explicit anomaly flags to feed ML models
- Specs in `specs/` define target behavior; steps in `steps/` progressively implement them

## Notes for contributors and LLMs

- Treat `specs/` as the contract; ensure new code matches the documented order and idempotence
- Preserve suspicious signals as flags; avoid silent fixes that hide anomalies from models
- When modifying step order or behavior, update test expectations and expand cases under `test/in`/`out`
- Keep logging via `loguru` at appropriate levels (DEBUG for telemetry, INFO for anomalies)
