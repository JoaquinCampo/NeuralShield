# NeuralShield – Flag Specification & Report

_v1.0 • 2025‑09‑20_

This report documents **every flag** emitted by the preprocessing pipeline: where it is produced, exact trigger conditions, output placement, examples, interplay across stages, and known edge cases. It is meant to be the single source of truth for maintainers and for aligning tests.

---

## 0) Pipeline & Output Conventions

**Stage order (write path):**

1. **RemoveFramingArtifacts** →
2. **RequestStructurer** →
3. **FlagsNormalizer** →
4. **QueryProcessor** →
5. **PathStructureNormalizer** →
6. **DangerousCharacterDetector**

**Line types:**

- `[URL]`: canonicalized request-target path (and query if still raw).
- `[QUERY]`: raw query line (only before QueryProcessor).
- `[QSEP]`, `[QPARAM ...]`, `[QMETA]`: structured query output (replace `[QUERY]`).
- `[HEADER name]`: original header name + value (with folding preserved by structurer).
- `FLAGS: ...`: a flag line **immediately following** the component line it annotates (URL/QUERY/HEADER), unless otherwise noted.

**Component scope:** Unless stated, flags apply to **the component line they follow**. Query-wide aggregates appear in `[QMETA]`.

**Examples reference:** Filenames like `001_basic_get`, `201_comprehensive`, etc., refer to our test corpus.

---

## 1) Flags by Stage

### 1.1 RemoveFramingArtifacts

- **Purpose:** Trim capture wrappers (e.g., proxy markers), normalize line endings.
- **Flags:** _None produced._

### 1.2 RequestStructurer

- **Purpose:** Lay out lines in canonical order; preserve header folding (obs-fold) verbatim.
- **Flags:** _None produced._ (Downstream detectors read its output.)

### 1.3 FlagsNormalizer

_Applies over `[URL]` and (pre-structured) `[QUERY]`. Emits a `FLAGS:` line **right after** the processed line. Query flags from this stage are later propagated into `[QMETA]` by QueryProcessor._

| Flag          | Component(s) | Trigger (precise)                                                                                                           | Notes & Examples                                                                                                |
| ------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **FULLWIDTH** | URL, QUERY   | Unicode **NFKC** alters the string **or** any character from the Fullwidth/compatibility forms is present.                  | Signals confusable/width variants. e.g., `101_unicode_edge_cases`, `010_combined_anomalies`.                    |
| **CONTROL**   | URL, QUERY   | String contains a **control char** (Unicode General Category `Cc`) **or** a literal `%00`.                                  | `%00` is not decoded here; decoding is one-pass elsewhere. e.g., `102_control_chars_edge`, `201_comprehensive`. |
| **HTMLENT**   | URL, QUERY   | Presence of HTML entity forms (`&name;`, `&#d;`, `&#xhh;`) **and** a one-pass decode **changes** the text.                  | Logic exists; may be hidden later if representation is replaced. No current `.out` surface.                     |
| **DOUBLEPCT** | URL, QUERY   | After **one** percent-decoding pass, the result **still contains decodable** `%[0-9A-Fa-f]{2}` sequences (excluding `%00`). | Detects multi-level encoding. e.g., `104_percent_encoding_edge`, `201_comprehensive` (query meta).              |

**Output placement:** A `FLAGS:` line follows the relevant `[URL]`/`[QUERY]`. For query, these flags are also summarized under `[QMETA]` after QueryProcessor runs.

---

### 1.4 QueryProcessor

_Replaces the raw `[QUERY]` line with structured lines: one global separator decision, parameter lines, and a meta summary. Also **projects** relevant normalization flags into `[QMETA]`._

**Separator decision:**

- **`[QSEP]`** with either `QSEMISEP` or `QRAWSEMI`:

  - **QSEMISEP** → treat both `;` and `&` as separators when a **semicolon‑dominant** `k=v;k=v` pattern is detected (more `;` than `&`, ≥1 semicolon‑separated `k=v`). Example: `302_query_semicolon_dominant`.
  - **QRAWSEMI** → semicolons appear but are **not** dominant; keep `&` as separator; record presence. Example: `001_basic_get`, `999_ultimate_comprehensive`.

**Parameter lines:**

- **`[QPARAM key=value]`** per decoded pair (single-decoded; blanks preserved via `keep_blank_values=True`).
- Per‑parameter flags:

  - **QBARE** → token without `=`. Example: `003_duplicate_params`, `103_html_entities_edge` (`malformed`).
  - **QEMPTYVAL** → key with empty value (has `=` but nothing after). Example: `003_duplicate_params`, `108_edge_combinations`.
  - **QARRAY\:key** → key ends with `[]` (rendered `QARRAY:items[]` in `[QMETA]`). Example: `305_query_comprehensive_flags`.
  - **QREPEAT\:key** → repeated key after its first appearance. For empty key, display as `<empty>`. Example: `003_duplicate_params`, `301_query_basic_ampersand`, `108_edge_combinations`.
  - **QNUL** → NUL byte present in **value** after a single decode. Mirrors component‑level `NUL`. Example: `102_control_chars_edge`, `109_boundary_conditions`.
  - **QNONASCII** → any non‑ASCII in key or value. Example: `001_basic_get` (`multi`), `202_script_mixing`.
  - **QLONG** → value length `> long_value_threshold` (default **1024**). _Not triggered_ by current tests.

**Meta line:**

- **`[QMETA ...]`** aggregates query‑wide counters and flags (**excludes** `QREPEAT:*`). Includes: counts, `QSEMISEP`/`QRAWSEMI`, `QBARE`, `QEMPTYVAL`, `QNUL`, `QNONASCII`, `QLONG`, and summarized arrays as `QARRAY:…`.

**Propagation of normalizer flags:** When FlagsNormalizer raised `FULLWIDTH`, `CONTROL`, `DOUBLEPCT`, `HTMLENT` on the query, their presence is reflected in `[QMETA]` (so they survive after `[QUERY]` is replaced).

---

### 1.5 PathStructureNormalizer

| Flag              | Component(s) | Trigger (precise)                                              | Notes & Examples                                                                                                               |
| ----------------- | ------------ | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **MULTIPLESLASH** | URL          | Any occurrence of `//` that collapses to `/`.                  | We **collapse** repeats, then append flag. e.g., `007_path_structure`, `105_path_structure_complex`, `010_combined_anomalies`. |
| **DOTDOT**        | URL          | Any `..` segment appears in the path; **do not resolve** away. | Preserves traversal signals. e.g., `008_path_traversal`, `107_mixed_encoding_attacks`.                                         |
| **HOME**          | URL          | Normalized path is `/` **or** input path empty.                | e.g., `009_root_path`, `106_empty_minimal`, `109_boundary_conditions`.                                                         |

---

### 1.6 DangerousCharacterDetector

_Applies to `[URL]`, `[HEADER]`, and (only if still present) `[QUERY]`. Emits a `FLAGS:` line after the component. For query, see “Visibility” note below._

| Flag            | Component(s)               | Trigger (precise)                                                              | Notes & Examples                                                           |                                                                                  |
| --------------- | -------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **ANGLE**       | URL, QUERY, HEADER         | `<` or `>` present, including `%3C`/`%3E` detected post‑decode.                | `201_dangerous_chars_basic` (path), `205_combined_dangers`.                |                                                                                  |
| **QUOTE**       | URL, QUERY, HEADER         | `'` or `"` or `%27`/`%22`.                                                     | Common in ETag/If‑None‑Match. e.g., `002_root_no_query` header.            |                                                                                  |
| **SEMICOLON**   | URL, QUERY                 | `;` or `%3B`. **Not emitted** for headers.                                     | URL example: `001_basic_get`.                                              |                                                                                  |
| **PAREN**       | URL, QUERY, HEADER         | `(` or `)` or `%28`/`%29`.                                                     | User‑Agent strings: `(Macintosh; …)`.                                      |                                                                                  |
| **BRACE**       | URL, QUERY, HEADER         | `{` or `}` or `%7B`/`%7D`.                                                     | Present in inputs; rare in current `.out`.                                 |                                                                                  |
| **PIPE**        | URL, QUERY, HEADER         | \`                                                                             | `or`%7C\`.                                                                 | Often in shell‑style payloads; may not surface if query is replaced (see below). |
| **BACKSLASH**   | URL, QUERY, HEADER         | `\` or `%5C`.                                                                  | Windows paths like `C:\Windows`.                                           |                                                                                  |
| **SPACE**       | URL                        | Literal space or `%20`.                                                        | Only for URL component. e.g., `204_spaces_in_path`, `006_edge_cases`.      |                                                                                  |
| **NUL**         | URL, QUERY, HEADER         | Literal NUL or `%00`.                                                          | For query, `QNUL` is also emitted by QueryProcessor (after single decode). |                                                                                  |
| **MIXEDSCRIPT** | URL, HEADER (value), QUERY | Mixed scripts (e.g., Latin + Cyrillic/Greek) after decoding and normalization. | e.g., `202_script_mixing` (URL & Host header), `205_combined_dangers`.     |                                                                                  |

**Header‑specific behavior:**

- `SEMICOLON` **suppressed** for headers (too noisy; common in UA/comment tokens).
- `MIXEDSCRIPT` checks **header value only** (post‑colon).

**Query visibility caveat:** Since QueryProcessor replaces `[QUERY]` with `[QSEP]/[QPARAM]/[QMETA]`, DangerousCharacterDetector usually **won’t** see query text anymore; thus these flags will rarely appear attached to query **component** lines. Query issues should be reflected via `[QMETA]`.

---

## 2) Cross‑Stage Interactions & Precedence

- **Single‑decode invariant:** All query parsing and most detectors assume **one** percent‑decode pass. `DOUBLEPCT` marks when a second decode would still change content.
- **NUL vs CONTROL:**

  - `CONTROL` (FlagsNormalizer) marks any control char (incl. `%00` literal) found early in URL/QUERY.
  - `NUL` (DangerousCharacterDetector) annotates specific components later. For query values, `QNUL` is additionally set by QueryProcessor.

- **Query flag surfacing:** To keep query anomalies visible after structuring, **use `[QMETA]`**. Do not rely on a `[QUERY]`‐adjacent `FLAGS:` line in final output.
- **Header semicolons:** Never flagged (`SEMICOLON`), but parentheses/quotes remain flagged to capture likely injection delimiters.

---

## 3) Quick Reference Table (All Flags)

| Flag          | Stage                      | Component                | Purpose (1‑liner)                                  |                   |
| ------------- | -------------------------- | ------------------------ | -------------------------------------------------- | ----------------- |
| FULLWIDTH     | FlagsNormalizer            | URL, QUERY               | Compatibility/width forms detected (NFKC changed). |                   |
| CONTROL       | FlagsNormalizer            | URL, QUERY               | Control characters (Cc) or `%00` literal present.  |                   |
| HTMLENT       | FlagsNormalizer            | URL, QUERY               | HTML entity decode would change text.              |                   |
| DOUBLEPCT     | FlagsNormalizer            | URL, QUERY               | Multi‑level percent encoding likely.               |                   |
| MULTIPLESLASH | PathStructureNormalizer    | URL                      | Collapsed repeated `/`.                            |                   |
| DOTDOT        | PathStructureNormalizer    | URL                      | `..` segment present (no resolution).              |                   |
| HOME          | PathStructureNormalizer    | URL                      | Path is `/` or empty.                              |                   |
| QSEMISEP      | QueryProcessor             | `[QSEP]`                 | Treat `;` as separator along with `&`.             |                   |
| QRAWSEMI      | QueryProcessor             | `[QSEP]`                 | Semicolons present but not used as separators.     |                   |
| QBARE         | QueryProcessor             | `[QPARAM]`, `[QMETA]`    | Token without `=`.                                 |                   |
| QEMPTYVAL     | QueryProcessor             | `[QPARAM]`, `[QMETA]`    | Key with empty value.                              |                   |
| QNUL          | QueryProcessor             | `[QPARAM]`, `[QMETA]`    | NUL in value post single decode.                   |                   |
| QNONASCII     | QueryProcessor             | `[QPARAM]`, `[QMETA]`    | Non‑ASCII in key/value.                            |                   |
| QARRAY\:key   | QueryProcessor             | `[QPARAM]`, `[QMETA]`    | Array‑style key (e.g., `items[]`).                 |                   |
| QLONG         | QueryProcessor             | `[QPARAM]`, `[QMETA]`    | Very long value (>1024 by default).                |                   |
| QREPEAT\:key  | QueryProcessor             | `[QPARAM]`               | Repeated key occurrence.                           |                   |
| ANGLE         | DangerousCharacterDetector | URL, QUERY, HEADER       | `<`/`>` seen (or %3C/%3E).                         |                   |
| QUOTE         | DangerousCharacterDetector | URL, QUERY, HEADER       | `'`/`"` seen (or %27/%22).                         |                   |
| SEMICOLON     | DangerousCharacterDetector | URL, QUERY               | `;` seen (or %3B); **not** on headers.             |                   |
| PAREN         | DangerousCharacterDetector | URL, QUERY, HEADER       | `(`/`)` seen (or %28/%29).                         |                   |
| BRACE         | DangerousCharacterDetector | URL, QUERY, HEADER       | `{`/`}` seen (or %7B/%7D).                         |                   |
| PIPE          | DangerousCharacterDetector | URL, QUERY, HEADER       | \`                                                 | \` seen (or %7C). |
| BACKSLASH     | DangerousCharacterDetector | URL, QUERY, HEADER       | `\` seen (or %5C).                                 |                   |
| SPACE         | DangerousCharacterDetector | URL                      | Space or `%20` present in path.                    |                   |
| NUL           | DangerousCharacterDetector | URL, QUERY, HEADER       | Literal NUL or `%00` present.                      |                   |
| MIXEDSCRIPT   | DangerousCharacterDetector | URL, HEADER value, QUERY | Mixed Unicode scripts (e.g., Latin + Cyrillic).    |                   |

---

## 4) Output Placement Rules (Authoritative)

1. A `FLAGS:` line follows the component line it annotates.
2. For **query**, after structuring:

   - `[QSEP]` appears once.
   - One `[QPARAM …]` per decoded pair; per‑param Q\* flags appear **inline** next to the parameter or on the following `FLAGS:` line (depending on renderer), and are also summarized in `[QMETA]`.
   - `[QMETA]` is the query‑wide summary (counts + Q\* aggregate + projected normalizer flags).

3. URL‑level flags from FlagsNormalizer remain attached to the `[URL]` section even after later stages.
4. Header flags attach to each `[HEADER name]` line; no semicolon flagging for headers by design.

---

## 5) Known Edge Cases & Gaps

1. **HTMLENT not surfacing:** Although detected by FlagsNormalizer, the final `.out` may not display it if later stages re‑emit a transformed representation. _Action:_ Teach QueryProcessor to carry `HTMLENT` into `[QMETA]` when present.
2. **Query visibility for Dangerous chars:** Since `[QUERY]` is replaced, DangerousCharacterDetector rarely annotates query content. _Action:_ Extend the detector to scan decoded values from `[QPARAM]` lines, emitting per‑param flags (e.g., `QPAREN`, `QQUOTE`) **or** mirror them into `[QMETA]` with component tags (e.g., `DANGER:ANGLE`).
3. **Double counting NUL:** `CONTROL` (early) + `NUL`/`QNUL` (later) can co‑exist. _Action:_ In summaries, de‑duplicate by type while preserving stage provenance.
4. **QSEMISEP threshold:** Dominance heuristic currently “more `;` than `&` and ≥1 semicolon `k=v` token”. _Action:_ Make ratio configurable (e.g., `>= 60%` semicolons) for noisy payloads.
5. **QLONG threshold:** Documented default is **1024** bytes. _Action:_ Promote to config & surface the actual threshold in `[QMETA]` when triggered (e.g., `QLONG>1024`).

---

## 6) Security Rationale (Why These Flags Exist)

- **Normalization flags (FULLWIDTH, DOUBLEPCT, HTMLENT, CONTROL):** catch canonicalization mismatches and decoder differentials that often lead to WAF bypasses.
- **Path structure (MULTIPLESLASH, DOTDOT, HOME):** structural anomalies and traversal indicators that are both **model‑useful** and helpful for downstream rules.
- **Dangerous characters:** broadly indicative of injection delimiters or parser ambiguities shared across vectors (URL, headers, query).
- \*_Query flags (Q_):\*\* stabilize query parsing under adversarial separator choices and malformed tokens, surfacing key signals (arrays, repeats, bare/empty pairs, NULs, non‑ASCII, long values).

---

## 7) Modeling Hints (Tokenization‑Aware)

- Place **`FLAGS:`** lines **immediately after** the component they annotate so an encoder with limited window attends to the right context.
- Keep flag **lexemes short and uppercase** (stable tokens). Avoid punctuation inside names except `:` for keyed variants (e.g., `QARRAY:items[]`, `QREPEAT:login`).
- Prefer a **fixed order** within a `FLAGS:` line (alphabetical) to improve consistency for the tokenizer.
- In `[QMETA]`, list counts first, then booleans, then keyed summaries (arrays, repeats), then propagated normalizer flags.

---

## 8) Test Coverage Map (by Example)

- **URL:** `MULTIPLESLASH` (`007_path_structure`), `DOTDOT` (`008_path_traversal`), `HOME` (`009_root_path`), `SPACE` (`204_spaces_in_path`), `ANGLE`/`QUOTE` (`201_dangerous_chars_basic`).
- **QUERY:** `QSEMISEP` (`302_query_semicolon_dominant`), `QRAWSEMI` (`001_basic_get`), `QBARE`/`QEMPTYVAL`/`QREPEAT` (`003_duplicate_params`), `QNONASCII` (`202_script_mixing`), `QNUL` (`102_control_chars_edge`), `DOUBLEPCT` (`104_percent_encoding_edge`).
- **HEADERS:** `QUOTE` (ETag in `002_root_no_query`), `PAREN` (User‑Agent), `MIXEDSCRIPT` (Host variant in `202_script_mixing`).

---

## 9) Proposed Small Improvements (Optional)

1. **Carry `HTMLENT` into `[QMETA]`.**
2. **Scan `[QPARAM]` values for dangerous chars.** Either emit per‑param flags (prefixed `Q`), or aggregate into `[QMETA]` as `DANGER:<NAME>` with counts.
3. **Expose thresholds in meta.** e.g., `QLONG>1024`, `SEMIDOM>=0.6`.
4. **De‑dupe control/NUL flags in summaries** while keeping per‑component detail in place.

---

## 10) Appendix A – Formal Triggers (Pseudocode)

> Pseudocode only; the implementation may differ in structure but should be equivalent.

- **FULLWIDTH:** `if NFKC(s) != s or contains_fullwidth(s): emit(FULLWIDTH)`
- **CONTROL:** `if any(ch in Cc for ch in s) or '%00' in s: emit(CONTROL)`
- **HTMLENT:** `if html_unescape_once(s) != s: emit(HTMLENT)`
- **DOUBLEPCT:** `if pct_decode_once(s) contains /%[0-9A-Fa-f]{2}/: emit(DOUBLEPCT)` (except `%00`).
- **MULTIPLESLASH:** `if '//' in path: path = collapse_slashes(path); emit(MULTIPLESLASH)`
- **DOTDOT:** `if any(seg == '..' for seg in split_path(path)): emit(DOTDOT)`
- **HOME:** `if path == '' or path == '/': path = '/'; emit(HOME)`
- **QSEMISEP:** `if semicolons > ampersands and has_semicolon_kv: separator=';&'; emit(QSEMISEP) else if semicolons>0: emit(QRAWSEMI)`
- **QBARE:** `if '=' not in token` → per‑param flag.
- **QEMPTYVAL:** `if token.endswith('=')` → per‑param flag.
- **QARRAY\:key:** `if key.endswith('[]')` → per‑param & summarize.
- **QREPEAT\:key:** `if key in seen_keys` → per‑param flag.
- **QNUL:** `if '\x00' in value` after single decode.
- **QNONASCII:** `if any(ord(ch)>127 for ch in key+value)`
- **QLONG:** `if len(value) > LONG_VALUE_THRESHOLD` (default 1024).
- **ANGLE/QUOTE/SEMICOLON/PAREN/BRACE/PIPE/BACKSLASH/SPACE/NUL:** presence checks over decoded component; header suppresses `SEMICOLON`.
- **MIXEDSCRIPT:** `if count(distinct scripts in token excluding Common/Inherited) >= 2`.

---

### End of Report
