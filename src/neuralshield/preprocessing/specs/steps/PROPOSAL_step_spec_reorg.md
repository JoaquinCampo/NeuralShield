## Step Spec Reorganization Proposal (single-purpose, one .py per step)

### Goals

- Eliminate overlaps between specs by defining single-responsibility steps.
- Ensure one-to-one mapping: one `.md` spec → one `.py` step.
- Make inputs/outputs and emitted flags explicit per step; avoid cross-step side effects.

### Principles

- Deterministic, idempotent steps; each step only does what its spec states.
- Detection vs aggregation: detection steps emit local per-line flags; a dedicated aggregator performs global roll-up.
- No step performs decoding/normalization outside its remit (e.g., dangerous-char detection does not decode).
- Steps may depend on prior steps’ canonical forms; dependencies are ordered and listed.

## Proposed step taxonomy and order

1) 00-framing-cleanup.md → 00_framing_cleanup.py
- Scope: Remover BOM y caracteres de control solo en los bordes del request crudo.
- Emits: none
- Input: raw request string; Output: cleaned raw request string.

2) 01-request-structurer.md → 01_request_structurer.py
- Scope: Convertir request crudo a formato canónico por líneas: `[METHOD]`, `[URL]`, `[QUERY]`, `[HEADER]` (sin interpretar valores).
- Emits: syntax errors como excepciones (no flags).
- Depends-on: 00

3) 02-eol-anomaly-annotator.md → 02_eol_annotator.py
- Scope: Detectar CR/LF/CRLF mixtos y EOF sin CRLF; anotar por línea sin modificar contenido.
- Emits: `EOL_BARELF`, `EOL_BARECR`, `EOLMIX`, `EOL_EOF_NOCRLF`
- Depends-on: 01

4) 03-bytes-to-unicode.md → 03_bytes_to_unicode.py
- Scope: Política de decodificación bytes→Unicode (UTF-8 tolerante) y marca `BADUTF8` si hubo sustituciones.
- Emits: `BADUTF8`
- Depends-on: 01 (si aplica a entradas byte-level) o se especifica no-op si ya es `str`.

5) 04-unicode-nfkc-and-control.md → 04_unicode_nfkc.py
- Scope: Normalización Unicode NFKC en campos estructurales; detectar `FULLWIDTH` y `CONTROL` (categoría Cc), sin tocar delimitadores RFC.
- Emits: `FULLWIDTH`, `CONTROL`
- Depends-on: 03 (o 01 si no hay 03)

6) 05-percent-decode-once.md → 05_percent_decode_once.py
- Scope: Aplicar percent-decode exactamente una vez por componente (path por segmentos y query por clave/valor) sin romper delimitadores preservados.
- Emits: `DOUBLEPCT`, `PCTSLASH`, `PCTBACKSLASH`
- Depends-on: 04

7) 06-html-entity-decode-once.md → 06_html_entity_decode_once.py
- Scope: Decodificar entidades HTML una sola vez en URL/QUERY tras la pasada de percent; respetar política de delimitadores.
- Emits: `HTMLENT`
- Depends-on: 05

8) 07-query-parser-and-flags.md → 07_query_parser.py
- Scope: Detección de separadores (`&`/heurística `;&`), tokenización determinista, percent-decode exacta por token (si no se aplicó en 05 a nivel token), y flags de estructura/valores.
- Emits: `QSEMISEP`, `QRAWSEMI`, `QBARE`, `QEMPTYVAL`, `QREPEAT:<k>`, `QNUL`, `QNONASCII`, `QARRAY:<k>`, `QLONG`
- Output: `[QPARAM]`, `[QSEP]`, `[QMETA]` (o la representación que se elija, pero única por este paso).
- Depends-on: 06

9) 08-path-structure-normalizer.md → 08_path_structure_normalizer.py
- Scope: Colapsar `//` y `/.`, preservar `..` sin resolver, reconstruir path canónico.
- Emits: `MULTIPLESLASH`, `DOTDOT`, `HOME`
- Depends-on: 06

10) 09-header-unfold-obs-fold.md → 09_header_unfold_obs_fold.py
- Scope: Detectar y desplegar obs-fold (continuaciones con SP/HTAB), normalizar separador a un solo espacio, tratar CR/LF incrustados.
- Emits: `OBSFOLD`, `BADCRLF`, `BADHDRCONT`
- Depends-on: 01

11) 10-header-normalization-and-duplicates.md → 10_header_normalizer.py
- Scope: Lowercase nombres, validar token, ordenar/emisión, merge de duplicados cuando corresponde, hop-by-hop.
- Emits: `BADHDRNAME:<name>`, `DUPHDR:<name>`, `HOPBYHOP:<name>`
- Depends-on: 09

12) 11-whitespace-collapse.md → 11_whitespace_collapse.py
- Scope: Colapsar runs de `\t` y espacios a un único espacio en campos textuales ya canónicos (p. ej., valores de headers), sin alterar estructura.
- Emits: `WSPAD`
- Depends-on: 09–10

13) 12-dangerous-characters-and-script-mixing.md → 12_dangerous_chars.py
- Scope: Detectar caracteres peligrosos y mezcla de alfabetos (tras decodes previos) en `[URL]`, `[QPARAM]`/`[QUERY]`, `[HEADER]`.
- Emits: `ANGLE`, `QUOTE`, `SEMICOLON`, `PAREN`, `BRACE`, `PIPE`, `BACKSLASH`, `SPACE` (solo URL), `NUL`, `QNUL`, `MIXEDSCRIPT`
- Depends-on: 06, 07, 10

14) 13-absolute-url-builder.md → 13_absolute_url_builder.py
- Scope: Construir `U:scheme://host[:port]/path?query` desde origin/absolute/authority-form, validar contra `Host`, IDNA.
- Emits: `HOSTMISMATCH`, `IDNA`, `BADHOST`
- Depends-on: 10 (headers normalizados), 08 (path), 07 (query)

15) 14-length-bucketing.md → 14_length_bucketing.py
- Scope: Métricas y bucketing de longitud (`PLEN`, `PMAX`, `HCNT`, `HLEN`) post-normalización.
- Emits: `PLEN:{len}@{bucket}`, `PMAX:{len}@{bucket}`, `HCNT:{n}`, `HLEN:{len}@{bucket}` (como líneas de métrica, no flags de error)
- Depends-on: 08, 10, 11

99) 99-flags-rollup-aggregator.md → 99_flags_rollup.py
- Scope: Agregar todas las flags emitidas por pasos anteriores en una línea global `FLAGS:[...]`, deduplicada y ordenada.
- Emits: `FLAGS:[...]`
- Depends-on: todos los anteriores

## Overlaps resolved (old → new)

- `normalizar-flags.md` + `flags-por-rarezas.md` → `99-flags-rollup-aggregator.md` (política de emisión y agregación global; detección queda en pasos dedicados 04–15).
- `encodings-y-decodificaciones-raras.md` → dividido en `03-bytes-to-unicode.md`, `05-percent-decode-once.md`, `06-html-entity-decode-once.md` (cada uno con sus flags: `BADUTF8`, `DOUBLEPCT`/`PCTSLASH`/`PCTBACKSLASH`, `HTMLENT`).
- `percent-decode-una-vez.md` → `05-percent-decode-once.md` (solo percent; sin HTML entities ni flags nucleares ajenos).
- `query-decodificar-una-vez.md` + `separadores-parseo-robusto.md` → `07-query-parser-and-flags.md` (separadores, tokenización, decode por token, Q*).
- `colapsar-slash-dot-no-resolver-dotdot.md` → `08-path-structure-normalizer.md`.
- `lineas-plegadas-obs-fold.md` → `09-header-unfold-obs-fold.md`.
- `headers-nombres-orden-duplicados.md` → `10-header-normalization-and-duplicates.md`.
- `separadores-espacios.md` → `11-whitespace-collapse.md` (aplicar después del unfold/merge; sin tocar delimitadores).
- `caracteres-peligrosos-script-mixing.md` → `12-dangerous-characters-and-script-mixing.md` (sin decode propio; depende de 06/07/10).
- `url-absoluta-desde-relativa.md` → `13-absolute-url-builder.md`.
- `longitudes-bucketing.md` → `14-length-bucketing.md`.
- `no-traducir-plus-espacio.md` → se integra como regla explícita en `05-percent-decode-once.md` y `07-query-parser-and-flags.md` (no requiere paso aparte).
- `shape-longitud-redaccion.md` → (opcional, futuro) dividir en `value-redaction-and-shapes.md` si se implementa; no incluido en este conjunto base para mantener una .py por función ortogonal.

## Step contracts (per-file template)

Cada nueva spec debe definir, al inicio, esta plantilla:

- Input lines considered: e.g., `[URL]`, `[QPARAM]`, `[HEADER]`.
- Transformations allowed: exact rules; what is preserved.
- Flags emitted: complete list, parámetros si aplica.
- Dependencies: prior steps required and why.
- Idempotence: proof sketch or rationale.
- Examples: mínimo 3 casos con entrada/salida.

## Migration guide (docs-only)

1) Mantener los `.md` actuales como referencia histórica.
2) Crear nuevos `.md` listados arriba con la plantilla de contrato clara.
3) En cada `.md` nuevo, añadir una sección "Supersedes" con los nombres de los `.md` antiguos que cubre total/parcialmente.
4) Al implementar, crear un `.py` por cada `.md` nuevo, respetando dependencias y sin reintroducir solapamientos.



## Existing implementations → proposed steps (keep all functionality)

This section maps current code to the new single-responsibility steps. The intent is to reuse code with minimal refactors, splitting where a file currently mixes concerns.

- 00_framing_cleanup.py
  - Reuse: `neuralshield.preprocessing.steps.pre_parse_over_raw.RemoveFramingArtifacts`
  - Notes: Behavior already matches edge-only cleanup (BOM, control chars at edges).

- 01_request_structurer.py
  - Reuse: `neuralshield.preprocessing.steps.pre_parse_over_raw.RequestStructurer`
  - Notes: Output schema `[METHOD]`, `[URL]`, `[QUERY]`, `[HEADER]` is aligned. Keep HTML-entity–aware splitting for initial `[QUERY]` tokens.

- 02_eol_annotator.py
  - Reuse: `neuralshield.preprocessing.steps.pre_parse_over_raw.LineJumpCatcher`
  - Notes: Already emits `EOL_BARELF`, `EOL_BARECR`, `EOLMIX`, `EOL_EOF_NOCRLF`. Ensure it remains purely annotative (no content rewriting) and idempotent.

- 03_bytes_to_unicode.py
  - Reuse: N/A (new). If upstream inputs are already `str`, this step can be a no-op scaffold that only detects/flags invalid UTF-8 if integrating a bytes path later.

- 04_unicode_nfkc.py
  - Reuse: from `neuralshield.preprocessing.steps.flag_normalizer.FlagsNormalizer` helpers:
    - `unicode_normalizer` (FULLWIDTH), `control_char_detector` (CONTROL)
  - Refactor: Extract these into a dedicated step that operates on `[URL]` and `[QUERY]` (and optionally header names if desired later). Do not perform percent or HTML entity decoding here.

- 05_percent_decode_once.py
  - Reuse: from `FlagsNormalizer.percent_encoding_analyzer`
  - Refactor: Convert to a pure, single-responsibility decode-once step:
    - Apply exactly-once percent-decoding per component.
    - Keep `%00` undecoded.
    - Detect residual valid `%hh` → `DOUBLEPCT`.
    - Add path-specific residual flags `PCTSLASH`/`PCTBACKSLASH` per spec (new; not in current code).
    - Avoid coupling to NFKC-change heuristics; decoding policy should not depend on previous normalization to ensure determinism.

- 06_html_entity_decode_once.py
  - Reuse: from `FlagsNormalizer.html_entity_decoder`
  - Refactor: Isolate into its own step, executed after 05; respect delimiter preservation policy.

- 07_query_parser.py
  - Reuse: `neuralshield.preprocessing.steps.query_processor.QueryProcessor`
  - Notes: Already implements separator heuristics (`&` vs `;&`), exact-once per-token decoding, multiplicidad, and Q* flags. Output `[QPARAM]`, `[QSEP]`, `[QMETA]` aligns with a single step owning query representation.
  - Minor alignment: Ensure `+` remains literal (already covered). Keep HTML-entity protected splitting consistent with earlier steps.

- 08_path_structure_normalizer.py
  - Reuse: `neuralshield.preprocessing.steps.path_structure_normalizer.PathStructureNormalizer`
  - Notes: Already collapses `//`, removes `/.`, preserves `..`; emits `MULTIPLESLASH`, `DOTDOT`, `HOME`.

- 09_header_unfold_obs_fold.py
  - Reuse: N/A (new). To be implemented per spec `lineas-plegadas-obs-fold.md`.

- 10_header_normalizer.py
  - Reuse: N/A (new). To be implemented per spec `headers-nombres-orden-duplicados.md`.

- 11_whitespace_collapse.py
  - Reuse: N/A (new). To be implemented per spec `separadores-espacios.md` (headers/values whitespace).

- 12_dangerous_chars.py
  - Reuse: `neuralshield.preprocessing.steps.dangerous_character_detector.DangerousCharacterDetector`
  - Notes: Already flags `ANGLE`, `QUOTE`, `SEMICOLON` (skip for headers), `PAREN`, `BRACE`, `PIPE`, `BACKSLASH`, `SPACE` (URL only), `NUL`/`QNUL`, and `MIXEDSCRIPT`. Ensure it runs after 06/07/10 so it analyzes normalized content.

- 13_absolute_url_builder.py
  - Reuse: N/A (new). To be implemented per spec `url-absoluta-desde-relativa.md`.

- 14_length_bucketing.py
  - Reuse: Partial from `QueryProcessor` (`QLONG` for values). Extend to compute `PLEN`/`PMAX` (paths) and `HCNT`/`HLEN` (headers) per spec `longitudes-bucketing.md`.

- 99_flags_rollup.py
  - Reuse: N/A (new). Aggregates all flags into a global `FLAGS:[...]`, deduped and sorted; existing steps currently emit per-line flags only.

### Compatibility notes (to avoid loss of behavior)

- Preserve the existing line-based flag emission immediately after the content line in each detection step (idempotent, no duplication on re-run).
- Where `FlagsNormalizer` mixed responsibilities, split its helpers into 04, 05, 06 while keeping the same flag names and semantics.
- Keep `QueryProcessor` as the single owner of query representation and Q* flags; earlier steps must not alter `[QUERY]/[QPARAM]` semantics.
- `DangerousCharacterDetector` remains detection-only; do not decode in that step.

