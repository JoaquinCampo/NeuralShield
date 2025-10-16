# Preprocessing Pipeline Rule Map

This table summarises which RFC 9112 rules (or related specifications) each
preprocessing step enforces and the flags or records emitted. It mirrors the
style of `Coverage.md` but focuses on implementation responsibilities rather
than completion status.

| Step Module | RFC / Spec Rule | Behaviour / Flags |
| --- | --- | --- |
| `00_framing_cleanup.py` | Pre-RFC hygiene | Trim BOM/edge controls and `_rule_trim_surplus_crlf` removes extra framing CRLF. |
| `01_request_structurer.py` | §2.2 – bare CR tolerance | `_rule_2_2_detect_bare_cr` adds `BARECR` without mutating payload. Emits `[VERSION]` for downstream inspectors. |
|  | §3.1 – lenient separators | `_rule_3_1_parse_request_line` flags `LENIENTSEP` when separators exceed single SP. |
|  | §3.2 – target whitespace | `_rule_3_2_detect_target_whitespace` flags `TARGETSPACE`. |
|  | §2.3 – HTTP-version grammar | `_rule_2_3_validate_http_version` flags `BADVERSION`. |
|  | Structuring | Emits `[METHOD]`, `[URL]`, `[QUERY]`, `[HEADER]` scaffold for downstream steps. |
| `02_header_unfold_obs_fold.py` | §5.2 – obs-fold handling | Unfolds continuations, flags `OBSFOLD`, `BADCRLF`, `BADHDRCONT`. |
| `03_header_normalization_duplicates.py` | §5 token rules, §7 | Lowercases names (`HDRNORM`), validates tokens (`BADHDRNAME`), merges duplicates (`DUPHDR`, `HDRMERGE`), tags hop-by-hop headers (`HOPBYHOP`), emits `[HAGG]`, `[HGF]`. |
| `04_whitespace_collapse.py` | Spec step – header value whitespace | Collapses extra whitespace, flags `WSPAD` when adjustments occur (skips redacted values). |
| `13_body_framing_inspector.py` | §6 – body framing headers | Validates `Content-Length` (BADCL, CLMISMATCH), checks TE/CL conflicts (`TECLCONFLICT`), forbids TE on HTTP/1.0 (`TEHTTP10`), ensures TE ends with `chunked` (`TEBADEND`), emits `[BFR]`. |
| `05_dangerous_characters_script_mixing.py` | Spec step – dangerous characters & mixed scripts | Flags context-aware anomalies (`ANGLE`, `QUOTE`, `NUL`, `PIPE`, `MIXEDSCRIPT`, etc.) across URL/QUERY/HEADER lines. |
| `06_absolute_url_builder.py` | §3.2 – Host requirements | Validates origin/absolute/authority/asterisk forms: `HOSTMISSING`, `HOSTMISMATCH`, `HOSTNOTEMPTY`, `EMPTYHOST`, `BADHOST`, `IDNA`. Emits `[URL_ABS]`. |
| `07_unicode_nkfc_and_control.py` | Spec step – Unicode normalisation | Applies NFKC, flags `FULLWIDTH`, `CONTROL`, `UNICODE_FORMAT`, `MATH_UNICODE`, `INVALID_UNICODE`. |
| `08_percent_decode_once.py` | Spec step – controlled percent decode | Performs selective decode, flags `DOUBLEPCT`, `PCTSLASH`, `PCTBACKSLASH`, `PCTSPACE`, `PCTCONTROL`, `PCTNULL`, `PCTSUSPICIOUS`. |
| `09_html_entity_decode_once.py` | Spec step – HTML entity detection | Flags `HTMLENT` for URL/QUERY lines without decoding the entity. |
| `10_query_parser_and_flags.py` | Spec step – query semantics | Splits parameters with heuristics (`QSEMISEP`, `QRAWSEMI`), flags `QREPEAT:<key>`, `QBARE`, `QEMPTYVAL`, `QNUL`, `QNONASCII`, `QARRAY:<key>`, `QLONG`, redacts/shapes sensitive values, emits `[QSEP]`, `[QMETA]`. |
| `11_path_structure_normalizer.py` | Spec step – path normalisation | Collapses redundant segments while flagging traversal evidence (`MULTIPLESLASH`, `DOTCUR`, `DOTDOT`, `HOME`). |

## Test Fixtures Highlighting These Rules

- `line_terminators`, `request_line_separators`, `host_authority_matrix`, `origin_missing_host`, `asterisk_host`, `connect_mismatch`, `header_field_syntax`, `body_framing_headers`, `transfer_coding_rules`, `extra_crlf`, `start_line_whitespace` ensure the pipeline conforms to the mapped rules.
- `comprehensive_test` exercises the end-to-end flow across multiple rule categories.

Keep `Coverage.md` as the authoritative status tracker; update both files in tandem when new rules or flags are introduced.
