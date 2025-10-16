# RFC 9112 Implementation Plan

This plan maps each outstanding requirement from `RFC9112.md` to concrete
pipeline work. It names the target step (new or existing), the expected flags,
and the regression fixture that will exercise the behaviour.

## Legend

- **Step** — Pipeline component to add or extend (final module names follow the
  `NN_description.py` pattern).
- **Rules** — Shorthand reference to `RFC9112.md`.
- **Action** — Implementation notes.
- **Flag(s)** — New or existing evidence markers to emit.
- **Fixture** — `.in/.out` pair under `src/neuralshield/preprocessing/test`.

## Planned Work

| Step | Rules | Action | Flag(s) | Fixture |
| --- | --- | --- | --- | --- |
| 00 Framing Cleanup (existing) | §2.2 bare CR | Normalise bare `\r` (implemented in RequestStructurer `_normalize_line_endings`) and surface evidence. | `BARECR` | `line_terminators`, `comprehensive_test` |
| 01 Request Structurer (existing) | §2.3 version grammar, §3.1 separators, §3.2 whitespace | Tighten HTTP-version regex, accept lenient separators but add evidence, reject whitespace in request-target. | `LENIENTSEP`, `BADVERSION`, `TARGETSPACE` | `request_line_separators`, `comprehensive_test` |
| 06 Absolute URL Builder (existing) | §3.2 host presence/match (all forms) | Validate host header across origin/absolute/authority/asterisk forms, ensure empty host when required. | `BADHOST`, `HOSTMISMATCH`, `HOSTMISSING`, `HOSTNOTEMPTY`, `EMPTYHOST`, `IDNA` | `host_authority_matrix`, `origin_missing_host`, `asterisk_host`, `connect_mismatch` |
| 12 Header Field Syntax (new) | §5 field-name grammar, colon spacing | New step after `HeaderUnfoldObsFold` to enforce header name token rules and whitespace before colon. | `BADFIELD`, `PRECOLONWS`, `BADFIELDNAME` | `header_field_syntax` |
| 13 Body Framing Inspector (new) | §6 TE vs CL, HTTP/1.0 + TE, CL validity | Inspect header set for invalid combinations and numeric ranges. | `TECLCONFLICT`, `TEHTTP10`, `BADCL`, `DUPCL` | `body_framing_headers` |
| 14 Transfer Coding Sanitiser (new) | §7 chunked parameters, duplicate codings, TE header rules | Parse `Transfer-Encoding`, disallow `chunked;param`, duplicate `chunked`, and enforce `TE`/`Connection` coupling. | `CHUNKEDPARAM`, `CHUNKEDDUP`, `TECHUNKED`, `TEMISSCONN` | `transfer_coding_rules` |
| 15 Incomplete Message Sentinel (future) | §8 incomplete headers/body | Placeholder step — will require streaming context; defer but keep fixture ready for future integration. | `INCOMPLETE_HEADERS`, etc. | `incomplete_messages` (TODO) |

## Test Roadmap

- Populate each fixture under `src/neuralshield/preprocessing/test/in` and `out`
  with current pipeline behaviour; update expected outputs alongside step
  implementations.
- Expand `tests/preprocessing/` with focused unit tests per step once the
  implementation lands.
