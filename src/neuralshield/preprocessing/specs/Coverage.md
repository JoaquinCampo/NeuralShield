| RFC Reference & Rule | Status | Implementation / Notes |
| --- | --- | --- |
| §2.2 – Accept bare LF as line terminator | ⚠️ Partial | `RequestStructurer.process` splits on `"\n"` but leaves stray `\r` intact (`src/neuralshield/preprocessing/steps/01_request_structurer.py:29`). |
| §2.2 – Flag/replace bare CR outside content | ✅ Implemented | RequestStructurer normalises bare CR and emits `BARECR` (`src/neuralshield/preprocessing/steps/01_request_structurer.py:219`). |
| §2.2 – Detect extra leading/trailing CRLF around requests | ✅ Implemented | Framing cleanup trims surplus CR/LF (`src/neuralshield/preprocessing/steps/00_framing_cleanup.py:66`). |
| §2.2 – Forbid whitespace between start-line and first header (ignore offending lines) | ✅ Implemented | RequestStructurer/obs-fold skip leading blank lines before headers (`src/neuralshield/preprocessing/steps/01_request_structurer.py:203`, `src/neuralshield/preprocessing/steps/02_header_unfold_obs_fold.py:36`). |
| §2.3 – HTTP-version grammar (`HTTP/` + DIGIT.DIGIT) | ✅ Implemented | Strict regex validation with `BADVERSION` evidence (`src/neuralshield/preprocessing/steps/01_request_structurer.py:139`). |
| §3.1 – Request-line structure (method SP target SP version) | ✅ Implemented | Strict token count enforced in `_rule_3_1_parse_request_line` (`src/neuralshield/preprocessing/steps/01_request_structurer.py:85`). |
| §3.1 – Method token must be recognised | ✅ Implemented | Allowed set enforced in `VALID_METHODS` (`src/neuralshield/preprocessing/steps/01_request_structurer.py:10`). |
| §3.1 – Lenient separators MAY be accepted (flag if used) | ✅ Implemented | Non-SP separators trigger `LENIENTSEP` (`src/neuralshield/preprocessing/steps/01_request_structurer.py:117`). |
| §3.2 – No whitespace in request-target | ✅ Implemented | Linear whitespace surfaces as `TARGETSPACE` (`src/neuralshield/preprocessing/steps/01_request_structurer.py:126`). |
| §3.2 – Host header MUST be present | ✅ Implemented | Missing/invalid host yields `BADHOST` (`src/neuralshield/preprocessing/steps/06_absolute_url_builder.py:360`). |
| §3.2 – Host MUST match authority when provided | ✅ Implemented | Form-specific checks emit `HOSTMISMATCH`, `HOSTMISSING`, etc. (`src/neuralshield/preprocessing/steps/06_absolute_url_builder.py:321`). |
| §3.2 – Host MUST be empty when authority absent (asterisk-form) | ✅ Implemented | Non-empty host flagged as `HOSTNOTEMPTY` (`src/neuralshield/preprocessing/steps/06_absolute_url_builder.py:386`). |
| §3.2 – CONNECT authority-form consistency | ✅ Implemented | CONNECT targets validated against Host header with mismatch evidence (`src/neuralshield/preprocessing/steps/06_absolute_url_builder.py:334`). |
| §5 – Header line MUST follow `field-name ":" ...` | ✅ Implemented | HeaderFieldSyntax flags `BADFIELD` (`src/neuralshield/preprocessing/steps/12_header_field_syntax.py:38`). |
| §5 – No whitespace before colon | ✅ Implemented | HeaderFieldSyntax flags `PRECOLONWS` (`src/neuralshield/preprocessing/steps/12_header_field_syntax.py:44`). |
| §5 – obs-fold forbidden (flag when present) | ✅ Implemented | Continuations emit `OBSFOLD` / `BADHDRCONT` (`src/neuralshield/preprocessing/steps/02_header_unfold_obs_fold.py:18`). |
| §6 – `Transfer-Encoding` vs `Content-Length` conflicts | ✅ Implemented | `BodyFramingInspector` emits `TECLCONFLICT` (`src/neuralshield/preprocessing/steps/13_body_framing_inspector.py:101`). |
| §6 – HTTP/1.0 request MUST NOT send `Transfer-Encoding` | ✅ Implemented | `BodyFramingInspector` flags `TEHTTP10` when version is HTTP/1.0 (`src/neuralshield/preprocessing/steps/13_body_framing_inspector.py:104`). |
| §6 – `Content-Length` value validation & agreement | ✅ Implemented | `BodyFramingInspector` validates/compares CL headers (`src/neuralshield/preprocessing/steps/13_body_framing_inspector.py:86`). |
| §6 – `Transfer-Encoding` must end with `chunked` | ✅ Implemented | `BodyFramingInspector` adds `TEBADEND` when chains end incorrectly (`src/neuralshield/preprocessing/steps/13_body_framing_inspector.py:107`). |
