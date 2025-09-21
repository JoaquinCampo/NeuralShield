### 04 Unicode NFKC and Control Detection

**Step Contract:**
- Input lines considered: `[URL]`, `[QUERY]` lines
- Transformations allowed: NFKC normalization of content
- Flags emitted: `FULLWIDTH`, `CONTROL`
- Dependencies: 01 (request structurer)
- Idempotence: multiple applications produce same result

---

### Scope

Apply Unicode NFKC normalization to structural fields and detect anomalies:
- Normalize fullwidth characters to their standard equivalents
- Detect control characters (Unicode category Cc)
- Emit flags for detected anomalies

---

### Rules

1. **NFKC Normalization**:
   - Apply `unicodedata.normalize("NFKC", text)` to URL and QUERY content
   - **Always preserve original encoding evidence**: Flag any transformation
   - Detect fullwidth characters before normalization (U+FF00–U+FFEF range)

2. **Fullwidth Detection and Flagging**:
   - Check for fullwidth characters before normalization
   - Emit `FULLWIDTH` flag if any detected or if normalization changed the text
   - **Security rationale**: Fullwidth chars often used for filter evasion

3. **Control Character Detection**:
   - Scan for Unicode category Cc characters in normalized text
   - Include `%00` sequences (null bytes) without decoding them
   - Emit `CONTROL` flag if any detected
   - **Security rationale**: Control chars indicate potential injection attempts

4. **Flag Emission**:
   - Emit flags immediately after the processed line
   - Sort flags alphabetically
   - Skip lines that are already flag lines

---

### Examples

Input:
```
[METHOD] GET
[URL] /ｐａｔｈ/file
[QUERY] param=％76alue
[HEADER] Host: example.com
```

Output:
```
[METHOD] GET
[URL] /path/file
FULLWIDTH
[QUERY] param=value
FULLWIDTH
[HEADER] Host: example.com
```

Control character example:
```
[URL] /path%00file
```

Output:
```
[URL] /path%00file
CONTROL
```

---

### Implementation Notes

- Only process `[URL]` and `[QUERY]` lines
- Pass through `[METHOD]` and `[HEADER]` lines unchanged
- Do not decode percent-encoding or HTML entities - that's for later steps
- Log detected anomalies for debugging

---

### Supersedes

- `unicode_normalizer` and `control_char_detector` from `FlagsNormalizer`

---

### Good vs Bad Examples

Good (ASCII-only URL and queries):
```
[URL] /ascii/path
[QUERY] key=value
```

Bad (fullwidth characters normalized → flagged):
```
[URL] /ｐａｔｈ/file
FULLWIDTH
```

Bad (control characters present → flagged):
```
[QUERY] name=alice%00
CONTROL
```
