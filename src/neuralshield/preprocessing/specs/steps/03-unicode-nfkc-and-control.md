### 04 Unicode NFKC and Control Detection

**Step Contract:**

- Input lines considered: `[URL]`, `[QUERY]` lines
- Transformations allowed: NFKC normalization of content
- Flags emitted: `FULLWIDTH`, `CONTROL`
- Dependencies: 01 (request structurer)
- Idempotence: multiple applications produce same result

---

### Scope

Apply Unicode NFKC normalization to structural fields and detect comprehensive Unicode security anomalies:

- Normalize fullwidth characters to their standard equivalents
- Detect control characters (Unicode category Cc)
- Detect zero-width and formatting characters (invisible text manipulation)
- Detect bidirectional text controls (text direction attacks)
- Detect mathematical alphanumeric symbols (homoglyph attacks)
- Detect private use and invalid Unicode characters
- Emit flags for all detected anomalies

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

4. **Zero-width and Formatting Character Detection**:

   - Detect invisible zero-width characters: U+200B (ZWSP), U+200C (ZWNJ), U+200D (ZWJ), U+FEFF (BOM)
   - Detect bidirectional text controls: U+202A–U+202E (RTL override, etc.)
   - Emit `UNICODE_FORMAT` flag if any detected
   - **Security rationale**: Invisible characters can hide malicious content or manipulate text rendering

5. **Mathematical Alphanumeric Detection**:

   - Detect mathematical alphanumeric symbols: U+1D400–U+1D7FF (𝐀𝐁𝐂..., etc.)
   - Emit `MATH_UNICODE` flag if any detected
   - **Security rationale**: Mathematical symbols visually similar to letters can create homoglyph attacks

6. **Private Use and Invalid Character Detection**:

   - Detect private use characters: U+E000–U+F8FF
   - Detect non-characters: U+FFFE, U+FFFF, and other invalid code points
   - Emit `INVALID_UNICODE` flag if any detected
   - **Security rationale**: Private use and invalid characters may contain hidden malicious data

7. **Flag Emission**:
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

Zero-width character example:

```
[URL] /path/file‌hidden
```

Output:

```
[URL] /path/file‌hidden
UNICODE_FORMAT
```

Mathematical unicode example:

```
[URL] /𝐩𝐚𝐭𝐡/file
```

Output:

```
[URL] /𝐩𝐚𝐭𝐡/file
MATH_UNICODE
```

Invalid unicode example:

```
[URL] /path/file󾠀
```

Output:

```
[URL] /path/file󾠀
INVALID_UNICODE
```

Multiple anomalies example:

```
[URL] /ｐａｔｈ%00𝐟𝐢𝐥𝐞‌
```

Output:

```
[URL] /path%00𝐟𝐢𝐥𝐞‌
CONTROL FULLWIDTH INVALID_UNICODE MATH_UNICODE UNICODE_FORMAT
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
