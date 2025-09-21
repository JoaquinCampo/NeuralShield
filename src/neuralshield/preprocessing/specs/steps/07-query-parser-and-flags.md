### 07 Query Parser and Flags

**Step Contract:**
- Input lines considered: `[QUERY]` lines from request structurer
- Transformations allowed: parse query parameters and emit structured format
- Flags emitted: `QSEMISEP`, `QRAWSEMI`, `QBARE`, `QEMPTYVAL`, `QREPEAT:<k>`, `QNUL`, `QNONASCII`, `QARRAY:<k>`, `QLONG`
- Dependencies: 06 (HTML entity decode once), 05 (percent decode once)
- Idempotence: multiple applications produce same result

---

### Scope

Comprehensive query parameter parsing with anomaly detection (preservation-first):
- Intelligent separator detection (`&` vs `;` heuristics)
- No decoding here: decoding evidence and flags are handled in step 05
- Multiplicidad preservation and repetition detection
- Security-focused parameter analysis
- Output format transformation to `[QPARAM]`, `[QSEP]`, `[QMETA]`

---

### Separator Detection Heuristics

1. **Default**: Use `&` as primary separator
2. **Semicolon Dominant**: If `;` appears and pattern is `k=v(;k=v)+` with few `&`, use mixed `;&` separator and emit `QSEMISEP`
3. **Raw Semicolon**: If `;` appears but doesn't match dominant pattern, keep `&` separator and emit `QRAWSEMI`

---

### Parameter Analysis

For each parameter, detect:
- **Structure**: Bare keys (no `=`), empty values (`key=`)
- **Encoding**: Rely on PCT* and HTMLENT flags produced earlier; do not decode
- **Content**: Null bytes (from decoding step's `PCTNULL` → map to `QNUL` when value contains NUL), non-ASCII characters
- **Format**: Array notation (`key[]`), long values
- **Repetition**: Multiple occurrences of same key

---

### Flags Emitted

**Separator Flags**:
- `QSEMISEP`: Semicolon recognized as dominant separator
- `QRAWSEMI`: Semicolon present but not recognized as separator

**Parameter Flags**:
- `QBARE`: Parameter without `=` (bare key)
- `QEMPTYVAL`: Parameter with `=` but empty value
- `QREPEAT:<k>`: Key appears multiple times
- `QNUL`: Value contains null byte after decode
- `QNONASCII`: Key or value contains non-ASCII characters
- `QARRAY:<k>`: Key has array suffix `[]`
- `QLONG`: Value exceeds length threshold (default 1024 bytes)
- `DOUBLEPCT`: Double percent encoding detected

---

### Output Format

Replaces `[QUERY]` lines with structured format:

```
[QPARAM] {key} [flags...]
[QSEP] [separator_flags...]
[QMETA] count=N [global_flags...]
```

---

### Examples

Basic ampersand-separated query:
```
[QUERY] name=John
[QUERY] age=25
[QUERY] city=
```

Output:
```
[QPARAM] name
[QPARAM] age  
[QPARAM] city QEMPTYVAL
[QMETA] count=3 QEMPTYVAL
```

Semicolon-dominant pattern:
```
[QUERY] mode=1;user=alice;token=xyz
```

Output:
```
[QPARAM] mode
[QPARAM] user
[QPARAM] token
[QSEP] QSEMISEP
[QMETA] count=3
```

Complex query with anomalies:
```
[QUERY] items[]=1
[QUERY] items[]=2
[QUERY] bare
[QUERY] encoded=%252F
[QUERY] unicode=αβγ
```

Output:
```
[QPARAM] items[] QARRAY:items[] QREPEAT:items[]
[QPARAM] items[] QARRAY:items[] QREPEAT:items[]
[QPARAM] bare QBARE
[QPARAM] encoded DOUBLEPCT
[QPARAM] unicode QNONASCII
[QMETA] count=5 QARRAY:items[] QBARE DOUBLEPCT QNONASCII QREPEAT:items[]
```

---

### Implementation Notes

- Uses HTML-entity aware splitting to preserve entities during tokenization
- Applies exact-once percent decoding per token after splitting
- Tracks key repetitions for multiplicidad analysis
- Configurable length threshold for `QLONG` detection
- Maintains order and preserves all parameter instances

---

### Supersedes

- Original `QueryProcessor` with enhanced format and comprehensive flag detection

---

### Good vs Bad Examples

Good (well-formed key/value pairs):
```
[QUERY] x=1
[QUERY] y=2
```

Bad (structure anomalies):
```
[QUERY] bare
QBARE
[QUERY] empty=
QEMPTYVAL
```
