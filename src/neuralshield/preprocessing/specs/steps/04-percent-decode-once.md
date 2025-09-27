### 05 Percent Decode Once

**Step Contract:**
- Input lines considered: `[URL]`, `[QUERY]` lines
- Transformations allowed: apply percent-decode exactly once with context awareness
- Flags emitted: `DOUBLEPCT`, `PCTSLASH`, `PCTBACKSLASH`
- Dependencies: 04 (Unicode NFKC)
- Idempotence: multiple applications produce same result

---

### Scope

Apply percent-decode exactly once per component with intelligent preservation:
- Decode `%hh` sequences to their character equivalents
- Preserve dangerous characters that should remain encoded
- Detect double/multiple encoding attempts
- Use context-aware decoding (URLs vs queries)

---

### Rules

1. **Single Pass**: Apply percent-decode exactly once per component
2. **Context Awareness**: URLs preserve more characters than queries
3. **Null Preservation**: Always preserve `%00` (null bytes)
4. **Space Preservation**: Preserve `%20` in URLs, decode in queries
5. **Control Preservation**: Preserve control characters (`%01`-`%1F`) in URLs
6. **Double Encoding Detection**: Flag when valid `%hh` sequences remain after decoding
7. **Delimiter Detection**: Flag preserved path delimiters (`%2F`, `%5C`)

---

### Preservation Rules by Context

**URL Context** (conservative):
- Preserve: `%00`, `%20`, `%01`-`%1F`, `%2F`, `%5C`, and other suspicious encodings
- Decode: Only clearly safe alphanumeric sequences
- Flag: All preserved sequences with specific flags

**Query Context** (conservative - changed from permissive):
- Preserve: `%00`, `%20`, control characters, and suspicious encodings
- Decode: Only clearly safe alphanumeric sequences  
- Flag: All preserved sequences to maintain evidence of original encoding

---

### Flags Emitted

- `DOUBLEPCT`: Valid percent sequences remain after one decode pass
- `PCTSLASH`: Path contains `%2F` after decoding
- `PCTBACKSLASH`: Path contains `%5C` after decoding
- `PCTSPACE`: Contains `%20` (space encoding detected)
- `PCTCONTROL`: Contains control character encodings (`%01`-`%1F`)
- `PCTNULL`: Contains `%00` (null byte encoding detected)
- `PCTSUSPICIOUS`: Contains other suspicious percent encodings

---

### Examples

Double encoding in query:
```
[QUERY] param=foo%252Ebar
```

Output:
```
[QUERY] param=foo%2Ebar
DOUBLEPCT
```

Spaces in URL (preserved and flagged):
```
[URL] /path%20with%20spaces
```

Output:
```
[URL] /path%20with%20spaces
PCTSPACE
```

Spaces in query (preserved and flagged):
```
[QUERY] text=hello%20world
```

Output:
```
[QUERY] text=hello%20world
PCTSPACE
```

Control characters in URL (preserved and flagged):
```
[URL] /path%01%09test
```

Output:
```
[URL] /path%01%09test
PCTCONTROL
```

Path delimiter preservation:
```
[URL] /path%2Ftraversal
```

Output:
```
[URL] /path%2Ftraversal
PCTSLASH
```

---

### Implementation Notes

- Uses regex pattern `%[0-9A-Fa-f]{2}` to identify valid sequences
- Invalid sequences like `%2G` are left unchanged
- Multi-level encoding is handled by applying decode twice when appropriate
- Context parameter determines preservation strategy

---

### Supersedes

- `percent_encoding_analyzer` from `FlagsNormalizer`

---

### Good vs Bad Examples

Good (no percent encodings):
```
[URL] /plain/path
[QUERY] a=b
```

Bad (suspicious encodings preserved and flagged):
```
[URL] /path%20with%20spaces
PCTSPACE
[QUERY] text=hello%20world
PCTSPACE
```
