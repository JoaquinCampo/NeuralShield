### 00 Framing Cleanup

**Step Contract:**
- Input lines considered: raw HTTP request string
- Transformations allowed: remove BOM and control characters only at absolute edges
- Flags emitted: `BOMREMOVED`, `EDGECTRLREMOVED`
- Dependencies: none (first step)
- Idempotence: multiple applications produce same result

---

### Scope

Remove framing artifacts from the absolute borders of the HTTP request string (and flag any change):
- BOM (Byte Order Mark) at the beginning
- Non-printable control characters (Unicode category Cc, excluding `\t\r\n`) at the beginning and end

Preserve all content within the HTTP message structure and only modify the absolute edges to ensure robust parsing downstream.

---

### Rules

1. **Evidence Preservation**: ALWAYS flag when a removal occurs
2. **BOM Removal**: If the string starts with `\ufeff`, remove it and emit `BOMREMOVED`
3. **Leading Control Chars**: Remove Unicode category Cc characters from the beginning (excluding `\t\r\n`) and emit `EDGECTRLREMOVED` if any were removed
4. **Trailing Control Chars**: Remove Unicode category Cc characters from the end (excluding `\t\r\n`) and emit `EDGECTRLREMOVED` if any were removed
5. **Preservation**: Leave all internal content untouched

---

### Flags Emitted

- `BOMREMOVED`: A BOM was present and removed
- `EDGECTRLREMOVED`: One or more edge control characters were removed

---

### Examples

Input with BOM and control chars:
```
\ufeff\x01GET /path HTTP/1.1\r\nHost: example.com\r\n\r\n\x02
```

Output:
```
GET /path HTTP/1.1\r\nHost: example.com\r\n\r\n
BOMREMOVED EDGECTRLREMOVED
```

---

### Implementation Notes

- Emit flags summarizing what was removed
- Should log telemetry about what was removed for debugging
- Must be idempotent: running twice produces same result

---

### Good vs Bad Examples

Good (no framing artifacts):
```
GET /ok HTTP/1.1\r\nHost: example.com\r\n\r\n
```

Bad (BOM and edge controls present → flagged and removed):
```
\ufeff\x01GET /ok HTTP/1.1\r\nHost: example.com\r\n\r\n\x02
BOMREMOVED EDGECTRLREMOVED
```

---

### Supersedes

- Part of original `RemoveFramingArtifacts` functionality
