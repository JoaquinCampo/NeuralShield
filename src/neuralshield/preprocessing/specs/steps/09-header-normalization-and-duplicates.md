### 10 Header Normalization and Duplicates

**Step Contract:**
- Input lines considered: `[HEADER]` lines (after obs-fold unfolding)
- Transformations allowed: normalize names, handle duplicates, validate format
- Flags emitted: `BADHDRNAME:<name>`, `DUPHDR:<name>`, `HOPBYHOP:<name>`
- Dependencies: 09 (header unfold obs-fold)
- Idempotence: multiple applications produce same result

---

### Scope

Normalize header names and handle duplicate headers according to HTTP semantics:
- Lowercase header names for consistent processing
- Validate header name format (RFC 9110 token rules)
- Merge or flag duplicate headers based on semantics
- Detect hop-by-hop headers in requests
- Emit headers in canonical sorted order

---

### Normalization Rules

1. **Name Lowercasing**: Convert all header names to lowercase
2. **Whitespace Trimming**: Remove leading/trailing spaces from names and values  
3. **Token Validation**: Ensure names contain only valid token characters
4. **Canonical Ordering**: Sort headers by name (except `set-cookie`)
5. **Transformation Flagging**: Flag ANY normalization that changes original format
6. **Evidence Preservation**: Maintain indicators of original suspicious formatting

---

### Valid Header Name Characters

Per RFC 9110 token definition:
- **Allowed**: `!#$%&'*+-.^_`|~` plus alphanumeric
- **Forbidden**: Control characters, separators, spaces
- **Special Case**: Underscores may be flagged depending on policy

---

### Duplicate Handling

**Comma-Mergeable Headers** (merge with `,`):
- `accept`, `accept-encoding`, `accept-language`
- `cache-control`, `pragma`, `link`
- `www-authenticate` (with syntax respect)

**Never Merge**:
- `set-cookie` (maintain separate lines in arrival order)
- Other headers (flag as duplicate but keep separate)

---

### Flags Emitted

- `BADHDRNAME:<name>`: Header name contains invalid characters
- `DUPHDR:<name>`: Duplicate header detected (after merge if applicable)  
- `HOPBYHOP:<name>`: Hop-by-hop header found in request
- `HDRNORM`: Header names were normalized (case changes detected)
- `HDRMERGE`: Headers were merged due to duplication

---

### Hop-by-Hop Headers

Headers that should not appear in requests:
- `connection`, `te`, `upgrade`, `trailer`

---

### Examples

Basic normalization:
```
[HEADER] Host: Example.com
[HEADER] User-Agent: Mozilla/5.0
```

Output:
```
[HEADER] host: Example.com
[HEADER] user-agent: Mozilla/5.0
```

Duplicate mergeable headers:
```
[HEADER] Accept: text/html
[HEADER] Accept: application/json
```

Output:
```
[HEADER] accept: text/html, application/json
DUPHDR:accept
```

Invalid header name:
```
[HEADER] X_Custom: value
```

Output:
```
[HEADER] x_custom: value
BADHDRNAME:x_custom
```

Hop-by-hop in request:
```
[HEADER] Connection: keep-alive
[HEADER] Host: example.com
```

Output:
```
[HEADER] connection: keep-alive
[HEADER] host: example.com
HOPBYHOP:connection
```

Set-cookie preservation:
```
[HEADER] Set-Cookie: session=abc123
[HEADER] Set-Cookie: theme=dark
```

Output:
```
[HEADER] set-cookie: session=abc123
[HEADER] set-cookie: theme=dark
```

---

### Implementation Details

1. **Two-Pass Processing**: First pass collects and normalizes, second pass emits
2. **Merge Logic**: Apply comma-joining only to known-safe headers
3. **Order Preservation**: Maintain `set-cookie` order, sort others
4. **Flag Generation**: Track anomalies during processing
5. **Case Sensitivity**: All name comparisons case-insensitive

---

### Edge Cases

- **Empty Values**: Headers with empty values are valid
- **Whitespace Values**: Leading/trailing spaces trimmed from values
- **Mixed Case**: `Content-Type`, `content-type`, `CONTENT-TYPE` all normalized
- **Unknown Headers**: Headers not in merge list are flagged if duplicated

---

### Security Considerations

- **Duplicate Attacks**: Multiple headers can confuse downstream processing
- **Name Validation**: Invalid characters may indicate injection attempts
- **Hop-by-hop**: Presence in requests may indicate proxy manipulation

---

### Supersedes

- New step implementing `headers-nombres-orden-duplicados.md` specification

---

### Good vs Bad Examples

Good (valid names, no duplicates):
```
[HEADER] host: example.com
[HEADER] user-agent: Test/1.0
```

Bad (invalid name and duplicates):
```
[HEADER] X_Custom: v
[HEADER] Accept: text/html
[HEADER] Accept: */*
BADHDRNAME:x_custom
DUPHDR:accept
```
