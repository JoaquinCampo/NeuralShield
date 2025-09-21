### 13 Absolute URL Builder

**Step Contract:**
- Input lines considered: `[METHOD]`, `[URL]`, `[HEADER]` lines
- Transformations allowed: construct absolute URL from components (additive only)
- Flags emitted: `HOSTMISMATCH`, `IDNA`, `BADHOST`, `URLBUILT`
- Dependencies: 10 (header normalization), 08 (path), 07 (query)
- Idempotence: multiple applications produce same result

---

### Scope

Construct canonical absolute URLs from HTTP request components (without mutating originals):
- Build `scheme://host[:port]/path[?query]` from origin-form requests
- Handle absolute-form, authority-form, and asterisk-form targets
- Validate Host header consistency
- Apply IDNA encoding for internationalized domains
- Emit single `[URL_ABS]` line with complete URL
- Emit `URLBUILT` to indicate synthesis occurred

---

### Request Target Forms (RFC 9110)

**Origin-form**: `/path?query` (most common)
- Extract host from `Host` header
- Infer scheme from context (default: `http`)

**Absolute-form**: `http://host:port/path?query` (proxy requests)
- Parse all components from target
- Validate against `Host` header

**Authority-form**: `host:port` (CONNECT method)
- Used for tunnel establishment
- No path/query components

**Asterisk-form**: `*` (OPTIONS method)
- Server-wide options request
- Combine with Host header

---

### URL Construction Rules

1. **Scheme Determination**: Default to `http` unless configured otherwise
2. **Host Extraction**: From `Host` header (origin-form) or target (absolute-form)
3. **Port Handling**: Omit default ports (80 for http, 443 for https)
4. **Path Normalization**: Use normalized path from step 08
5. **Query Reconstruction**: Rebuild from parsed parameters if available
6. **IDNA Processing**: Convert Unicode domains to ASCII

---

### Flags Emitted

- `HOSTMISMATCH`: Absolute-form target host differs from Host header
- `IDNA`: Host header contained Unicode characters, converted to punycode
- `BADHOST`: Host header is missing, invalid, or malformed

---

### Examples

Origin-form with default port:
```
[METHOD] GET
[URL] /api/users
[HEADER] host: example.com:80
```

Output:
```
[METHOD] GET
[URL] /api/users
[URL_ABS] http://example.com/api/users
[HEADER] host: example.com:80
```

Origin-form with non-standard port:
```
[METHOD] GET
[URL] /app
[HEADER] host: api.example.com:8080
```

Output:
```
[METHOD] GET
[URL] /app
[URL_ABS] http://api.example.com:8080/app
[HEADER] host: api.example.com:8080
```

Absolute-form with Host mismatch:
```
[METHOD] GET
[URL] http://target.com/path
[HEADER] host: different.com
```

Output:
```
[METHOD] GET
[URL] http://target.com/path
[URL_ABS] http://target.com/path
[HEADER] host: different.com
HOSTMISMATCH
```

Unicode domain (IDNA):
```
[METHOD] GET
[URL] /search
[HEADER] host: пример.com
```

Output:
```
[METHOD] GET
[URL] /search
[URL_ABS] http://xn--e1afmkfd.com/search
[HEADER] host: пример.com
IDNA
```

Authority-form (CONNECT):
```
[METHOD] CONNECT
[URL] database.internal:5432
```

Output:
```
[METHOD] CONNECT
[URL] database.internal:5432
[URL_ABS] database.internal:5432
```

Asterisk-form (OPTIONS):
```
[METHOD] OPTIONS
[URL] *
[HEADER] host: api.example.com
```

Output:
```
[METHOD] OPTIONS
[URL] *
[URL_ABS] http://api.example.com/*
[HEADER] host: api.example.com
```

---

### Implementation Details

1. **Host Parsing**: Extract host and optional port from Host header
2. **Validation**: Check for required components based on request method
3. **Port Normalization**: Remove default ports, preserve non-standard
4. **IDNA Conversion**: Apply punycode encoding to Unicode domains
5. **Consistency Checking**: Compare absolute-form with Host header

---

### Error Handling

**Missing Host** (origin-form):
```
[METHOD] GET
[URL] /path
```

Output:
```
[METHOD] GET
[URL] /path
BADHOST
```

**Invalid Host Format**:
```
[HEADER] host: invalid..domain
```

Output:
```
[HEADER] host: invalid..domain
BADHOST
```

**Malformed Port**:
```
[HEADER] host: example.com:abc
```

Output:
```
[HEADER] host: example.com:abc
BADHOST
```

---

### Security Considerations

- **Host Header Injection**: Validate host format to prevent injection
- **SSRF Prevention**: Canonical URLs help identify suspicious targets
- **Consistency Validation**: Detect potential proxy manipulation

---

### Supersedes

- New step implementing `url-absoluta-desde-relativa.md` specification

---

### Good vs Bad Examples

Good (origin-form with valid Host):
```
[METHOD] GET
[URL] /ok
[HEADER] host: example.com
[URL_ABS] http://example.com/ok
URLBUILT
```

Bad (missing Host in origin-form):
```
[METHOD] GET
[URL] /ok
BADHOST
```
