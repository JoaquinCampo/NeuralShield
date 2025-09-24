### 13 Absolute URL Builder

**Step Contract:**

- Input lines considered: `[METHOD]`, `[URL]`, `[HEADER]` lines
- Transformations allowed: construct absolute URL from components (additive only)
- New lines added: `[URL_ABS]` with canonical absolute URL
- Global flags emitted: `HOSTMISMATCH`
- Inline flags emitted: `IDNA`, `BADHOST`
- Dependencies: 10 (header normalization), 08 (path), 07 (query)
- Idempotence: multiple applications produce same result

---

### Scope

Construct canonical absolute URLs from HTTP request components (without mutating originals):

- Build `scheme://host[:port]/path[?query]` from origin-form requests
- Handle absolute-form, authority-form, and asterisk-form targets
- Validate Host header consistency with security flag emission
- Apply IDNA encoding for internationalized domains with evidence flags
- Emit single `[URL_ABS]` line with complete canonical URL
- Emit global flags for security issues and processing status
- Preserve all original request structure for forensic analysis

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

### Flag Emission Strategy

**Global Flags** (emitted at end of request output):

- `HOSTMISMATCH`: Absolute-form target host differs from Host header (security issue)

**Inline Flags** (emitted with affected component):

- `IDNA`: Host header contained Unicode characters, converted to punycode (evidence flag)
- `BADHOST`: Host header is missing, invalid, or malformed (validation failure)

### New Structured Lines

- `[URL_ABS] <canonical-url>`: Canonical absolute URL representation
  - Added when URL construction succeeds
  - Uses normalized components from previous steps
  - Never replaces the original `[URL]` line

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
[HEADER] host: пример.com IDNA
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
[HEADER] host: BADHOST
```

**Invalid Host Format**:

```
[HEADER] host: invalid..domain
```

Output:

```
[HEADER] host: invalid..domain BADHOST
```

**Malformed Port**:

```
[HEADER] host: example.com:abc
```

Output:

```
[HEADER] host: example.com:abc BADHOST
```

---

### Security Considerations

- **Host Header Injection**: Validate host format to prevent injection
- **SSRF Prevention**: Canonical URLs help identify suspicious targets
- **Consistency Validation**: Detect potential proxy manipulation

---

### Implementation Approach

**Flag Emission Strategy:**

- `HOSTMISMATCH`: Global flag for request-level security issues (relationship between components)
- `IDNA`: Inline flag with affected header (evidence of Unicode processing)
- `BADHOST`: Inline flag with affected header (validation failure of specific component)
- Hybrid approach balances ML parseability with debugging clarity
- Follows NeuralShield patterns (similar to Step 9: inline + global flags)

**Integration with Pipeline:**

- Complements Steps 8-11 security validations
- Provides canonical URLs for downstream processing
- Enables SSRF detection and host header attack prevention
- Supports international domain security through IDNA

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
```

Bad (missing Host in origin-form):

```
[METHOD] GET
[URL] /ok
[HEADER] host: BADHOST
```
